use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use glidinghorse::causal::engine::CausalEngine;
use glidinghorse::causal::fused::FusedRootCauseEngine;
use glidinghorse::causal::store::CausalModelStore;
use glidinghorse::config::{McpServerConfig, McpStdioServerConfig};
use glidinghorse::core::agent_runner::TaskResult;
use glidinghorse::core::event_bus::{Event, EventBus};
use glidinghorse::core::sa::SupervisorAgent;
use glidinghorse::core::ApplicationPromptProfile;
use glidinghorse::gateway::UnifiedGateway;
use glidinghorse::graph_backend::{GraphBackend, PetgraphBackend, SkillGraphSnapshotBackend};
use glidinghorse::graph_features::features::FeatureExtractor;
use glidinghorse::knowledge_graph::code_ast::CodeAstExtractor;
use glidinghorse::knowledge_graph::store::KnowledgeGraphStore;
use glidinghorse::memory::consistency_engine::ConsistencyEngine;
use glidinghorse::memory::embedding_service::{
    create_embedding_service_from_config, FallbackEmbeddingService,
};
use glidinghorse::memory::hyperspace_store::HyperspaceStore;
use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l1_session::EvictionConfig;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::memory::memory_bus::MemoryBus;
use glidinghorse::memory::memory_manager::MemoryManager;
use glidinghorse::memory::scheduler::MemoryScheduler;
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
use tracing::{debug, info, warn};

use crate::config::CliConfig;

pub type StartupReporter = Arc<dyn Fn(&str, Option<f64>) + Send + Sync>;

fn report_startup(reporter: &Option<StartupReporter>, stage: &str, progress: Option<f64>) {
    if let Some(reporter) = reporter {
        reporter(stage, progress);
    }
}

#[cfg(target_os = "linux")]
fn current_rss_kb() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

#[cfg(not(target_os = "linux"))]
fn current_rss_kb() -> Option<u64> {
    None
}

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
    /// Persistent vector store; checkpointed after each task to bound WAL growth.
    vector_store: Arc<HyperspaceStore>,
    /// Embedding backend used by the vector store; kept for startup health probing
    /// and to expose the active provider to status/health surfaces.
    embedding: Arc<dyn glidinghorse::memory::embedding_service::EmbeddingService>,
    /// True once the embedding backend has been probed for connectivity this run.
    embedding_health_checked: AtomicBool,
    /// Set when the startup probe failed; makes `embedding_status()` report degraded.
    embedding_degraded: AtomicBool,
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
    learning_snapshot_max_files: usize,
    learning_snapshot_max_bytes: u64,
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
    ".pytest_cache/",
    ".venv/",
    "venv/",
    ".next/",
    ".gliding_horse/",
];

const GLIDINGCODE_WORKFLOW_SKILL_IRI: &str = "iri://skills/glidingcode-workflow";

fn glidingcode_learning_skill_node() -> glidinghorse::skill_graph::types::SkillGraphNode {
    use glidinghorse::skill_graph::types::{Skill5W2H, SkillGraphNode};

    SkillGraphNode::new(
        GLIDINGCODE_WORKFLOW_SKILL_IRI,
        "glidingcode software delivery workflow",
        "Application-level planning, implementation, executable verification, and acceptance workflow for software tasks.",
    )
    .with_5w2h(
        Skill5W2H::new(
            "software task delivery workflow",
            "Reuse CA-validated implementation and verification knowledge without bypassing current-task audit",
        )
        .with_phase("Plan")
        .with_phase("Do")
        .with_agent_role("PA"),
    )
    .with_tag("glidingcode")
    .with_tag("application-workflow")
    .with_tag("non-executable-learning-skill")
}

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

/// Hash the substantive workspace state for controlled replay comparability.
/// This deliberately runs only when a learning pair ID is supplied; normal
/// interactive tasks retain the constant-time workspace identity fingerprint.
fn workspace_state_fingerprint(
    root: &std::path::Path,
    exclusion_patterns: &[String],
    max_files: usize,
    max_bytes: u64,
) -> anyhow::Result<String> {
    use std::io::Read;

    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = pending.pop() {
        let mut entries = std::fs::read_dir(&dir)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                continue;
            }
            let relative = path.strip_prefix(root).unwrap_or(path.as_path());
            if file_type.is_dir() {
                if !code_path_is_excluded(relative, true, exclusion_patterns) {
                    pending.push(path);
                }
            } else if file_type.is_file()
                && !code_path_is_excluded(relative, false, exclusion_patterns)
            {
                files.push(path);
                if files.len() > max_files {
                    anyhow::bail!("workspace snapshot exceeds {max_files} files");
                }
            }
        }
    }
    files.sort();

    let mut total_bytes = 0u64;
    let mut digest = Sha256::new();
    for path in files {
        let relative = path
            .strip_prefix(root)
            .unwrap_or(path.as_path())
            .to_string_lossy()
            .replace('\\', "/");
        let metadata = std::fs::metadata(&path)?;
        total_bytes = total_bytes.saturating_add(metadata.len());
        if total_bytes > max_bytes {
            anyhow::bail!("workspace snapshot exceeds {max_bytes} bytes");
        }
        digest.update((relative.len() as u64).to_le_bytes());
        digest.update(relative.as_bytes());
        digest.update(metadata.len().to_le_bytes());
        let mut file = std::fs::File::open(&path)?;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
    }
    Ok(format!("sha256:{}", hex::encode(&digest.finalize()[..12])))
}

impl CodeCliEngine {
    pub fn new(config: CliConfig) -> anyhow::Result<Self> {
        Self::new_with_startup_reporter(config, None)
    }

    pub fn new_with_startup_reporter(
        mut config: CliConfig,
        startup_reporter: Option<StartupReporter>,
    ) -> anyhow::Result<Self> {
        let startup_started = Instant::now();
        report_startup(&startup_reporter, "Resolving workspace", None);
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
        // Load agent-os config before constructing memory layers so their
        // storage and capacity settings are effective from the first write.
        let loaded_settings = glidinghorse::config::Settings::load().ok();
        let settings = loaded_settings.clone().unwrap_or_default();

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

        let l0_file = std::path::Path::new(&l0_path).join("l0.redb");
        let l0_size_gib = std::fs::metadata(&l0_file)
            .map(|metadata| metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0);
        report_startup(
            &startup_reporter,
            &format!("Opening L0 memory ({l0_size_gib:.1} GiB)"),
            None,
        );
        let l0_repair_callback = startup_reporter.as_ref().map(|reporter| {
            let reporter = reporter.clone();
            Arc::new(move |progress: f64| {
                reporter("Recovering L0 after an unclean exit", Some(progress));
            }) as Arc<dyn Fn(f64) + Send + Sync>
        });

        let l0 = Arc::new(
            L0Store::with_config_and_repair_callback(
                glidinghorse::memory::l0_store::L0Config {
                    path: l0_path,
                    max_entries: settings.memory.l0.max_entries as usize,
                    compression: settings.memory.l0.compression,
                    blob_inline_threshold: settings.memory.l0.blob_inline_threshold,
                    cache_size_bytes: settings.memory.l0.cache_size_bytes,
                    quick_repair: settings.memory.l0.quick_repair,
                },
                l0_repair_callback,
            )
            .map_err(|e| anyhow::anyhow!("L0Store 创建失败: {}", e))?,
        );

        // ── Unified Oxigraph Store — shared across Blackboard, SkillGraphStore,
        //    ToolExecutor (KnowledgeGraphStore), and KnowledgeBridge so that all
        //    subsystems operate on the same RDF store via named-graph isolation.
        report_startup(&startup_reporter, "Opening unified knowledge graph", None);
        let unified = Arc::new(
            match &persistent_root {
                Some(root) => UnifiedGraphStore::new_persistent(root.join("unified-graph")),
                None => UnifiedGraphStore::new(),
            }
            .map_err(|e| anyhow::anyhow!("UnifiedGraphStore 创建失败: {}", e))?,
        );

        let l2 = Arc::new(
            Blackboard::with_store_and_queue_capacity(
                unified.store(),
                settings.memory.l2.sync_queue_capacity,
            )
            .map_err(|e| anyhow::anyhow!("Blackboard 创建失败: {}", e))?,
        );
        l2.set_max_memory_mb(settings.memory.l2.max_memory_mb);

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
        report_startup(&startup_reporter, "Opening semantic vector memory", None);
        let vector_store = Arc::new(
            HyperspaceStore::open(std::path::Path::new(&hyperspace_path), embed.clone())
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
        report_startup(&startup_reporter, "Opening ontology memory", None);
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
        proj.set_max_memory_mb(settings.memory.l3.max_memory_mb);
        let core_config = CoreConfig {
            max_node_size: settings.memory.l2.max_node_size,
            max_projection_size: agent_settings.max_projection_size,
            l1_token_budget: settings.memory.l1.max_tokens.max(1),
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
                        max_low_relevance_refs: l1.max_low_relevance_refs,
                        reload_preview_chars: l1.reload_preview_chars,
                    })
                } else {
                    None
                }
            },
            l1_max_low_relevance_refs: Some(settings.memory.l1.max_low_relevance_refs),
            l1_reload_preview_chars: Some(settings.memory.l1.reload_preview_chars),
        };
        let mm = Arc::new(tokio::sync::Mutex::new(MemoryManager::with_vector_store(
            l0.clone(),
            l2.clone(),
            proj.clone(),
            core_config.clone(),
            Some(vector_store.clone()),
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
                Ok(count) => {
                    info!(count, path = %skill_dir, "Loaded external skills from --skill-dir")
                }
                Err(error) => {
                    warn!(path = %skill_dir, %error, "Failed to load skills from --skill-dir")
                }
            }
        }

        let workspace_root = std::path::PathBuf::from(&config.workspace);
        let code_scan_exclude_patterns =
            load_code_scan_exclusions(&workspace_root, &settings.workspace.exclude_patterns);

        // ── TimelineStore (temporal event recording for graph mutations) ──
        // Created before SkillGraphStore so the store can attach it and record
        // every structural mutation (otherwise TL: pending stays at 0).
        report_startup(&startup_reporter, "Restoring timeline metadata", None);
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
        report_startup(
            &startup_reporter,
            "Restoring skill and knowledge graphs",
            None,
        );
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

        // This is application workflow knowledge, not an executable kernel
        // tool. Keeping it outside SkillRegistry prevents accidental syscall
        // authority while giving validated fragments one stable graph home.
        if skill_graph
            .get_skill(GLIDINGCODE_WORKFLOW_SKILL_IRI)
            .is_none()
        {
            if let Err(error) = skill_graph.register_skill(glidingcode_learning_skill_node()) {
                warn!(%error, "Failed to register glidingcode learning workflow skill");
            }
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
            // Restrict this bootstrap heuristic to the registered application
            // tool skills. Learned/generalized nodes are linked by governed
            // evolution; including the whole accumulated graph here made each
            // startup O(total_skill_count^2).
            let registered_iris = skills
                .list_all_skills()
                .into_iter()
                .map(|skill| skill.skill_iri)
                .collect::<std::collections::HashSet<_>>();
            let nodes = skill_graph
                .list_all_skills()
                .into_iter()
                .filter(|skill| registered_iris.contains(&skill.skill_iri))
                .collect::<Vec<_>>();
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
                        let already_linked = a.links.iter().any(|link| {
                            link.target_iri == b.skill_iri
                                && link.link_type == SkillLinkType::Related
                                && link.strength == strength
                                && link.description == desc
                        });
                        if !already_linked
                            && skill_graph
                                .add_link(
                                    &a.skill_iri,
                                    &b.skill_iri,
                                    SkillLinkType::Related,
                                    strength,
                                    &desc,
                                )
                                .is_ok()
                        {
                            link_count += 1;
                        }
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
        .with_application_prompt(glidingcode_prompt_profile())
        .with_prompt_loader(glidinghorse::core::prompt_loader::PromptLoader::new(
            Default::default(),
            tmpl.clone(),
        ))
        .with_workspace_root(workspace_root.clone())
        .with_token_optimization(settings.token_optimization.clone())
        .with_tool_result_router_settings(settings.tool_result_router.clone());

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
                .with_usage_persistence(l0.clone())
                .with_causal_engine(causal_engine.clone()),
        ));

        let event_bus = Arc::new(EventBus::new(100));
        // The mature Runner emits tool-call/result events through this bus.
        runner.set_event_bus(event_bus.clone());

        // ── MemoryScheduler with HyperspaceStore: activates context_request_with_decay ──
        let memory_bus = Arc::new(MemoryBus::new(event_bus.clone()));
        let consistency_engine = Arc::new(ConsistencyEngine::new(
            memory_bus.clone(),
            l0.clone(),
            l2.clone(),
            proj.clone(),
        ));
        let scheduler = Arc::new(MemoryScheduler::with_hyperspace(
            l0.clone(),
            l2.clone(),
            proj.clone(),
            consistency_engine.clone(),
            memory_bus.clone(),
            Some(vector_store.clone()),
        ));
        {
            let mut mm_lock = mm.blocking_lock();
            mm_lock.set_scheduler(scheduler.clone());
        }

        // TimelineStore EventBus subscription deferred — requires a Tokio runtime.
        // Subscribe via start_async_components() in process_task().

        // 初始化 WorkspaceMonitor — 从 settings.workspace 读取配置
        report_startup(&startup_reporter, "Opening workspace monitor", None);
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
                // Show the TUI before walking and indexing a large workspace.
                // The generic monitor starts the metadata-only scan when the
                // first async task context becomes available.
                defer_initial_scan: true,
                poll_interval_ms: settings.workspace.poll_interval_ms,
                debounce_ms: settings.workspace.debounce_ms,
                max_debounce_wait_ms: settings.workspace.max_debounce_wait_ms,
                initial_scan_wait_ms: settings.workspace.initial_scan_wait_ms,
                change_history_capacity: settings.workspace.change_history_capacity,
                effect_snapshot_max_files: settings.workspace.effect_snapshot_max_files,
                effect_snapshot_max_bytes: settings.workspace.effect_snapshot_max_bytes,
                exclude_patterns: settings.workspace.exclude_patterns.clone(),
                db_path: Some(ws_db_path),
                ..Default::default()
            };
            match WorkspaceMonitor::initialize(ws_config, Some(l2.clone()), Some(event_bus.clone()))
            {
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
            .with_unified_graph_store(unified_kg_store)
            .with_learning_mode(config.learning_mode);

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
        .with_memory(Some(l2), None, Some(scheduler))
        .with_execution_timeout(agent_settings.sa_execution_timeout_secs)
        .with_perception_hyperspace(vector_store.clone())
        .with_perception_store(Arc::new(runner_perception))
        .with_discovery_engine(discovery_engine.clone())
        .with_learning_mode(config.learning_mode)
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

        report_startup(&startup_reporter, "Finalizing TUI services", None);
        info!(
            model = %config.model,
            workspace = %config.workspace,
            max_iterations = config.max_iterations,
            mcp_servers = config.mcp_servers.len(),
            startup_ms = startup_started.elapsed().as_millis(),
            rss_kb = current_rss_kb().unwrap_or(0),
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
            vector_store: vector_store.clone(),
            embedding: embed,
            embedding_health_checked: AtomicBool::new(false),
            embedding_degraded: AtomicBool::new(false),
            feature_extractor,
            causal_engine,
            evolution_engine,
            timeline,
            core_config,
            oxi_store: unified.store(),
            code_scan_exclude_patterns,
            learning_snapshot_max_files: settings.workspace.learning_snapshot_max_files,
            learning_snapshot_max_bytes: settings.workspace.learning_snapshot_max_bytes,
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

    /// Probe the configured embedding backend exactly once per process so a
    /// dead Ollama/OneAPI endpoint is surfaced loudly on the first task instead
    /// of silently degrading every vector read to keyword hashing.
    async fn ensure_embedding_healthy(&self) {
        if self.embedding_health_checked.swap(true, Ordering::AcqRel) {
            return;
        }
        if self.embedding.provider() == "fallback" {
            return;
        }
        if let Err(error) = self.embedding.health_check().await {
            self.embedding_degraded.store(true, Ordering::Release);
            warn!(
                provider = self.embedding.provider(),
                %error,
                "语义检索已降级为关键词匹配: embedding 后端不可达"
            );
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
        self.ensure_embedding_healthy().await;
        self.ensure_skill_vectors_indexed().await;
        // 首次进入 async 上下文时完成 WorkspaceMonitor 的异步初始化
        if let Some(ref wm) = self.workspace_monitor {
            wm.start_async_components();
            // On first use, scan and watcher installation are already running
            // in the background. Later, rescan only as a watcher fallback.
            if wm.scan_complete() && !wm.watch_engine_active() {
                wm.rescan();
            }
            let indexed = wm.wait_for_initial_scan().await;
            debug!(
                indexed,
                generation = wm.generation(),
                "Workspace metadata readiness checked"
            );
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
            .with_original_task(user_input);
            let ctx = with_learning_experiment_constraints(
                with_glidingcode_task_constraints(ctx, user_input),
                &self.config,
                self.learning_snapshot_max_files,
                self.learning_snapshot_max_bytes,
            )
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
            let ctx = with_learning_experiment_constraints(
                with_glidingcode_task_constraints(ctx, user_input),
                &self.config,
                self.learning_snapshot_max_files,
                self.learning_snapshot_max_bytes,
            );
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
        if let (true, Ok(mut ee)) = (
            self.config.learning_mode.updates_learning(),
            self.evolution_engine.try_lock(),
        ) {
            let success = result.status == "completed" || result.status == "success";
            let mut affected_skill_iris = Vec::new();
            let mut skill_outcomes = Vec::new();

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
                let before = ee.get_usage_stats(&skill_iri);
                let mut usage = glidinghorse::skill_graph::evolution::UsageRecord::new(
                    &skill_iri,
                    &task_iri,
                    &action.agent_role,
                    action_success,
                )
                .with_context_tag(&result.status)
                .with_context_tag(&format!("tool:{}", action.tool_name))
                .with_context_tag(&format!(
                    "task-family:{}",
                    glidinghorse::core::policy_learning::learning_policy_context(user_input)
                ))
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
                    affected_skill_iris.push(skill_iri.clone());
                    let after = ee.get_usage_stats(&skill_iri);
                    let assessment = glidinghorse::skill_graph::evolution::assess_outcome(
                        before.success_rate,
                        after.success_rate,
                        before.total_usage,
                        after.total_usage,
                    );
                    skill_outcomes.push(serde_json::json!({
                        "skill_iri": skill_iri,
                        "task_iri": task_iri,
                        "action_success": action_success,
                        "before_usage": before.total_usage,
                        "before_success_rate": before.success_rate,
                        "after_usage": after.total_usage,
                        "after_success_rate": after.success_rate,
                        "success_rate_delta": assessment.success_rate_delta,
                        "evidence_verdict": assessment.verdict,
                        "duration_seconds": action.duration_secs,
                        "task_status": result.status,
                    }));
                }
            }

            if !skill_outcomes.is_empty() {
                let outcome_iri = format!("{}#skill-outcomes", task_iri);
                let outcome_json = serde_json::json!({
                    "@id": outcome_iri,
                    "@type": "SkillOutcomeEvidence",
                    "task_iri": task_iri,
                    "outcomes": skill_outcomes,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                });
                if let Err(error) = self.l2_bb.write_node(
                    &outcome_iri,
                    &outcome_json.to_string(),
                    &self.core_config,
                ) {
                    warn!(task_iri = %task_iri, %error, "Failed to persist skill outcome evidence");
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
                        // Causal knowledge fragments are governed proposals,
                        // not merely log entries. They become reusable graph
                        // knowledge only after explicit approval/validation.
                        let proposal_store = EvolutionProposalStore::new(self.l0.clone());
                        for suggestion in &suggestions {
                            if suggestion.patch.is_none() {
                                continue;
                            }
                            let Ok(serialized) = serde_json::to_vec(suggestion) else {
                                continue;
                            };
                            let key = format!(
                                "{}:causal-evolution:{}",
                                task_iri,
                                hex::encode(Sha256::digest(serialized))
                            );
                            if let Err(error) = proposal_store.create_or_get(
                                &key,
                                suggestion.clone(),
                                self.skill_graph.as_ref(),
                            ) {
                                warn!(task_iri = %task_iri, %error, "Failed to persist causal knowledge proposal");
                            }
                        }
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
            if let (true, Ok(mut ee)) = (
                self.config.learning_mode.updates_learning(),
                self.evolution_engine.try_lock(),
            ) {
                let affected_skill_iris = result
                    .tracked_actions
                    .iter()
                    .filter_map(|action| self.skills.skill_iri_for_tool_name(&action.tool_name))
                    .collect::<Vec<_>>();
                let improvements = ee
                    .suggest_improvements_for_skills(&affected_skill_iris)
                    .await;
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

        if let Err(error) = self.vector_store.checkpoint().await {
            warn!(task_iri = %task_iri, error = %error, "Failed to checkpoint hyperspace store after task");
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

    /// Active embedding provider ("ollama" | "oneapi" | "fallback") for status surfaces.
    pub fn embedding_provider(&self) -> &'static str {
        self.embedding.provider()
    }

    /// Semantic-search health: degraded when the configured backend is a placeholder
    /// or failed its startup connectivity probe.
    pub fn embedding_status(&self) -> &'static str {
        if self.embedding.provider() == "fallback"
            || self.embedding_degraded.load(Ordering::Acquire)
        {
            return "degraded";
        }
        if self.embedding_health_checked.load(Ordering::Acquire) {
            "healthy"
        } else {
            "unknown"
        }
    }

    pub async fn list_checkpoints(
        &self,
    ) -> anyhow::Result<Vec<glidinghorse::core::checkpoint::CheckpointData>> {
        let prefix = "iri://checkpoint/";
        let entries = self.l0.scan_iri_prefix(prefix, 100_000)?;
        let mut results: Vec<glidinghorse::core::checkpoint::CheckpointData> = entries
            .iter()
            .filter_map(|e| serde_json::from_str(&e.content).ok())
            .collect();
        results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        results.truncate(20);
        Ok(results)
    }

    /// Return task-level treatment evidence used to audit whether accumulated
    /// knowledge/skills helped a later task. Intermediate BizAgent traces are
    /// intentionally excluded from this prefix.
    pub fn list_learning_evaluations(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut results = self
            .l0
            .scan_iri_prefix("iri://learning/evaluations/", 10_000)?
            .into_iter()
            .filter_map(|entry| serde_json::from_str(&entry.content).ok())
            .collect::<Vec<serde_json::Value>>();
        results.sort_by(|left, right| {
            left.get("timestamp")
                .and_then(serde_json::Value::as_str)
                .cmp(&right.get("timestamp").and_then(serde_json::Value::as_str))
        });
        Ok(results)
    }

    /// Management-command fast path: open only the workspace's durable L0
    /// database. This avoids constructing embeddings, graph stores, templates,
    /// watchers and the complete TUI engine merely to read audit records.
    pub fn list_learning_evaluations_from_config(
        config: &CliConfig,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let Some(base) = config.data_dir.as_deref() else {
            return Ok(Vec::new());
        };
        let workspace = std::fs::canonicalize(&config.workspace)
            .unwrap_or_else(|_| std::path::PathBuf::from(&config.workspace));
        let digest = Sha256::digest(workspace.to_string_lossy().as_bytes());
        let l0_path = std::path::Path::new(base)
            .join(format!("workspace-{}", hex::encode(&digest[..12])))
            .join("l0");
        if !l0_path.exists() {
            return Ok(Vec::new());
        }
        let l0 = L0Store::new(&l0_path.to_string_lossy())?;
        let mut results = l0
            .scan_iri_prefix("iri://learning/evaluations/", 10_000)?
            .into_iter()
            .filter_map(|entry| serde_json::from_str(&entry.content).ok())
            .collect::<Vec<serde_json::Value>>();
        results.sort_by(|left, right| {
            left.get("timestamp")
                .and_then(serde_json::Value::as_str)
                .cmp(&right.get("timestamp").and_then(serde_json::Value::as_str))
        });
        Ok(results)
    }

    /// Aggregate treatment evidence by normalized family/mode/action and
    /// audit controlled replay pairs. Percentiles are reported only from
    /// observed samples; no counterfactual value is synthesized.
    pub fn summarize_learning_evaluations(&self) -> anyhow::Result<serde_json::Value> {
        let evaluations = self.list_learning_evaluations()?;
        Ok(Self::summarize_learning_evaluation_values(evaluations))
    }

    pub fn summarize_learning_evaluation_values(
        evaluations: Vec<serde_json::Value>,
    ) -> serde_json::Value {
        let percentile = |mut values: Vec<f64>, quantile: f64| -> Option<f64> {
            if values.is_empty() {
                return None;
            }
            values.sort_by(|left, right| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            });
            let index = ((values.len() as f64 * quantile).ceil() as usize)
                .saturating_sub(1)
                .min(values.len() - 1);
            Some(values[index])
        };

        let mut groups: std::collections::BTreeMap<
            (String, String, String),
            Vec<&serde_json::Value>,
        > = std::collections::BTreeMap::new();
        for evaluation in &evaluations {
            let family = evaluation
                .get("policy_context")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            let mode = evaluation
                .get("mode")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            let action = evaluation
                .get("policy_action")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            groups
                .entry((family.to_string(), mode.to_string(), action.to_string()))
                .or_default()
                .push(evaluation);
        }
        let arms = groups
            .into_iter()
            .map(|((family, mode, action), samples)| {
                let number = |sample: &&serde_json::Value, field: &str| {
                    sample.get(field).and_then(serde_json::Value::as_f64)
                };
                let rewards = samples
                    .iter()
                    .filter_map(|sample| {
                        sample
                            .get("reward")
                            .and_then(|reward| reward.get("total"))
                            .and_then(serde_json::Value::as_f64)
                    })
                    .collect::<Vec<_>>();
                let success_count = samples
                    .iter()
                    .filter(|sample| {
                        sample
                            .get("status")
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|status| matches!(status, "success" | "completed"))
                    })
                    .count();
                let metric = |field: &str| {
                    let values = samples
                        .iter()
                        .filter_map(|sample| number(sample, field))
                        .collect::<Vec<_>>();
                    serde_json::json!({
                        "p50": percentile(values.clone(), 0.50),
                        "p95": percentile(values, 0.95),
                    })
                };
                serde_json::json!({
                    "family": family,
                    "mode": mode,
                    "action": action,
                    "samples": samples.len(),
                    "success_rate": if samples.is_empty() { 0.0 } else { success_count as f64 / samples.len() as f64 },
                    "reward": {
                        "p50": percentile(rewards.clone(), 0.50),
                        "p95": percentile(rewards, 0.95),
                    },
                    "elapsed_ms": metric("elapsed_ms"),
                    "prompt_tokens": metric("prompt_tokens"),
                    "turn_count": metric("turn_count"),
                    "tool_call_count": metric("tool_call_count"),
                })
            })
            .collect::<Vec<_>>();

        let mut pair_groups: std::collections::BTreeMap<String, Vec<&serde_json::Value>> =
            std::collections::BTreeMap::new();
        for evaluation in &evaluations {
            if let Some(pair_id) = evaluation
                .pointer("/treatment/experiment_pair_id")
                .and_then(serde_json::Value::as_str)
            {
                pair_groups
                    .entry(pair_id.to_string())
                    .or_default()
                    .push(evaluation);
            }
        }
        let pairs = pair_groups
            .into_iter()
            .map(|(pair_id, samples)| {
                let mut modes = samples
                    .iter()
                    .filter_map(|sample| sample.get("mode").and_then(serde_json::Value::as_str))
                    .map(str::to_string)
                    .collect::<Vec<_>>();
                modes.sort();
                modes.dedup();
                let stable_field = |pointer: &str| {
                    let values = samples
                        .iter()
                        .filter_map(|sample| sample.pointer(pointer))
                        .map(serde_json::Value::to_string)
                        .collect::<std::collections::BTreeSet<_>>();
                    values.len() == 1
                        && values.iter().next().is_some_and(|value| {
                            value != "null" && !value.trim_matches('"').starts_with("unavailable:")
                        })
                };
                let mut issues = Vec::new();
                for (pointer, label) in [
                    ("/treatment/objective_fingerprint", "objective"),
                    ("/treatment/workspace_fingerprint", "workspace"),
                    ("/treatment/experiment_model", "model"),
                    ("/treatment/experiment_seed", "seed"),
                    ("/treatment/orchestration_mode", "orchestration_mode"),
                ] {
                    if !stable_field(pointer) {
                        issues.push(format!("missing_or_mismatched_{label}"));
                    }
                }
                let required_modes = ["active", "baseline", "shadow"];
                let missing_modes = required_modes
                    .iter()
                    .filter(|required| !modes.iter().any(|mode| mode == **required))
                    .copied()
                    .collect::<Vec<_>>();
                serde_json::json!({
                    "pair_id": pair_id,
                    "samples": samples.len(),
                    "modes": modes,
                    "comparable": issues.is_empty() && modes.len() >= 2,
                    "complete_three_arm_replay": issues.is_empty() && missing_modes.is_empty(),
                    "missing_modes": missing_modes,
                    "issues": issues,
                })
            })
            .collect::<Vec<_>>();

        serde_json::json!({
            "evaluation_count": evaluations.len(),
            "arms": arms,
            "pairs": pairs,
            "gate": {
                "same_family_only": true,
                "candidate_trial_after_same_family_baseline_samples": 1,
                "promotion_minimum_baseline_samples": 5,
                "promotion_minimum_candidate_samples": 5,
                "unpromoted_model_role": "shadow_or_bounded_candidate",
            }
        })
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
        self.ensure_embedding_healthy().await;
        self.ensure_skill_vectors_indexed().await;
        // Match the normal task path: TUI/resume must populate the inventory
        // before SA decides between direct execution and verify-first routing.
        if let Some(ref wm) = self.workspace_monitor {
            wm.start_async_components();
            if wm.scan_complete() && !wm.watch_engine_active() {
                wm.rescan();
            }
        }
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
        let ctx = with_learning_experiment_constraints(
            with_glidingcode_task_constraints(ctx, user_input),
            &self.config,
            self.learning_snapshot_max_files,
            self.learning_snapshot_max_bytes,
        );
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
        let post_task_started = Instant::now();
        self.record_tui_task_evolution(task_iri, user_input, &result)
            .await;
        let evolution_ms = post_task_started.elapsed().as_millis();
        self.write_task_metadata(task_iri, user_input, &result);
        let metadata_ms = post_task_started
            .elapsed()
            .as_millis()
            .saturating_sub(evolution_ms);
        let checkpoint_started = Instant::now();
        if let Err(error) = self.vector_store.checkpoint().await {
            warn!(task_iri = %task_iri, %error, "Failed to checkpoint TUI hyperspace store after task");
        }
        let checkpoint_ms = checkpoint_started.elapsed().as_millis();

        // A timeline snapshot represents graph evolution, not merely task
        // completion. Avoid serializing an identical full skill graph after
        // every TUI task when no governed graph mutation occurred.
        if self.timeline.pending_mutations() > 0 || self.timeline.snapshot_count() == 0 {
            let backend = SkillGraphSnapshotBackend::new(self.skill_graph.clone());
            self.timeline
                .create_snapshot(&backend, &format!("task:{}", result.status.as_str()));
        } else {
            debug!(task_iri = %task_iri, "Skipped unchanged skill-graph timeline snapshot");
        }
        info!(
            task_iri = %task_iri,
            evolution_ms,
            metadata_ms,
            checkpoint_ms,
            post_task_ms = post_task_started.elapsed().as_millis(),
            "Completed TUI post-task persistence"
        );

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
    async fn record_tui_task_evolution(
        &self,
        task_iri: &str,
        user_input: &str,
        result: &TaskResult,
    ) {
        if !self.config.learning_mode.updates_learning() {
            return;
        }
        let Ok(mut evolution) = self.evolution_engine.try_lock() else {
            return;
        };
        let mut affected_skill_iris = Vec::new();
        let mut skill_outcomes = Vec::new();
        for action in &result.tracked_actions {
            let Some(skill_iri) = self.skills.skill_iri_for_tool_name(&action.tool_name) else {
                warn!(task_iri = %task_iri, tool = %action.tool_name, "No skill IRI for TUI tracked action");
                continue;
            };
            let succeeded = matches!(
                action.status,
                glidinghorse::core::tracked_action::ActionStatus::Success
            );
            let before = evolution.get_usage_stats(&skill_iri);
            let mut usage = glidinghorse::skill_graph::evolution::UsageRecord::new(
                &skill_iri,
                task_iri,
                &action.agent_role,
                succeeded,
            )
            .with_context_tag(&result.status)
            .with_context_tag(&format!("tool:{}", action.tool_name))
            .with_context_tag(&format!(
                "task-family:{}",
                glidinghorse::core::policy_learning::learning_policy_context(user_input)
            ))
            .with_duration(action.duration_secs.ceil().min(u32::MAX as f64) as u32);
            if let Some(error) = action.error.as_deref() {
                usage = usage.with_error(error);
            }
            if let Err(error) = evolution.record_usage(usage) {
                warn!(task_iri = %task_iri, skill_iri = %skill_iri, %error, "Failed to record TUI skill usage");
            } else if !affected_skill_iris.contains(&skill_iri) {
                affected_skill_iris.push(skill_iri.clone());
                let after = evolution.get_usage_stats(&skill_iri);
                let assessment = glidinghorse::skill_graph::evolution::assess_outcome(
                    before.success_rate,
                    after.success_rate,
                    before.total_usage,
                    after.total_usage,
                );
                skill_outcomes.push(serde_json::json!({
                    "skill_iri": skill_iri,
                    "task_iri": task_iri,
                    "action_success": succeeded,
                    "before_usage": before.total_usage,
                    "before_success_rate": before.success_rate,
                    "after_usage": after.total_usage,
                    "after_success_rate": after.success_rate,
                    "success_rate_delta": assessment.success_rate_delta,
                    "evidence_verdict": assessment.verdict,
                    "duration_seconds": action.duration_secs,
                    "task_status": result.status,
                }));
            }
        }

        if !skill_outcomes.is_empty() {
            let outcome_iri = format!("{}#skill-outcomes", task_iri);
            let outcome = serde_json::json!({
                "@id": outcome_iri,
                "@type": "SkillOutcomeEvidence",
                "task_iri": task_iri,
                "outcomes": skill_outcomes,
                "entrypoint": "tui_or_resume",
                "timestamp": chrono::Utc::now().to_rfc3339(),
            });
            if let Err(error) =
                self.l2_bb
                    .write_node(&outcome_iri, &outcome.to_string(), &self.core_config)
            {
                warn!(task_iri = %task_iri, %error, "Failed to persist TUI skill outcome evidence");
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

        let improvements = evolution
            .suggest_improvements_for_skills(&affected_skill_iris)
            .await;
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

/// Application-level software-engineering contract. Kernel policy, role
/// permissions, and lifecycle rules remain authoritative and are injected
/// separately by AgentRunner.
fn glidingcode_prompt_profile() -> ApplicationPromptProfile {
    ApplicationPromptProfile::new(
        "glidingcode",
        "v1",
        r#"Software-engineering application rules:
- Treat the configured workspace as the only project scope. Ignore .git, .gliding_horse, caches, generated binaries, and unrelated projects unless the task explicitly names them.
- For maintenance, prefer targeted reads and minimal changes. For greenfield or broad feature work, do not use minimal-change guidance to reduce requested scope.
- DA must move from targeted inspection to incremental artifact creation early enough to leave time for executable verification; a read-only diagnosis is not a completed implementation or correction.
- Before writing or replacing tests, inspect the existing public API, fixtures, and test conventions and run the relevant baseline command. Extend the real interface; do not invent incompatible constructors, method names, or CLI wiring.
- Treat every non-zero build/test exit as an immediate repair signal. Fix the concrete failure before broad reading or adding more speculative tests.
- For code changes, identify affected files and the relevant test command before claiming completion.
- DA must report changed file paths, commands run, and command results. CA must verify the declared acceptance criteria with direct evidence.
- PA application extension: identify the smallest relevant code scope, target files, dependencies, and executable acceptance checks.
- DA application extension: implement the full declared scope, preserve unrelated behavior, and record changed artifacts plus verification commands. If blocked, report `FAILED:` instead of claiming a no-change fix.
- CA application extension: verify each declared acceptance criterion with concrete file or command evidence; do not invent coverage, performance, or security requirements.
- AA application extension: decide from the structured CA evidence. Do not modify code or explore files unless the runtime explicitly enables correction or challenge mode.
- Test failures, build failures, incomplete code analysis, or unavailable external services are evidence to report; do not hide them or claim success without a valid fallback.
- Code AST/knowledge-graph information is auxiliary evidence. Its absence or partial failure does not fail an ordinary file-editing task unless graph analysis is an explicit requirement.
- Do not modify repository metadata, internal runtime state, backup copies, or diagnostic-output files unless explicitly requested."#,
    )
    .with_optimized_contract(
        r#"Software-engineering application contract:

Scope
- Treat the configured workspace as the only project scope. Ignore .git, .gliding_horse, caches, generated binaries, and unrelated projects unless explicitly named.
- For maintenance, prefer targeted reads and the smallest sufficient change. For greenfield or explicitly broad feature work, "smallest change" must not reduce requested scope: implement every acceptance criterion with a coherent architecture.
- DA should establish the relevant file shape quickly, then create or modify artifacts incrementally. Do not spend an implementation pass only researching or repeatedly reading files.
- For tasks requiring tests or runnable acceptance checks, first inspect the existing public interfaces, fixtures, CLI parser, and test conventions and run the relevant baseline command. Then create a compatible test/check skeleton early and run a representative check before the midpoint of the execution budget; reserve the remaining budget for implementation, full verification, and repair.
- New tests must exercise the repository's real constructors, field names, methods, persistence model, and CLI entry point. Do not invent a parallel API merely to satisfy a guessed test shape.
- A non-zero build/test exit immediately moves work into repair: use its concrete traceback or diagnostic, fix that defect, and rerun the focused check before unrelated inspection or additional speculative tests.
- Do not re-read an entire file that DA just wrote when the needed facts are already in the active context; use targeted ranges or search for later inspection.
- Reserve enough execution budget for build/test and repair. If a recursive sub-task requests implementation, inspection alone is not completion.

Execution evidence
- PA identifies target files, dependencies, risks, and executable acceptance checks.
- DA reports changed paths, commands run, exit results, and any unverified assumption.
- A corrective DA is an executor, not another auditor: implement the identified gap, then verify it. If no change can be made, begin the final result with `FAILED:` and state the blocker; never report a read-only diagnosis as a completed fix.
- CA verifies every declared criterion with direct file or command evidence. A failed, skipped, unavailable, or incomplete check remains visible as such.
- AA decides only from structured CA evidence and the declared criteria. It does not silently repair, explore, or expand scope.

Engineering boundaries
- AST/knowledge-graph information is auxiliary evidence. Missing or partial graph data does not fail ordinary file-editing unless graph analysis is explicitly required.
- Do not invent coverage, performance, or security requirements. Do not modify repository metadata, runtime state, backup copies, or diagnostic-output files unless explicitly requested.
- If tests/builds/external services fail, report the exact failure and distinguish it from a code defect.

Required handoff shape
- `completion_state`: `complete`, `incomplete`, or `blocked`
- `criteria`: each declared criterion and status (`pass`, `fail`, `blocked`, or `unverified`)
- `evidence`: file paths, relevant excerpts, commands, and exit results
- `changes`: files actually changed (empty for read-only work)
- `verification`: checks actually executed and their results
- `pending_effects`: unresolved executable work as objects with `objective`, optional `target`, `reason`, and `effect_policy`
- `blockers`: exact external or capability blockers
- Put these fields in a machine-readable JSON object named `completion`; use an empty `pending_effects` array only when execution is complete.
- Final status must be supported by the evidence; never claim success from an absent check."#,
    )
}

/// Declare when the software application expects DA to leave concrete
/// workspace effects. The kernel consumes the generic `required_effect`
/// contract, while this application owns the domain/language classification.
fn with_glidingcode_task_constraints(
    ctx: glidinghorse::core::agent_runner::TaskContext,
    user_input: &str,
) -> glidinghorse::core::agent_runner::TaskContext {
    let normalized = user_input.to_lowercase();
    let explicitly_read_only = [
        "不要修改",
        "不修改任何文件",
        "只读分析",
        "仅分析",
        "do not modify",
        "without modifying",
        "read-only review",
    ]
    .iter()
    .any(|marker| normalized.contains(marker));
    let requests_change = [
        "创建",
        "编写",
        "实现",
        "修复",
        "修改",
        "开发",
        "优化",
        "重构",
        "新增",
        "增加",
        "删除",
        "生成",
        "搭建",
        "完善",
        "解决",
        "implement",
        "create",
        "build",
        "fix",
        "modify",
        "develop",
        "optimize",
        "refactor",
        "add",
        "remove",
        "generate",
        "write",
    ]
    .iter()
    .any(|marker| normalized.contains(marker));

    let conditional_change = [
        "检查并修复",
        "确认并修复",
        "核对并补齐",
        "验证并修复",
        "测试并修复",
        "分析并修复",
        "check and fix",
        "verify and repair",
        "validate and fix",
        "test and fix",
    ]
    .iter()
    .any(|marker| normalized.contains(marker));

    if explicitly_read_only {
        ctx.with_effect_policy(glidinghorse::core::effect::EffectPolicy::EvidenceOnly)
            .with_constraint("effect_policy", "evidence_only")
    } else if requests_change && conditional_change {
        ctx.with_effect_policy(
            glidinghorse::core::effect::EffectPolicy::conditional_workspace_mutation(
                "mutate only when verification finds a task-relevant defect",
            ),
        )
        .with_constraint("effect_policy", "conditional_workspace_mutation")
    } else if requests_change {
        ctx.with_effect_policy(
            glidinghorse::core::effect::EffectPolicy::required_workspace_mutation(),
        )
        // Preserve compatibility with checkpoints and older skill discovery.
        .with_constraint("required_effect", "workspace_mutation")
        .with_constraint("effect_policy", "required_workspace_mutation")
    } else {
        ctx
    }
}

fn with_learning_experiment_constraints(
    mut ctx: glidinghorse::core::agent_runner::TaskContext,
    config: &CliConfig,
    snapshot_max_files: usize,
    snapshot_max_bytes: u64,
) -> glidinghorse::core::agent_runner::TaskContext {
    let workspace_identity = std::fs::canonicalize(&config.workspace)
        .unwrap_or_else(|_| std::path::PathBuf::from(&config.workspace))
        .to_string_lossy()
        .to_string();
    let workspace_identity_fingerprint = format!(
        "sha256:{}",
        hex::encode(&Sha256::digest(workspace_identity.as_bytes())[..12])
    );
    let workspace_fingerprint = if config.learning_pair_id.is_some() {
        let root = std::path::Path::new(&config.workspace);
        let exclusions = load_code_scan_exclusions(root, &[]);
        workspace_state_fingerprint(root, &exclusions, snapshot_max_files, snapshot_max_bytes)
            .unwrap_or_else(|error| {
                let error_digest = Sha256::digest(error.to_string().as_bytes());
                format!("unavailable:sha256:{}", hex::encode(&error_digest[..12]))
            })
    } else {
        workspace_identity_fingerprint
    };
    ctx = ctx
        .with_constraint("learning_skill_iri", GLIDINGCODE_WORKFLOW_SKILL_IRI)
        .with_constraint("learning_model", &config.model)
        .with_constraint("learning_workspace_fingerprint", &workspace_fingerprint);
    if let Some(pair_id) = config.learning_pair_id.as_deref() {
        ctx = ctx.with_constraint("learning_pair_id", pair_id);
    }
    if let Some(seed) = config.learning_seed.as_deref() {
        ctx = ctx.with_constraint("learning_seed", seed);
    }
    ctx
}

#[cfg(test)]
mod tests {
    use super::{
        collect_workspace_code_files, glidingcode_learning_skill_node, glidingcode_prompt_profile,
        load_code_scan_exclusions, with_glidingcode_task_constraints, workspace_state_fingerprint,
        GLIDINGCODE_WORKFLOW_SKILL_IRI,
    };

    #[test]
    fn glidingcode_prompt_is_application_specific_and_preserves_kernel_boundary() {
        let prompt = glidingcode_prompt_profile().render();
        assert!(prompt.contains("Application: glidingcode"));
        assert!(prompt.contains("Software-engineering application contract"));
        assert!(prompt.contains("tests/builds") || prompt.contains("test command"));
        assert!(prompt.contains("AST/knowledge-graph"));
        assert!(prompt.contains("criteria"));
        assert!(prompt.contains("existing public interfaces"));
        assert!(prompt.contains("non-zero build/test exit"));
        assert!(!prompt.contains("Constitution"));
        assert!(!prompt.contains("PDCA"));
    }

    #[test]
    fn glidingcode_declares_workspace_effect_only_for_change_tasks() {
        let change = glidinghorse::core::agent_runner::TaskContext::new("t", "x", 10);
        let change = with_glidingcode_task_constraints(change, "实现一个可运行的功能并测试");
        assert_eq!(
            change
                .constraints
                .get("required_effect")
                .map(String::as_str),
            Some("workspace_mutation")
        );
        assert_eq!(
            change.effective_effect_policy(),
            glidinghorse::core::effect::EffectPolicy::required_workspace_mutation()
        );

        let review = glidinghorse::core::agent_runner::TaskContext::new("t", "x", 10);
        let review = with_glidingcode_task_constraints(review, "只读分析当前实现，不修改任何文件");
        assert!(!review.constraints.contains_key("required_effect"));
        assert_eq!(
            review.effective_effect_policy(),
            glidinghorse::core::effect::EffectPolicy::EvidenceOnly
        );

        let conditional = glidinghorse::core::agent_runner::TaskContext::new("t", "x", 10);
        let conditional =
            with_glidingcode_task_constraints(conditional, "检查并修复发现的实现问题");
        assert!(matches!(
            conditional.effective_effect_policy(),
            glidinghorse::core::effect::EffectPolicy::Conditional { .. }
        ));
    }

    #[test]
    fn validated_knowledge_has_a_non_executable_application_skill_home() {
        let node = glidingcode_learning_skill_node();
        assert_eq!(node.skill_iri, GLIDINGCODE_WORKFLOW_SKILL_IRI);
        assert!(node.tags.iter().any(|tag| tag == "application-workflow"));
        assert!(node
            .tags
            .iter()
            .any(|tag| tag == "non-executable-learning-skill"));
    }

    #[test]
    fn learning_summary_reports_percentiles_and_controlled_pair_comparability() {
        let sample = |mode: &str, elapsed_ms: u64, reward: f64| {
            serde_json::json!({
                "policy_context": "planning:v2:ops=test;kinds=data",
                "policy_action": if mode == "active" { "knowledge_first" } else { "baseline" },
                "mode": mode,
                "status": "success",
                "elapsed_ms": elapsed_ms,
                "prompt_tokens": 1000,
                "turn_count": 5,
                "tool_call_count": 3,
                "reward": {"total": reward},
                "treatment": {
                    "experiment_pair_id": "pair-1",
                    "experiment_seed": "fixed-42",
                    "experiment_model": "deepseek-test",
                    "workspace_fingerprint": "sha256:workspace",
                    "objective_fingerprint": "sha256:objective",
                    "orchestration_mode": "pdca"
                }
            })
        };
        let summary = super::CodeCliEngine::summarize_learning_evaluation_values(vec![
            sample("baseline", 300, 0.6),
            sample("shadow", 250, 0.7),
            sample("active", 200, 0.9),
        ]);

        assert_eq!(summary["evaluation_count"], 3);
        assert_eq!(summary["arms"].as_array().unwrap().len(), 3);
        assert_eq!(summary["pairs"][0]["comparable"], true);
        assert_eq!(summary["pairs"][0]["complete_three_arm_replay"], true);
        assert!(summary["pairs"][0]["issues"].as_array().unwrap().is_empty());
    }

    #[test]
    fn controlled_replay_fingerprint_tracks_content_but_ignores_runtime_state() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        std::fs::write(workspace.path().join("main.py"), "print('one')\n").unwrap();
        let exclusions = load_code_scan_exclusions(workspace.path(), &[]);
        let first =
            workspace_state_fingerprint(workspace.path(), &exclusions, 100, 1_000_000).unwrap();

        std::fs::create_dir_all(workspace.path().join(".gliding_horse")).unwrap();
        std::fs::write(
            workspace.path().join(".gliding_horse/runtime-state"),
            "changes between executions",
        )
        .unwrap();
        let runtime_changed =
            workspace_state_fingerprint(workspace.path(), &exclusions, 100, 1_000_000).unwrap();
        assert_eq!(first, runtime_changed);

        std::fs::write(workspace.path().join("main.py"), "print('two')\n").unwrap();
        let source_changed =
            workspace_state_fingerprint(workspace.path(), &exclusions, 100, 1_000_000).unwrap();
        assert_ne!(first, source_changed);
    }

    #[test]
    fn controlled_pair_rejects_unavailable_workspace_snapshot() {
        let sample = |mode: &str| {
            serde_json::json!({
                "policy_context": "planning:v2:ops=test;kinds=code",
                "policy_action": "baseline",
                "mode": mode,
                "status": "success",
                "reward": {"total": 1.0},
                "treatment": {
                    "experiment_pair_id": "pair-unavailable",
                    "experiment_seed": "42",
                    "experiment_model": "deepseek-test",
                    "workspace_fingerprint": "unavailable:sha256:reason",
                    "objective_fingerprint": "sha256:objective",
                    "orchestration_mode": "pdca"
                }
            })
        };
        let summary = super::CodeCliEngine::summarize_learning_evaluation_values(vec![
            sample("baseline"),
            sample("active"),
        ]);
        assert_eq!(summary["pairs"][0]["comparable"], false);
        assert_eq!(
            summary["pairs"][0]["issues"][0],
            "missing_or_mismatched_workspace"
        );
    }

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
