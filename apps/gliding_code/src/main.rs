use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "glidingcode",
    version = env!("CARGO_PKG_VERSION"),
    about = "Agent OS Console - AI Coding Assistant"
)]
struct Cli {
    #[arg(help = "Single prompt (omit for interactive mode)")]
    prompt: Option<String>,

    #[arg(
        short = 'm',
        long = "model",
        default_value = "deepseek-v4-flash",
        help = "Model to use"
    )]
    model: String,

    #[arg(
        short = 'w',
        long = "workspace",
        default_value = ".",
        help = "Working directory"
    )]
    workspace: String,

    #[arg(
        long = "max-iterations",
        default_value = "50",
        help = "Maximum ReAct turns per agent invocation (the PDCA task total can be higher)"
    )]
    max_iterations: u32,

    #[arg(
        long = "max-pdca-cycles",
        default_value = "7",
        help = "Maximum SA-level PDCA cycles (verify-first plans require a verification and fallback cycle)"
    )]
    max_pdca_cycles: u32,

    #[arg(
        long = "learning-mode",
        value_name = "MODE",
        help = "Continuous learning treatment: active, baseline, or shadow (env: GLIDING_LEARNING_MODE)"
    )]
    learning_mode: Option<String>,

    #[arg(
        long = "learning-pair-id",
        value_name = "ID",
        help = "Controlled replay pair identifier shared by baseline/shadow/active runs"
    )]
    learning_pair_id: Option<String>,

    #[arg(
        long = "learning-seed",
        value_name = "LABEL",
        help = "Audit label for fixed randomness/configuration in a controlled replay"
    )]
    learning_seed: Option<String>,

    #[arg(
        long = "api-key",
        help = "API key (takes precedence over DEEPSEEK_API_KEY env var)"
    )]
    api_key: Option<String>,

    #[arg(
        long = "api-url",
        help = "API URL (takes precedence over DEEPSEEK_API_URL env var)"
    )]
    api_url: Option<String>,

    #[arg(short = 'v', long = "verbose", help = "Show verbose logs")]
    verbose: bool,

    #[arg(long = "debug", help = "Show debug logs (more detailed)")]
    debug: bool,

    #[arg(
        long = "resume",
        help = "Resume task from checkpoint (provide task_iri)"
    )]
    resume: Option<String>,

    #[arg(long = "list-checkpoints", help = "List all checkpoints")]
    list_checkpoints: bool,

    #[arg(
        long = "list-learning-evaluations",
        help = "List durable continuous-learning treatment outcomes as JSON"
    )]
    list_learning_evaluations: bool,

    #[arg(
        long = "summarize-learning-evaluations",
        help = "Report same-family P50/P95 treatment metrics and paired replay comparability"
    )]
    summarize_learning_evaluations: bool,

    #[arg(
        long = "list-offline-retrieval-evaluations",
        help = "List durable independent-label retrieval verdicts as JSON"
    )]
    list_offline_retrieval_evaluations: bool,

    #[arg(
        long = "verify-task-evidence",
        value_name = "TASK_IRI",
        help = "Verify one task's hash-linked execution evidence and terminal seal as JSON"
    )]
    verify_task_evidence: Option<String>,

    #[arg(
        long = "inspect-ann-health",
        help = "Run a read-only exact-vs-ANN health probe and persist aggregate audit evidence"
    )]
    inspect_ann_health: bool,

    #[arg(
        long = "list-learning-health",
        help = "List family-scoped learning health reports as JSON"
    )]
    list_learning_health: bool,

    #[arg(
        long = "list-learning-deltas",
        help = "List auditable retrieval-policy delta lifecycles as JSON"
    )]
    list_learning_deltas: bool,

    #[arg(
        long = "rollback-learning-delta",
        value_name = "ID",
        help = "Record an approved rollback for a frozen learning delta (requires --delta-approver)"
    )]
    rollback_learning_delta: Option<String>,

    #[arg(
        long = "delta-approver",
        value_name = "IDENTITY",
        help = "Audit identity for --rollback-learning-delta; not authentication"
    )]
    delta_approver: Option<String>,

    #[arg(
        long = "delta-comment",
        value_name = "TEXT",
        help = "Optional audit comment for --rollback-learning-delta"
    )]
    delta_comment: Option<String>,

    #[arg(
        long = "evaluate-candidate-graph-rerank",
        value_name = "FILE",
        help = "Run an offline candidate-graph rerank experiment from JSON; stores only IRI-based verdicts"
    )]
    evaluate_candidate_graph_rerank: Option<String>,

    #[arg(
        long = "propose-candidate-graph-rerank",
        value_name = "FILE",
        help = "Create a proposed reranker Delta from an admitted graph-vs-exact evaluation JSON"
    )]
    propose_candidate_graph_rerank: Option<String>,

    #[arg(
        long = "list-evolution-proposals",
        help = "List durable skill-evolution proposals for this workspace"
    )]
    list_evolution_proposals: bool,

    #[arg(
        long = "approve-evolution-proposal",
        value_name = "ID",
        help = "Record an explicit approval for a proposal (requires --proposal-approver)"
    )]
    approve_evolution_proposal: Option<String>,

    #[arg(
        long = "validate-evolution-proposal",
        value_name = "ID",
        help = "Run no-side-effect gates for an approved proposal"
    )]
    validate_evolution_proposal: Option<String>,

    #[arg(
        long = "commit-evolution-proposal",
        value_name = "ID",
        help = "Commit a validated AddLink proposal"
    )]
    commit_evolution_proposal: Option<String>,

    #[arg(
        long = "proposal-approver",
        value_name = "IDENTITY",
        help = "Audit identity recorded with --approve-evolution-proposal; not authentication"
    )]
    proposal_approver: Option<String>,

    #[arg(
        long = "proposal-comment",
        value_name = "TEXT",
        help = "Optional audit comment for --approve-evolution-proposal"
    )]
    proposal_comment: Option<String>,

    #[arg(
        long = "workflow",
        help = "Path to JSON-LD workflow definition file (optional, replaces LLM-generated plan)"
    )]
    workflow: Option<String>,

    #[arg(
        long = "skill-dir",
        help = "Directory of external skill definitions (scans skills/*/skill.jsonld)"
    )]
    skill_dir: Option<String>,

    #[arg(
        long = "daemon",
        help = "Run in daemon mode (Agent OS Worker — processes tasks from a durable filesystem queue)"
    )]
    daemon: bool,

    #[arg(
        long = "mcp-server",
        value_name = "NAME=URL",
        help = "MCP server config (repeatable, format name=url, e.g. --mcp-server chrome=http://localhost:3000/sse)"
    )]
    mcp_server: Vec<String>,

    #[arg(
        long = "mcp-server-stdio",
        value_name = "NAME=JSON",
        help = "MCP Stdio server config (repeatable, format name=json, e.g. --mcp-server-stdio my-server='{\"command\":\"npx\",\"args\":[\"-y\",\"@modelcontextprotocol/server-filesystem\"]}')"
    )]
    mcp_server_stdio: Vec<String>,
}

impl Cli {
    /// Commands in this set operate only on local durable state. They must
    /// remain available when an API key is intentionally absent, for example
    /// during incident evidence collection or offline audit.
    fn is_local_management_command(&self) -> bool {
        self.list_checkpoints
            || self.list_learning_evaluations
            || self.summarize_learning_evaluations
            || self.list_offline_retrieval_evaluations
            || self.verify_task_evidence.is_some()
            || self.inspect_ann_health
            || self.list_learning_health
            || self.list_learning_deltas
            || self.rollback_learning_delta.is_some()
            || self.evaluate_candidate_graph_rerank.is_some()
            || self.propose_candidate_graph_rerank.is_some()
            || self.list_evolution_proposals
            || self.approve_evolution_proposal.is_some()
            || self.validate_evolution_proposal.is_some()
            || self.commit_evolution_proposal.is_some()
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.daemon {
        return run_daemon();
    }
    let local_management_command = cli.is_local_management_command();

    let log_level = if cli.debug {
        "debug"
    } else if cli.verbose {
        "info"
    } else {
        "warn"
    };

    // Capture all tracing output into a shared buffer so the TUI can display it
    // in the log panel instead of sending it to stderr where it corrupts the display.
    let log_buffer = std::sync::Arc::new(code_cli::log_buffer::LogBuffer::new());
    let shared_log = code_cli::log_buffer::SharedLogBuffer(log_buffer.clone());

    // In single-shot (--prompt) mode there is no TUI log panel, so mirror every
    // log line to stderr in real time. Without this, long-running tasks hold all
    // logs in the in-memory buffer until the task ends, making the agent appear
    // frozen / "jam generated nothing" while it is actually making progress.
    if cli.prompt.is_some() {
        log_buffer.set_mirror_to_stderr(true);
    }

    // tui-markdown 0.3 spams "Could not find syntax for code block: ''" on
    // every render when encountering fenced ``` or indented (4-space) code blocks.
    // Suppress its warnings to keep the log panel clean.
    let filter_with_suppressions = |level: &str| format!("{},tui_markdown=error", level);

    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new(filter_with_suppressions(&log_level))
            }),
        )
        .with_writer(shared_log)
        .with_target(false)
        .init();

    let learning_mode = cli
        .learning_mode
        .or_else(|| std::env::var("GLIDING_LEARNING_MODE").ok())
        .unwrap_or_else(|| "active".to_string())
        .parse::<glidinghorse::core::policy_learning::LearningMode>()
        .map_err(anyhow::Error::msg)?;

    let config_builder = if local_management_command {
        code_cli::config::CliConfig::from_env_and_args_without_required_api_key
    } else {
        code_cli::config::CliConfig::from_env_and_args
    };
    let mut config = config_builder(
        cli.api_key.clone(),
        cli.api_url.clone(),
        cli.model,
        cli.workspace.clone(),
        cli.max_iterations,
        cli.max_pdca_cycles,
        learning_mode,
        cli.workflow,
        cli.skill_dir,
    );
    for entry in &cli.mcp_server {
        if let Some((name, url)) = entry.split_once('=') {
            let name = name.to_lowercase();
            if let Some(existing) = config
                .mcp_servers
                .iter_mut()
                .find(|server| server.name == name)
            {
                existing.url = url.to_string();
            } else {
                config.mcp_servers.push(code_cli::config::McpServerEntry {
                    name,
                    url: url.to_string(),
                });
            }
        }
    }
    for entry in &cli.mcp_server_stdio {
        if let Some((name, json_value)) = entry.split_once('=') {
            let parsed: code_cli::config::McpStdioServerEntry = serde_json::from_str(json_value)
                .map_err(|error| {
                    anyhow::anyhow!("invalid stdio MCP config '{}': {}", name, error)
                })?;
            let name = name.to_lowercase();
            if let Some(existing) = config
                .mcp_stdio_servers
                .iter_mut()
                .find(|(server_name, _)| server_name == &name)
            {
                existing.1 = parsed;
            } else {
                config.mcp_stdio_servers.push((name, parsed));
            }
        }
    }
    if cli.learning_pair_id.is_some() {
        config.learning_pair_id = cli.learning_pair_id.clone();
    }
    if cli.learning_seed.is_some() {
        config.learning_seed = cli.learning_seed.clone();
    }

    if cli.list_checkpoints {
        list_checkpoints(&config)?;
        return Ok(());
    }

    if cli.list_learning_evaluations {
        list_learning_evaluations(&config)?;
        return Ok(());
    }

    if cli.summarize_learning_evaluations {
        summarize_learning_evaluations(&config)?;
        return Ok(());
    }

    if cli.list_offline_retrieval_evaluations {
        list_offline_retrieval_evaluations(&config)?;
        return Ok(());
    }

    if let Some(ref task_iri) = cli.verify_task_evidence {
        verify_task_evidence(&config, task_iri)?;
        return Ok(());
    }

    if cli.inspect_ann_health {
        inspect_ann_health(config)?;
        return Ok(());
    }

    if cli.list_learning_health {
        list_learning_health(&config)?;
        return Ok(());
    }

    if cli.list_learning_deltas {
        list_learning_deltas(&config)?;
        return Ok(());
    }

    if let Some(ref delta_id) = cli.rollback_learning_delta {
        let approver = cli.delta_approver.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--rollback-learning-delta requires --delta-approver")
        })?;
        rollback_learning_delta(&config, delta_id, approver, cli.delta_comment.as_deref())?;
        return Ok(());
    }

    if let Some(ref input_path) = cli.evaluate_candidate_graph_rerank {
        evaluate_candidate_graph_rerank(&config, input_path)?;
        return Ok(());
    }

    if let Some(ref input_path) = cli.propose_candidate_graph_rerank {
        propose_candidate_graph_rerank(&config, input_path)?;
        return Ok(());
    }

    if cli.list_evolution_proposals {
        list_evolution_proposals(&config)?;
        return Ok(());
    }

    if let Some(ref proposal_id) = cli.approve_evolution_proposal {
        let approver = cli.proposal_approver.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--approve-evolution-proposal requires --proposal-approver")
        })?;
        approve_evolution_proposal(&config, proposal_id, approver, cli.proposal_comment.clone())?;
        return Ok(());
    }

    if let Some(ref proposal_id) = cli.validate_evolution_proposal {
        validate_evolution_proposal(&config, proposal_id)?;
        return Ok(());
    }

    if let Some(ref proposal_id) = cli.commit_evolution_proposal {
        commit_evolution_proposal(&config, proposal_id)?;
        return Ok(());
    }

    if let Some(ref task_iri) = cli.resume {
        resume_task(config, task_iri, log_buffer)?;
        return Ok(());
    }

    if let Some(prompt) = cli.prompt {
        if cli.debug {
            run_single_with_logs(config, &prompt, log_buffer)?;
        } else {
            run_single(config, &prompt)?;
        }
    } else {
        recover_terminal_for_interactive_session();
        code_cli::tui::App::new(config, log_buffer, None)?.run()?;
    }

    return Ok(());

    // Run in daemon mode: spawn an Agent OS Worker that processes tasks
    // from a durable filesystem queue.
    fn run_daemon() -> anyhow::Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        let config = glidinghorse::worker::WorkerConfig::from_env();
        eprintln!(
            "Agent OS Worker starting (queue={}, concurrency={})...",
            config.queue_base_path, config.concurrency
        );
        rt.block_on(glidinghorse::worker::run_worker(config))
            .map_err(|error| anyhow::anyhow!("Agent OS Worker terminated with error: {error}"))
    }
}

/// Best-effort recovery for a terminal left in TUI mode by an interrupted
/// interactive session.  This deliberately runs only when starting a new TUI:
/// audit and one-shot commands must keep stdout machine-readable.
fn recover_terminal_for_interactive_session() {
    let _ = crossterm::terminal::disable_raw_mode();
    let mut stdout = std::io::stdout();
    let _ = crossterm::execute!(
        stdout,
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture,
    );
}

fn run_single(config: code_cli::config::CliConfig, prompt: &str) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;

    let mut engine = {
        // Construct the engine inside the runtime context so subsystems
        // that capture a tokio Handle at init (e.g. WatchEngine) work.
        let _rt_guard = rt.enter();
        code_cli::engine::CodeCliEngine::new(config)?
    };
    println!("Code CLI - Agent OS");
    println!(
        "Model: {} | Workspace: {}",
        engine.model(),
        engine.workspace()
    );
    println!();

    let result = rt.block_on(engine.process_task(prompt));

    let exit_status = match result {
        Ok((_, tr)) => {
            let icon = match tr.status.as_str() {
                "success" => "✅",
                _ => "❌",
            };
            println!(
                "{} {} | Turns: {} | Tools: {}",
                icon,
                tr.status.to_uppercase(),
                tr.turn_count,
                tr.tool_call_count
            );
            println!("📁 Output: {}", engine.workspace());
            println!();
            println!("{}", tr.summary);
            task_exit_status(&tr)
        }
        Err(error) => Err(error),
    };

    rt.block_on(engine.shutdown());

    exit_status
}

/// Run a single task and dump all captured logs before exiting.
/// Used for testing/verification when --debug is passed.
#[allow(dead_code)]
fn run_single_with_logs(
    config: code_cli::config::CliConfig,
    prompt: &str,
    log_buffer: std::sync::Arc<code_cli::log_buffer::LogBuffer>,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;

    let mut engine = {
        // Construct the engine inside the runtime context so subsystems
        // that capture a tokio Handle at init (e.g. WatchEngine) work.
        let _rt_guard = rt.enter();
        code_cli::engine::CodeCliEngine::new(config)?
    };
    println!("Code CLI - Agent OS");
    println!(
        "Model: {} | Workspace: {}",
        engine.model(),
        engine.workspace()
    );
    println!();

    let result = rt.block_on(engine.process_task(prompt));

    // Mirror mode writes every log line to stderr in real time, so the final
    // dump below only needs to run when mirroring is off (e.g. interactive).
    let mirrored = log_buffer.mirrors_to_stderr();
    // Dump logs before result so they appear in chronological order
    let logs = log_buffer.drain();
    if mirrored && !logs.is_empty() {
        eprintln!("--- END LOG DUMP ---");
    }
    if !mirrored && !logs.is_empty() {
        eprintln!("--- LOG DUMP ({} lines) ---", logs.len());
        for line in logs {
            eprintln!("{}", line);
        }
        eprintln!("--- END LOG DUMP ---");
    }

    let exit_status = match result {
        Ok((_, tr)) => {
            let icon = match tr.status.as_str() {
                "success" => "✅",
                _ => "❌",
            };
            println!(
                "{} {} | Turns: {} | Tools: {}",
                icon,
                tr.status.to_uppercase(),
                tr.turn_count,
                tr.tool_call_count
            );
            println!("📁 Output: {}", engine.workspace());
            println!();
            println!("{}", tr.summary);
            task_exit_status(&tr)
        }
        Err(error) => Err(error),
    };

    rt.block_on(engine.shutdown());

    exit_status
}

/// A one-shot command is successful only when the supervisory quality gate
/// explicitly returns `success`.  Printing a failed result while returning
/// exit code zero made shell scripts and CI treat failed tasks as completed.
fn task_exit_status(result: &glidinghorse::core::agent_runner::TaskResult) -> anyhow::Result<()> {
    if result.status == "success" {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "Task finished with status '{}' after {} turn(s) and {} tool call(s)",
            result.status,
            result.turn_count,
            result.tool_call_count
        ))
    }
}

fn list_checkpoints(config: &code_cli::config::CliConfig) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    let engine = {
        // Construct the engine inside the runtime context so subsystems
        // that capture a tokio Handle at init (e.g. WatchEngine) work.
        let _rt_guard = rt.enter();
        code_cli::engine::CodeCliEngine::new(config.clone())?
    };

    let checkpoints = rt.block_on(engine.list_checkpoints())?;
    if checkpoints.is_empty() {
        println!("No checkpoints found.");
    } else {
        println!("Checkpoints:");
        for cp in &checkpoints {
            println!(
                "  {}  {}  turns={}  {}",
                cp.created_at, cp.name, cp.node_count, cp.task_iri
            );
        }
        println!("\nUse glidingcode --resume <task_iri> to resume");
    }
    Ok(())
}

fn list_evolution_proposals(config: &code_cli::config::CliConfig) -> anyhow::Result<()> {
    let engine = code_cli::engine::CodeCliEngine::new(config.clone())?;
    let proposals = engine.list_evolution_proposals()?;
    if proposals.is_empty() {
        println!("No evolution proposals found.");
    } else {
        for proposal in proposals {
            println!(
                "{}  {:?}  {}  {}",
                proposal.proposal_id,
                proposal.status,
                proposal.suggestion.skill_iri,
                proposal.suggestion.description,
            );
        }
    }
    Ok(())
}

fn list_learning_evaluations(config: &code_cli::config::CliConfig) -> anyhow::Result<()> {
    let evaluations =
        code_cli::engine::CodeCliEngine::list_learning_evaluations_from_config(config)?;
    println!("{}", serde_json::to_string_pretty(&evaluations)?);
    Ok(())
}

fn summarize_learning_evaluations(config: &code_cli::config::CliConfig) -> anyhow::Result<()> {
    let evaluations =
        code_cli::engine::CodeCliEngine::list_learning_evaluations_from_config(config)?;
    let summary =
        code_cli::engine::CodeCliEngine::summarize_learning_evaluation_values(evaluations);
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn list_offline_retrieval_evaluations(config: &code_cli::config::CliConfig) -> anyhow::Result<()> {
    let evaluations =
        code_cli::engine::CodeCliEngine::list_offline_retrieval_evaluations_from_config(config)?;
    println!("{}", serde_json::to_string_pretty(&evaluations)?);
    Ok(())
}

fn verify_task_evidence(
    config: &code_cli::config::CliConfig,
    task_iri: &str,
) -> anyhow::Result<()> {
    let verification =
        code_cli::engine::CodeCliEngine::verify_task_evidence_from_config(config, task_iri)?;
    println!("{}", serde_json::to_string_pretty(&verification)?);
    if verification.valid && verification.sealed {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "task evidence is not a valid sealed chain: {}",
            verification.failures.join("; ")
        ))
    }
}

fn inspect_ann_health(config: code_cli::config::CliConfig) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    let engine = {
        let _guard = runtime.enter();
        code_cli::engine::CodeCliEngine::new(config)?
    };
    if engine.embedding_provider() == "fallback" {
        return Err(anyhow::anyhow!(
            "ANN health probe is unavailable because semantic embeddings are not configured \
             (provider=fallback). Configure embedding.provider as 'ollama' or 'oneapi' before \
             interpreting ANN recall metrics."
        ));
    }
    let evidence = runtime.block_on(engine.inspect_ann_health())?;
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    Ok(())
}

fn list_learning_health(config: &code_cli::config::CliConfig) -> anyhow::Result<()> {
    let reports = code_cli::engine::CodeCliEngine::list_learning_health_from_config(config)?;
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}

fn list_learning_deltas(config: &code_cli::config::CliConfig) -> anyhow::Result<()> {
    let deltas = code_cli::engine::CodeCliEngine::list_learning_deltas_from_config(config)?;
    println!("{}", serde_json::to_string_pretty(&deltas)?);
    Ok(())
}

fn rollback_learning_delta(
    config: &code_cli::config::CliConfig,
    delta_id: &str,
    approver: &str,
    comment: Option<&str>,
) -> anyhow::Result<()> {
    let delta = code_cli::engine::CodeCliEngine::rollback_learning_delta_from_config(
        config, delta_id, approver, comment,
    )?;
    println!(
        "Rolled back learning delta {} ({:?}); its policy family remains frozen at baseline.",
        delta.delta_id, delta.state
    );
    Ok(())
}

fn evaluate_candidate_graph_rerank(
    config: &code_cli::config::CliConfig,
    input_path: &str,
) -> anyhow::Result<()> {
    let admission = code_cli::engine::CodeCliEngine::evaluate_candidate_graph_rerank_from_config(
        config, input_path,
    )?;
    println!("{}", serde_json::to_string_pretty(&admission)?);
    Ok(())
}

fn propose_candidate_graph_rerank(
    config: &code_cli::config::CliConfig,
    input_path: &str,
) -> anyhow::Result<()> {
    let delta = code_cli::engine::CodeCliEngine::propose_candidate_graph_rerank_from_config(
        config, input_path,
    )?;
    println!(
        "Proposed candidate graph reranker delta {} ({:?}); it is not connected to runtime retrieval.",
        delta.delta_id, delta.state
    );
    Ok(())
}

fn approve_evolution_proposal(
    config: &code_cli::config::CliConfig,
    proposal_id: &str,
    approver: &str,
    comment: Option<String>,
) -> anyhow::Result<()> {
    let engine = code_cli::engine::CodeCliEngine::new(config.clone())?;
    let proposal = engine.approve_evolution_proposal(proposal_id, approver, comment)?;
    println!(
        "Approved proposal {} ({:?})",
        proposal.proposal_id, proposal.status
    );
    Ok(())
}

fn validate_evolution_proposal(
    config: &code_cli::config::CliConfig,
    proposal_id: &str,
) -> anyhow::Result<()> {
    let engine = code_cli::engine::CodeCliEngine::new(config.clone())?;
    let proposal = engine.validate_evolution_proposal(proposal_id)?;
    println!(
        "Validated proposal {} ({:?})",
        proposal.proposal_id, proposal.status
    );
    Ok(())
}

fn commit_evolution_proposal(
    config: &code_cli::config::CliConfig,
    proposal_id: &str,
) -> anyhow::Result<()> {
    let engine = code_cli::engine::CodeCliEngine::new(config.clone())?;
    let proposal = engine.commit_evolution_proposal(proposal_id)?;
    println!(
        "Committed proposal {} ({:?})",
        proposal.proposal_id, proposal.status
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    fn task_result(status: &str) -> glidinghorse::core::agent_runner::TaskResult {
        glidinghorse::core::agent_runner::TaskResult {
            task_iri: "iri://task/test".to_string(),
            status: status.to_string(),
            verdict: None,
            summary: String::new(),
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: Vec::new(),
            turn_count: 2,
            tool_call_count: 1,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: None,
        }
    }

    #[test]
    fn parses_durable_evolution_governance_commands() {
        let cli = Cli::try_parse_from([
            "glidingcode",
            "--workspace",
            "/tmp/gliding-code-test",
            "--approve-evolution-proposal",
            "proposal-42",
            "--proposal-approver",
            "reviewer@example.test",
            "--proposal-comment",
            "reviewed locally",
        ])
        .expect("governance command should be registered");

        assert_eq!(cli.workspace, "/tmp/gliding-code-test");
        assert_eq!(
            cli.approve_evolution_proposal.as_deref(),
            Some("proposal-42")
        );
        assert_eq!(
            cli.proposal_approver.as_deref(),
            Some("reviewer@example.test")
        );
        assert_eq!(cli.proposal_comment.as_deref(), Some("reviewed locally"));
    }

    #[test]
    fn parses_non_mutating_proposal_list_command() {
        let cli = Cli::try_parse_from(["glidingcode", "--list-evolution-proposals"])
            .expect("list command should be registered");

        assert!(cli.list_evolution_proposals);
        assert!(cli.approve_evolution_proposal.is_none());
        assert!(cli.commit_evolution_proposal.is_none());
    }

    #[test]
    fn parses_controlled_learning_treatment() {
        let cli = Cli::try_parse_from(["glidingcode", "--learning-mode", "shadow"])
            .expect("learning treatment should be registered");
        assert_eq!(cli.learning_mode.as_deref(), Some("shadow"));
    }

    #[test]
    fn parses_learning_evaluation_audit_command() {
        let cli = Cli::try_parse_from(["glidingcode", "--list-learning-evaluations"])
            .expect("learning evaluation command should be registered");
        assert!(cli.list_learning_evaluations);
    }

    #[test]
    fn parses_learning_health_and_human_rollback_commands() {
        let list = Cli::try_parse_from(["glidingcode", "--list-learning-health"])
            .expect("learning health command should be registered");
        assert!(list.list_learning_health);

        let rollback = Cli::try_parse_from([
            "glidingcode",
            "--rollback-learning-delta",
            "delta_42",
            "--delta-approver",
            "operator@example.test",
            "--delta-comment",
            "reviewed regression",
        ])
        .expect("learning delta rollback command should be registered");
        assert_eq!(
            rollback.rollback_learning_delta.as_deref(),
            Some("delta_42")
        );
        assert_eq!(
            rollback.delta_approver.as_deref(),
            Some("operator@example.test")
        );
    }

    #[test]
    fn parses_offline_candidate_graph_rerank_commands() {
        let list = Cli::try_parse_from(["glidingcode", "--list-offline-retrieval-evaluations"])
            .expect("offline retrieval evaluation list command should be registered");
        assert!(list.list_offline_retrieval_evaluations);

        let evaluate = Cli::try_parse_from([
            "glidingcode",
            "--evaluate-candidate-graph-rerank",
            "experiment.json",
        ])
        .expect("offline graph rerank evaluation command should be registered");
        assert_eq!(
            evaluate.evaluate_candidate_graph_rerank.as_deref(),
            Some("experiment.json")
        );

        let propose = Cli::try_parse_from([
            "glidingcode",
            "--propose-candidate-graph-rerank",
            "proposal.json",
        ])
        .expect("offline graph rerank proposal command should be registered");
        assert_eq!(
            propose.propose_candidate_graph_rerank.as_deref(),
            Some("proposal.json")
        );
    }

    #[test]
    fn parses_task_evidence_and_ann_health_commands() {
        let verify = Cli::try_parse_from([
            "glidingcode",
            "--verify-task-evidence",
            "iri://task/example",
        ])
        .expect("task evidence verification command should be registered");
        assert_eq!(
            verify.verify_task_evidence.as_deref(),
            Some("iri://task/example")
        );

        let inspect = Cli::try_parse_from(["glidingcode", "--inspect-ann-health"])
            .expect("ANN health inspection command should be registered");
        assert!(inspect.inspect_ann_health);
    }

    #[test]
    fn local_management_commands_do_not_require_an_llm_credential() {
        let verify = Cli::try_parse_from([
            "glidingcode",
            "--verify-task-evidence",
            "iri://task/example",
        ])
        .expect("task evidence command should parse");
        assert!(verify.is_local_management_command());

        let prompt = Cli::try_parse_from(["glidingcode", "answer the task"])
            .expect("one-shot prompt should parse");
        assert!(!prompt.is_local_management_command());
    }

    #[test]
    fn parses_paired_learning_summary_command() {
        let cli = Cli::try_parse_from([
            "glidingcode",
            "--summarize-learning-evaluations",
            "--learning-pair-id",
            "pair-17",
            "--learning-seed",
            "fixed-42",
        ])
        .expect("paired learning audit options should be registered");
        assert!(cli.summarize_learning_evaluations);
        assert_eq!(cli.learning_pair_id.as_deref(), Some("pair-17"));
        assert_eq!(cli.learning_seed.as_deref(), Some("fixed-42"));
    }

    #[test]
    fn one_shot_exit_status_requires_full_success() {
        assert!(super::task_exit_status(&task_result("success")).is_ok());
        let error = super::task_exit_status(&task_result("partial_success"))
            .expect_err("partial success must return a non-zero CLI result");
        assert!(error.to_string().contains("partial_success"));
    }
}

fn resume_task(
    config: code_cli::config::CliConfig,
    task_iri: &str,
    log_buffer: std::sync::Arc<code_cli::log_buffer::LogBuffer>,
) -> anyhow::Result<()> {
    println!("Resuming task from checkpoint: {}", task_iri);
    println!("Opening console...\n");
    code_cli::tui::App::new(config, log_buffer, Some(task_iri.to_string()))?.run()?;
    Ok(())
}
