use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "gliding", about = "Agent OS Console - AI Coding Assistant")]
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
        help = "Maximum iterations"
    )]
    max_iterations: u32,

    #[arg(
        long = "max-pdca-cycles",
        default_value = "7",
        help = "Maximum PDCA cycle re-entry count for recursive tasks"
    )]
    max_pdca_cycles: u32,

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
        help = "Run in daemon mode (Agent OS Worker — processes tasks from a Unix socket queue)"
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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let log_level = if cli.debug {
        "debug"
    } else if cli.verbose {
        "info"
    } else {
        "warn"
    };

    // ── Terminal crash recovery ──
    // If the previous glidingcode instance was killed by SIGKILL (OOM killer),
    // the terminal may still be in raw mode + alternate screen.
    // Clean up here BEFORE any mode-specific code runs, so --help, one-shot,
    // and interactive mode all recover from a prior crash.
    let _ = crossterm::terminal::disable_raw_mode();
    let mut crash_recovery_stdout = std::io::stdout();
    let _ = crossterm::execute!(
        crash_recovery_stdout,
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture,
    );

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

    if let Some(key) = cli.api_key {
        std::env::set_var("DEEPSEEK_API_KEY", key);
    }
    if let Some(url) = cli.api_url {
        std::env::set_var("DEEPSEEK_API_URL", url);
    }

    // Parse --mcp-server args into MCP_SERVER__{NAME} env vars
    for entry in &cli.mcp_server {
        if let Some((name, url)) = entry.split_once('=') {
            let env_key = format!("MCP_SERVER__{}", name);
            std::env::set_var(env_key, url);
        }
    }

    // Parse --mcp-server-stdio args into MCP_STDIO__{NAME} env vars
    for entry in &cli.mcp_server_stdio {
        if let Some((name, json_val)) = entry.split_once('=') {
            let env_key = format!("MCP_STDIO__{}", name);
            std::env::set_var(env_key, json_val);
        }
    }

    let config = code_cli::config::CliConfig::from_env_and_args(
        cli.model,
        cli.workspace.clone(),
        cli.max_iterations,
        cli.max_pdca_cycles,
        cli.workflow,
        cli.skill_dir,
    );

    if cli.daemon {
        return run_daemon();
    }

    if cli.list_checkpoints {
        list_checkpoints(&config)?;
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
        code_cli::tui::App::new(config, log_buffer, None)?.run()?;
    }

    return Ok(());

    // Run in daemon mode: spawn an Agent OS Worker that processes tasks
    // from a Unix socket queue.
    fn run_daemon() -> anyhow::Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        let config = glidinghorse::worker::WorkerConfig::from_env();
        eprintln!(
            "Agent OS Worker starting (queue={}, concurrency={})...",
            config.queue_base_path, config.concurrency
        );
        if let Err(e) = rt.block_on(glidinghorse::worker::run_worker(config)) {
            eprintln!("Agent OS Worker terminated with error: {}", e);
        }
        Ok(())
    }
}

fn run_single(config: code_cli::config::CliConfig, prompt: &str) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;

    let mut engine = code_cli::engine::CodeCliEngine::new(config)?;
    println!("Code CLI - Agent OS");
    println!(
        "Model: {} | Workspace: {}",
        engine.model(),
        engine.workspace()
    );
    println!();

    let result = rt.block_on(engine.process_task(prompt));

    match result {
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
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
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

    let mut engine = code_cli::engine::CodeCliEngine::new(config)?;
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

    match result {
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
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

fn list_checkpoints(config: &code_cli::config::CliConfig) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    let engine = code_cli::engine::CodeCliEngine::new(config.clone())?;

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
