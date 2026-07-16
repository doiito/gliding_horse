use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "gliding", about = "Agent OS Console - AI Coding Assistant")]
struct Cli {
    #[arg(help = "Single prompt (omit for interactive mode)")]
    prompt: Option<String>,

    #[arg(short = 'm', long = "model", default_value = "deepseek-v4-flash", help = "Model to use")]
    model: String,

    #[arg(short = 'w', long = "workspace", default_value = ".", help = "Working directory")]
    workspace: String,

    #[arg(long = "max-iterations", default_value = "50", help = "Maximum iterations")]
    max_iterations: u32,

    #[arg(long = "max-pdca-cycles", default_value = "7", help = "Maximum PDCA cycle re-entry count for recursive tasks")]
    max_pdca_cycles: u32,

    #[arg(long = "api-key", help = "API key (takes precedence over DEEPSEEK_API_KEY env var)")]
    api_key: Option<String>,

    #[arg(long = "api-url", help = "API URL (takes precedence over DEEPSEEK_API_URL env var)")]
    api_url: Option<String>,

    #[arg(short = 'v', long = "verbose", help = "Show verbose logs")]
    verbose: bool,

    #[arg(long = "debug", help = "Show debug logs (more detailed)")]
    debug: bool,

    #[arg(long = "resume", help = "Resume task from checkpoint (provide task_iri)")]
    resume: Option<String>,

    #[arg(long = "list-checkpoints", help = "List all checkpoints")]
    list_checkpoints: bool,

    #[arg(long = "workflow", help = "Path to JSON-LD workflow definition file (optional, replaces LLM-generated plan)")]
    workflow: Option<String>,

    #[arg(long = "daemon", help = "Run in daemon mode (Agent OS Worker — processes tasks from a Unix socket queue)")]
    daemon: bool,

    #[arg(long = "mcp-server", value_name = "NAME=URL", help = "MCP server config (repeatable, format name=url, e.g. --mcp-server chrome=http://localhost:3000/sse)")]
    mcp_server: Vec<String>,

    #[arg(long = "mcp-server-stdio", value_name = "NAME=JSON", help = "MCP Stdio server config (repeatable, format name=json, e.g. --mcp-server-stdio my-server='{\"command\":\"npx\",\"args\":[\"-y\",\"@modelcontextprotocol/server-filesystem\"]}')")]
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

    // Capture all tracing output into a shared buffer so the TUI can display it
    // in the log panel instead of sending it to stderr where it corrupts the display.
    let log_buffer = std::sync::Arc::new(code_cli::log_buffer::LogBuffer::new());
    let shared_log = code_cli::log_buffer::SharedLogBuffer(log_buffer.clone());

    // tui-markdown 0.3 spams "Could not find syntax for code block: ''" on
    // every render when encountering fenced ``` or indented (4-space) code blocks.
    // Suppress its warnings to keep the log panel clean.
    let filter_with_suppressions = |level: &str| {
        format!("{},tui_markdown=error", level)
    };

    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter_with_suppressions(&log_level))),
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
    );

    if cli.daemon {
        return run_daemon();
    }

    if cli.list_checkpoints {
        list_checkpoints(&config)?;
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
    eprintln!("Agent OS Worker starting (queue={}, concurrency={})...",
        config.queue_base_path, config.concurrency);
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
    println!("Model: {} | Workspace: {}", engine.model(), engine.workspace());
    println!();

    let result = rt.block_on(engine.process_task(prompt));

    match result {
        Ok((_, tr)) => {
            let icon = match tr.status.as_str() { "success" => "✅", _ => "❌" };
            println!("{} {} | Turns: {} | Tools: {}", icon, tr.status.to_uppercase(), tr.turn_count, tr.tool_call_count);
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
    println!("Model: {} | Workspace: {}", engine.model(), engine.workspace());
    println!();

    let result = rt.block_on(engine.process_task(prompt));

    // Dump logs before result so they appear in chronological order
    let logs = log_buffer.drain();
    if !logs.is_empty() {
        eprintln!("--- LOG DUMP ({} lines) ---", logs.len());
        for line in logs {
            eprintln!("{}", line);
        }
        eprintln!("--- END LOG DUMP ---");
    }

    match result {
        Ok((_, tr)) => {
            let icon = match tr.status.as_str() { "success" => "✅", _ => "❌" };
            println!("{} {} | Turns: {} | Tools: {}", icon, tr.status.to_uppercase(), tr.turn_count, tr.tool_call_count);
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
            println!("  {}  {}  turns={}  {}", cp.created_at, cp.name, cp.node_count, cp.task_iri);
        }
        println!("\nUse glidingcode --resume <task_iri> to resume");
    }
    Ok(())
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
