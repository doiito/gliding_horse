use std::cell::RefCell;
use std::sync::Arc;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use glidinghorse::core::agent_runner::TaskResult;
use glidinghorse::core::event_bus::EventBus;
use glidinghorse::core::execution_event::{ExecutionEvent, ExecutionEventKind};
use glidinghorse::gateway::unified_gateway::ChatMessage;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Clear, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation,
    ScrollbarState, Wrap,
};
use ratatui::Frame;
use ratatui::Terminal;
use serde_json::Value;
use tokio::sync::mpsc;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::config::CliConfig;
use crate::log_buffer::LogBuffer;

#[derive(Clone, Copy, PartialEq)]
enum MessageRole {
    User,
    Assistant,
    System,
    Error,
}

struct Message {
    role: MessageRole,
    content: String,
    timestamp: String,
    mermaid_blocks: Vec<MermaidBlock>,
    /// Full raw payload for expandable execution events (tool JSON, full thought, full result).
    full_raw: Option<String>,
    /// True if this message has a `full_raw` that can be shown on expand.
    can_expand: bool,
}

struct MermaidBlock {
    source: String,
    svg: Option<String>,
}

struct StatusEvent {
    event_type: String,
    payload: String,
}

pub struct App {
    engine: Arc<tokio::sync::Mutex<super::engine::CodeCliEngine>>,
    event_bus: Arc<EventBus>,
    log_buffer: Arc<LogBuffer>,
    model_name: String,
    embedding_provider: &'static str,
    workspace_path: String,
    max_iter: u32,
    input: String,
    cursor_position: usize,
    messages: Vec<Message>,
    status_events: Vec<StatusEvent>,
    log_lines: Vec<String>,
    current_phase: String,
    current_task_iri: Option<String>,
    /// 从 checkpoint 恢复的历史消息（用于 resume 模式）
    resumed_messages: Option<Vec<glidinghorse::gateway::unified_gateway::ChatMessage>>,
    /// 标记当前会话是否为 resume 模式（防止事件重置计数）
    is_resume_session: bool,
    session_turn_count: u32,
    session_tool_call_count: u32,
    is_processing: bool,
    should_quit: bool,
    expanded: std::collections::HashSet<usize>,
    line_map_cache: RefCell<Vec<(usize, bool)>>,
    panel_top: RefCell<u16>,
    panel_vh: RefCell<usize>,
    panel_start: RefCell<usize>,
    rt: tokio::runtime::Runtime,
    scroll_offset: usize,
    auto_scroll: bool,
    /// Memory subsystem usage (queried from engine before each render)
    l1_count: u64,
    l2_count: u64,
    l3_count: u64,
    total_tokens: u64,
    prompt_tok: u64,
    completion_tok: u64,
    /// 最后一次 API 调用的 token 数（单次，非累计）
    last_prompt_tok: u64,
    last_completion_tok: u64,
    /// 上一帧 prompt token 值（用于计算 delta）
    prev_last_prompt_tok: u64,
    /// delta 显示状态：变化时更新，无变化时保持（避免闪烁）
    display_delta_arrow: RefCell<String>,
    display_delta_val: RefCell<String>,
    /// 模型上下文窗口上限（用于计算占比）
    context_limit: u64,
    /// Checkpoint 恢复的 token 基数（resume 模式下使用）
    resume_prompt_base: u64,
    resume_completion_base: u64,
    /// Memory limits (MB) from config
    max_l1_mb: u64,
    max_l2_mb: u64,
    max_l3_mb: u64,
    /// Lock-free handles for memory stats (no engine lock needed)
    l2_bb: Arc<glidinghorse::memory::l2_blackboard::Blackboard>,
    proj: Arc<glidinghorse::memory::l3_projection::ProjectionEngine>,
    mm: Arc<tokio::sync::Mutex<glidinghorse::memory::memory_manager::MemoryManager>>,
    /// Token counter Arcs (lock-free reads from AgentRunner)
    prompt_tokens: Arc<std::sync::atomic::AtomicU64>,
    completion_tokens: Arc<std::sync::atomic::AtomicU64>,
    last_prompt_tokens: Arc<std::sync::atomic::AtomicU64>,
    last_completion_tokens: Arc<std::sync::atomic::AtomicU64>,
    status_rx: Option<mpsc::UnboundedReceiver<StatusEvent>>,
    result_rx: Option<tokio::sync::oneshot::Receiver<anyhow::Result<(String, TaskResult)>>>,
    /// Last user input string (for topic shift detection)
    last_user_input: String,
    /// WorkspaceMonitor handle (for resetting perception on topic shift)
    workspace_monitor: Option<Arc<glidinghorse::tools::workspace_monitor::WorkspaceMonitor>>,
    // ── Skill Graph subsystem handles (lock-free read via Arc<>) ──
    skill_graph: Arc<glidinghorse::skill_graph::graph_store::SkillGraphStore>,
    // Kept alive to prevent drop — AgentRunner registered these internally.
    #[allow(dead_code)]
    discovery_engine: Arc<glidinghorse::skill_graph::discovery::SkillDiscoveryEngine>,
    #[allow(dead_code)]
    feature_extractor: Arc<glidinghorse::graph_features::features::FeatureExtractor>,
    causal_engine: Arc<glidinghorse::causal::engine::CausalEngine>,
    timeline: Arc<glidinghorse::snapshots::timeline::TimelineStore>,
    /// Cached skill graph stats (refreshed each frame)
    sg_nodes: usize,
    sg_edges: usize,
    sg_snapshots: usize,
    causal_observations: u64,
    timeline_pending: usize,
}

fn extract_mermaid_blocks(content: &str) -> Vec<MermaidBlock> {
    let mut blocks = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim() == "```mermaid" {
            let mut src = Vec::new();
            i += 1;
            while i < lines.len() && lines[i].trim() != "```" {
                src.push(lines[i]);
                i += 1;
            }
            let source = src.join("\n");
            let svg = mermaid_rs_renderer::render(&source).ok();
            blocks.push(MermaidBlock { source, svg });
        }
        i += 1;
    }
    blocks
}

/// Extract keywords from text for topic shift detection (stop-word filtered)
fn extract_keywords(text: &str) -> Vec<String> {
    let stop_words = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "each", "every", "both", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just", "because", "and", "but",
        "or", "if", "while", "that", "this", "it", "its", "i", "me", "my", "we", "our", "you",
        "your", "he", "she", "they", "what", "which", "who", "about", "use", "used",
    ];

    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| w.len() > 2 && !stop_words.contains(&w.as_str()))
        .collect()
}

/// Compute keyword overlap ratio between two texts (0.0 = completely different, 1.0 = identical)
fn keyword_overlap(a: &str, b: &str) -> f64 {
    let ka = extract_keywords(a);
    let kb = extract_keywords(b);

    if ka.is_empty() && kb.is_empty() {
        return 1.0;
    }
    if ka.is_empty() || kb.is_empty() {
        return 0.0;
    }

    let intersection = ka.iter().filter(|k| kb.contains(k)).count() as f64;
    let union = (ka.len() + kb.len()) as f64 - intersection;
    if union == 0.0 {
        1.0
    } else {
        intersection / union
    }
}

fn strip_mermaid_fences(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut in_mermaid = false;
    for line in content.lines() {
        let t = line.trim();
        if t == "```mermaid" {
            in_mermaid = true;
            out.push_str("```\n");
            continue;
        }
        if in_mermaid && t == "```" {
            in_mermaid = false;
            out.push_str("```\n");
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Replace bare ``` fences (no language) with ```txt so syntect
/// can resolve it to "Plain Text" syntax instead of logging a warning.
fn default_code_lang(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut in_fence = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if !in_fence && trimmed.starts_with("```") {
            let rest = trimmed[3..].trim();
            if rest.is_empty() {
                out.push_str("```txt\n");
            } else {
                out.push_str(line);
                out.push('\n');
            }
            in_fence = !trimmed.ends_with("```") || rest.is_empty();
        } else if in_fence && trimmed == "```" {
            out.push_str(line);
            out.push('\n');
            in_fence = false;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn markdown_to_owned_lines(content: &str) -> Vec<Line<'static>> {
    let prepared = default_code_lang(content);
    let text = tui_markdown::from_str(&prepared);
    text.lines
        .into_iter()
        .map(|line| {
            let spans: Vec<Span<'static>> = line
                .spans
                .into_iter()
                .map(|span| {
                    // tui-markdown uses ratatui_core::Style (separate crate) which is
                    // layout-compatible with ratatui::Style when underline-color is
                    // disabled (our build: default-features=false). Transmute is safe
                    // because both structs have {fg, bg, add_modifier, sub_modifier}
                    // with identical Color/Modifier representations.
                    let style = span.style;
                    let style: ratatui::style::Style = unsafe { std::mem::transmute(style) };
                    Span::styled(span.content.into_owned(), style)
                })
                .collect();
            Line::from(spans)
        })
        .collect()
}

fn mermaid_block_lines(mb: &MermaidBlock) -> Vec<Line<'static>> {
    let mut buf = Vec::new();
    buf.push(Line::from(Span::styled(
        "  Mermaid Diagram",
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    )));
    if let Some(ref svg) = mb.svg {
        buf.push(Line::from(Span::styled(
            format!("     Rendered to SVG ({} bytes)", svg.len()),
            Style::default().fg(Color::Green),
        )));
    } else {
        buf.push(Line::from(Span::styled(
            "     Could not render",
            Style::default().fg(Color::Yellow),
        )));
    }
    for src_line in mb.source.lines() {
        buf.push(Line::from(Span::styled(
            format!("       {}", src_line),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::DIM),
        )));
    }
    buf
}

/// Truncate a string so its display width does not exceed `max_width`.
/// The ellipsis "…" (width 1) is only added if it fits within `max_width`.
fn width_truncate(s: &str, max_width: usize) -> String {
    let mut out = String::new();
    let mut w = 0;
    for g in s.graphemes(true) {
        let gw = g.width();
        if w + gw > max_width {
            // Only add ellipsis if it fits — avoids terminal auto-wrap on overflow
            if w + 1 <= max_width {
                out.push_str("…");
            }
            break;
        }
        w += gw;
        out.push_str(g);
    }
    out
}

/// Truncate to `max_width` display columns, then right-pad with spaces to exactly
/// `width` display columns.  Ensures every output line occupies exactly `width`
/// terminal columns, preventing ratatui diff from leaving character residues.
///
/// ⚠️  Do NOT use `format!("{:<N$}", s)` for this — it pads by byte length, not
/// display width.  Multi-byte chars (↑ ↓ ∞ ▶ ✔ …) would cause insufficient padding
/// and column-alignment drift.
fn pad_to_width(s: &str, width: usize) -> String {
    let truncated = width_truncate(s, width);
    let dw = truncated.width();
    if dw < width {
        let mut r = truncated;
        r.push_str(&" ".repeat(width - dw));
        r
    } else {
        truncated.to_string()
    }
}

/// Split a single Line into multiple Lines at display-width boundaries so that
/// ratatui's Paragraph wrapping does not need to add extra visual rows (which
/// would break the 1:1 mapping between line_map entries and screen rows).
fn prewrap_line(line: Line<'static>, max_width: usize) -> Vec<Line<'static>> {
    struct Chunk {
        text: String,
        width: usize,
        style: Style,
    }
    let total: usize = line.spans.iter().map(|s| s.content.as_ref().width()).sum();
    if total <= max_width {
        return vec![line];
    }

    // Flatten spans into grapheme chunks with their display width and style.
    let mut chunks: Vec<Chunk> = Vec::new();
    for span in line.spans {
        for g in span.content.as_ref().graphemes(true) {
            chunks.push(Chunk {
                text: g.to_string(),
                width: g.width(),
                style: span.style,
            });
        }
    }

    let mut out: Vec<Line<'static>> = Vec::new();
    let mut i = 0;
    while i < chunks.len() {
        let mut w = 0usize;
        let mut j = i;
        while j < chunks.len() && w + chunks[j].width <= max_width {
            w += chunks[j].width;
            j += 1;
        }
        if j == i {
            j = i + 1;
        } // single grapheme wider than max_width — force it

        // Merge consecutive chunks with the same style into one Span.
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut buf = String::new();
        let mut cur_style = chunks[i].style;
        for k in i..j {
            if chunks[k].style != cur_style && !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), cur_style));
                cur_style = chunks[k].style;
            }
            buf.push_str(&chunks[k].text);
        }
        if !buf.is_empty() {
            spans.push(Span::styled(buf, cur_style));
        }
        out.push(Line::from(spans));
        i = j;
    }
    out
}

fn timestamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

fn role_color(role: &str) -> Color {
    match role {
        "SA" => Color::Blue,
        "PA" => Color::Cyan,
        "DA" => Color::Magenta,
        "CA" => Color::Yellow,
        "AA" => Color::Green,
        _ => Color::DarkGray,
    }
}

fn action_color(action: &str) -> Color {
    match action {
        "TOOL_CALL" => Color::LightYellow,
        "TOOL_RESULT" => Color::DarkGray,
        "ERROR" => Color::Red,
        _ => Color::DarkGray, // THOUGHT etc.
    }
}

fn tool_color(tool: &str) -> Color {
    if tool.starts_with("web_") {
        Color::Blue
    } else if tool.starts_with("file_") || tool == "file_read" {
        Color::Green
    } else if tool.starts_with("bash") || tool == "command" {
        Color::Cyan
    } else if tool == "glob" || tool == "grep" || tool.starts_with("ast_") {
        Color::Magenta
    } else if tool.starts_with("write") || tool.starts_with("edit") {
        Color::Yellow
    } else {
        Color::White
    }
}

/// Try to parse an execution‑event line and return styled spans for
/// `[icon] AGENT:ROLE:ACTION` and the remaining body.
/// Returns `Some((prefix_spans, body_text))` on success, `None` for non‑event lines.
///
/// Uses `AGENT:` as anchor instead of exact emoji matching — any non‑whitespace
/// icon preceding `AGENT:` is accepted, making the parser tolerant of emoji
/// encoding variations.
fn parse_execution_event_line(text: &str) -> Option<(Vec<Span<'static>>, &str)> {
    // Find "AGENT:" — anything before it is the icon
    let agent_pos = text.find("AGENT:")?;
    if agent_pos == 0 {
        return None; // no icon prefix
    }
    let raw_prefix = &text[..agent_pos];
    let icon = raw_prefix.trim();
    if icon.is_empty() {
        return None;
    }

    // Parse rest: ROLE:ACTION body
    let after_agent = &text[agent_pos + 6..].trim_start();

    // ROLE — everything before first colon
    let role_end = after_agent.find(':')?;
    let role = &after_agent[..role_end];

    // ACTION — after role colon, before first space (or colon)
    let after_role = &after_agent[role_end + 1..].trim_start();
    let action_end = after_role.find(' ').unwrap_or(after_role.len());
    let action = &after_role[..action_end];
    let rest = after_role[action_end..].trim_start();

    let rc = role_color(role);
    let ac = action_color(action);

    let mut prefix_spans = Vec::new();
    prefix_spans.push(Span::styled(icon.to_string(), Style::default().fg(rc)));
    prefix_spans.push(Span::raw(" "));
    prefix_spans.push(Span::styled(
        "AGENT:",
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
    ));
    prefix_spans.push(Span::styled(
        role.to_string(),
        Style::default().fg(rc).add_modifier(Modifier::BOLD),
    ));
    prefix_spans.push(Span::styled(
        format!(":{}", action),
        Style::default().fg(ac).add_modifier(Modifier::BOLD),
    ));

    Some((prefix_spans, rest))
}

/// Try to parse a phase‑transition line like `▶ SA`, `🔄 PA`, `✅ DA (success)`.
/// Returns `Some((colored_prefix_spans, rest))` on success.
fn parse_phase_line(text: &str) -> Option<(Vec<Span<'static>>, &str)> {
    let icon_end = text.find(char::is_whitespace)?;
    let icon = &text[..icon_end];
    if icon.is_empty() {
        return None;
    }
    let after_icon = text[icon_end..].trim_start();
    if after_icon.is_empty() {
        return None;
    }
    let role_end = after_icon
        .find(|c: char| !c.is_ascii_uppercase())
        .unwrap_or(after_icon.len());
    let role = &after_icon[..role_end];
    if !["SA", "PA", "DA", "CA", "AA"].contains(&role) {
        return None;
    }
    let rest = after_icon[role_end..].trim_start();
    let rc = role_color(role);
    let spans = vec![
        Span::styled(icon.to_string(), Style::default().fg(rc)),
        Span::raw(" "),
        Span::styled(
            role.to_string(),
            Style::default().fg(rc).add_modifier(Modifier::BOLD),
        ),
    ];
    Some((spans, rest))
}

/// Extract short role (SA/PA/DA/CA/AA) from an agent_id like `agent_plan_<uuid>`.
fn agent_id_to_role(agent_id: &str) -> &str {
    // agent_id 格式: cycle_role_uuid，role 用 AgentRole::Display (PA/DA/CA/AA)
    // 形如: "cycle_1_PA_550e8400-e29b-41d4-a716-446655440000"
    if agent_id.contains("_PA_") {
        "PA"
    } else if agent_id.contains("_DA_") {
        "DA"
    } else if agent_id.contains("_CA_") {
        "CA"
    } else if agent_id.contains("_AA_") {
        "AA"
    } else if agent_id.contains("SA") || agent_id.contains("sa") || agent_id.contains("supervisor")
    {
        "SA"
    } else {
        "?"
    }
}

/// Return phase only for major-phase events (SA/PA/DA/CA/AA).
fn detect_phase(et: &str) -> Option<String> {
    if et == "TASK_START" || et.contains("CYCLE_STARTED") || et.contains("SA_STARTED") {
        Some("SA".into())
    } else if et.contains("Plan_STARTED") || et == "PA_STARTED" {
        Some("PA".into())
    } else if et.contains("Plan_COMPLETED") || et == "PA_COMPLETED" {
        Some("PA".into())
    } else if et.contains("Do_STARTED") || et == "DA_STARTED" {
        Some("DA".into())
    } else if et.contains("Do_COMPLETED") || et == "DA_COMPLETED" {
        Some("DA".into())
    } else if et.contains("Check_STARTED") || et == "CA_STARTED" {
        Some("CA".into())
    } else if et.contains("Check_COMPLETED") || et == "CA_COMPLETED" {
        Some("CA".into())
    } else if et.contains("Act_STARTED") || et == "AA_STARTED" {
        Some("AA".into())
    } else if et.contains("Act_COMPLETED") || et == "AA_COMPLETED" {
        Some("AA".into())
    } else {
        None
    }
}

fn event_icon(et: &str) -> (&'static str, Color) {
    if et == "CYCLE_STARTED" || et == "TASK_START" {
        ("\u{25B6}", Color::Blue)
    } else if et.contains("COMPLETED") || et == "COMPLETE" {
        ("\u{2714}", Color::Green)
    } else if et.contains("_STARTED") {
        ("\u{25B6}", Color::Cyan)
    } else if et.contains("ERROR") || et.contains("BLOCKED") {
        ("\u{2716}", Color::Red)
    } else if et.contains("SKIPPED") {
        ("\u{229D}", Color::DarkGray)
    } else if et.contains("ABORTED") || et.contains("FROZEN") {
        ("\u{2744}", Color::Yellow)
    } else {
        ("\u{2022}", Color::DarkGray)
    }
}

/// RAII guard: restores terminal on Drop no matter how run() exits.
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = std::io::stdout();
        let _ = execute!(
            stdout,
            LeaveAlternateScreen,
            crossterm::event::DisableMouseCapture
        );
    }
}

impl App {
    pub fn new(
        config: CliConfig,
        log_buffer: Arc<LogBuffer>,
        resume_task_iri: Option<String>,
    ) -> anyhow::Result<Self> {
        let max_l1_mb = config.max_l1_mb;
        let max_l2_mb = config.max_l2_mb;
        let max_l3_mb = config.max_l3_mb;
        let rt = tokio::runtime::Runtime::new()?;
        let engine = {
            // Construct the engine inside the runtime context so subsystems
            // that capture a tokio Handle at init (e.g. WatchEngine) work.
            let _rt_guard = rt.enter();
            super::engine::CodeCliEngine::new(config)?
        };
        let l0 = engine.l0();
        let l2_bb = engine.l2_bb();
        let proj = engine.proj();
        let mm = engine.mm();
        let (prompt_tokens, completion_tokens, last_prompt_tokens, last_completion_tokens) =
            engine.token_arcs();
        let event_bus = engine.event_bus();
        let model_name = engine.model().to_string();
        let embedding_provider = engine.embedding_provider();
        let workspace_path = std::path::Path::new(engine.workspace())
            .canonicalize()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| engine.workspace().to_string());
        let max_iter = engine.max_iterations();
        let context_limit = engine.context_limit();
        let workspace_monitor = engine.workspace_monitor();
        // Probe memory subsystem stats (exercises the CodeCliEngine::memory_stats() path)
        let _mem_stats = engine.memory_stats();
        let skill_graph = engine.skill_graph();
        let discovery_engine = engine.discovery_engine();
        let feature_extractor = engine.feature_extractor();
        let causal_engine = engine.causal_engine();
        let timeline = engine.timeline();

        let mut app = Self {
            engine: Arc::new(tokio::sync::Mutex::new(engine)),
            event_bus,
            log_buffer,
            model_name,
            embedding_provider,
            workspace_path,
            max_iter,
            input: String::new(),
            cursor_position: 0,
            messages: Vec::new(),
            status_events: Vec::new(),
            log_lines: Vec::new(),
            current_phase: "Idle".into(),
            current_task_iri: None,
            resumed_messages: None,
            is_resume_session: false,
            session_turn_count: 0,
            session_tool_call_count: 0,
            is_processing: false,
            should_quit: false,
            expanded: std::collections::HashSet::new(),
            line_map_cache: RefCell::new(Vec::new()),
            panel_top: RefCell::new(0),
            panel_vh: RefCell::new(0),
            panel_start: RefCell::new(0),
            rt,
            scroll_offset: 0,
            auto_scroll: true,
            l1_count: 0,
            l2_count: 0,
            l3_count: 0,
            total_tokens: 0,
            prompt_tok: 0,
            completion_tok: 0,
            last_prompt_tok: 0,
            last_completion_tok: 0,
            prev_last_prompt_tok: 0,
            display_delta_arrow: RefCell::new(String::new()),
            display_delta_val: RefCell::new(String::new()),
            context_limit,
            resume_prompt_base: 0,
            resume_completion_base: 0,
            max_l1_mb,
            max_l2_mb,
            max_l3_mb,
            l2_bb,
            proj,
            mm,
            prompt_tokens,
            completion_tokens,
            last_prompt_tokens,
            last_completion_tokens,
            status_rx: None,
            result_rx: None,
            last_user_input: String::new(),
            workspace_monitor,
            skill_graph,
            discovery_engine,
            feature_extractor,
            causal_engine,
            timeline,
            sg_nodes: 0,
            sg_edges: 0,
            sg_snapshots: 0,
            causal_observations: 0,
            timeline_pending: 0,
        };

        let welcome = format!(
            "## Agent OS Programming Console\n\
             \nCommands: `/help` for help  |  `Esc` to quit",
        );
        app.messages.push(Message {
            role: MessageRole::System,
            content: welcome,
            full_raw: None,
            can_expand: false,
            timestamp: timestamp(),
            mermaid_blocks: Vec::new(),
        });

        // Resume mode: load checkpoint from L0 and restore conversation
        if let Some(ref task_iri) = resume_task_iri {
            let cm = glidinghorse::core::checkpoint::CheckpointManager::with_persistence(l0);
            if let Ok(Some(cp)) = cm.restore_latest(task_iri) {
                // Restore current_task_iri so new input continues the same task
                app.current_task_iri = Some(task_iri.clone());

                // Parse session_messages_json into TUI Messages
                if let Ok(msgs) =
                    serde_json::from_str::<Vec<ChatMessage>>(&cp.session_messages_json)
                {
                    // 保存恢复的历史消息用于传递给 AgentRunner
                    app.resumed_messages = Some(msgs.clone());
                    app.is_resume_session = true;

                    // 恢复 turn/tool 计数
                    app.session_turn_count =
                        msgs.iter().filter(|m| m.role == "assistant").count() as u32;
                    app.session_tool_call_count = msgs
                        .iter()
                        .filter(|m| m.role == "tool" || m.tool_call_id.is_some())
                        .count() as u32;

                    for msg in &msgs {
                        let role = match msg.role.as_str() {
                            "user" => MessageRole::User,
                            "assistant" => MessageRole::Assistant,
                            _ => continue,
                        };
                        app.messages.push(Message {
                            role,
                            content: msg.content.clone(),
                            full_raw: msg.reasoning_content.clone(),
                            can_expand: msg.reasoning_content.is_some(),
                            timestamp: timestamp(),
                            mermaid_blocks: extract_mermaid_blocks(&msg.content),
                        });
                    }
                }

                // 恢复 token 计数（从 agent_state_json）
                if let Ok(state) = serde_json::from_str::<serde_json::Value>(&cp.agent_state_json) {
                    let p = state
                        .get("prompt_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let c = state
                        .get("completion_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    app.prompt_tok = p;
                    app.completion_tok = c;
                    app.total_tokens = p + c;
                    // 保存基数，用于后续 tick 累加新执行的增量
                    app.resume_prompt_base = p;
                    app.resume_completion_base = c;
                }

                // Only show resume banner if we actually restored messages
                if app
                    .messages
                    .iter()
                    .any(|m| matches!(m.role, MessageRole::User | MessageRole::Assistant))
                {
                    let phase_label =
                        glidinghorse::core::checkpoint::parse_checkpoint_phase(&cp.name);
                    let turn_str = serde_json::from_str::<serde_json::Value>(&cp.agent_state_json)
                        .ok()
                        .and_then(|v| v.get("turn").and_then(|t| t.as_u64()))
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "?".to_string());
                    let role_str = cp.current_role.as_deref().unwrap_or("?");
                    let info = format!(
                        "📋 已恢复任务 ({} 条消息)\n  task: `{}`\n  阶段: {} | 角色: {} | Turns: {}",
                        app.messages.len(),
                        task_iri,
                        phase_label,
                        role_str,
                        turn_str,
                    );
                    app.messages.push(Message {
                        role: MessageRole::System,
                        content: info,
                        full_raw: None,
                        can_expand: false,
                        timestamp: timestamp(),
                        mermaid_blocks: Vec::new(),
                    });
                }
            }
        }

        Ok(app)
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        // 注意: 前一次崩溃的终端状态清理已移到 main.rs 入口处，
        // 这样 --help 和单命令模式也能恢复终端。
        // 此处不再重复。

        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            crossterm::event::EnableMouseCapture
        )?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Guard: Drop 时始终恢复终端，即使 run() 因错误提前返回
        let _guard = TerminalGuard;

        // Panic hook: restore terminal if anything panics, so Windows console
        // doesn't get stuck in raw mode (which causes crash on title bar click).
        let orig_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let mut stdout = std::io::stdout();
            let _ = execute!(
                stdout,
                LeaveAlternateScreen,
                crossterm::event::DisableMouseCapture
            );
            orig_hook(info);
        }));

        // Signal handler: catch SIGTERM/SIGINT for graceful shutdown.
        // SIGKILL (OOM killer) can't be caught — the only defence is reducing
        // memory pressure (see checkpoint truncation below).
        let sigquit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        #[cfg(unix)]
        {
            let sigquit_clone = sigquit.clone();
            self.rt.spawn(async move {
                use tokio::signal::unix::{signal, SignalKind};
                let mut term = match signal(SignalKind::terminate()) {
                    Ok(s) => s,
                    Err(_) => return,
                };
                let mut int = match signal(SignalKind::interrupt()) {
                    Ok(s) => s,
                    Err(_) => return,
                };
                // Wait for EITHER signal
                tokio::select! {
                    _ = term.recv() => {}
                    _ = int.recv() => {}
                }
                sigquit_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                // Best-effort terminal restore from signal context (may fail, that's OK)
                let _ = disable_raw_mode();
                let mut stdout = std::io::stdout();
                let _ = execute!(
                    stdout,
                    LeaveAlternateScreen,
                    crossterm::event::DisableMouseCapture
                );
            });
        }

        #[cfg(not(unix))]
        {
            let sigquit_clone = sigquit.clone();
            self.rt.spawn(async move {
                match tokio::signal::ctrl_c().await {
                    Ok(()) => {
                        sigquit_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                        // Best-effort terminal restore
                        let _ = disable_raw_mode();
                        let mut stdout = std::io::stdout();
                        let _ = execute!(
                            stdout,
                            LeaveAlternateScreen,
                            crossterm::event::DisableMouseCapture
                        );
                    }
                    Err(_) => {}
                }
            });
        }

        loop {
            // Check signal-triggered quit (SIGTERM / SIGINT)
            if sigquit.load(std::sync::atomic::Ordering::SeqCst) {
                self.should_quit = true;
            }

            // Drain incoming status events from the background processing task
            self.drain_events();

            let mut new_logs = self.log_buffer.drain();
            let n = new_logs.len();
            if n > 0 {
                self.log_lines = new_logs.split_off(n.saturating_sub(3));
            }

            // Check if the background task has produced a result
            if let Some(rx) = &mut self.result_rx {
                match rx.try_recv() {
                    Ok(result) => {
                        self.result_rx = None;
                        self.complete_task(result);
                    }
                    Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                        // Sender dropped — the background task panicked or was cancelled
                        self.result_rx = None;
                        self.status_rx = None; // drop mpsc receiver → kills zombie event listener
                        self.is_processing = false;
                        self.current_phase = "Idle".into();
                    }
                    Err(_) => {}
                }
            }

            // 不要用 ? — draw 错误后继续循环，让 cleanup 有机会执行
            if self.auto_scroll {
                self.scroll_offset = 0;
            }
            // Read memory stats directly from the Arcs — no engine lock needed
            self.l2_count = self.l2_bb.total_bytes();
            {
                let _cs = self.proj.cache_stats();
            }
            self.l3_count = self.proj.list_frames().len() as u64;
            self.l1_count = self
                .mm
                .try_lock()
                .map(|g| g.l1_session_count())
                .unwrap_or(self.l1_count);
            // 读取 token 计数
            let current_prompt = self
                .prompt_tokens
                .load(std::sync::atomic::Ordering::Relaxed);
            let current_completion = self
                .completion_tokens
                .load(std::sync::atomic::Ordering::Relaxed);
            if self.is_resume_session {
                self.prompt_tok = self.resume_prompt_base + current_prompt;
                self.completion_tok = self.resume_completion_base + current_completion;
                self.total_tokens = self.prompt_tok + self.completion_tok;
            } else {
                self.total_tokens = current_prompt + current_completion;
                self.prompt_tok = current_prompt;
                self.completion_tok = current_completion;
            }
            // 保存旧值用于计算 delta
            self.prev_last_prompt_tok = self.last_prompt_tok;
            self.last_prompt_tok = self
                .last_prompt_tokens
                .load(std::sync::atomic::Ordering::Relaxed);
            self.last_completion_tok = self
                .last_completion_tokens
                .load(std::sync::atomic::Ordering::Relaxed);
            // Lock-free reads from skill graph subsystems
            self.sg_nodes = self.skill_graph.skill_count();
            self.sg_edges = self
                .skill_graph
                .list_all_skills()
                .iter()
                .map(|s| s.links.len())
                .sum();
            self.sg_snapshots = self.timeline.snapshot_count();
            self.causal_observations = self.causal_engine.store().total_observations() as u64;
            self.timeline_pending = self.timeline.pending_mutations();
            let _ = terminal.draw(|f| self.ui(f));
            if self.should_quit {
                break;
            }

            // 同样，event 错误也吞掉。Windows 标题栏交互可能让 poll/read 返回 Err，
            // 如果 ? 传播出去会跳过 disable_raw_mode + LeaveAlternateScreen → 窗口闪退。
            let timeout = std::time::Duration::from_millis(100);
            if matches!(event::poll(timeout), Ok(true)) {
                if let Ok(ev) = event::read() {
                    match ev {
                        Event::Key(key) if key.kind == KeyEventKind::Press => {
                            self.handle_key(key.code, key.modifiers);
                        }
                        Event::Mouse(me) => {
                            let column = me.column;
                            let row = me.row;
                            match me.kind {
                                // ScrollDown = wheel away from user = want newer content
                                crossterm::event::MouseEventKind::ScrollDown => {
                                    self.auto_scroll = false;
                                    self.scroll_offset = self.scroll_offset.saturating_sub(3);
                                }
                                // ScrollUp = wheel toward user = want older content
                                crossterm::event::MouseEventKind::ScrollUp => {
                                    self.auto_scroll = false;
                                    self.scroll_offset = self.scroll_offset.saturating_add(3);
                                }
                                crossterm::event::MouseEventKind::Down(
                                    crossterm::event::MouseButton::Left,
                                )
                                | crossterm::event::MouseEventKind::Up(
                                    crossterm::event::MouseButton::Left,
                                ) => {
                                    if column <= 4 {
                                        self.handle_expand_click(row, column);
                                    }
                                }
                                _ => {}
                            }
                        }
                        Event::Resize(_, _) => {}
                        _ => {}
                    }
                }
            }
        }

        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            crossterm::event::DisableMouseCapture
        )?;
        Ok(())
    }

    /// Find the byte index of the character just before `cursor_pos`.
    /// `cursor_pos` must already be on a char boundary.
    fn prev_char_boundary(s: &str, cursor_pos: usize) -> usize {
        assert!(cursor_pos <= s.len());
        let mut i = cursor_pos.saturating_sub(1);
        while i > 0 && !s.is_char_boundary(i) {
            i -= 1;
        }
        i
    }

    /// Handle mouse click on expand marker area (column 0-3).
    /// `row` and `col` are terminal coords from the mouse event.
    fn handle_expand_click(&mut self, row: u16, _col: u16) {
        let top = *self.panel_top.borrow();
        if row < top {
            return;
        }
        let relative_y = (row - top) as usize;
        if relative_y >= *self.panel_vh.borrow() {
            return;
        }
        let click_global = self.panel_start.borrow().saturating_add(relative_y);

        let line_map = self.line_map_cache.borrow();
        if click_global >= line_map.len() {
            return;
        }
        let (msg_idx, is_header) = line_map[click_global];
        if !is_header {
            return;
        }
        if let Some(msg) = self.messages.get(msg_idx) {
            if msg.can_expand {
                if !self.expanded.remove(&msg_idx) {
                    self.expanded.insert(msg_idx);
                }
            }
        }
    }

    /// Find the byte index of the character just after `cursor_pos`.
    fn next_char_boundary(s: &str, cursor_pos: usize) -> usize {
        assert!(cursor_pos <= s.len());
        let mut i = cursor_pos + 1;
        while i < s.len() && !s.is_char_boundary(i) {
            i += 1;
        }
        i
    }

    fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        if self.is_processing {
            if code == KeyCode::Esc
                || (modifiers == KeyModifiers::CONTROL && code == KeyCode::Char('c'))
            {
                self.should_quit = true;
                return;
            }
            if code == KeyCode::Enter && !self.input.trim().is_empty() {
                let input = std::mem::take(&mut self.input);
                self.cursor_position = 0;

                // Show the supplementary input as a user message
                let mb = extract_mermaid_blocks(&input);
                self.messages.push(Message {
                    role: MessageRole::User,
                    content: format!("(supplementary) {}", input),
                    full_raw: None,
                    can_expand: false,
                    timestamp: timestamp(),
                    mermaid_blocks: mb,
                });

                // Send supplementary input to the running SA via event bus
                if let Some(ref task_iri) = self.current_task_iri {
                    let bus = self.event_bus.clone();
                    let ti = task_iri.clone();
                    let inp = input;
                    self.rt.spawn(async move {
                        bus.emit(&ti, "USER_SUPPLEMENTARY_INPUT", "code_cli", &inp)
                            .await;
                    });
                }
                return;
            }
            if code == KeyCode::Enter {
                return;
            }
        }
        if self.cursor_position > self.input.len() {
            self.cursor_position = self.input.len();
        }
        if !self.input.is_char_boundary(self.cursor_position) {
            let mut i = self.cursor_position;
            while i > 0 && !self.input.is_char_boundary(i) {
                i -= 1;
            }
            self.cursor_position = i;
        }

        match code {
            KeyCode::Char(c) => {
                if modifiers == KeyModifiers::CONTROL && (c == 'd' || c == 'c') {
                    self.should_quit = true;
                } else if modifiers == KeyModifiers::CONTROL && c == 'u' {
                    self.input.clear();
                    self.cursor_position = 0;
                } else if modifiers == KeyModifiers::CONTROL && c == 'w' {
                    let before = &self.input[..self.cursor_position];
                    if let Some(pos) = before
                        .char_indices()
                        .rev()
                        .skip(1)
                        .find(|(_, ch)| ch.is_whitespace())
                        .map(|(idx, _)| idx)
                        .or_else(|| if before.is_empty() { None } else { Some(0) })
                    {
                        let end = Self::next_char_boundary(before, pos);
                        self.input.drain(end..self.cursor_position);
                        self.cursor_position = end;
                    } else {
                        self.input.drain(..self.cursor_position);
                        self.cursor_position = 0;
                    }
                } else if c == 'e' && self.input.is_empty() {
                    // Toggle expand on the most recent expandable message.
                    // Only fire when input is empty so 'e' can be typed in slash commands.
                    if let Some(idx) = self.messages.iter().rposition(|m| m.can_expand) {
                        if !self.expanded.remove(&idx) {
                            self.expanded.insert(idx);
                        }
                    }
                } else {
                    self.input.insert(self.cursor_position, c);
                    self.cursor_position += c.len_utf8();
                }
            }
            KeyCode::Backspace if self.cursor_position > 0 => {
                let start = Self::prev_char_boundary(&self.input, self.cursor_position);
                self.input.drain(start..self.cursor_position);
                self.cursor_position = start;
            }
            KeyCode::Delete if self.cursor_position < self.input.len() => {
                let end = Self::next_char_boundary(&self.input, self.cursor_position);
                self.input.drain(self.cursor_position..end);
            }
            KeyCode::Left if self.cursor_position > 0 => {
                self.cursor_position = Self::prev_char_boundary(&self.input, self.cursor_position);
            }
            KeyCode::Right if self.cursor_position < self.input.len() => {
                self.cursor_position = Self::next_char_boundary(&self.input, self.cursor_position);
            }
            KeyCode::Home => self.cursor_position = 0,
            KeyCode::End => self.cursor_position = self.input.len(),
            KeyCode::Enter if !self.input.trim().is_empty() => {
                let input = std::mem::take(&mut self.input);
                self.cursor_position = 0;
                self.start_task(&input);
            }
            KeyCode::Esc => self.should_quit = true,
            _ => {}
        }
    }

    /// Start processing a task in the background on `self.rt`.
    /// The UI continues to run and receives events + result asynchronously.
    fn start_task(&mut self, input: &str) {
        let input = input.trim().to_string();
        if input.starts_with('/') {
            self.handle_command(&input);
            return;
        }

        let mermaid_blocks = extract_mermaid_blocks(&input);
        self.messages.push(Message {
            role: MessageRole::User,
            content: input.clone(),
            full_raw: None,
            can_expand: false,
            timestamp: timestamp(),
            mermaid_blocks,
        });
        self.is_processing = true;
        self.current_phase = "SA".into();
        self.status_events.clear();
        let preview: String = input.chars().take(60).collect();
        self.add_event("TASK_START", &preview);

        // Topic shift detection: compare new input with previous input
        const TOPIC_SHIFT_THRESHOLD: f64 = 0.3;
        let is_topic_shift = if self.current_task_iri.is_some() && !self.last_user_input.is_empty()
        {
            let overlap = keyword_overlap(&self.last_user_input, &input);
            overlap < TOPIC_SHIFT_THRESHOLD
        } else {
            // First task or no previous context → new topic
            true
        };

        let task_iri = if is_topic_shift {
            // Topic shift: reset workspace perception, generate new task_iri
            if let Some(ref wm) = self.workspace_monitor {
                wm.reset_inventory();
                tracing::info!("topic shift detected, resetting workspace perception");
            }
            let task_id = uuid::Uuid::new_v4().to_string();
            format!("iri://task/{}", task_id)
        } else {
            // Same topic: reuse existing task_iri for continuity
            self.current_task_iri.take().unwrap_or_else(|| {
                let task_id = uuid::Uuid::new_v4().to_string();
                format!("iri://task/{}", task_id)
            })
        };
        self.current_task_iri = Some(task_iri.clone());
        self.last_user_input = input.clone();

        let (status_tx, status_rx) = mpsc::unbounded_channel::<StatusEvent>();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let engine = self.engine.clone();
        let input2 = input.clone();
        let task_iri_bg = task_iri.clone();
        // 仅在同任务延续时传递 resumed_messages，主题切换时丢弃
        let resumed = if is_topic_shift {
            self.resumed_messages = None; // 新任务：丢弃之前对话历史
            None
        } else {
            self.resumed_messages.take() // 同任务：传递历史用于上下文延续
        };

        self.rt.spawn(async move {
            let mut guard = engine.lock().await;
            let mut receiver = guard.subscribe();

            let stx = status_tx.clone();
            tokio::spawn(async move {
                loop {
                    match receiver.recv().await {
                        Ok(ev) => {
                            if stx
                                .send(StatusEvent {
                                    event_type: ev.event_type,
                                    payload: ev.payload,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    }
                }
            });

            let result = guard
                .process_task_with_iri_and_messages(&input2, &task_iri_bg, resumed)
                .await;
            let _ = result_tx.send(result.map(|tr| (task_iri_bg, tr)));
        });

        self.status_rx = Some(status_rx);
        self.result_rx = Some(result_rx);
    }

    fn complete_task(&mut self, result: anyhow::Result<(String, TaskResult)>) {
        // Keep current_task_iri alive for task continuity & topic shift detection on next input
        match result {
            Ok((_task_iri, tr)) => {
                // Resume 模式下不覆盖计数（事件已增量累加）
                // 非 resume 模式下同步 TaskResult 计数
                if !self.is_resume_session {
                    self.session_turn_count = tr.turn_count;
                    self.session_tool_call_count = tr.tool_call_count;
                }
                // 任务完成后清除 resume 标志
                self.is_resume_session = false;
                let (icon, role) = match tr.status.as_str() {
                    "success" => ("\u{2705}", MessageRole::Assistant),
                    "partial" => ("\u{26A0}\u{FE0F}", MessageRole::Assistant),
                    _ => ("\u{274C}", MessageRole::Error),
                };
                let output_text = tr
                    .output
                    .as_ref()
                    .and_then(|v| v.as_str())
                    .unwrap_or(&tr.summary);
                let summary = format!(
                    "{} **{}** | Turns: {} | Tools: {}\n\n{}",
                    icon,
                    tr.status.to_uppercase(),
                    tr.turn_count,
                    tr.tool_call_count,
                    output_text,
                );
                let mermaid_blocks = extract_mermaid_blocks(&summary);
                self.messages.push(Message {
                    role,
                    content: summary,
                    full_raw: None,
                    can_expand: false,
                    timestamp: timestamp(),
                    mermaid_blocks,
                });
                self.add_event("COMPLETE", &format!("{} {}", icon, tr.status));

                // ── Multi-turn conversation carryover ──
                // Capture this turn's conversation history so the next user input
                // is processed with prior context. The Vec starts with a dummy
                // "system" entry that BizAgent's resume code skips (skip(1)).
                let mut conversation: Vec<ChatMessage> = vec![ChatMessage {
                    role: "system".to_string(),
                    content: String::new(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                }];
                // Previous turns' history (if any) accumulated in resumed_messages
                if let Some(prev) = self.resumed_messages.take() {
                    // Skip the dummy system entry at index 0 — keep real user/assistant pairs
                    conversation.extend(prev.into_iter().skip(1));
                }
                // Current turn: user input → assistant response
                conversation.push(ChatMessage {
                    role: "user".to_string(),
                    content: self.last_user_input.clone(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
                conversation.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: output_text.to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
                self.resumed_messages = Some(conversation);
            }
            Err(e) => {
                self.messages.push(Message {
                    role: MessageRole::Error,
                    content: format!("\u{274C} **Error**: {}", e),
                    full_raw: None,
                    can_expand: false,
                    timestamp: timestamp(),
                    mermaid_blocks: Vec::new(),
                });
            }
        }
        self.status_rx = None; // drop mpsc receiver → kills zombie event listener
        self.is_processing = false;
        self.current_phase = "Idle".into();
        self.scroll_offset = 0;
    }

    fn drain_events(&mut self) {
        // Collect events into local vec to avoid borrow conflicts
        let batch: Vec<StatusEvent> = if let Some(rx) = &mut self.status_rx {
            let mut v = Vec::new();
            while let Ok(ev) = rx.try_recv() {
                v.push(ev);
            }
            v
        } else {
            return;
        };

        for ev in &batch {
            // Sidebar: only show major phase events (SA/PA/DA/CA/AA start/end)
            let is_major_phase = matches!(
                ev.event_type.as_str(),
                "TASK_START"
                    | "CYCLE_STARTED"
                    | "COMPLETE"
                    | "Plan_STARTED"
                    | "Plan_COMPLETED"
                    | "Do_STARTED"
                    | "Do_COMPLETED"
                    | "Check_STARTED"
                    | "Check_COMPLETED"
                    | "Act_STARTED"
                    | "Act_COMPLETED"
                    | "PA_STARTED"
                    | "PA_COMPLETED"
                    | "DA_STARTED"
                    | "DA_COMPLETED"
                    | "CA_STARTED"
                    | "CA_COMPLETED"
                    | "AA_STARTED"
                    | "AA_COMPLETED"
            );

            if is_major_phase {
                self.status_events.push(StatusEvent {
                    event_type: ev.event_type.clone(),
                    payload: ev.payload.clone(),
                });
                if self.status_events.len() > 100 {
                    self.status_events.remove(0);
                }
            }

            // Stats tracking from execution events
            match ev.event_type.as_str() {
                "TASK_START" | "CYCLE_STARTED" => {
                    // Resume 模式下不重置计数（已从 checkpoint 恢复）
                    // 使用 is_resume_session 标志判断
                    if !self.is_resume_session {
                        self.session_turn_count = 0;
                        self.session_tool_call_count = 0;
                    }
                }
                "THOUGHT" => self.session_turn_count += 1,
                "TOOL_CALL" => self.session_tool_call_count += 1,
                _ => {}
            }

            // Phase bar update from major phase events only
            if let Some(phase) = detect_phase(&ev.event_type) {
                self.current_phase = phase;
            }

            // Messages panel: show phase transitions + execution details
            if let Some((role, msg, full_raw)) = self.format_ui_message(&ev.event_type, &ev.payload)
            {
                let can_expand = full_raw.is_some();
                self.messages.push(Message {
                    role,
                    content: msg,
                    full_raw,
                    can_expand,
                    timestamp: timestamp(),
                    mermaid_blocks: Vec::new(),
                });
            }
        }
    }

    /// Try to extract a file path from tool-call arguments JSON or tool result JSON.
    /// Returns `None` when no path-like field is found or parsing fails.
    fn extract_file_path_from_args(args_json: &str) -> Option<String> {
        let v: Value = serde_json::from_str(args_json).ok()?;
        let obj = v.as_object()?;
        for key in &["path", "filePath", "pattern"] {
            if let Some(Value::String(s)) = obj.get(*key) {
                if !s.is_empty() {
                    return Some(Self::shorten_path(s));
                }
            }
        }
        if let Some(file_obj) = obj.get("file").and_then(|f| f.as_object()) {
            if let Some(Value::String(s)) = file_obj.get("filePath") {
                if !s.is_empty() {
                    return Some(Self::shorten_path(s));
                }
            }
        }
        None
    }

    fn shorten_path(s: &str) -> String {
        width_truncate(s, 60)
    }

    /// Parse TOOL_CALL arguments JSON and return a short human-readable summary.
    fn summarize_tool_args(tool_name: &str, args_json: &str) -> Option<String> {
        let v: Value = serde_json::from_str(args_json).ok()?;
        let obj = v.as_object()?;
        match tool_name {
            "bash" => {
                let cmd = obj.get("command").and_then(|c| c.as_str()).unwrap_or("");
                let desc = obj.get("description").and_then(|d| d.as_str());
                let truncated = width_truncate(cmd, 80);
                let text = format!("`{}`", truncated);
                if let Some(d) = desc {
                    if !d.is_empty() {
                        return Some(format!("{} — {}", text, d));
                    }
                }
                Some(text)
            }
            "file_read" | "file_write" | "file_edit" | "file_list" => {
                let path = obj.get("path").and_then(|p| p.as_str()).unwrap_or("");
                Some(format!("`{}`", Self::shorten_path(path)))
            }
            "grep_search" => {
                let pattern = obj.get("pattern").and_then(|p| p.as_str()).unwrap_or("");
                let path = obj.get("path").and_then(|p| p.as_str());
                let pat = width_truncate(pattern, 60);
                if let Some(p) = path {
                    if !p.is_empty() {
                        Some(format!("`{}` in `{}`", pat, Self::shorten_path(p)))
                    } else {
                        Some(format!("`{}`", pat))
                    }
                } else {
                    Some(format!("`{}`", pat))
                }
            }
            "glob_search" => {
                let p = obj.get("pattern").and_then(|p| p.as_str()).unwrap_or("");
                Some(format!("`{}`", width_truncate(p, 60)))
            }
            _ => {
                // Generic: show first string param
                for (_k, v) in obj.iter() {
                    if let Value::String(s) = v {
                        if !s.is_empty() && s.len() < 80 {
                            return Some(width_truncate(s, 60));
                        }
                    }
                }
                None
            }
        }
    }

    /// Parse TOOL_RESULT JSON and return a short human-readable preview.
    fn summarize_tool_result(tool_name: &str, result: &str) -> Option<String> {
        let v: Value = serde_json::from_str(result).ok()?;
        let obj = v.as_object()?;
        match tool_name {
            "bash" => {
                let ec = obj.get("exit_code").and_then(|c| c.as_i64()).unwrap_or(-1);
                let dur = obj.get("duration_ms").and_then(|d| d.as_u64()).unwrap_or(0);
                let stdout = obj.get("stdout").and_then(|s| s.as_str()).unwrap_or("");
                let stderr = obj.get("stderr").and_then(|s| s.as_str()).unwrap_or("");
                let snippet = if !stdout.is_empty() {
                    let s = stdout.trim();
                    width_truncate(s, 80)
                } else if !stderr.is_empty() {
                    let s = stderr.trim();
                    width_truncate(s, 80)
                } else {
                    String::new()
                };
                let base = format!("exit:{} {}ms", ec, dur);
                if snippet.is_empty() {
                    Some(base)
                } else {
                    Some(format!("{}, `{}`", base, snippet))
                }
            }
            "file_read" => {
                // execute_file_read in tool_executor.rs returns flat JSON:
                // {path, total_lines, offset, lines: [...], returned}
                let fp = obj.get("path").and_then(|p| p.as_str()).unwrap_or("");
                let total = obj.get("total_lines").and_then(|n| n.as_u64()).unwrap_or(0);
                let ret = obj.get("returned").and_then(|n| n.as_u64()).unwrap_or(0);
                // Show a preview of the first few lines
                let preview: String = obj
                    .get("lines")
                    .and_then(|l| l.as_array())
                    .map(|arr| {
                        arr.iter()
                            .take(3)
                            .filter_map(|v| v.as_str())
                            .map(|s| {
                                let t = s.trim();
                                if t.len() > 60 {
                                    width_truncate(t, 60)
                                } else {
                                    t.to_string()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default();
                let base = format!("`{}` ({}:{})", Self::shorten_path(fp), ret, total);
                if preview.is_empty() {
                    Some(base)
                } else {
                    Some(format!("{} `{}`", base, preview))
                }
            }
            _ => {
                // Generic: show first short string field
                for (_k, v) in obj.iter() {
                    if let Value::String(s) = v {
                        if !s.is_empty() && s.len() < 100 {
                            return Some(width_truncate(s, 80));
                        }
                    }
                }
                None
            }
        }
    }

    /// Format an event bus event into a clean human-readable message for the
    /// messages panel. Returns `(role, summary_text, optional_full_raw)` or
    /// `None` if the event should be silently consumed.
    fn format_ui_message(
        &self,
        event_type: &str,
        payload: &str,
    ) -> Option<(MessageRole, String, Option<String>)> {
        let max_cols = 160usize;

        // ── Phase transition events ──────────────────────────────────────────
        match event_type {
            "TASK_START" | "CYCLE_STARTED" => {
                return Some((MessageRole::System, "▶ SA".into(), None))
            }
            s if s == "Plan_STARTED" => return Some((MessageRole::System, "🔄 PA".into(), None)),
            s if s == "Plan_COMPLETED" => return Some((MessageRole::System, "✅ PA".into(), None)),
            s if s == "Do_STARTED" => return Some((MessageRole::System, "🔄 DA".into(), None)),
            s if s == "Do_COMPLETED" => return Some((MessageRole::System, "✅ DA".into(), None)),
            s if s == "Check_STARTED" => return Some((MessageRole::System, "🔄 CA".into(), None)),
            s if s == "Check_COMPLETED" => {
                return Some((MessageRole::System, "✅ CA".into(), None))
            }
            s if s == "Act_STARTED" => return Some((MessageRole::System, "🔄 AA".into(), None)),
            s if s == "Act_COMPLETED" => return Some((MessageRole::System, "✅ AA".into(), None)),
            "COMPLETE" => return Some((MessageRole::System, format!("✅ SA {}", payload), None)),
            _ => {}
        }

        // ── Detailed execution events (THOUGHT / TOOL_CALL / TOOL_RESULT / ERROR) ──
        match event_type {
            "THOUGHT" | "TOOL_CALL" | "TOOL_RESULT" | "EXECUTION_ERROR" => {}
            _ => return None,
        }

        let ee: ExecutionEvent = serde_json::from_str(payload).ok()?;
        let short_role = match &ee.event {
            ExecutionEventKind::Thought(t) => agent_id_to_role(&t.agent_id),
            ExecutionEventKind::ToolCall(t) => agent_id_to_role(&t.agent_id),
            ExecutionEventKind::ToolResult(tr) => agent_id_to_role(&tr.agent_id),
            ExecutionEventKind::Error(e) => agent_id_to_role(&e.agent_id),
            _ => return None,
        };

        match &ee.event {
            ExecutionEventKind::Thought(th) => {
                let preview = width_truncate(&th.thought, max_cols);
                let content = format!("◆ AGENT:{}:THOUGHT {}", short_role, preview);
                Some((MessageRole::System, content, Some(th.thought.clone())))
            }
            ExecutionEventKind::ToolCall(tc) => {
                let summary = Self::summarize_tool_args(&tc.tool_name, &tc.arguments_json)
                    .unwrap_or_else(|| {
                        width_truncate(&tc.arguments_json, max_cols.saturating_sub(40))
                    });
                let fp = Self::extract_file_path_from_args(&tc.arguments_json);
                let content = if let Some(ref p) = fp {
                    format!(
                        "⚡ AGENT:{}:TOOL_CALL **{}** {} `{}`",
                        short_role, tc.tool_name, summary, p
                    )
                } else {
                    format!(
                        "⚡ AGENT:{}:TOOL_CALL **{}** {}",
                        short_role, tc.tool_name, summary
                    )
                };
                Some((
                    MessageRole::System,
                    content,
                    Some(tc.arguments_json.clone()),
                ))
            }
            ExecutionEventKind::ToolResult(tr) => {
                let preview = Self::summarize_tool_result(&tr.tool_name, &tr.result)
                    .unwrap_or_else(|| width_truncate(&tr.result, max_cols.saturating_sub(40)));
                let fp = Self::extract_file_path_from_args(&tr.result);
                let content = if let Some(ref p) = fp {
                    format!(
                        "◆ AGENT:{}:TOOL_RESULT **{}** → {} `{}`",
                        short_role, tr.tool_name, preview, p
                    )
                } else {
                    format!(
                        "◆ AGENT:{}:TOOL_RESULT **{}** → {}",
                        short_role, tr.tool_name, preview
                    )
                };
                Some((MessageRole::System, content, Some(tr.result.clone())))
            }
            ExecutionEventKind::Error(err) => {
                let content = format!(
                    "❌ AGENT:{}:ERROR **{}**: {}",
                    short_role, err.error_type, err.message
                );
                Some((MessageRole::Error, content, None))
            }
            _ => None,
        }
    }

    fn add_msg(&mut self, role: MessageRole, content: String) {
        self.messages.push(Message {
            role,
            content,
            full_raw: None,
            can_expand: false,
            timestamp: timestamp(),
            mermaid_blocks: Vec::new(),
        });
    }

    fn handle_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();

        match parts[0] {
            "/exit" | "/quit" => self.should_quit = true,
            "/help" => self.add_msg(
                MessageRole::System,
                "\
**Commands**\n\
`/model <name>`  - Switch model\n\
`/apikey <key>`  - Set API key\n\
`/apiurl <url>`  - Set API URL\n\
`/clear`         - Clear history\n\
`/stats`         - Show stats\n\
`/exit`          - Quit\n\
\n\
**Keys**\n\
`Enter`  - Send\n\
`Esc`    - Quit\n\
`Ctrl+C` - Quit\n\
`Ctrl+D` - Quit\n\
`Ctrl+U` - Clear line\n\
`Ctrl+W` - Delete word\n\
`\u{2190} \u{2192}`    - Move cursor\n\
`Home`   - Line start\n\
`End`    - Line end"
                    .to_string(),
            ),
            "/model" if parts.len() > 1 => {
                if self.is_processing {
                    self.add_msg(
                        MessageRole::Error,
                        "Cannot change model while processing".to_string(),
                    );
                    return;
                }
                let new_model = parts[1].trim().to_string();
                let result = self
                    .engine
                    .blocking_lock()
                    .rebuild_with_model(new_model.clone());
                match result {
                    Ok(_) => {
                        self.model_name = new_model;
                        self.add_msg(
                            MessageRole::System,
                            format!("Model: **{}**", self.model_name),
                        );
                    }
                    Err(e) => self.add_msg(MessageRole::Error, format!("Error: {}", e)),
                }
            }
            "/apikey" if parts.len() > 1 => {
                if self.is_processing {
                    self.add_msg(
                        MessageRole::Error,
                        "Cannot change API key while processing".to_string(),
                    );
                    return;
                }
                let new_key = parts[1].trim().to_string();
                let result = self.engine.blocking_lock().rebuild_with_api_key(new_key);
                match result {
                    Ok(_) => self.add_msg(MessageRole::System, "API key updated".to_string()),
                    Err(e) => self.add_msg(MessageRole::Error, format!("Error: {}", e)),
                }
            }
            "/apiurl" if parts.len() > 1 => {
                if self.is_processing {
                    self.add_msg(
                        MessageRole::Error,
                        "Cannot change API URL while processing".to_string(),
                    );
                    return;
                }
                let new_url = parts[1].trim().to_string();
                let result = self
                    .engine
                    .blocking_lock()
                    .rebuild_with_api_url(new_url.clone());
                match result {
                    Ok(_) => self.add_msg(MessageRole::System, format!("API URL: **{}**", new_url)),
                    Err(e) => self.add_msg(MessageRole::Error, format!("Error: {}", e)),
                }
            }
            "/clear" => {
                self.messages.clear();
                self.status_events.clear();
            }
            "/stats" => {
                let (key_masked, api_url) = {
                    let engine = self.engine.blocking_lock();
                    let key_masked = if engine.api_key().len() > 8 {
                        format!(
                            "{}...{}",
                            &engine.api_key()[..4],
                            &engine.api_key()[engine.api_key().len() - 4..]
                        )
                    } else {
                        "***".to_string()
                    };
                    (key_masked, engine.api_url().to_string())
                };
                let msg = format!(
                    "**Session**\n- Model: `{}`\n- API URL: `{}`\n- API Key: `{}`\n- Workspace: `{}`\n- Max iterations: `{}`\n- Messages: `{}`",
                    self.model_name, api_url, key_masked, self.workspace_path, self.max_iter, self.messages.len()
                );
                self.add_msg(MessageRole::System, msg);
            }
            _ => self.add_msg(
                MessageRole::Error,
                format!("Unknown: `{}`. Try `/help`.", parts[0]),
            ),
        }
    }

    fn add_event(&mut self, event_type: &str, payload: &str) {
        self.status_events.push(StatusEvent {
            event_type: event_type.to_string(),
            payload: payload.to_string(),
        });
    }

    fn ui(&self, f: &mut Frame) {
        let area = f.area();

        // Top-level: status bar (full width) | everything else
        let vert = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Fill(1)])
            .split(area);

        self.render_status_bar(f, vert[0]);

        // Below status: left column (messages + input + log) | right column (sidebar stats)
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(vert[1]);

        // Left column: messages | input | log
        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Fill(1),
                Constraint::Length(5),
                Constraint::Length(4),
            ])
            .split(columns[0]);

        self.render_messages(f, left[0]);
        self.render_input(f, left[1]);
        self.render_log_panel(f, left[2]);

        // Right column: sidebar takes full remaining height
        self.render_sidebar(f, columns[1]);
    }

    fn render_status_bar(&self, f: &mut Frame, area: Rect) {
        let pc = match self.current_phase.as_str() {
            "SA" => Color::Blue,
            "PA" => Color::Cyan,
            "DA" => Color::Magenta,
            "CA" => Color::Yellow,
            "AA" => Color::Green,
            _ => Color::DarkGray,
        };
        let sc = if self.is_processing {
            Color::Yellow
        } else {
            Color::Green
        };
        let dot = if self.is_processing {
            "\u{25CF}"
        } else {
            "\u{25CB}"
        };

        // Fixed prefix width: dot + Ready/Running + separators + Model + Phase
        let status_w = if self.is_processing {
            " Running "
        } else {
            " Ready "
        }
        .width();
        let emb_w = 5 + self.embedding_provider.width() + 1;
        let prefix_w = 1 + 1 + status_w + 1 + 8 + self.model_name.width() + 2 + emb_w + 2 + 8 + 6 + 2 + 12;
        let max_path_w = (area.width as usize).saturating_sub(prefix_w);
        let workspace_display = width_truncate(&self.workspace_path, max_path_w);

        f.render_widget(Clear, area);
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(dot, Style::default().fg(sc).add_modifier(Modifier::BOLD)),
                Span::raw(" "),
                Span::styled(
                    if self.is_processing {
                        " Running "
                    } else {
                        " Ready "
                    },
                    Style::default().fg(sc),
                ),
                Span::styled("|", Style::default().fg(Color::DarkGray)),
                Span::raw(" Model: "),
                Span::styled(
                    &self.model_name,
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" |", Style::default().fg(Color::DarkGray)),
                Span::raw(" Emb: "),
                Span::styled(
                    self.embedding_provider,
                    Style::default()
                        .fg(if self.embedding_provider == "fallback" {
                            Color::Yellow
                        } else {
                            Color::Green
                        })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" |", Style::default().fg(Color::DarkGray)),
                Span::raw(" Phase: "),
                Span::styled(
                    format!("{:<6}", self.current_phase),
                    Style::default().fg(pc).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" |", Style::default().fg(Color::DarkGray)),
                Span::raw(" Workspace: "),
                Span::styled(workspace_display, Style::default().fg(Color::Green)),
            ]))
            .style(Style::default().bg(Color::Rgb(30, 30, 40))),
            area,
        );
    }

    fn render_messages(&self, f: &mut Frame, area: Rect) {
        let mut all_lines: Vec<Line<'static>> = Vec::new();
        // Track which rendered line belongs to which message index (for expand clicks).
        // Each entry: (message_index, is_header_line). Header lines of expandable messages
        // are clickable at column 0-3.
        let mut line_map: Vec<(usize, bool)> = Vec::new();

        for (idx, msg) in self.messages.iter().enumerate() {
            let (color, prefix) = match msg.role {
                MessageRole::User => (Color::Cyan, "\u{25B6} You"),
                MessageRole::Assistant => (Color::Green, "\u{25C0} Agent OS"),
                MessageRole::System => (Color::DarkGray, "\u{2139} System"),
                MessageRole::Error => (Color::Red, "\u{2716} Error"),
            };

            // Expand/collapse indicator + header line
            let expand_marker = if msg.can_expand {
                if self.expanded.contains(&idx) {
                    "[-] "
                } else {
                    "[+] "
                }
            } else {
                "    "
            };
            all_lines.push(Line::from(vec![
                Span::styled(expand_marker, Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("[{}] ", msg.timestamp),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    prefix,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
            ]));
            line_map.push((idx, msg.can_expand));

            let clean = strip_mermaid_fences(&msg.content);

            if msg.can_expand && self.expanded.contains(&idx) {
                // Expanded: extract content, detect mermaid blocks, render markdown.
                if let Some(ref raw) = msg.full_raw {
                    let display = extract_expand_content(raw);
                    let expand_mermaid = extract_mermaid_blocks(&display);
                    let clean_md = strip_mermaid_fences(&display);
                    for line in markdown_to_owned_lines(&clean_md) {
                        all_lines.push(line);
                        line_map.push((idx, false));
                    }
                    for mb in &expand_mermaid {
                        all_lines.extend(mermaid_block_lines(mb));
                        line_map.push((idx, false));
                    }
                }
                for mb in &msg.mermaid_blocks {
                    all_lines.extend(mermaid_block_lines(mb));
                    line_map.push((idx, false));
                }
                all_lines.push(Line::from(""));
                line_map.push((idx, false));
                continue;
            }

            // Collapsed (or non-expandable): show summary
            // Phase transitions (▶ SA, 🔄 PA, etc.) → colorized prefix
            // Execution events with AGENT: → colorized prefix
            if msg.role == MessageRole::System || msg.role == MessageRole::Error {
                if clean.contains("AGENT:") {
                    // Execution event
                    for line_str in clean.lines() {
                        if let Some((mut spans, rest)) = parse_execution_event_line(line_str) {
                            if !rest.is_empty() {
                                spans.push(Span::raw(" "));
                                if let Some(body) = rest.strip_prefix("**") {
                                    if let Some(tool_end) = body.find("**") {
                                        let tool_name = &body[..tool_end];
                                        let after_tool = &body[tool_end + 2..];
                                        spans.push(Span::styled(
                                            tool_name.to_string(),
                                            Style::default()
                                                .fg(tool_color(tool_name))
                                                .add_modifier(Modifier::BOLD),
                                        ));
                                        spans.push(Span::styled(
                                            after_tool.to_string(),
                                            Style::default().fg(Color::White),
                                        ));
                                    } else {
                                        spans.push(Span::styled(
                                            rest.to_string(),
                                            Style::default().fg(Color::White),
                                        ));
                                    }
                                } else {
                                    spans.push(Span::styled(
                                        rest.to_string(),
                                        Style::default().fg(Color::White),
                                    ));
                                }
                            }
                            all_lines.push(Line::from(spans));
                            line_map.push((idx, false));
                        } else {
                            for line in markdown_to_owned_lines(line_str) {
                                all_lines.push(line);
                                line_map.push((idx, false));
                            }
                        }
                    }
                } else if let Some((spans, rest)) = parse_phase_line(&clean) {
                    // Phase transition line → colored icon + role
                    let mut spans = spans;
                    if !rest.is_empty() {
                        spans.push(Span::raw(" "));
                        spans.push(Span::styled(
                            rest.to_string(),
                            Style::default().fg(Color::White),
                        ));
                    }
                    all_lines.push(Line::from(spans));
                    line_map.push((idx, false));
                } else {
                    // Plain markdown
                    for line in markdown_to_owned_lines(&clean) {
                        all_lines.push(line);
                        line_map.push((idx, false));
                    }
                }
            } else {
                // User / Assistant messages → markdown
                for line in markdown_to_owned_lines(&clean) {
                    all_lines.push(line);
                    line_map.push((idx, false));
                }
            }

            for mb in &msg.mermaid_blocks {
                all_lines.extend(mermaid_block_lines(mb));
                line_map.push((idx, false));
            }
            all_lines.push(Line::from(""));
            line_map.push((idx, false));
        }

        // Pre-wrap long lines so Paragraph::wrap does not add extra visual
        // rows that would break the 1:1 line_map ↔ screen-row mapping.
        // 预留足够列给右侧滚动条，避免滚动条区域出现正文内容字符残留。
        // 之前使用 area.width - 3 (仅1列缓冲)，在某些终端/字体/宽字符
        // 组合下正文仍会渗透到滚动条列，因此增加到 4 列缓冲。
        let content_w = (area.width.saturating_sub(6)).max(20) as usize;
        {
            let old_lines = std::mem::take(&mut all_lines);
            let old_map = std::mem::take(&mut line_map);
            for (line, entry) in old_lines.into_iter().zip(old_map) {
                for split in prewrap_line(line, content_w) {
                    all_lines.push(split);
                    line_map.push(entry);
                }
            }
        }

        let vh = area.height.saturating_sub(2) as usize;
        let all_lines_cnt = all_lines.len();
        let (visible, pct, start_line) = if all_lines_cnt <= vh {
            (all_lines, 0, 0usize)
        } else {
            let max_start = all_lines.len() - vh;
            let scroll = self.scroll_offset.min(max_start);
            let start = max_start - scroll;
            let pct = scroll * 100 / max_start;
            let visible: Vec<Line<'static>> = all_lines.into_iter().skip(start).take(vh).collect();
            (visible, pct, start)
        };

        // Store panel info for click handler
        *self.line_map_cache.borrow_mut() = line_map;
        *self.panel_top.borrow_mut() = area.y + 1; // +1 for top border
        *self.panel_vh.borrow_mut() = vh;
        *self.panel_start.borrow_mut() = start_line;

        f.render_widget(Clear, area);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(format!(" Messages ({}) [{}%] ", self.messages.len(), pct))
            .title_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            );
        f.render_widget(
            Paragraph::new(Text::from(visible))
                .block(block.clone())
                .wrap(Wrap { trim: false }),
            area,
        );

        // 滚动条独占最右列（text 已预留 1 列不写入）
        if all_lines_cnt > vh {
            let mut sb_state = ScrollbarState::new(all_lines_cnt)
                .position(start_line)
                .viewport_content_length(vh);
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(Some("│"))
                .thumb_symbol("▓")
                .style(Style::default().fg(Color::DarkGray));
            f.render_stateful_widget(
                scrollbar,
                area.inner(Margin {
                    vertical: 1,
                    horizontal: 0,
                }),
                &mut sb_state,
            );
        }
    }

    fn render_sidebar(&self, f: &mut Frame, area: Rect) {
        // ⚠️ 必须用 Min(0) 而非 Min(3)。
        //
        // cassowary 求解器优先级: MIN_SIZE_GE(强度~100k) >> LENGTH_SIZE_EQ(~10k)
        // 如果 Events 用 Min(3), 当 sidebar < 13 行时求解器会缩减 Length(10)
        // 来满足 Events 的 3 行需求, 导致 Stats 内容行被静默截断.
        //
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(13), Constraint::Fill(1)])
            .split(area);
        self.render_session_panel(f, chunks[0]);
        self.render_events_panel(f, chunks[1]);
    }

    /// Format memory usage as "used/max MB" with percentage.
    fn mem_ratio(&self, used_mb: u64, max_mb: u64) -> String {
        if max_mb == 0 {
            format!("{}/∞ MB", used_mb)
        } else {
            let pct = (used_mb as f64 / max_mb as f64 * 100.0).min(100.0);
            format!("{}/{} MB ({:.0}%)", used_mb, max_mb, pct)
        }
    }

    fn fmt_l2(&self, used_bytes: u64, max_mb: u64) -> String {
        let used_mb = used_bytes as f64 / (1024.0 * 1024.0);
        if max_mb == 0 {
            format!("{:.1}/∞ MB", used_mb)
        } else {
            let pct = (used_mb / max_mb as f64 * 100.0).min(100.0);
            format!("{:.1}/{} MB ({:.0}%)", used_mb, max_mb, pct)
        }
    }

    fn render_session_panel(&self, f: &mut Frame, area: Rect) {
        let cw = (area.width.saturating_sub(2)).max(1) as usize;
        let content_h = (area.height.saturating_sub(2)).max(1) as usize;
        let sid = self.current_task_iri.as_deref().unwrap_or("N/A");

        // 固定宽度填充，防止 ratatui diff 字符残留
        let fw = |s: &str| -> String { pad_to_width(s, cw) };

        let mut lines: Vec<Line<'static>> = Vec::with_capacity(content_h);

        lines.push(Line::from(vec![Span::styled(
            fw("Session ID"),
            Style::default().fg(Color::DarkGray),
        )]));
        lines.push(Line::from(vec![Span::styled(
            fw(sid),
            Style::default().fg(Color::Cyan),
        )]));
        lines.push(Line::from(vec![Span::styled(
            fw(&format!(
                "Turns:{:>5} Tools:{:>5}",
                self.session_turn_count, self.session_tool_call_count
            )),
            Style::default().fg(Color::White),
        )]));
        lines.push(Line::from(vec![Span::styled(
            fw(&format!(
                "L1: {}",
                self.mem_ratio(self.l1_count, self.max_l1_mb)
            )),
            Style::default().fg(Color::Yellow),
        )]));
        lines.push(Line::from(vec![Span::styled(
            fw(&format!(
                "L2: {}",
                self.fmt_l2(self.l2_count, self.max_l2_mb)
            )),
            Style::default().fg(Color::Yellow),
        )]));
        lines.push(Line::from(vec![Span::styled(
            fw(&format!(
                "L3: {}",
                self.mem_ratio(self.l3_count, self.max_l3_mb)
            )),
            Style::default().fg(Color::Yellow),
        )]));
        // Skill Graph stats
        lines.push(Line::from(vec![Span::styled(
            fw(&format!("SG: {}N {}E", self.sg_nodes, self.sg_edges)),
            Style::default().fg(Color::Green),
        )]));
        lines.push(Line::from(vec![Span::styled(
            fw(&format!(
                "TL: {}snap {:>4}pend",
                self.sg_snapshots, self.timeline_pending
            )),
            Style::default().fg(Color::Green),
        )]));
        lines.push(Line::from(vec![Span::styled(
            fw(&format!("CA: {}obs", self.causal_observations)),
            Style::default().fg(Color::Green),
        )]));
        lines.push(Line::from(vec![Span::styled(
            fw(&format!(
                "Total:{:>8}  P:{:>8}  C:{:>8}",
                fmt_k(self.total_tokens),
                fmt_k(self.prompt_tok),
                fmt_k(self.completion_tok)
            )),
            Style::default().fg(Color::White),
        )]));
        // 当前上下文用量 = 最后一次 API 调用的 prompt 体量
        let ctx_pct = if self.last_prompt_tok > 0 {
            (self.last_prompt_tok as f64 / self.context_limit as f64 * 100.0).min(99.9)
        } else {
            0.0
        };
        let ctx_label = if ctx_pct < 0.1 && self.last_prompt_tok > 0 {
            format!("{:.1}%", ctx_pct)
        } else {
            format!("{:.0}%", ctx_pct)
        };
        // prompt delta：变化时更新持久显示，无变化时保持旧值（避免闪烁）
        let p_delta = self.last_prompt_tok as i64 - self.prev_last_prompt_tok as i64;
        if p_delta > 0 {
            *self.display_delta_arrow.borrow_mut() = "↑".to_string();
            *self.display_delta_val.borrow_mut() = fmt_k(p_delta as u64);
        } else if p_delta < 0 {
            *self.display_delta_arrow.borrow_mut() = "↓".to_string();
            *self.display_delta_val.borrow_mut() = fmt_k((-p_delta) as u64);
        }
        // p_delta == 0: 保持当前显示不变
        let fg = if ctx_pct > 50.0 {
            Color::Red
        } else if ctx_pct > 30.0 {
            Color::Yellow
        } else {
            Color::White
        };
        lines.push(Line::from(vec![Span::styled(
            fw(&format!(
                "Ctx:{:>8}/{:>8} ({:>5})  {}{}",
                fmt_k(self.last_prompt_tok),
                fmt_k_short(self.context_limit),
                ctx_label,
                self.display_delta_arrow.borrow(),
                self.display_delta_val.borrow()
            )),
            Style::default().fg(fg),
        )]));

        // 用空行填充剩余空间，确保 Clear + 固定宽度占位符消除字符残留
        while lines.len() < content_h {
            lines.push(Line::from(vec![Span::raw(" ".repeat(cw))]));
        }

        // 先 clear 再 render，确保 ratatui diff 不会残留上一帧的旧字符
        f.render_widget(Clear, area);
        f.render_widget(
            Paragraph::new(Text::from(lines))
                // 左侧紧邻 messages 面板的右边框，不再重复绘制左边框
                .block(
                    Block::default()
                        .borders(Borders::ALL.difference(Borders::LEFT))
                        .border_style(Style::default().fg(Color::DarkGray))
                        .title(" Stats ")
                        .title_style(
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                ),
            area,
        );
    }

    fn render_events_panel(&self, f: &mut Frame, area: Rect) {
        let cw = (area.width.saturating_sub(4)).max(4) as usize;
        let max = (area.height.saturating_sub(2)).max(1) as usize;
        // 固定宽度填充，防止 ratatui diff 字符残留
        let fw = |s: &str| -> String { pad_to_width(s, cw) };
        let mut items: Vec<ListItem> = self
            .status_events
            .iter()
            .rev()
            .take(max)
            .map(|ev| {
                let (ic, clr) = event_icon(&ev.event_type);
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{} ", ic), Style::default().fg(clr)),
                    Span::styled(
                        fw(&format!("{} {}", ev.event_type, ev.payload)),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]))
            })
            .collect();

        // 用空行填充到固定高度，避免帧间内容变化导致字符残留
        while items.len() < max {
            items.push(ListItem::new(Line::from(vec![Span::raw(" ".repeat(cw))])));
        }

        f.render_widget(Clear, area);
        f.render_widget(
            List::new(items).block(
                Block::default()
                    .borders(Borders::ALL.difference(Borders::LEFT))
                    .border_style(Style::default().fg(Color::DarkGray))
                    .title(" Events ")
                    .title_style(
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
            ),
            area,
        );
    }

    fn render_input(&self, f: &mut Frame, area: Rect) {
        let prefix = "\u{276F} ";
        let full_text = format!("{}{}", prefix, self.input);

        let title = if self.is_processing {
            " Input (processing, Esc=quit) "
        } else {
            " Input (Enter=send, Esc=quit, Ctrl+U=clear) "
        };
        let style = Style::default().fg(Color::White);

        f.render_widget(Clear, area);
        f.render_widget(
            Paragraph::new(full_text.as_str())
                .style(style)
                .wrap(Wrap { trim: false })
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::DarkGray))
                        .title(title)
                        .title_style(
                            Style::default()
                                .fg(Color::DarkGray)
                                .add_modifier(Modifier::DIM),
                        ),
                ),
            area,
        );

        let content_w = (area.width.saturating_sub(2)).max(1) as usize;
        let prefix_w = prefix.width();
        let before_cursor = &self.input[..self.cursor_position];
        let cursor_w = before_cursor.width();
        let visual_pos = prefix_w + cursor_w;
        let row = visual_pos / content_w;
        let col = visual_pos % content_w;
        let content_h = (area.height.saturating_sub(2)).max(1) as usize;
        let row = row.min(content_h.saturating_sub(1));
        f.set_cursor_position((area.x + 1 + col as u16, area.y + 1 + row as u16));
    }

    fn render_log_panel(&self, f: &mut Frame, area: Rect) {
        if area.height < 2 || area.width < 4 {
            return;
        }
        f.render_widget(Clear, area);

        let block = Block::default()
            .title(" Log ")
            .borders(Borders::TOP | Borders::RIGHT)
            .border_style(Style::default().fg(Color::DarkGray))
            .title_style(
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM),
            );
        let inner = block.inner(area);
        f.render_widget(block, area);

        let cw = inner.width as usize;
        let max_lines = (inner.height as usize).max(1);
        let start = self.log_lines.len().saturating_sub(max_lines);
        let mut lines: Vec<Line> = self.log_lines[start..]
            .iter()
            .map(|s| {
                let cleaned = strip_log_prefix(s);
                let fixed = pad_to_width(&cleaned, cw);
                Line::from(Span::raw(fixed))
            })
            .collect();
        // 固定高度填充，清除帧间字符残留
        while lines.len() < max_lines {
            lines.push(Line::from(Span::raw(" ".repeat(cw))));
        }

        f.render_widget(
            Paragraph::new(Text::from(lines)).style(Style::default().fg(Color::DarkGray)),
            inner,
        );
    }
}

fn strip_ansi_escapes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut s = s;
    while let Some(c) = s.chars().next() {
        if c == '\x1b' {
            let rest = &s[1..];
            if let Some(next) = rest.chars().next() {
                match next {
                    '[' => {
                        let mut ci = rest[1..].char_indices();
                        let skip = loop {
                            match ci.next() {
                                Some((_off, ch)) if ((ch as u8) < 0x40 || (ch as u8) > 0x7E) => {}
                                Some((off, ch)) => break off + ch.len_utf8(),
                                None => break rest.len(),
                            }
                        };
                        s = &rest[1 + skip..];
                        continue;
                    }
                    ']' => {
                        let mut ci = rest[1..].char_indices();
                        let skip = loop {
                            match ci.next() {
                                Some((off, '\x07')) => break off + 1,
                                Some((off, '\x1b')) => {
                                    if ci.next().map_or(false, |(_, c2)| c2 == '\\') {
                                        break off + 2;
                                    }
                                }
                                Some((_off, _)) => {}
                                None => break rest.len(),
                            }
                        };
                        s = &rest[1 + skip..];
                        continue;
                    }
                    'P' | '_' | '^' | 'X' => {
                        let mut ci = rest[1..].char_indices();
                        let skip = loop {
                            match ci.next() {
                                Some((off, '\x1b')) => {
                                    if ci.next().map_or(false, |(_, c2)| c2 == '\\') {
                                        break off + 2;
                                    }
                                }
                                Some((_off, _)) => {}
                                None => break rest.len(),
                            }
                        };
                        s = &rest[1 + skip..];
                        continue;
                    }
                    _ => {
                        out.push(next);
                        s = &rest[1..];
                        continue;
                    }
                }
            } else {
                break;
            }
        } else {
            out.push(c);
            s = &s[c.len_utf8()..];
        }
    }
    out
}

fn strip_log_prefix(s: &str) -> String {
    let s = s.trim();
    let s = strip_ansi_escapes(s);
    // ISO 时间戳 + 可选 <module> + 空格 + LEVEL → 提取 LEVEL 之后的内容
    // 例如: "2026-06-12T11:14:14.6382504333<module>    WARN     [tool] ..."
    if let Some(level_end) = s
        .rfind("WARN")
        .or_else(|| s.rfind("INFO"))
        .or_else(|| s.rfind("ERRO"))
        .or_else(|| s.rfind("DEBG"))
        .or_else(|| s.rfind("TRACE"))
    {
        let after_level = &s[level_end + 4..].trim_start();
        if !after_level.is_empty() {
            return after_level.to_string();
        }
    }
    // 无 tracing 前缀的普通行
    s.to_string()
}

fn fmt_k(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1000 {
        format!("{:.1}K", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

fn fmt_k_short(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1000 {
        format!("{}K", n / 1000)
    } else {
        n.to_string()
    }
}

/// Extract user-facing content from a JSON-wrapped expandable payload.
/// Priority: stdout (bash output) > content (file content) > lines (file_read) > command.
/// Otherwise returns the pretty-printed JSON or the raw string.
fn extract_expand_content(raw: &str) -> String {
    let val = try_parse_json_in_text(raw);

    match val {
        Some(serde_json::Value::Object(ref obj)) => {
            if let Some(serde_json::Value::String(s)) = obj.get("content") {
                let trimmed = s.trim_start();
                if trimmed.starts_with('{') || trimmed.starts_with('[') {
                    let inner = extract_expand_content(s);
                    if !inner.is_empty() && inner != *s {
                        return inner;
                    }
                }
            }

            if let Some(serde_json::Value::String(stdout)) = obj.get("stdout") {
                let mut output = stdout.clone();
                if let Some(serde_json::Value::String(stderr)) = obj.get("stderr") {
                    if !stderr.is_empty() {
                        if !output.is_empty() {
                            output.push('\n');
                        }
                        output.push_str("stderr:\n");
                        output.push_str(stderr);
                    }
                }
                if !output.is_empty() {
                    return output;
                }
            }

            if let Some(serde_json::Value::String(s)) = obj.get("content") {
                return s.clone();
            }

            if let Some(serde_json::Value::Array(arr)) = obj.get("lines") {
                let joined: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                if !joined.is_empty() {
                    return joined.join("\n");
                }
            }

            if let Some(serde_json::Value::String(s)) = obj.get("command") {
                return s.clone();
            }

            serde_json::to_string_pretty(&serde_json::Value::Object(obj.clone()))
                .unwrap_or_else(|_| raw.to_string())
        }
        Some(other) => serde_json::to_string_pretty(&other).unwrap_or_else(|_| raw.to_string()),
        None => raw.to_string(),
    }
}

/// Try to parse raw as JSON; if that fails, scan for a balanced JSON object
/// within the text (useful when the result is wrapped in an injection message).
fn try_parse_json_in_text(raw: &str) -> Option<serde_json::Value> {
    if let Ok(v) = serde_json::from_str(raw) {
        return Some(v);
    }

    let bytes = raw.as_bytes();
    let len = bytes.len();
    for start in 0..len {
        if bytes[start] == b'{' {
            let mut depth: i32 = 0;
            let mut in_string = false;
            let mut escaped = false;
            for end in start..len {
                let c = bytes[end];
                if escaped {
                    escaped = false;
                } else if c == b'\\' && in_string {
                    escaped = true;
                } else if c == b'"' {
                    in_string = !in_string;
                } else if !in_string {
                    match c {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                let candidate = &raw[start..=end];
                                if let Ok(v) = serde_json::from_str(candidate) {
                                    return Some(v);
                                }
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    None
}
