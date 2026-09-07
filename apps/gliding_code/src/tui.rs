use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::io::Write;
use std::sync::Arc;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{cursor, style, terminal};
use glidinghorse::core::agent_runner::{TaskResult, TaskVerdict};
use glidinghorse::core::event_bus::{Event as AgentBusEvent, EventBus, EventFilter};
use glidinghorse::core::execution_event::{ExecutionEvent, ExecutionEventKind, ToolResult};
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

mod markdown;
#[cfg(test)]
use markdown::convert_markdown_style;
use markdown::markdown_to_owned_lines;

// `Runtime::new()` gives Tokio worker threads a small default stack.  The
// interactive path intentionally keeps the complete SA/PDCA future, stream
// accumulator, persistence hand-off and event forwarding on those workers;
// real planning requests can therefore exceed the default before an `await`
// unwinds the stack.  Keep this explicit and local to the TUI so a large
// terminal task cannot abort the entire process with a worker stack overflow.
const TUI_WORKER_STACK_BYTES: usize = 8 * 1024 * 1024;

fn build_tui_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("glidingcode-tui")
        .thread_stack_size(TUI_WORKER_STACK_BYTES)
        .build()
}

#[derive(Clone, Copy, PartialEq)]
enum MessageRole {
    User,
    Assistant,
    System,
    Warning,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct StatusEvent {
    task_iri: String,
    source_agent_iri: String,
    sequence: u64,
    event_type: String,
    payload: String,
}

const TUI_EVENT_STREAM_LAGGED: &str = "TUI_EVENT_STREAM_LAGGED";
const TUI_STATUS_CHANNEL_CAPACITY: usize = 1024;
const TUI_EVENT_DRAIN_BATCH: usize = 512;
const TUI_LOG_HISTORY_MAX_LINES: usize = 200;
const TUI_SEQUENCE_DEDUP_CAPACITY: usize = 32 * 1024;
const TUI_TERMINAL_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

impl From<AgentBusEvent> for StatusEvent {
    fn from(event: AgentBusEvent) -> Self {
        Self {
            task_iri: event.task_iri,
            source_agent_iri: event.source_agent_iri,
            sequence: event.sequence,
            event_type: event.event_type,
            payload: event.payload,
        }
    }
}

fn event_belongs_to_root_task(root_task_iri: &str, event_task_iri: &str) -> bool {
    event_task_iri == root_task_iri
        || event_task_iri
            .strip_prefix(root_task_iri)
            .is_some_and(|suffix| suffix.starts_with("/biz-agent-child/"))
}

fn is_react_turn_start_event(event: &StatusEvent) -> bool {
    event.event_type == "REACT_TURN_STARTED"
}

/// Update task progress exclusively from authoritative EventBus event kinds.
/// Provider call IDs intentionally play no part: they are request-local
/// protocol correlation values and may repeat in independent Agent/L1
/// sessions. EventBus sequence de-duplication happens before this reducer.
fn apply_progress_event(turns: &mut u32, tools: &mut u32, event: &StatusEvent) {
    if is_react_turn_start_event(event) {
        *turns = turns.saturating_add(1);
    } else if event.event_type == "TOOL_CALL" {
        *tools = tools.saturating_add(1);
    }
}

fn append_bounded_log_history(history: &mut Vec<String>, incoming: Vec<String>, max_lines: usize) {
    if max_lines == 0 {
        history.clear();
        return;
    }
    history.extend(incoming);
    let overflow = history.len().saturating_sub(max_lines);
    if overflow > 0 {
        history.drain(..overflow);
    }
}

fn task_scoped_token_total(resume_base: u64, process_current: u64, process_start: u64) -> u64 {
    resume_base.saturating_add(process_current.saturating_sub(process_start))
}

fn reset_last_context_counters(
    last_prompt: &std::sync::atomic::AtomicU64,
    last_completion: &std::sync::atomic::AtomicU64,
) {
    last_prompt.store(0, std::sync::atomic::Ordering::Relaxed);
    last_completion.store(0, std::sync::atomic::Ordering::Relaxed);
}

fn terminal_progress_uses_live_fallback(status: &str, verdict: Option<TaskVerdict>) -> bool {
    verdict == Some(TaskVerdict::Timeout) || status.eq_ignore_ascii_case("timeout")
}

/// A completed TaskResult owns the canonical aggregate. Timeout results are
/// the sole exception: cancellation can interrupt aggregation after EventBus
/// progress was already emitted, so retaining the larger observation is safer.
fn terminal_progress_count(
    observed: u32,
    terminal: u32,
    status: &str,
    verdict: Option<TaskVerdict>,
) -> u32 {
    if terminal_progress_uses_live_fallback(status, verdict) {
        observed.max(terminal)
    } else {
        terminal
    }
}

fn latest_event_sequence(event_bus: &EventBus) -> Option<u64> {
    event_bus.event_count().checked_sub(1)
}

struct EventSequenceWindow {
    baseline: Option<u64>,
    high_water: Option<u64>,
    order: VecDeque<u64>,
    seen: HashSet<u64>,
}

impl EventSequenceWindow {
    fn new(baseline: Option<u64>) -> Self {
        Self {
            baseline,
            high_water: baseline,
            order: VecDeque::new(),
            seen: HashSet::new(),
        }
    }

    fn insert(&mut self, sequence: u64) -> bool {
        if self.baseline.is_some_and(|baseline| sequence <= baseline) || !self.seen.insert(sequence)
        {
            return false;
        }
        self.order.push_back(sequence);
        self.high_water = Some(
            self.high_water
                .map_or(sequence, |current| current.max(sequence)),
        );
        while self.order.len() > TUI_SEQUENCE_DEDUP_CAPACITY {
            if let Some(expired) = self.order.pop_front() {
                self.seen.remove(&expired);
            }
        }
        true
    }

    fn high_water(&self) -> Option<u64> {
        self.high_water
    }

    fn after_baseline(&self, sequence: u64) -> bool {
        self.baseline.is_none_or(|baseline| sequence > baseline)
    }

    fn has_seen(&self, sequence: u64) -> bool {
        self.baseline.is_some_and(|baseline| sequence <= baseline) || self.seen.contains(&sequence)
    }
}

struct EventHistoryRecovery {
    events: Vec<AgentBusEvent>,
    recovered_missed: u64,
    unrecoverable: u64,
}

/// Build a deterministic, duplicate-free replay batch and measure recovery
/// only against the exact sequence interval reported missing by broadcast.
/// Already delivered history and newer concurrently emitted events are not
/// part of this gap and must neither be replayed here nor hide a real gap.
fn plan_event_history_recovery(
    sequences: &EventSequenceWindow,
    mut history: Vec<AgentBusEvent>,
    expected_start: u64,
    skipped: u64,
) -> EventHistoryRecovery {
    let expected_end = expected_start.saturating_add(skipped.saturating_sub(1));
    let expected_range = expected_start..=expected_end;
    history.sort_by_key(|event| event.sequence);

    let mut unique = HashSet::new();
    history.retain(|event| {
        expected_range.contains(&event.sequence)
            && sequences.after_baseline(event.sequence)
            && !sequences.has_seen(event.sequence)
            && unique.insert(event.sequence)
    });

    let recovered_missed = history.len() as u64;
    let unrecoverable = skipped.saturating_sub(recovered_missed.min(skipped));

    EventHistoryRecovery {
        events: history,
        recovered_missed,
        unrecoverable,
    }
}

/// Bridge the task EventBus into the bounded TUI channel without hiding
/// broadcast loss. A lagged broadcast receiver first replays the bounded
/// process-local EventBus history by global sequence. Duplicate broadcast
/// deliveries are then ignored. Only an actual history gap produces a warning.
async fn forward_status_events(
    mut receiver: tokio::sync::broadcast::Receiver<AgentBusEvent>,
    event_bus: Arc<EventBus>,
    root_task_iri: String,
    initial_sequence: Option<u64>,
    status_tx: mpsc::Sender<StatusEvent>,
    mut stop_rx: tokio::sync::oneshot::Receiver<Option<u64>>,
) {
    let mut sequences = EventSequenceWindow::new(initial_sequence);
    let mut total_unrecoverable = 0u64;

    loop {
        let received = tokio::select! {
            biased;
            boundary = &mut stop_rx => {
                // Every task event is inserted into EventBus history before
                // process_task sends its terminal oneshot result. Replaying
                // unseen history through that captured sequence closes the
                // broadcast-to-mpsc scheduling race without waiting forever
                // for an observer stream which otherwise remains open.
                if let Ok(Some(boundary)) = boundary {
                    let history_len = event_bus.history_len();
                    let mut history = event_bus.recent_events(
                        &EventFilter::default(),
                        history_len,
                    );
                    history.sort_by_key(|event| event.sequence);
                    for event in history {
                        if event.sequence > boundary || !sequences.insert(event.sequence) {
                            continue;
                        }
                        if event_belongs_to_root_task(&root_task_iri, &event.task_iri)
                            && status_tx.send(StatusEvent::from(event)).await.is_err()
                        {
                            break;
                        }
                    }
                }
                break;
            }
            received = receiver.recv() => received,
        };

        match received {
            Ok(event) => {
                if !sequences.insert(event.sequence) {
                    continue;
                }
                if event_belongs_to_root_task(&root_task_iri, &event.task_iri)
                    && status_tx.send(StatusEvent::from(event)).await.is_err()
                {
                    break;
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                let expected_start = sequences
                    .high_water()
                    .map_or(0, |sequence| sequence.saturating_add(1));
                let history_len = event_bus.history_len();
                // Query the whole history because concurrent emitters need
                // not have inserted it in sequence order. The recovery plan
                // sorts and deduplicates it, but counts only unseen events in
                // the exact broadcast gap as recovered.
                let recovery = plan_event_history_recovery(
                    &sequences,
                    event_bus.recent_events(&EventFilter::default(), history_len),
                    expected_start,
                    skipped,
                );
                let recovered_missed = recovery.recovered_missed;
                let unrecoverable = recovery.unrecoverable;

                if unrecoverable > 0 {
                    total_unrecoverable = total_unrecoverable.saturating_add(unrecoverable);
                    tracing::warn!(
                        skipped,
                        recovered = recovered_missed,
                        unrecoverable,
                        total_unrecoverable,
                        "TUI event listener lagged and bounded history could not recover every event"
                    );
                    let warning = StatusEvent {
                        task_iri: root_task_iri.clone(),
                        source_agent_iri: "TUI".to_string(),
                        sequence: sequences.high_water().unwrap_or_default(),
                        event_type: TUI_EVENT_STREAM_LAGGED.to_string(),
                        payload: serde_json::json!({
                            "skipped": unrecoverable,
                            "broadcast_skipped": skipped,
                            "recovered": recovered_missed,
                            "total_lagged": total_unrecoverable,
                        })
                        .to_string(),
                    };
                    if status_tx.send(warning).await.is_err() {
                        break;
                    }
                }

                let mut channel_closed = false;
                for event in recovery.events {
                    if !sequences.insert(event.sequence) {
                        continue;
                    }
                    if event_belongs_to_root_task(&root_task_iri, &event.task_iri)
                        && status_tx.send(StatusEvent::from(event)).await.is_err()
                    {
                        channel_closed = true;
                        break;
                    }
                }
                if channel_closed {
                    break;
                }
            }
        }
    }
}

/// Drain the bounded UI status channel while the forwarding task performs its
/// terminal history replay. The deadline keeps a damaged listener from
/// blocking the render loop; after cancellation, every event already queued
/// in the channel is still collected before returning.
async fn drain_status_until_listener_stops(
    mut status_rx: mpsc::Receiver<StatusEvent>,
    mut listener: tokio::task::JoinHandle<()>,
    timeout: std::time::Duration,
) -> (Vec<StatusEvent>, bool) {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut events = Vec::new();
    let mut channel_closed = false;
    let graceful = loop {
        tokio::select! {
            joined = &mut listener => break joined.is_ok(),
            event = status_rx.recv(), if !channel_closed => {
                match event {
                    Some(event) => events.push(event),
                    None => channel_closed = true,
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                listener.abort();
                let _ = listener.await;
                break false;
            }
        }
    };

    while let Ok(event) = status_rx.try_recv() {
        events.push(event);
    }
    (events, graceful)
}

fn abort_task_handle_bounded(
    runtime: &tokio::runtime::Runtime,
    handle: tokio::task::JoinHandle<()>,
    timeout: std::time::Duration,
) -> bool {
    handle.abort();
    runtime
        .block_on(async { tokio::time::timeout(timeout, handle).await })
        .is_ok()
}

/// Map the persisted task status to its terminal presentation. Partial
/// completion is a usable result with a warning marker, not a runtime error.
/// Keeping this policy separate from `complete_task` makes it difficult for a
/// future UI refactor to silently turn `partial_success` red again.
fn task_status_presentation(status: &str) -> (&'static str, MessageRole) {
    match status {
        "success" => ("\u{2705}", MessageRole::Assistant),
        "partial" | "partial_success" => ("\u{26A0}\u{FE0F}", MessageRole::Assistant),
        _ => ("\u{274C}", MessageRole::Error),
    }
}

/// Crossterm reports the terminal line-feed byte (`0x0a`) as Ctrl+J while a
/// carriage return is `KeyCode::Enter`. Treat both conventional terminal
/// submit sequences alike; otherwise pasted/automated input can acquire a
/// literal trailing `j` instead of being submitted.
fn is_submit_key(code: KeyCode, modifiers: KeyModifiers) -> bool {
    code == KeyCode::Enter
        || (code == KeyCode::Char('j') && modifiers.contains(KeyModifiers::CONTROL))
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
    /// Bounded prior turns. `resumed_state` independently determines whether
    /// these messages belong to durable replay or ordinary conversation.
    conversation_history: Option<Vec<glidinghorse::gateway::unified_gateway::ChatMessage>>,
    /// Canonical checkpoint state.  This is present only for an explicit
    /// startup resume, never for ordinary multi-turn conversation carryover.
    resumed_state: Option<glidinghorse::core::checkpoint::TaskResumeState>,
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
    /// Durable checkpoint token totals carried into the current resumed task.
    resume_prompt_base: u64,
    resume_completion_base: u64,
    /// Process counters are lifetime totals. Capture their value at task start
    /// so the sidebar reports this user task instead of every prior task in the
    /// same TUI process.
    task_prompt_counter_start: u64,
    task_completion_counter_start: u64,
    /// Byte-backed memory limits (MB) from config
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
    status_rx: Option<mpsc::Receiver<StatusEvent>>,
    event_listener: Option<tokio::task::JoinHandle<()>>,
    event_listener_stop: Option<tokio::sync::oneshot::Sender<Option<u64>>>,
    result_rx: Option<tokio::sync::oneshot::Receiver<anyhow::Result<(String, TaskResult)>>>,
    /// Own the task future so closing the TUI cancels it before shutdown tries
    /// to acquire the engine mutex held by that future.
    task_handle: Option<tokio::task::JoinHandle<()>>,
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
        "POLICY_GUIDANCE" => Color::Yellow,
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

#[derive(Debug)]
struct StartupDisplayState {
    stage: String,
    progress: Option<f64>,
}

/// Owns a lightweight alternate-screen renderer while the synchronous engine
/// constructor opens persistent stores. In particular, redb crash recovery
/// may scan a large L0 file several times; the user must see that progress
/// before the normal TUI can exist.
struct StartupScreen {
    state: Arc<std::sync::Mutex<StartupDisplayState>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl StartupScreen {
    fn start(workspace: &str) -> Option<Self> {
        use std::sync::atomic::Ordering;

        let state = Arc::new(std::sync::Mutex::new(StartupDisplayState {
            stage: "Starting Agent OS".to_string(),
            progress: None,
        }));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let state_for_thread = state.clone();
        let stop_for_thread = stop.clone();
        let workspace = workspace.to_string();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);

        let handle = std::thread::spawn(move || {
            let mut stdout = std::io::stdout();
            let entered = execute!(
                stdout,
                EnterAlternateScreen,
                terminal::Clear(terminal::ClearType::All),
                cursor::Hide
            )
            .is_ok();
            let _ = ready_tx.send(entered);
            if !entered {
                return;
            }

            let started = std::time::Instant::now();
            let spinner = ['◴', '◷', '◶', '◵'];
            let mut frame = 0usize;
            while !stop_for_thread.load(Ordering::Acquire) {
                let (stage, progress) = state_for_thread
                    .lock()
                    .map(|state| (state.stage.clone(), state.progress))
                    .unwrap_or_else(|_| ("Initializing".to_string(), None));
                let elapsed = started.elapsed().as_secs_f32();
                let progress_line = progress.map_or_else(
                    || "Preparing persistent services".to_string(),
                    |value| {
                        let percent = (value.clamp(0.0, 1.0) * 100.0).round() as u32;
                        let filled = (percent as usize * 32) / 100;
                        format!(
                            "[{}{}] {percent}%",
                            "█".repeat(filled),
                            "░".repeat(32usize.saturating_sub(filled))
                        )
                    },
                );
                let _ = execute!(
                    stdout,
                    cursor::MoveTo(0, 0),
                    terminal::Clear(terminal::ClearType::All),
                    cursor::MoveTo(2, 1),
                    style::Print("glidingcode · Agent OS"),
                    cursor::MoveTo(2, 3),
                    style::Print(format!("{}  {}", spinner[frame % spinner.len()], stage)),
                    cursor::MoveTo(2, 5),
                    style::Print(progress_line),
                    cursor::MoveTo(2, 7),
                    style::Print(format!("Workspace: {workspace}")),
                    cursor::MoveTo(2, 8),
                    style::Print(format!("Elapsed: {elapsed:.1}s")),
                    cursor::MoveTo(2, 10),
                    style::Print(
                        "Large stores may need recovery after an unclean exit; do not terminate the process."
                    ),
                );
                let _ = stdout.flush();
                frame += 1;
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            let _ = execute!(
                stdout,
                cursor::Show,
                LeaveAlternateScreen,
                crossterm::event::DisableMouseCapture
            );
        });

        match ready_rx.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(true) => Some(Self {
                state,
                stop,
                handle: Some(handle),
            }),
            _ => {
                stop.store(true, Ordering::Release);
                let _ = handle.join();
                None
            }
        }
    }

    fn reporter(&self) -> super::engine::StartupReporter {
        let state = self.state.clone();
        Arc::new(move |stage, progress| {
            if let Ok(mut state) = state.lock() {
                state.stage = stage.to_string();
                state.progress = progress;
            }
        })
    }

    fn update(&self, stage: &str) {
        if let Ok(mut state) = self.state.lock() {
            state.stage = stage.to_string();
            state.progress = None;
        }
    }
}

impl Drop for StartupScreen {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl App {
    pub fn new(
        config: CliConfig,
        log_buffer: Arc<LogBuffer>,
        resume_task_iri: Option<String>,
    ) -> anyhow::Result<Self> {
        let startup_screen = StartupScreen::start(&config.workspace);
        let startup_reporter = startup_screen.as_ref().map(StartupScreen::reporter);
        let max_l2_mb = config.max_l2_mb;
        let max_l3_mb = config.max_l3_mb;
        let rt = build_tui_runtime()?;
        let engine = {
            // Construct the engine inside the runtime context so subsystems
            // that capture a tokio Handle at init (e.g. WatchEngine) work.
            let _rt_guard = rt.enter();
            super::engine::CodeCliEngine::new_with_startup_reporter(config, startup_reporter)?
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

        if let Some(screen) = &startup_screen {
            screen.update("Preparing initial TUI state");
        }
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
            conversation_history: None,
            resumed_state: None,
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
            task_prompt_counter_start: 0,
            task_completion_counter_start: 0,
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
            event_listener: None,
            event_listener_stop: None,
            result_rx: None,
            task_handle: None,
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
            if let Ok(Some(restored)) = cm.restore_task(task_iri) {
                let cp = restored.checkpoint;
                // Restore current_task_iri so new input continues the same task
                app.current_task_iri = Some(task_iri.clone());

                let msgs = restored.messages;
                // 保存恢复的历史消息和结构化状态用于传递给 AgentRunner
                app.conversation_history = Some(compact_chat_history(msgs.clone()));
                app.resumed_state = Some(restored.state.clone());
                app.is_resume_session = true;

                // Restore counters from the same validated state used by SA.
                app.session_turn_count = restored.state.observed_turn_count();
                app.session_tool_call_count = restored.state.observed_tool_call_count();

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
                    let turn_str = restored.state.observed_turn_count().to_string();
                    let role_str = restored.state.current_role.as_deref().unwrap_or("?");
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

            append_bounded_log_history(
                &mut self.log_lines,
                self.log_buffer.drain(),
                TUI_LOG_HISTORY_MAX_LINES,
            );

            // Check if the background task has produced a result
            if let Some(rx) = &mut self.result_rx {
                match rx.try_recv() {
                    Ok(result) => {
                        self.result_rx = None;
                        self.settle_event_listener();
                        self.complete_task(result);
                    }
                    Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                        self.result_rx = None;
                        self.settle_event_listener();
                        self.complete_task_channel_closed();
                    }
                    Err(_) => {}
                }
            }
            self.enforce_message_budget();

            // 不要用 ? — draw 错误后继续循环，让 cleanup 有机会执行
            if self.auto_scroll {
                self.scroll_offset = 0;
            }
            // Read memory stats directly from the Arcs — no engine lock needed
            self.l2_count = self.l2_bb.total_bytes();
            self.l3_count = self.proj.cache_stats().total_size_bytes as u64;
            self.l1_count = self
                .mm
                .try_lock()
                .map(|g| g.l1_session_count())
                .unwrap_or(self.l1_count);
            // Convert process-lifetime atomics into current-task totals. A
            // resumed task also carries the durable totals saved at checkpoint.
            let current_prompt = self
                .prompt_tokens
                .load(std::sync::atomic::Ordering::Relaxed);
            let current_completion = self
                .completion_tokens
                .load(std::sync::atomic::Ordering::Relaxed);
            self.prompt_tok = task_scoped_token_total(
                self.resume_prompt_base,
                current_prompt,
                self.task_prompt_counter_start,
            );
            self.completion_tok = task_scoped_token_total(
                self.resume_completion_base,
                current_completion,
                self.task_completion_counter_start,
            );
            self.total_tokens = self.prompt_tok + self.completion_tok;
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
                                    self.scroll_offset = self.scroll_offset.saturating_sub(3);
                                    self.auto_scroll = self.scroll_offset == 0;
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
        self.stop_event_listener();
        self.cancel_active_task();
        self.rt.block_on(async {
            let lock =
                tokio::time::timeout(std::time::Duration::from_secs(5), self.engine.lock()).await;
            match lock {
                Ok(mut engine) => {
                    if tokio::time::timeout(std::time::Duration::from_secs(5), engine.shutdown())
                        .await
                        .is_err()
                    {
                        tracing::warn!("TUI engine shutdown exceeded its bounded wait");
                    }
                }
                Err(_) => tracing::warn!("TUI engine remained busy after task cancellation"),
            }
        });
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
        let submit = is_submit_key(code, modifiers);
        if self.is_processing {
            if code == KeyCode::Esc
                || (modifiers == KeyModifiers::CONTROL && code == KeyCode::Char('c'))
            {
                self.should_quit = true;
                return;
            }
            if submit && !self.input.trim().is_empty() {
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

                // Submit synchronously to the reliable task command inbox.
                // This makes the acknowledgement truthful and closes the
                // race where terminal SA processing finished before a spawned
                // emitter was first polled.
                let submission = self.current_task_iri.as_ref().map(|task_iri| {
                    self.event_bus
                        .submit_supplementary_command(task_iri, "code_cli", &input)
                });
                let acknowledgement = match submission {
                    Some(Ok(_)) => "↳ 补充指令已接收，等待当前步骤结束后合并".to_string(),
                    Some(Err(error)) => format!("↳ 补充指令未接收：{error}"),
                    None => "↳ 当前没有可接收补充指令的任务".to_string(),
                };
                self.messages.push(Message {
                    role: MessageRole::System,
                    content: acknowledgement,
                    full_raw: None,
                    can_expand: false,
                    timestamp: timestamp(),
                    mermaid_blocks: Vec::new(),
                });
                return;
            }
            if submit {
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

        if submit && !self.input.trim().is_empty() {
            let input = std::mem::take(&mut self.input);
            self.cursor_position = 0;
            self.start_task(&input);
            return;
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
        self.auto_scroll = true;
        self.scroll_offset = 0;
        self.current_phase = "SA".into();
        self.status_events.clear();
        // The log panel is task-scoped, just like the Turns/Tools/token
        // counters beside it. Retaining process-startup or prior-task lines
        // made an old ToolGuard warning appear to belong to a new task and
        // created apparent contradictions such as Tools=0 next to a stale
        // tool error. Durable logs remain available on disk; only the live
        // task view is reset here.
        self.log_lines.clear();
        let _discarded_prior_scope = self.log_buffer.drain();
        // Reset once per user task. A task may run several SA-level PDCA
        // cycles; CYCLE_STARTED must not erase facts from earlier cycles.
        if !self.is_resume_session {
            self.session_turn_count = 0;
            self.session_tool_call_count = 0;
            self.resume_prompt_base = 0;
            self.resume_completion_base = 0;
        }
        self.task_prompt_counter_start = self
            .prompt_tokens
            .load(std::sync::atomic::Ordering::Relaxed);
        self.task_completion_counter_start = self
            .completion_tokens
            .load(std::sync::atomic::Ordering::Relaxed);
        self.prompt_tok = self.resume_prompt_base;
        self.completion_tok = self.resume_completion_base;
        self.total_tokens = self.prompt_tok + self.completion_tok;
        // LastCtx is task-local presentation state. Clear both the displayed
        // values and their lock-free source before the new task is spawned, so
        // the previous task is not shown while the first request is in flight.
        reset_last_context_counters(&self.last_prompt_tokens, &self.last_completion_tokens);
        self.last_prompt_tok = 0;
        self.last_completion_tok = 0;
        self.prev_last_prompt_tok = 0;
        *self.display_delta_arrow.borrow_mut() = String::new();
        *self.display_delta_val.borrow_mut() = String::new();
        let preview: String = input.chars().take(60).collect();
        self.add_event("TASK_START", &preview);

        // Topic shift detection: compare new input with previous input
        const TOPIC_SHIFT_THRESHOLD: f64 = 0.3;
        // An explicit checkpoint resume owns one canonical original task.
        // The text typed to start the resumed run is not a new task and must
        // not discard the structured state via ordinary topic-shift logic.
        let is_checkpoint_resume = self.resumed_state.is_some();
        let is_topic_shift = if is_checkpoint_resume {
            false
        } else if self.current_task_iri.is_some() && !self.last_user_input.is_empty() {
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

        self.stop_event_listener();
        let (status_tx, status_rx) = mpsc::channel::<StatusEvent>(TUI_STATUS_CHANNEL_CAPACITY);
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let engine = self.engine.clone();
        let input2 = input.clone();
        let task_iri_bg = task_iri.clone();
        // Preserve prior turns across ordinary same-topic requests. A
        // structured state, when present, upgrades this one dispatch to an
        // explicit checkpoint replay.
        let prior_messages = if is_topic_shift {
            self.conversation_history = None;
            self.resumed_state = None;
            None
        } else {
            self.conversation_history.clone()
        };
        let resumed_state = if is_topic_shift {
            None
        } else {
            self.resumed_state.take()
        };

        let receiver = self.event_bus.subscribe();
        let (event_listener_stop_tx, event_listener_stop_rx) = tokio::sync::oneshot::channel();
        // The task has not been spawned yet, so every sequence up to this
        // baseline belongs to pre-existing traffic and must not be replayed.
        let initial_sequence = latest_event_sequence(&self.event_bus);
        self.event_listener = Some(self.rt.spawn(forward_status_events(
            receiver,
            self.event_bus.clone(),
            task_iri.clone(),
            initial_sequence,
            status_tx,
            event_listener_stop_rx,
        )));
        self.event_listener_stop = Some(event_listener_stop_tx);

        let task_handle = self.rt.spawn(async move {
            let mut guard = engine.lock().await;
            let result = guard
                .process_task_with_iri_and_resume_state(
                    &input2,
                    &task_iri_bg,
                    prior_messages,
                    resumed_state,
                )
                .await;
            let _ = result_tx.send(result.map(|tr| (task_iri_bg, tr)));
        });

        self.status_rx = Some(status_rx);
        self.result_rx = Some(result_rx);
        self.task_handle = Some(task_handle);
    }

    fn complete_task(&mut self, result: anyhow::Result<(String, TaskResult)>) {
        self.task_handle.take();
        // Keep current_task_iri alive for task continuity & topic shift detection on next input
        match result {
            Ok((_task_iri, tr)) => {
                // A normal terminal result is the canonical aggregate. A
                // phase timeout cancels the in-flight future, however, and its
                // synthetic result cannot recover counters owned inside that
                // future. Keep event-observed progress monotonically in that
                // case instead of making the sidebar jump back to zero.
                let observed_turns = self.session_turn_count;
                let observed_tools = self.session_tool_call_count;
                let timeout_fallback = terminal_progress_uses_live_fallback(&tr.status, tr.verdict);
                if observed_turns != tr.turn_count || observed_tools != tr.tool_call_count {
                    tracing::debug!(
                        task_iri = %tr.task_iri,
                        status = %tr.status,
                        verdict = ?tr.verdict,
                        observed_turns,
                        terminal_turns = tr.turn_count,
                        observed_tools,
                        terminal_tools = tr.tool_call_count,
                        resolution = if timeout_fallback { "timeout_max" } else { "terminal_aggregate" },
                        "TUI reconciled live progress with terminal task accounting"
                    );
                }
                self.session_turn_count =
                    terminal_progress_count(observed_turns, tr.turn_count, &tr.status, tr.verdict);
                self.session_tool_call_count = terminal_progress_count(
                    observed_tools,
                    tr.tool_call_count,
                    &tr.status,
                    tr.verdict,
                );
                // 任务完成后清除 resume 标志
                self.is_resume_session = false;
                let (icon, role) = task_status_presentation(&tr.status);
                let output_text = tr
                    .output
                    .as_ref()
                    .and_then(|v| v.as_str())
                    .unwrap_or(&tr.summary);
                let summary = format!(
                    "{} **{}** | Turns: {} | Tools: {}\n\n{}",
                    icon,
                    tr.status.to_uppercase(),
                    self.session_turn_count,
                    self.session_tool_call_count,
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
                // Previous turns remain available while the asynchronous task
                // runs, so successful completion can append instead of
                // accidentally replacing the entire history with one turn.
                self.conversation_history = Some(append_completed_conversation_turn(
                    self.conversation_history.take(),
                    &self.last_user_input,
                    output_text,
                ));
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
        self.stop_event_listener();
        self.is_processing = false;
        self.current_phase = "Idle".into();
        self.scroll_offset = 0;
        self.auto_scroll = true;
    }

    fn complete_task_channel_closed(&mut self) {
        self.task_handle.take();
        const MESSAGE: &str =
            "Task result channel closed before a terminal result; the background task panicked or was cancelled";
        tracing::error!(
            reason = MESSAGE,
            "TUI background task ended without TaskResult"
        );
        self.messages.push(Message {
            role: MessageRole::Error,
            content: format!("\u{274C} **Error**: {MESSAGE}"),
            full_raw: None,
            can_expand: false,
            timestamp: timestamp(),
            mermaid_blocks: Vec::new(),
        });
        self.add_event("ERROR", MESSAGE);
        self.stop_event_listener();
        self.is_processing = false;
        self.current_phase = "Error".into();
        self.scroll_offset = 0;
        self.auto_scroll = true;
    }

    fn drain_events(&mut self) {
        // Collect events into local vec to avoid borrow conflicts
        let batch: Vec<StatusEvent> = if let Some(rx) = &mut self.status_rx {
            let mut v = Vec::new();
            while v.len() < TUI_EVENT_DRAIN_BATCH {
                let Ok(ev) = rx.try_recv() else {
                    break;
                };
                v.push(ev);
            }
            v
        } else {
            return;
        };

        self.apply_status_events(batch);
    }

    fn apply_status_events(&mut self, batch: Vec<StatusEvent>) {
        for ev in &batch {
            if ev.event_type == "SA_STREAM_DELTA" {
                self.append_sa_stream_delta(&ev.payload);
                continue;
            }

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
                self.status_events.push(ev.clone());
                if self.status_events.len() > 100 {
                    self.status_events.remove(0);
                }
            }

            // Stats tracking from execution events
            match ev.event_type.as_str() {
                "TASK_START" => {
                    // Resume 模式下不重置计数（已从 checkpoint 恢复）
                    // 使用 is_resume_session 标志判断
                    if !self.is_resume_session {
                        self.session_turn_count = 0;
                        self.session_tool_call_count = 0;
                    }
                }
                "CYCLE_STARTED" => {}
                _ => apply_progress_event(
                    &mut self.session_turn_count,
                    &mut self.session_tool_call_count,
                    ev,
                ),
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

    /// Close the listener at the EventBus sequence captured after the task
    /// result became visible, drain its tail concurrently, then apply those
    /// events before terminal accounting is rendered.
    fn settle_event_listener(&mut self) {
        let Some(listener) = self.event_listener.take() else {
            self.event_listener_stop = None;
            self.drain_events();
            return;
        };

        let boundary = latest_event_sequence(&self.event_bus);
        if let Some(stop) = self.event_listener_stop.take() {
            let _ = stop.send(boundary);
        }

        let Some(status_rx) = self.status_rx.take() else {
            listener.abort();
            self.rt.spawn(async move {
                let _ = listener.await;
            });
            return;
        };

        let (events, graceful) = self.rt.block_on(drain_status_until_listener_stops(
            status_rx,
            listener,
            TUI_TERMINAL_DRAIN_TIMEOUT,
        ));
        if !graceful {
            tracing::warn!(
                boundary = ?boundary,
                timeout_ms = TUI_TERMINAL_DRAIN_TIMEOUT.as_millis(),
                drained_events = events.len(),
                "TUI terminal event drain exceeded its bounded wait"
            );
        }
        self.apply_status_events(events);
    }

    fn stop_event_listener(&mut self) {
        self.event_listener_stop = None;
        self.status_rx = None;
        if let Some(listener) = self.event_listener.take() {
            listener.abort();
            self.rt.spawn(async move {
                let _ = listener.await;
            });
        }
    }

    fn cancel_active_task(&mut self) {
        self.result_rx = None;
        let Some(handle) = self.task_handle.take() else {
            return;
        };
        if !abort_task_handle_bounded(&self.rt, handle, std::time::Duration::from_secs(2)) {
            tracing::warn!("TUI task cancellation exceeded its bounded wait");
        }
        self.is_processing = false;
    }

    fn enforce_message_budget(&mut self) {
        const MAX_MESSAGES: usize = 500;
        const MAX_MESSAGE_BYTES: usize = 2 * 1024 * 1024;
        let mut bytes: usize = self
            .messages
            .iter()
            .map(|message| {
                message.content.len()
                    + message
                        .full_raw
                        .as_ref()
                        .map_or(0, std::string::String::len)
            })
            .sum();
        let mut removed = 0;
        while self.messages.len().saturating_sub(removed) > MAX_MESSAGES
            || (bytes > MAX_MESSAGE_BYTES && self.messages.len().saturating_sub(removed) > 1)
        {
            let message = &self.messages[removed];
            bytes = bytes.saturating_sub(
                message.content.len()
                    + message
                        .full_raw
                        .as_ref()
                        .map_or(0, std::string::String::len),
            );
            removed += 1;
        }
        if removed > 0 {
            self.messages.drain(..removed);
            self.expanded.clear();
            self.line_map_cache.borrow_mut().clear();
        }
    }

    /// Append a coalesced SA SSE delta to one live TUI message. Logical turns
    /// are counted only from explicit REACT_TURN_STARTED events, so stream
    /// chunks and supervisor narration cannot inflate the counter.
    fn append_sa_stream_delta(&mut self, payload: &str) {
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            return;
        };
        let stage = value
            .get("stage")
            .and_then(Value::as_str)
            .unwrap_or("planning");
        let kind = value
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("content");
        let Some(delta) = value.get("delta").and_then(Value::as_str) else {
            return;
        };
        if delta.is_empty() {
            return;
        }

        let prefix = format!("◆ AGENT:SA:STREAM[{}:{}] ", stage, kind);
        if let Some(last) = self.messages.last_mut() {
            if last.role == MessageRole::System && last.content.starts_with(&prefix) {
                last.content.push_str(delta);
                if let Some(raw) = &mut last.full_raw {
                    raw.push_str(delta);
                }
                return;
            }
        }

        self.messages.push(Message {
            role: MessageRole::System,
            content: format!("{}{}", prefix, delta),
            full_raw: Some(delta.to_string()),
            can_expand: true,
            timestamp: timestamp(),
            mermaid_blocks: Vec::new(),
        });
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
    fn summarize_tool_result(tool_name: &str, result: &str, success: bool) -> Option<String> {
        let v: Value = serde_json::from_str(result).ok()?;
        let obj = v.as_object()?;
        // A shell process can execute and return a structured non-zero result
        // without a top-level `error`. Render its bounded process facts before
        // the generic failure path; dumping the whole JSON envelope makes a
        // normal verifier failure look like ToolGuard/log corruption.
        if !success
            && tool_name == "bash"
            && (obj.contains_key("exit_code") || obj.contains_key("timed_out"))
        {
            let exit = obj
                .get("exit_code")
                .and_then(Value::as_i64)
                .map_or_else(|| "timeout".to_string(), |code| code.to_string());
            let duration_ms = obj.get("duration_ms").and_then(Value::as_u64).unwrap_or(0);
            let diagnostic = obj
                .get("stderr")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .or_else(|| {
                    obj.get("stdout")
                        .and_then(Value::as_str)
                        .filter(|value| !value.trim().is_empty())
                })
                .map(|value| width_truncate(value.trim(), 80));
            let feedback = obj
                .get("_toolguard_validation_feedback")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(Value::as_object)
                .and_then(|item| item.get("classification"))
                .and_then(Value::as_str)
                .map(|classification| format!(" [{classification}]"))
                .unwrap_or_default();
            let base = format!("failed: exit:{exit} {duration_ms}ms{feedback}");
            return Some(match diagnostic {
                Some(diagnostic) => format!("{base}, `{diagnostic}`"),
                None => base,
            });
        }
        if !success {
            let reason = ["error", "message", "reason"]
                .into_iter()
                .find_map(|field| {
                    obj.get(field).map(|value| {
                        value
                            .as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| value.to_string())
                    })
                })
                .unwrap_or_else(|| result.to_string());
            return Some(format!("failed: {}", width_truncate(reason.trim(), 120)));
        }
        if tool_name.starts_with("read_full_result_") {
            let content = obj.get("content").and_then(Value::as_str).unwrap_or("");
            let char_offset = obj.get("char_offset").and_then(Value::as_u64).unwrap_or(0);
            let returned_chars = obj
                .get("returned_chars")
                .and_then(Value::as_u64)
                .unwrap_or_else(|| content.chars().count() as u64);
            let next = obj.get("next_char_offset").and_then(Value::as_u64);
            let snippet = width_truncate(content.trim(), 80);
            let page = if let Some(next) = next {
                format!(
                    "chars {}..{} next={}",
                    char_offset,
                    char_offset + returned_chars,
                    next
                )
            } else {
                format!(
                    "chars {}..{} complete",
                    char_offset,
                    char_offset + returned_chars
                )
            };
            return Some(if snippet.is_empty() {
                page
            } else {
                format!("{}, `{}`", page, snippet)
            });
        }
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
                let profile = obj
                    .get("execution_profile")
                    .and_then(Value::as_str)
                    .map(|profile| format!(" [{profile}]"))
                    .unwrap_or_default();
                let base = format!("exit:{} {}ms{}", ec, dur, profile);
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

    fn summarize_policy_guidance(result: &str) -> Option<String> {
        let value: Value = serde_json::from_str(result).ok()?;
        let object = value.as_object()?;
        let next_action = object
            .get("required_next_action")
            .and_then(Value::as_str)
            .unwrap_or("Revise the operation and continue the task.");
        let anti_pattern = object
            .get("guidance")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .and_then(Value::as_object)
            .and_then(|item| item.get("anti_pattern"))
            .and_then(Value::as_str);
        Some(match anti_pattern {
            Some(pattern) if !pattern.trim().is_empty() => format!(
                "policy guidance [{}]: {}",
                width_truncate(pattern.trim(), 48),
                width_truncate(next_action.trim(), 100)
            ),
            _ => format!(
                "policy guidance: {}",
                width_truncate(next_action.trim(), 110)
            ),
        })
    }

    /// Map a terminal tool event to presentation severity without changing
    /// its execution/accounting facts. Only the kernel-owned recoverable
    /// pre-execution reason is yellow guidance; every executed failure,
    /// post-hook denial, and generic policy skip remains a red error.
    fn format_tool_result_message(
        result: &ToolResult,
        short_role: &str,
        max_cols: usize,
    ) -> (MessageRole, String, Option<String>) {
        let is_guidance = result.is_recoverable_policy_guidance();
        let preview = if is_guidance {
            Self::summarize_policy_guidance(&result.result)
                .unwrap_or_else(|| "policy guidance: revise the operation and continue".to_string())
        } else {
            Self::summarize_tool_result(&result.tool_name, &result.result, result.success)
                .unwrap_or_else(|| width_truncate(&result.result, max_cols.saturating_sub(40)))
        };
        let file_path = Self::extract_file_path_from_args(&result.result);
        let (message_role, marker, action) = if result.success {
            (MessageRole::System, "◆", "TOOL_RESULT")
        } else if is_guidance {
            (MessageRole::Warning, "⚠", "POLICY_GUIDANCE")
        } else {
            (MessageRole::Error, "✖", "TOOL_RESULT")
        };
        let content = if let Some(path) = file_path {
            format!(
                "{} AGENT:{}:{} **{}** → {} `{}`",
                marker, short_role, action, result.tool_name, preview, path
            )
        } else {
            format!(
                "{} AGENT:{}:{} **{}** → {}",
                marker, short_role, action, result.tool_name, preview
            )
        };
        (message_role, content, Some(result.result.clone()))
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

        if event_type.starts_with("LLM_INTERACTION_")
            || matches!(
                event_type,
                "LLM_REQUEST_STARTED" | "LLM_REQUEST_COMPLETED" | "LLM_REQUEST_FAILED"
            )
        {
            return Self::format_llm_interaction_message(event_type, payload);
        }

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
            "DELIVERY_CONTRACT_UPDATED" => {
                let target = serde_json::from_str::<Value>(payload)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("target_path")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| "deliverable.md".to_string());
                return Some((
                    MessageRole::System,
                    format!("📄 交付已切换为工作区文件：`{target}`"),
                    Some(payload.to_string()),
                ));
            }
            "DELIVERY_RECONCILIATION_STARTED" => {
                let target = serde_json::from_str::<Value>(payload)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("target_path")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| "工作区文件".to_string());
                return Some((
                    MessageRole::System,
                    format!("📄 正在补做并验证工作区交付：`{target}`"),
                    Some(payload.to_string()),
                ));
            }
            "DELIVERY_RECONCILIATION_COMPLETED" => {
                let target = serde_json::from_str::<Value>(payload)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("target_path")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| "工作区文件".to_string());
                return Some((
                    MessageRole::System,
                    format!("✅ 工作区交付已补做并进入校验：`{target}`"),
                    Some(payload.to_string()),
                ));
            }
            "DELIVERY_RECONCILIATION_FAILED" => {
                let detail = Self::runtime_event_detail(payload, "工作区交付补做失败");
                return Some((
                    MessageRole::Error,
                    format!("❌ {detail}"),
                    Some(payload.to_string()),
                ));
            }
            TUI_EVENT_STREAM_LAGGED => {
                return Some(Self::format_event_stream_lag_message(payload));
            }
            "TURN_PERSISTENCE_STARTED" => {
                let detail = Self::runtime_event_detail(payload, "正在保存执行状态");
                return Some((
                    MessageRole::System,
                    format!("⏳ {detail}"),
                    Some(payload.to_string()),
                ));
            }
            "TURN_PERSISTENCE_COMPLETED" => {
                let detail = Self::runtime_event_detail(payload, "执行状态已保存");
                return Some((
                    MessageRole::System,
                    format!("✓ {detail}"),
                    Some(payload.to_string()),
                ));
            }
            "TURN_PERSISTENCE_FAILED" => {
                let detail = Self::runtime_event_detail(payload, "保存执行状态失败（任务继续）");
                return Some((
                    MessageRole::System,
                    format!("⚠ {detail}"),
                    Some(payload.to_string()),
                ));
            }
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
                let content = format!("◆ AGENT:{}:THOUGHT [{}] {}", short_role, th.action, preview);
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
                Some(Self::format_tool_result_message(tr, short_role, max_cols))
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

    fn runtime_event_detail(payload: &str, fallback: &str) -> String {
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            return fallback.to_string();
        };
        let role = value.get("role").and_then(Value::as_str);
        let operation = value
            .get("operation")
            .or_else(|| value.get("stage"))
            .and_then(Value::as_str)
            .unwrap_or(fallback);
        match role {
            Some(role) => format!("{role}: {operation}"),
            None => operation.to_string(),
        }
    }

    fn format_event_stream_lag_message(payload: &str) -> (MessageRole, String, Option<String>) {
        let value = serde_json::from_str::<Value>(payload).unwrap_or(Value::Null);
        let skipped = value.get("skipped").and_then(Value::as_u64).unwrap_or(0);
        let total = value
            .get("total_lagged")
            .and_then(Value::as_u64)
            .unwrap_or(skipped);
        (
            MessageRole::Error,
            format!(
                "⚠ 事件流过载：本次丢失 {skipped} 条，任务累计丢失 {total} 条；当前调试视图可能不完整"
            ),
            None,
        )
    }

    /// Render only an allowlisted subset of the metadata-only LLM lifecycle.
    /// Request/response bodies, reasoning, tool arguments and raw event JSON
    /// are deliberately never returned to the normal TUI message panel.
    fn format_llm_interaction_message(
        event_type: &str,
        payload: &str,
    ) -> Option<(MessageRole, String, Option<String>)> {
        match event_type {
            // Useful for a trace/details view, but too noisy for the normal
            // conversation panel. STARTED already proves dispatch happened.
            "LLM_INTERACTION_ASSEMBLED" | "LLM_INTERACTION_FIRST_TOKEN" => return None,
            // The interaction facade emits the same lifecycle with stronger
            // correlation. Ignore legacy mirrors to avoid two rows per turn.
            "LLM_REQUEST_STARTED" | "LLM_REQUEST_COMPLETED" | "LLM_REQUEST_FAILED" => {
                return None;
            }
            "LLM_INTERACTION_STARTED"
            | "LLM_INTERACTION_COMPLETED"
            | "LLM_INTERACTION_FAILED"
            | "LLM_INTERACTION_CANCELLED" => {}
            _ => return None,
        }

        let value = serde_json::from_str::<Value>(payload).unwrap_or(Value::Null);
        let scope = value.get("scope").and_then(Value::as_object);
        let optional_control_fallback = scope.and_then(|scope| {
            let complete_scope = ["interaction_id", "task_iri", "agent_id", "role", "stage"]
                .iter()
                .all(|field| {
                    scope
                        .get(*field)
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.trim().is_empty())
                });
            if !complete_scope {
                return None;
            }
            match scope.get("stage").and_then(Value::as_str) {
                Some("bizagent_aggregate") => Some("deterministic aggregation retained"),
                Some("bizagent_decompose") => Some("safe canonical/MONO fallback engaged"),
                _ => None,
            }
        });
        let role = Self::safe_metadata_field(
            scope
                .and_then(|scope| scope.get("role"))
                .and_then(Value::as_str),
            "LLM",
            12,
        );
        let stage = Self::safe_metadata_field(
            scope
                .and_then(|scope| scope.get("stage"))
                .and_then(Value::as_str),
            "unknown",
            28,
        );
        let interaction_id = Self::compact_metadata_id(
            scope
                .and_then(|scope| scope.get("interaction_id"))
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
        );
        let identity = format!("{role}/{stage} [{interaction_id}]");
        let elapsed = value
            .get("elapsed_ms")
            .and_then(Value::as_u64)
            .map(|millis| format!("{millis}ms"));
        let model_dispatch_count = value.get("model_dispatch_count").and_then(Value::as_u64);
        let billed_usage = value
            .get("billed_prompt_tokens")
            .and_then(Value::as_u64)
            .zip(
                value
                    .get("billed_completion_tokens")
                    .and_then(Value::as_u64),
            );
        let append_dispatch_billing = |details: &mut Vec<String>| {
            if let Some(dispatches) = model_dispatch_count.filter(|count| *count > 0) {
                details.push(if dispatches == 1 {
                    "1 model dispatch".to_string()
                } else {
                    format!("{dispatches} model dispatches")
                });
            }
            if let Some((prompt, completion)) = billed_usage {
                details.push(format!("billed tokens {prompt}+{completion}"));
            }
        };

        let (message_role, message) = match event_type {
            "LLM_INTERACTION_STARTED" => {
                let model = Self::safe_metadata_field(
                    value.get("model").and_then(Value::as_str),
                    "unknown-model",
                    36,
                );
                let message_count = value
                    .get("message_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let tool_count = value
                    .get("advertised_tool_names")
                    .and_then(Value::as_array)
                    .map(Vec::len)
                    .unwrap_or(0);
                let transport = if value
                    .get("streaming")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    "stream"
                } else {
                    "sync"
                };
                let context = Self::llm_context_receipt_summary(&value);
                let available_tool_schemas = if tool_count == 1 {
                    "1 available tool schema".to_string()
                } else {
                    format!("{tool_count} available tool schemas")
                };
                (
                    MessageRole::System,
                    format!(
                        "⏳ LLM {identity} · {transport} · {context} · {model} · {message_count} messages · {available_tool_schemas}"
                    ),
                )
            }
            "LLM_INTERACTION_COMPLETED" => {
                let prompt_tokens = value.get("prompt_tokens").and_then(Value::as_u64);
                let completion_tokens = value.get("completion_tokens").and_then(Value::as_u64);
                let finish_reason = value
                    .get("finish_reason")
                    .and_then(Value::as_str)
                    .map(|reason| Self::safe_metadata_field(Some(reason), "unknown", 24));
                let incomplete_finish = finish_reason.as_deref().is_some_and(|reason| {
                    matches!(
                        reason.to_ascii_lowercase().as_str(),
                        "length"
                            | "max_tokens"
                            | "max_output_tokens"
                            | "incomplete"
                            | "content_filter"
                            | "error"
                    )
                });
                let response_tool_kinds = value
                    .get("response_tool_names")
                    .and_then(Value::as_array)
                    .map(Vec::len)
                    .unwrap_or(0);
                // Schema v5+ separates exact calls from the de-duplicated tool
                // inventory. Fall back to the inventory length for older
                // persisted/event-stream payloads.
                let response_tool_count = value
                    .get("response_tool_call_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(response_tool_kinds as u64);
                let visible_content_empty = value
                    .get("content_empty")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                    && response_tool_count == 0
                    && response_tool_kinds == 0;
                let mut details = Vec::new();
                if let Some(elapsed) = elapsed {
                    details.push(elapsed);
                }
                append_dispatch_billing(&mut details);
                let accepted_usage = prompt_tokens.zip(completion_tokens);
                if billed_usage.is_none()
                    && (prompt_tokens.is_some() || completion_tokens.is_some())
                {
                    details.push(format!(
                        "tokens {}+{}",
                        prompt_tokens.unwrap_or(0),
                        completion_tokens.unwrap_or(0)
                    ));
                } else if accepted_usage.is_some_and(|usage| Some(usage) != billed_usage) {
                    let (prompt, completion) = accepted_usage.unwrap_or_default();
                    details.push(format!("accepted tokens {prompt}+{completion}"));
                }
                if response_tool_count > 0 {
                    let calls = if response_tool_count == 1 {
                        "1 tool call".to_string()
                    } else {
                        format!("{response_tool_count} tool calls")
                    };
                    if response_tool_kinds > 0 && response_tool_kinds as u64 != response_tool_count
                    {
                        let kinds = if response_tool_kinds == 1 {
                            "1 kind".to_string()
                        } else {
                            format!("{response_tool_kinds} kinds")
                        };
                        details.push(format!("{calls} ({kinds})"));
                    } else {
                        details.push(calls);
                    }
                }
                if incomplete_finish {
                    details.push(format!(
                        "finish {}",
                        finish_reason.as_deref().unwrap_or("incomplete")
                    ));
                }
                if visible_content_empty {
                    details.push("visible content empty".to_string());
                }
                let suffix = if details.is_empty() {
                    String::new()
                } else {
                    format!(" · {}", details.join(" · "))
                };
                if incomplete_finish || visible_content_empty {
                    (
                        MessageRole::Error,
                        format!("⚠ LLM {identity} completed without a usable response{suffix}"),
                    )
                } else {
                    (MessageRole::System, format!("✓ LLM {identity}{suffix}"))
                }
            }
            "LLM_INTERACTION_FAILED" => {
                let error_class = Self::safe_metadata_field(
                    value.get("error_class").and_then(Value::as_str),
                    "unknown_error",
                    48,
                );
                let error_summary = Self::llm_error_class_summary(&error_class);
                let mut details = elapsed.into_iter().collect::<Vec<_>>();
                // A successful provider response may later be rejected by a
                // response hook. Do not present that transport's HTTP 200 as
                // though it caused the failure; status/retryability metadata
                // is diagnostic only for provider HTTP failures.
                if error_class.starts_with("provider_http_") {
                    if let Some(status) = value
                        .get("http_status")
                        .and_then(Value::as_u64)
                        .filter(|status| (100..=599).contains(status))
                    {
                        details.push(format!("HTTP {status}"));
                    }
                    if let Some(retryable) = value.get("retryable").and_then(Value::as_bool) {
                        details.push(if retryable {
                            "retryable".to_string()
                        } else {
                            "not retryable".to_string()
                        });
                    }
                }
                append_dispatch_billing(&mut details);
                let suffix = (!details.is_empty())
                    .then(|| format!(" · {}", details.join(" · ")))
                    .unwrap_or_default();
                if let Some(fallback) = optional_control_fallback {
                    (
                        MessageRole::System,
                        format!("⚠ LLM {identity} failed: {error_summary}{suffix} · {fallback}"),
                    )
                } else {
                    (
                        MessageRole::Error,
                        format!("❌ LLM {identity} failed: {error_summary}{suffix}"),
                    )
                }
            }
            "LLM_INTERACTION_CANCELLED" => {
                let mut details = elapsed.into_iter().collect::<Vec<_>>();
                append_dispatch_billing(&mut details);
                let suffix = (!details.is_empty())
                    .then(|| format!(" · {}", details.join(" · ")))
                    .unwrap_or_default();
                if let Some(fallback) = optional_control_fallback {
                    (
                        MessageRole::System,
                        format!("⚠ LLM {identity} cancelled{suffix} · {fallback}"),
                    )
                } else {
                    (
                        MessageRole::Error,
                        format!("⚠ LLM {identity} cancelled{suffix}"),
                    )
                }
            }
            _ => unreachable!("event type was matched above"),
        };

        Some((message_role, width_truncate(&message, 160), None))
    }

    fn llm_context_receipt_summary(event: &Value) -> String {
        let null = Value::Null;
        let receipt = event.get("context_receipt").unwrap_or(&null);
        let messages = receipt
            .get("request_messages")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| {
                event
                    .get("message_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            });
        let chars = receipt
            .get("request_chars")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let dispositions = receipt.get("dispositions").unwrap_or(&null);
        let count = |name: &str| dispositions.get(name).and_then(Value::as_u64).unwrap_or(0);
        let truncated = count("truncated");
        let expired = count("dropped_expired");
        let scope = count("dropped_scope");
        let role = count("dropped_role_policy");
        let budget = count("dropped_budget");
        let dropped = expired
            .saturating_add(scope)
            .saturating_add(role)
            .saturating_add(budget);

        // This receipt counts model-visible ChatMessage content only. It is
        // intentionally not labelled as provider tokens or full wire bytes,
        // which also include tool schemas and transport framing.
        let mut parts = vec![format!("msgctx {messages}msg/{chars}ch")];
        if truncated > 0 {
            parts.push(format!("cut{truncated}"));
        }
        if dropped > 0 {
            parts.push(format!(
                "drop{dropped}(e{expired}/s{scope}/r{role}/b{budget})"
            ));
        }
        if receipt
            .get("required_budget_exceeded")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            parts.push("required!".to_string());
        }
        parts.join(" ")
    }

    fn llm_error_class_summary(error_class: &str) -> String {
        match error_class {
            "internal" => "internal (provider/transport/runtime; inspect Log)".to_string(),
            "output_token_limit" => {
                "output_token_limit (reasoning/output budget exhausted)".to_string()
            }
            "interaction_rejected" => "interaction_rejected (policy denied request)".to_string(),
            "request_hook_rejected" => "request_hook_rejected (pre-request policy)".to_string(),
            "response_hook_rejected" => "response_hook_rejected (post-response policy)".to_string(),
            "response_hook_retry_exhausted" => {
                "response_hook_retry_exhausted (policy retry budget)".to_string()
            }
            "transport_timeout" => "transport_timeout (provider request timed out)".to_string(),
            "transport_connect" => "transport_connect (provider connection failed)".to_string(),
            "transport_decode" => "transport_decode (provider body encoding failed)".to_string(),
            "transport_body" => "transport_body (provider response ended unexpectedly)".to_string(),
            "transport_request" => {
                "transport_request (provider request transport failed)".to_string()
            }
            "provider_http_transient" => {
                "provider_http_transient (retryable provider status)".to_string()
            }
            "provider_http_client" => {
                "provider_http_client (non-retryable provider status)".to_string()
            }
            "provider_response_json" => {
                "provider_response_json (success body was not JSON)".to_string()
            }
            "provider_response_invalid" => {
                "provider_response_invalid (response schema mismatch)".to_string()
            }
            "stream_protocol" => "stream_protocol (malformed provider SSE frame)".to_string(),
            "stream_transport_timeout" => {
                "stream_transport_timeout (provider stream timed out)".to_string()
            }
            "stream_transport_connect" => {
                "stream_transport_connect (stream connection failed)".to_string()
            }
            "stream_transport_decode" => {
                "stream_transport_decode (stream body encoding failed)".to_string()
            }
            "stream_transport_body" => {
                "stream_transport_body (stream body ended unexpectedly)".to_string()
            }
            "stream_transport" => "stream_transport (provider stream transport failed)".to_string(),
            // Retain a readable label for evidence produced by binaries prior
            // to structured stream failure classes.
            "stream_decode" => "stream_decode (legacy undifferentiated stream failure)".to_string(),
            other => other.to_string(),
        }
    }

    fn safe_metadata_field(value: Option<&str>, fallback: &str, max_width: usize) -> String {
        let value = value
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(fallback);
        let sanitized = value
            .chars()
            .map(|character| {
                if character.is_control() {
                    ' '
                } else {
                    character
                }
            })
            .collect::<String>();
        width_truncate(sanitized.trim(), max_width)
    }

    fn compact_metadata_id(value: &str) -> String {
        let sanitized = Self::safe_metadata_field(Some(value), "unknown", 128);
        let chars = sanitized.chars().collect::<Vec<_>>();
        if chars.len() <= 24 {
            return sanitized;
        }
        format!(
            "{}…{}",
            chars[..11].iter().collect::<String>(),
            chars[chars.len() - 10..].iter().collect::<String>()
        )
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
            task_iri: self.current_task_iri.clone().unwrap_or_default(),
            source_agent_iri: "TUI".to_string(),
            sequence: 0,
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
        let prefix_w =
            1 + 1 + status_w + 1 + 8 + self.model_name.width() + 2 + emb_w + 2 + 8 + 6 + 2 + 12;
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
                MessageRole::Warning => (Color::Yellow, "\u{26A0} Warning"),
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
            if matches!(
                msg.role,
                MessageRole::System | MessageRole::Warning | MessageRole::Error
            ) {
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
            fw(&format!("L1: {} active", self.l1_count)),
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
                self.fmt_l2(self.l3_count, self.max_l3_mb)
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
                "TaskTok:{:>6}  P:{:>8}  C:{:>8}",
                fmt_k(self.total_tokens),
                fmt_k(self.prompt_tok),
                fmt_k(self.completion_tok)
            )),
            Style::default().fg(Color::White),
        )]));
        // LastCtx is the prompt size of the most recently completed provider
        // call with usage metadata. It is not the aggregate or an in-flight
        // concurrent request's current context size.
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
                "LastCtx:{:>8}/{:>8} ({:>5})  {}{}",
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

fn truncate_utf8(value: &mut String, max_bytes: usize) {
    if value.len() <= max_bytes {
        return;
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value.truncate(end);
    value.push_str("\n[history truncated]");
}

fn compact_chat_history(mut messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    const MAX_HISTORY_MESSAGES: usize = 200;
    const MAX_HISTORY_BYTES: usize = 1024 * 1024;
    for message in &mut messages {
        truncate_utf8(&mut message.content, 256 * 1024);
        if let Some(reasoning) = &mut message.reasoning_content {
            truncate_utf8(reasoning, 128 * 1024);
        }
        if let Some(tool_calls) = &mut message.tool_calls {
            for call in tool_calls {
                truncate_utf8(&mut call.function.arguments, 128 * 1024);
            }
        }
    }

    let leading_system = messages
        .first()
        .filter(|message| message.role == "system")
        .cloned();
    let mut kept = Vec::new();
    let mut bytes = 0;
    for message in messages.into_iter().rev() {
        let size = serde_json::to_vec(&message).map_or(message.content.len(), |value| value.len());
        if kept.len() >= MAX_HISTORY_MESSAGES
            || (!kept.is_empty() && bytes + size > MAX_HISTORY_BYTES)
        {
            break;
        }
        bytes = bytes.saturating_add(size);
        kept.push(message);
    }
    kept.reverse();
    if let Some(system) = leading_system {
        if kept.first().is_none_or(|message| message.role != "system") {
            kept.insert(0, system);
        }
    }
    kept
}

fn append_completed_conversation_turn(
    prior: Option<Vec<ChatMessage>>,
    user: &str,
    assistant: &str,
) -> Vec<ChatMessage> {
    let mut conversation = vec![ChatMessage {
        role: "system".to_string(),
        content: String::new(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    if let Some(prior) = prior {
        // Stale system messages are never carried into a new task contract.
        conversation.extend(prior.into_iter().filter(|message| message.role != "system"));
    }
    conversation.push(ChatMessage {
        role: "user".to_string(),
        content: user.to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    conversation.push(ChatMessage {
        role: "assistant".to_string(),
        content: assistant.to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    compact_chat_history(conversation)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui_core::style::{Color as MarkdownColor, Modifier as MarkdownModifier};

    #[test]
    fn terminal_carriage_return_and_line_feed_both_submit_without_inserting_j() {
        assert!(is_submit_key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(is_submit_key(KeyCode::Char('j'), KeyModifiers::CONTROL));
        assert!(!is_submit_key(KeyCode::Char('j'), KeyModifiers::NONE));
    }

    #[test]
    fn aborting_pending_task_releases_engine_lock_without_hanging() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("test runtime");
        let engine = Arc::new(tokio::sync::Mutex::new(()));
        let task_engine = engine.clone();
        let (locked_tx, locked_rx) = std::sync::mpsc::sync_channel(0);

        // This models a provider call that never returns while the TUI task
        // owns the engine mutex -- the deadlock shape that the exit path must
        // break before it invokes engine shutdown.
        let handle = runtime.spawn(async move {
            let _engine_guard = task_engine.lock().await;
            locked_tx.send(()).expect("test receiver remains alive");
            std::future::pending::<()>().await;
        });
        locked_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("task must acquire the engine lock");

        let started = std::time::Instant::now();
        assert!(abort_task_handle_bounded(
            &runtime,
            handle,
            std::time::Duration::from_millis(500),
        ));
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "cancellation must remain bounded"
        );

        let lock_reacquired = runtime.block_on(async {
            tokio::time::timeout(std::time::Duration::from_millis(500), engine.lock()).await
        });
        assert!(
            lock_reacquired.is_ok(),
            "aborting the task must release the engine lock for shutdown"
        );
    }

    #[test]
    fn partial_success_is_rendered_as_assistant_warning_not_error() {
        for status in ["partial", "partial_success"] {
            let (icon, role) = task_status_presentation(status);
            assert_eq!(icon, "\u{26A0}\u{FE0F}");
            assert!(matches!(role, MessageRole::Assistant));
            assert!(!matches!(role, MessageRole::Error));
        }

        assert!(matches!(
            task_status_presentation("failed").1,
            MessageRole::Error
        ));
    }

    #[test]
    fn tool_result_presentation_distinguishes_policy_guidance_from_real_failures() {
        use glidinghorse::core::execution_event::tool_terminal_reason;
        use glidinghorse::core::execution_journal::ToolCallIdentity;

        let identity =
            ToolCallIdentity::new("cycle_1_DA_worker", "l1-da", "request-da-1", "call_0");
        let guidance_payload = serde_json::json!({
            "status": "not_executed",
            "classification": "recoverable_methodology_constraint",
            "recoverable": true,
            "original_operation_executed": false,
            "required_next_action": "Use a precise path-scoped command, then continue.",
            "guidance": [{"anti_pattern": "blind full scan"}]
        })
        .to_string();
        let guidance = ToolResult::from_identity(
            &identity,
            "bash",
            &guidance_payload,
            false,
            false,
            Some(tool_terminal_reason::RECOVERABLE_POLICY_GUIDANCE),
            guidance_payload.len() as u32,
            0,
        );
        let (guidance_role, guidance_message, _) =
            App::format_tool_result_message(&guidance, "DA", 160);
        assert!(matches!(guidance_role, MessageRole::Warning));
        assert!(guidance_message.starts_with("⚠ AGENT:DA:POLICY_GUIDANCE **bash**"));
        assert!(guidance_message.contains("blind full scan"));
        assert!(!guidance_message.contains("failed:"));

        let verifier_guidance_payload = serde_json::json!({
            "status": "not_executed",
            "reason": "verification_command_not_attributable",
            "required_next_action": "Invoke exactly one foreground verifier."
        })
        .to_string();
        let verifier_guidance = ToolResult::from_identity(
            &identity,
            "bash",
            &verifier_guidance_payload,
            false,
            false,
            Some(tool_terminal_reason::VERIFICATION_COMMAND_NOT_ATTRIBUTABLE),
            verifier_guidance_payload.len() as u32,
            0,
        );
        let (role, message, _) = App::format_tool_result_message(&verifier_guidance, "CA", 160);
        assert!(matches!(role, MessageRole::Warning));
        assert!(message.starts_with("⚠ AGENT:CA:POLICY_GUIDANCE **bash**"));
        assert!(message.contains("Invoke exactly one foreground verifier"));

        for (reason, executed) in [
            (tool_terminal_reason::EXECUTION_FAILED, true),
            (tool_terminal_reason::RESULT_DISCLOSURE_DENIED, true),
            (tool_terminal_reason::SKILL_BEFORE_SKIPPED, false),
        ] {
            let failed = ToolResult::from_identity(
                &identity,
                "bash",
                r#"{"error":"operation failed"}"#,
                false,
                executed,
                Some(reason),
                28,
                1,
            );
            let (role, message, _) = App::format_tool_result_message(&failed, "DA", 160);
            assert!(matches!(role, MessageRole::Error), "reason={reason}");
            assert!(message.starts_with("✖ AGENT:DA:TOOL_RESULT **bash**"));
            assert!(message.contains("failed:"));
        }
    }

    #[test]
    fn recoverable_policy_guidance_does_not_change_tool_or_turn_accounting() {
        let event = |sequence: u64, event_type: &str| StatusEvent {
            task_iri: "iri://task/root".to_string(),
            source_agent_iri: "cycle_1_DA_worker".to_string(),
            sequence,
            event_type: event_type.to_string(),
            payload: "{}".to_string(),
        };
        let mut turns = 0;
        let mut tools = 0;
        for item in [
            event(1, "REACT_TURN_STARTED"),
            event(2, "TOOL_CALL"),
            event(3, "TOOL_RESULT"),
        ] {
            apply_progress_event(&mut turns, &mut tools, &item);
        }
        assert_eq!((turns, tools), (1, 1));
    }

    #[test]
    fn markdown_style_conversion_preserves_colors_and_modifiers() {
        let source = ratatui_core::style::Style {
            fg: Some(MarkdownColor::Rgb(12, 34, 56)),
            bg: Some(MarkdownColor::Indexed(123)),
            add_modifier: MarkdownModifier::BOLD | MarkdownModifier::ITALIC,
            sub_modifier: MarkdownModifier::DIM | MarkdownModifier::UNDERLINED,
        };

        let converted = convert_markdown_style(source);

        assert_eq!(converted.fg, Some(Color::Rgb(12, 34, 56)));
        assert_eq!(converted.bg, Some(Color::Indexed(123)));
        assert_eq!(converted.add_modifier, Modifier::BOLD | Modifier::ITALIC);
        assert_eq!(converted.sub_modifier, Modifier::DIM | Modifier::UNDERLINED);
    }

    #[test]
    fn markdown_rendering_keeps_inline_emphasis() {
        let lines = markdown_to_owned_lines("# Heading\n\nplain **bold** and *italic*");
        let spans: Vec<&Span<'static>> = lines.iter().flat_map(|line| line.spans.iter()).collect();

        assert!(spans.iter().any(|span| {
            span.content.contains("bold") && span.style.add_modifier.contains(Modifier::BOLD)
        }));
        assert!(spans.iter().any(|span| {
            span.content.contains("italic") && span.style.add_modifier.contains(Modifier::ITALIC)
        }));
    }

    #[test]
    fn resumed_history_is_bounded_and_preserves_latest_unicode_message() {
        let mut messages = vec![ChatMessage {
            role: "system".to_string(),
            content: String::new(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        for index in 0..300 {
            messages.push(ChatMessage {
                role: "user".to_string(),
                content: format!("{index}:{}", "滑翔马".repeat(2000)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }

        let compacted = compact_chat_history(messages);
        assert!(compacted.len() <= 201);
        assert_eq!(compacted.first().unwrap().role, "system");
        assert!(compacted.last().unwrap().content.starts_with("299:"));
        let bytes = serde_json::to_vec(&compacted).unwrap().len();
        assert!(bytes <= 1024 * 1024 + 1024);
    }

    #[test]
    fn completed_turn_appends_history_and_discards_stale_system_messages() {
        let prior = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "STALE_POLICY".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "first request".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "first response".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let history = append_completed_conversation_turn(
            Some(prior),
            "write it to the workspace",
            "saved report.md",
        );

        assert_eq!(
            history
                .iter()
                .filter(|message| message.role == "system")
                .count(),
            1
        );
        assert!(!history
            .iter()
            .any(|message| message.content.contains("STALE_POLICY")));
        let contents = history
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>();
        assert!(contents.contains(&"first request"));
        assert!(contents.contains(&"first response"));
        assert!(contents.contains(&"write it to the workspace"));
        assert!(contents.contains(&"saved report.md"));
    }

    #[test]
    fn llm_interaction_summary_is_metadata_only_and_bounded() {
        let payload = serde_json::json!({
            "schema_version": 1,
            "scope": {
                "interaction_id": "llm_12345678-1234-1234-1234-123456789abc",
                "task_iri": "iri://task/secret-task-name",
                "agent_id": "agent_do_private",
                "role": "DA",
                "stage": "agent_react\nforged-line"
            },
            "model": "production-model",
            "streaming": false,
            "message_count": 4,
            "advertised_tool_names": ["file_read", "file_write"],
            "elapsed_ms": 127,
            "prompt_tokens": 321,
            "completion_tokens": 45,
            "billed_prompt_tokens": 321,
            "billed_completion_tokens": 45,
            "model_dispatch_count": 1,
            "response_tool_names": ["file_read"],
            "response_tool_call_count": 3,
            "request_prompt": "TOP_SECRET_PROMPT",
            "model_response": "TOP_SECRET_RESPONSE",
            "tool_arguments": "TOP_SECRET_ARGUMENTS"
        })
        .to_string();

        let (role, message, full_raw) =
            App::format_llm_interaction_message("LLM_INTERACTION_COMPLETED", &payload)
                .expect("completed interaction must be visible");

        assert!(matches!(role, MessageRole::System));
        assert!(message.contains("DA/agent_react forged-line"));
        assert!(message.contains("127ms"));
        assert!(message.contains("1 model dispatch"));
        assert!(message.contains("billed tokens 321+45"));
        assert!(message.contains("3 tool calls (1 kind)"));
        assert!(message.width() <= 160);
        assert!(!message.contains("secret-task-name"));
        assert!(!message.contains("agent_do_private"));
        assert!(!message.contains("TOP_SECRET"));
        assert!(!message.contains('\n'));
        assert!(
            full_raw.is_none(),
            "normal TUI must not expose raw lifecycle JSON"
        );
    }

    #[test]
    fn llm_retry_summary_distinguishes_billed_and_accepted_usage() {
        let payload = serde_json::json!({
            "scope": {
                "interaction_id": "llm_retry",
                "role": "PA",
                "stage": "decompose"
            },
            "elapsed_ms": 1500,
            "prompt_tokens": 30,
            "completion_tokens": 10,
            "billed_prompt_tokens": 60,
            "billed_completion_tokens": 20,
            "model_dispatch_count": 2,
            "content_empty": false,
            "response_tool_names": []
        })
        .to_string();

        let (role, message, _) =
            App::format_llm_interaction_message("LLM_INTERACTION_COMPLETED", &payload)
                .expect("retried interaction remains visible");
        assert!(matches!(role, MessageRole::System));
        assert!(message.contains("2 model dispatches"));
        assert!(message.contains("billed tokens 60+20"));
        assert!(message.contains("accepted tokens 30+10"));
    }

    #[test]
    fn llm_length_or_thinking_only_completion_is_a_visible_warning() {
        let payload = serde_json::json!({
            "scope": {
                "interaction_id": "llm_12345678-1234-1234-1234-123456789abc",
                "role": "SA",
                "stage": "plan_generation"
            },
            "elapsed_ms": 91209,
            "prompt_tokens": 1899,
            "completion_tokens": 8189,
            "finish_reason": "length",
            "content_empty": true,
            "response_tool_names": []
        })
        .to_string();

        let (role, message, full_raw) =
            App::format_llm_interaction_message("LLM_INTERACTION_COMPLETED", &payload)
                .expect("incomplete completion remains visible");

        assert!(matches!(role, MessageRole::Error));
        assert!(message.starts_with("⚠ LLM SA/plan_generation"));
        assert!(message.contains("finish length"));
        assert!(message.contains("visible content empty"));
        assert!(!message.contains("✓ LLM"));
        assert!(full_raw.is_none());
    }

    #[test]
    fn content_empty_tool_call_completion_is_not_misclassified_as_unusable() {
        let payload = serde_json::json!({
            "scope": {
                "interaction_id": "llm_tool_call",
                "role": "DA",
                "stage": "agent_react"
            },
            "finish_reason": "tool_calls",
            "content_empty": true,
            "response_tool_names": ["file_read"]
        })
        .to_string();

        let (role, message, _) =
            App::format_llm_interaction_message("LLM_INTERACTION_COMPLETED", &payload)
                .expect("tool call completion remains visible");
        assert!(matches!(role, MessageRole::System));
        assert!(message.starts_with("✓ LLM DA/agent_react"));
    }

    #[test]
    fn llm_interaction_normal_view_suppresses_noisy_phases_and_legacy_mirrors() {
        assert!(App::format_llm_interaction_message("LLM_INTERACTION_ASSEMBLED", "{}").is_none());
        assert!(App::format_llm_interaction_message("LLM_INTERACTION_FIRST_TOKEN", "{}").is_none());
        assert!(App::format_llm_interaction_message("LLM_REQUEST_STARTED", "{}").is_none());
        assert!(App::format_llm_interaction_message("LLM_REQUEST_COMPLETED", "{}").is_none());
    }

    #[test]
    fn llm_started_shows_bounded_context_anomalies_without_sensitive_metadata() {
        let payload = serde_json::json!({
            "scope": {
                "interaction_id": "llm_12345678-1234-1234-1234-123456789abc",
                "task_iri": "iri://task/TOP_SECRET_TASK",
                "agent_id": "TOP_SECRET_AGENT",
                "role": "CA",
                "stage": "agent_check"
            },
            "model": "model-a",
            "streaming": false,
            "message_count": 8,
            "advertised_tool_names": ["file_read"],
            "context_receipt": {
                "request_messages": 8,
                "request_chars": 4321,
                "dispositions": {
                    "included": 3,
                    "truncated": 1,
                    "dropped_expired": 1,
                    "dropped_scope": 0,
                    "dropped_role_policy": 2,
                    "dropped_budget": 1
                },
                "required_budget_exceeded": true,
                "sensitive_body": "TOP_SECRET_CONTEXT"
            }
        })
        .to_string();
        let (_, message, raw) =
            App::format_llm_interaction_message("LLM_INTERACTION_STARTED", &payload)
                .expect("started event");
        assert!(message.contains("msgctx 8msg/4321ch"));
        assert!(message.contains("1 available tool schema"));
        assert!(message.contains("cut1"));
        assert!(message.contains("drop4(e1/s0/r2/b1)"));
        assert!(message.contains("required!"));
        assert!(message.width() <= 160);
        assert!(!message.contains("TOP_SECRET"));
        assert!(raw.is_none());
    }

    #[test]
    fn failed_and_cancelled_llm_interactions_remain_visible_with_malformed_metadata() {
        let (failed_role, failed, failed_raw) =
            App::format_llm_interaction_message("LLM_INTERACTION_FAILED", "not-json")
                .expect("failed interaction must never be hidden");
        let (cancelled_role, cancelled, cancelled_raw) =
            App::format_llm_interaction_message("LLM_INTERACTION_CANCELLED", "not-json")
                .expect("cancelled interaction must never be hidden");

        assert!(matches!(failed_role, MessageRole::Error));
        assert!(failed.contains("failed: unknown_error"));
        assert!(failed_raw.is_none());
        assert!(matches!(cancelled_role, MessageRole::Error));
        assert!(cancelled.contains("cancelled"));
        assert!(cancelled_raw.is_none());

        let non_aggregation_payload = serde_json::json!({
            "scope": {
                "role": "DA",
                "stage": "react_turn",
                "interaction_id": "llm-other-cancel"
            },
            "elapsed_ms": 5000,
            "billed_prompt_tokens": 11,
            "billed_completion_tokens": 7,
            "model_dispatch_count": 2
        })
        .to_string();
        let (other_cancelled_role, other_cancelled, other_cancelled_raw) =
            App::format_llm_interaction_message(
                "LLM_INTERACTION_CANCELLED",
                &non_aggregation_payload,
            )
            .expect("non-aggregation cancellation must remain visible");
        assert!(matches!(other_cancelled_role, MessageRole::Error));
        assert!(other_cancelled.contains("cancelled"));
        assert!(other_cancelled.contains("2 model dispatches"));
        assert!(other_cancelled.contains("billed tokens 11+7"));
        assert!(!other_cancelled.contains("deterministic aggregation retained"));
        assert!(other_cancelled_raw.is_none());

        let internal_payload = serde_json::json!({
            "scope": {"role": "PA", "stage": "bizagent_decompose"},
            "error_class": "internal"
        })
        .to_string();
        let (internal_role, internal, _) =
            App::format_llm_interaction_message("LLM_INTERACTION_FAILED", &internal_payload)
                .expect("internal failure");
        assert!(matches!(internal_role, MessageRole::Error));
        assert!(internal.contains("internal (provider/transport/runtime; inspect Log)"));

        let provider_status_payload = serde_json::json!({
            "scope": {"role": "DA", "stage": "agent_react"},
            "error_class": "provider_http_client",
            "http_status": 400,
            "retryable": false
        })
        .to_string();
        let (_, provider_status, raw) =
            App::format_llm_interaction_message("LLM_INTERACTION_FAILED", &provider_status_payload)
                .expect("safe provider status metadata");
        assert!(provider_status.contains("HTTP 400"));
        assert!(provider_status.contains("not retryable"));
        assert!(raw.is_none());

        let hook_failure_payload = serde_json::json!({
            "scope": {"role": "DA", "stage": "agent_react"},
            "error_class": "response_hook_rejected",
            "http_status": 200,
            "retryable": false
        })
        .to_string();
        let (_, hook_failure, _) =
            App::format_llm_interaction_message("LLM_INTERACTION_FAILED", &hook_failure_payload)
                .expect("hook failure metadata");
        assert!(!hook_failure.contains("HTTP 200"));
        assert!(!hook_failure.contains("not retryable"));

        let incomplete_control_payload = serde_json::json!({
            "scope": {"role": "DA", "stage": "bizagent_aggregate"},
            "elapsed_ms": 5000,
            "error_class": "cancelled"
        })
        .to_string();
        let (incomplete_control_role, incomplete_control, _) = App::format_llm_interaction_message(
            "LLM_INTERACTION_CANCELLED",
            &incomplete_control_payload,
        )
        .expect("incomplete optional-control metadata must remain visible");
        assert!(matches!(incomplete_control_role, MessageRole::Error));
        assert!(!incomplete_control.contains("deterministic aggregation retained"));

        let output_limit_payload = serde_json::json!({
            "scope": {"role": "PA", "stage": "bizagent_decompose"},
            "error_class": "output_token_limit"
        })
        .to_string();
        let (_, output_limit, _) =
            App::format_llm_interaction_message("LLM_INTERACTION_FAILED", &output_limit_payload)
                .expect("output limit failure");
        assert!(output_limit.contains("reasoning/output budget exhausted"));

        for (error_class, expected) in [
            ("transport_connect", "provider connection failed"),
            ("provider_response_json", "success body was not JSON"),
            ("stream_protocol", "malformed provider SSE frame"),
            ("stream_transport_body", "stream body ended unexpectedly"),
        ] {
            let payload = serde_json::json!({
                "scope": {"role": "SA", "stage": "plan_generation"},
                "error_class": error_class,
                "sensitive_payload": "TOP_SECRET_MUST_NOT_RENDER"
            })
            .to_string();
            let (_, rendered, raw) =
                App::format_llm_interaction_message("LLM_INTERACTION_FAILED", &payload)
                    .expect("classified LLM failure");
            assert!(rendered.contains(expected));
            assert!(!rendered.contains("TOP_SECRET"));
            assert!(raw.is_none());
        }

        let (_, zero_tools, _) =
            App::format_llm_interaction_message("LLM_INTERACTION_STARTED", "{}")
                .expect("started interaction");
        assert!(zero_tools.contains("msgctx 0msg/0ch"));
        assert!(zero_tools.contains("0 available tool schemas"));
    }

    #[test]
    fn optional_bizagent_control_failures_are_rendered_as_nonfatal_fallbacks() {
        for (stage, fallback) in [
            ("bizagent_aggregate", "deterministic aggregation retained"),
            ("bizagent_decompose", "safe canonical/MONO fallback engaged"),
        ] {
            for (event_type, terminal_word) in [
                ("LLM_INTERACTION_CANCELLED", "cancelled"),
                ("LLM_INTERACTION_FAILED", "failed:"),
            ] {
                let payload = serde_json::json!({
                    "scope": {
                        "interaction_id": format!("llm-{stage}-failure"),
                        "task_iri": "iri://task/optional-control-fallback",
                        "agent_id": "da-parent",
                        "role": "DA",
                        "stage": stage
                    },
                    "elapsed_ms": 5000,
                    "error_class": "transport_timeout"
                })
                .to_string();

                let (role, message, raw) =
                    App::format_llm_interaction_message(event_type, &payload)
                        .expect("optional control failure must remain visible");

                assert!(matches!(role, MessageRole::System));
                assert!(message.contains(&format!("DA/{stage}")));
                assert!(message.contains(terminal_word));
                assert!(message.contains("5000ms"));
                assert!(message.contains(fallback));
                assert!(raw.is_none());
            }
        }
    }

    #[test]
    fn tui_task_filter_accepts_only_root_and_biz_agent_descendants() {
        let root = "iri://task/root";
        assert!(event_belongs_to_root_task(root, root));
        assert!(event_belongs_to_root_task(
            root,
            "iri://task/root/biz-agent-child/one"
        ));
        assert!(event_belongs_to_root_task(
            root,
            "iri://task/root/biz-agent-child/one/biz-agent-child/two"
        ));
        assert!(!event_belongs_to_root_task(root, "iri://task/root-other"));
        assert!(!event_belongs_to_root_task(
            root,
            "iri://task/root/recursive-child/one"
        ));
    }

    #[test]
    fn live_turn_counter_uses_explicit_react_turn_events() {
        let event = |event_type: &str, source: &str| StatusEvent {
            task_iri: "iri://task/root".to_string(),
            source_agent_iri: source.to_string(),
            sequence: 1,
            event_type: event_type.to_string(),
            payload: "{}".to_string(),
        };
        assert!(!is_react_turn_start_event(&event("THOUGHT", "SA")));
        assert!(!is_react_turn_start_event(&event(
            "THOUGHT",
            "agent_plan_1"
        )));
        assert!(is_react_turn_start_event(&event(
            "REACT_TURN_STARTED",
            "agent_plan_1"
        )));
    }

    #[test]
    fn repeated_provider_call_id_across_sessions_counts_each_event_once() {
        let event = |sequence: u64, event_type: &str, source: &str| StatusEvent {
            task_iri: "iri://task/root".to_string(),
            source_agent_iri: source.to_string(),
            sequence,
            event_type: event_type.to_string(),
            // Both independent sessions intentionally reuse the provider's
            // request-local correlation ID.
            payload: r#"{"call_id":"call_0"}"#.to_string(),
        };
        let events = [
            event(1, "REACT_TURN_STARTED", "agent_plan_1"),
            event(2, "TOOL_CALL", "agent_plan_1"),
            event(3, "REACT_TURN_STARTED", "agent_do_1"),
            event(4, "TOOL_CALL", "agent_do_1"),
            // History recovery can replay this exact EventBus event; its
            // sequence, not call_id, suppresses the duplicate.
            event(4, "TOOL_CALL", "agent_do_1"),
        ];
        let mut sequences = EventSequenceWindow::new(None);
        let mut turns = 0;
        let mut tools = 0;
        for event in &events {
            if sequences.insert(event.sequence) {
                apply_progress_event(&mut turns, &mut tools, event);
            }
        }

        assert_eq!(turns, 2);
        assert_eq!(tools, 2);
        assert_eq!(
            terminal_progress_count(turns, 1, "success", Some(TaskVerdict::Success)),
            1,
            "a normal terminal aggregate must correct a high live count"
        );
        assert_eq!(
            terminal_progress_count(tools, 3, "success", Some(TaskVerdict::Success)),
            3
        );
    }

    #[test]
    fn normal_terminal_progress_corrects_a_high_live_observation() {
        assert_eq!(
            terminal_progress_count(9, 4, "success", Some(TaskVerdict::Success)),
            4
        );
        assert_eq!(
            terminal_progress_count(9, 4, "partial_success", Some(TaskVerdict::PartialSuccess)),
            4
        );
    }

    #[test]
    fn routed_result_page_preview_shows_content_not_provider_call_id() {
        let result = serde_json::json!({
            "content": "29 passed in 0.42s",
            "total_lines": 1,
            "offset": 0,
            "returned": 1,
            "char_offset": 0,
            "returned_chars": 18,
            "next_char_offset": null,
            "truncated": false,
            "call_id": "call_00_PROVIDER_ONLY_METADATA",
            "routing_call_key": "s0000000000000000_c1111111111111111",
        })
        .to_string();
        let preview = App::summarize_tool_result(
            "read_full_result_s0000000000000000_c1111111111111111",
            &result,
            true,
        )
        .expect("routed page preview");
        assert!(preview.contains("29 passed"));
        assert!(preview.contains("complete"));
        assert!(!preview.contains("PROVIDER_ONLY_METADATA"));
    }

    #[test]
    fn routed_result_failure_shows_real_error_not_empty_complete_page() {
        let result = serde_json::json!({
            "error": "archived file cursor 0 is outside archived source range 92..168"
        })
        .to_string();
        let preview = App::summarize_tool_result(
            "read_full_result_s0000000000000000_c1111111111111111",
            &result,
            false,
        )
        .expect("failed routed page preview");
        assert!(preview.contains("failed:"));
        assert!(preview.contains("outside archived source range 92..168"));
        assert!(!preview.contains("chars 0..0 complete"));
    }

    #[test]
    fn failed_bash_preview_reports_process_facts_without_dumping_guard_envelope() {
        let result = serde_json::json!({
            "command": "python -m unittest -v test_calculator.py",
            "exit_code": 1,
            "duration_ms": 42,
            "stdout": "Ran 0 tests in 0.000s",
            "stderr": "verification did not execute tests",
            "_toolguard_validation_feedback": [{
                "classification": "tool_result_quality_failure",
                "blocks_disclosure": false,
                "verbose_internal_detail": "must remain in expandable raw only"
            }]
        })
        .to_string();

        let preview = App::summarize_tool_result("bash", &result, false).unwrap();
        assert!(preview.contains("failed: exit:1 42ms"));
        assert!(preview.contains("tool_result_quality_failure"));
        assert!(preview.contains("verification did not execute tests"));
        assert!(!preview.contains("command"));
        assert!(!preview.contains("verbose_internal_detail"));
        assert!(!preview.contains("{"));
    }

    #[test]
    fn successful_clean_verifier_preview_exposes_profile_not_temp_paths() {
        let result = serde_json::json!({
            "command": "python -m pytest -q",
            "exit_code": 0,
            "duration_ms": 18,
            "stdout": "23 passed in 0.04s",
            "execution_profile": "clean_pytest_verification",
            "isolated_environment": {
                "PYTHONPYCACHEPREFIX": true,
                "PYTEST_ADDOPTS": true
            }
        })
        .to_string();

        let preview = App::summarize_tool_result("bash", &result, true).unwrap();
        assert!(preview.contains("exit:0 18ms [clean_pytest_verification]"));
        assert!(preview.contains("23 passed"));
        assert!(!preview.contains("PYTHONPYCACHEPREFIX"));
    }

    #[test]
    fn tui_log_history_retains_the_newest_bounded_window() {
        let mut history = Vec::new();
        append_bounded_log_history(
            &mut history,
            (0..250).map(|index| format!("line-{index}")).collect(),
            TUI_LOG_HISTORY_MAX_LINES,
        );
        assert_eq!(history.len(), TUI_LOG_HISTORY_MAX_LINES);
        assert_eq!(history.first().map(String::as_str), Some("line-50"));
        assert_eq!(history.last().map(String::as_str), Some("line-249"));

        append_bounded_log_history(
            &mut history,
            (250..275).map(|index| format!("line-{index}")).collect(),
            TUI_LOG_HISTORY_MAX_LINES,
        );
        assert_eq!(history.len(), TUI_LOG_HISTORY_MAX_LINES);
        assert_eq!(history.first().map(String::as_str), Some("line-75"));
        assert_eq!(history.last().map(String::as_str), Some("line-274"));
    }

    #[test]
    fn token_totals_are_scoped_to_the_current_task_and_resume_base() {
        assert_eq!(task_scoped_token_total(0, 1_250, 1_000), 250);
        assert_eq!(task_scoped_token_total(800, 1_250, 1_000), 1_050);
        // A reset process counter must not underflow an imported checkpoint.
        assert_eq!(task_scoped_token_total(800, 10, 20), 800);
    }

    #[test]
    fn new_task_resets_last_context_source_counters() {
        let prompt = std::sync::atomic::AtomicU64::new(8_192);
        let completion = std::sync::atomic::AtomicU64::new(512);

        reset_last_context_counters(&prompt, &completion);

        assert_eq!(prompt.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(completion.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn timeout_terminal_progress_does_not_erase_live_counts() {
        assert_eq!(
            terminal_progress_count(12, 0, "timeout", Some(TaskVerdict::Timeout)),
            12
        );
        assert_eq!(terminal_progress_count(12, 15, "timeout", None), 15);
        assert_eq!(
            terminal_progress_count(12, 4, "failed", Some(TaskVerdict::Failed)),
            4,
            "ordinary failures still carry an authoritative aggregate"
        );
    }

    #[test]
    fn sequence_window_deduplicates_without_dropping_out_of_order_events() {
        let mut window = EventSequenceWindow::new(Some(9));
        assert!(!window.insert(9), "baseline traffic must remain excluded");
        assert!(window.insert(11));
        assert!(
            window.insert(10),
            "a late lower sequence is still a new event"
        );
        assert!(!window.insert(11), "replayed history must be deduplicated");
        assert_eq!(window.high_water(), Some(11));
    }

    #[tokio::test]
    async fn history_recovery_seen_and_future_events_cannot_hide_a_partial_gap() {
        let bus = EventBus::new(8);
        for index in 0..4 {
            bus.emit(
                "iri://task/recovery",
                "TEST_EVENT",
                "test",
                &index.to_string(),
            )
            .await;
        }
        let history = bus.recent_events(&EventFilter::default(), 8);
        assert_eq!(history.len(), 4);

        let mut sequences = EventSequenceWindow::new(None);
        assert!(sequences.insert(history[0].sequence));
        let recovery = plan_event_history_recovery(
            &sequences,
            vec![
                history[3].clone(),
                history[0].clone(),
                history[2].clone(),
                history[0].clone(),
            ],
            1,
            2,
        );

        assert_eq!(
            recovery
                .events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(recovery.recovered_missed, 1);
        assert_eq!(recovery.unrecoverable, 1);
    }

    #[tokio::test]
    async fn history_recovery_sorts_and_deduplicates_a_complete_gap() {
        let bus = EventBus::new(8);
        for index in 0..4 {
            bus.emit(
                "iri://task/recovery",
                "TEST_EVENT",
                "test",
                &index.to_string(),
            )
            .await;
        }
        let history = bus.recent_events(&EventFilter::default(), 8);
        assert_eq!(history.len(), 4);

        let sequences = EventSequenceWindow::new(Some(1));
        let recovery = plan_event_history_recovery(
            &sequences,
            vec![
                history[3].clone(),
                history[2].clone(),
                history[3].clone(),
                history[2].clone(),
            ],
            2,
            2,
        );

        assert_eq!(
            recovery
                .events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(recovery.recovered_missed, 2);
        assert_eq!(recovery.unrecoverable, 0);
    }

    #[tokio::test]
    async fn event_forwarder_recovers_history_deduplicates_and_filters_tasks() {
        let bus = Arc::new(EventBus::with_config(
            glidinghorse::core::event_bus::EventBusConfig {
                buffer_size: 1,
                max_history: 16,
                reliable_command_capacity: 8,
            },
        ));
        let receiver = bus.subscribe();
        let initial_sequence = latest_event_sequence(&bus);
        bus.emit("iri://task/root", "THOUGHT", "agent_plan", "root")
            .await;
        bus.emit("iri://task/foreign", "THOUGHT", "agent_foreign", "foreign")
            .await;
        bus.emit(
            "iri://task/root/biz-agent-child/one",
            "TOOL_CALL",
            "agent_child",
            "child",
        )
        .await;

        let (status_tx, mut status_rx) = mpsc::channel(8);
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
        let forwarder = tokio::spawn(forward_status_events(
            receiver,
            bus.clone(),
            "iri://task/root".to_string(),
            initial_sequence,
            status_tx,
            stop_rx,
        ));

        let root = tokio::time::timeout(std::time::Duration::from_secs(1), status_rx.recv())
            .await
            .expect("root event timeout")
            .expect("root event");
        assert_eq!(root.task_iri, "iri://task/root");
        assert_eq!(root.event_type, "THOUGHT");
        assert_eq!(root.payload, "root");

        let child = tokio::time::timeout(std::time::Duration::from_secs(1), status_rx.recv())
            .await
            .expect("child event timeout")
            .expect("child event");
        assert_eq!(child.task_iri, "iri://task/root/biz-agent-child/one");
        assert_eq!(child.source_agent_iri, "agent_child");
        assert_eq!(child.event_type, "TOOL_CALL");
        assert_eq!(child.payload, "child");
        assert!(child.sequence > root.sequence);

        tokio::task::yield_now().await;
        assert!(
            status_rx.try_recv().is_err(),
            "replayed broadcast duplicate"
        );
        let _ = stop_tx.send(latest_event_sequence(&bus));
        assert!(forwarder.await.is_ok());
    }

    #[tokio::test]
    async fn event_forwarder_reports_only_history_gaps_that_cannot_be_recovered() {
        let bus = Arc::new(EventBus::with_config(
            glidinghorse::core::event_bus::EventBusConfig {
                buffer_size: 1,
                max_history: 1,
                reliable_command_capacity: 8,
            },
        ));
        let receiver = bus.subscribe();
        for index in 0..3 {
            bus.emit(
                "iri://task/tui-lag",
                "TEST_EVENT",
                "test",
                &index.to_string(),
            )
            .await;
        }

        let (status_tx, mut status_rx) = mpsc::channel(8);
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
        let forwarder = tokio::spawn(forward_status_events(
            receiver,
            bus.clone(),
            "iri://task/tui-lag".to_string(),
            None,
            status_tx,
            stop_rx,
        ));

        let warning = tokio::time::timeout(std::time::Duration::from_secs(1), status_rx.recv())
            .await
            .expect("warning timeout")
            .expect("lag warning must precede the newest retained event");
        assert_eq!(warning.event_type, TUI_EVENT_STREAM_LAGGED);
        let warning_payload: Value = serde_json::from_str(&warning.payload).unwrap();
        assert_eq!(warning_payload["skipped"].as_u64(), Some(2));
        assert_eq!(warning_payload["broadcast_skipped"].as_u64(), Some(2));
        assert_eq!(warning_payload["recovered"].as_u64(), Some(0));
        assert_eq!(warning_payload["total_lagged"].as_u64(), Some(2));
        let (warning_role, warning_message, warning_raw) =
            App::format_event_stream_lag_message(&warning.payload);
        assert!(matches!(warning_role, MessageRole::Error));
        assert!(warning_message.contains("本次丢失 2 条"));
        assert!(warning_message.contains("累计丢失 2 条"));
        assert!(warning_raw.is_none());

        let retained = tokio::time::timeout(std::time::Duration::from_secs(1), status_rx.recv())
            .await
            .expect("retained event timeout")
            .expect("newest retained event must still be forwarded");
        assert_eq!(retained.event_type, "TEST_EVENT");
        assert_eq!(retained.payload, "2");
        let _ = stop_tx.send(latest_event_sequence(&bus));
        assert!(forwarder.await.is_ok());
    }

    async fn terminal_tail_events_after_result_state_are_drained(result_channel_closed: bool) {
        let bus = Arc::new(EventBus::with_config(
            glidinghorse::core::event_bus::EventBusConfig {
                buffer_size: 8,
                max_history: 16,
                reliable_command_capacity: 8,
            },
        ));
        let receiver = bus.subscribe();
        let initial_sequence = latest_event_sequence(&bus);
        let (status_tx, status_rx) = mpsc::channel(1);
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
        let listener = tokio::spawn(forward_status_events(
            receiver,
            bus.clone(),
            "iri://task/terminal-tail".to_string(),
            initial_sequence,
            status_tx,
            stop_rx,
        ));

        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel::<()>();
        bus.emit(
            "iri://task/terminal-tail",
            "REACT_TURN_STARTED",
            "agent_plan",
            "{}",
        )
        .await;
        bus.emit(
            "iri://task/terminal-tail",
            "TOOL_CALL",
            "agent_plan",
            r#"{"call_id":"call_0"}"#,
        )
        .await;

        if result_channel_closed {
            drop(result_tx);
            assert!(matches!(
                result_rx.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Closed)
            ));
        } else {
            result_tx.send(()).expect("result receiver remains live");
            assert_eq!(result_rx.try_recv(), Ok(()));
        }

        stop_tx
            .send(latest_event_sequence(&bus))
            .expect("terminal listener remains live");
        let (events, graceful) = drain_status_until_listener_stops(
            status_rx,
            listener,
            std::time::Duration::from_secs(1),
        )
        .await;
        assert!(graceful);
        assert_eq!(
            events
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec!["REACT_TURN_STARTED", "TOOL_CALL"]
        );

        let mut turns = 0;
        let mut tools = 0;
        for event in &events {
            apply_progress_event(&mut turns, &mut tools, event);
        }
        assert_eq!((turns, tools), (1, 1));
    }

    #[tokio::test]
    async fn terminal_result_drains_events_still_in_listener_tail() {
        terminal_tail_events_after_result_state_are_drained(false).await;
    }

    #[tokio::test]
    async fn closed_result_channel_still_drains_events_in_listener_tail() {
        terminal_tail_events_after_result_state_are_drained(true).await;
    }
}
