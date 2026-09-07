use crate::config::settings::{ContextWindowSettings, ToolResultCompressorSettings};
use crate::gateway::unified_gateway::ChatMessage;
use crate::tools::result_router::ResultRoutingIdentity;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultEntry {
    pub turn: u32,
    pub tool_name: String,
    /// Provider call ID retained for diagnostics only.
    pub provider_tool_call_id: String,
    /// Session-scoped key used for all process-wide matching.
    pub routing_call_key: String,
    pub session_scope: String,
    /// Hash of the exact routed content originally appended to ChatMessage.
    /// This is an ambiguity-safe fallback for legacy messages that lack an
    /// embedded routing IRI; it is never used alone when multiple calls match.
    #[serde(default)]
    pub original_content_sha256: String,
    pub content: String,
    pub is_compressed: bool,
    /// True only when AgentRunner confirmed that the exact session-scoped
    /// reader was registered after routing this result. A routing key alone
    /// is not execution authority and must not produce a capability hint.
    #[serde(default)]
    pub reader_available: bool,
}

pub struct ToolResultCompressor {
    enabled: bool,
    /// Number of most-recent ReAct tool-result batches (identified by turn)
    /// that remain complete. A batch is an atomic observation unit: splitting
    /// sibling results makes multi-file comparisons oscillate between reads.
    max_full_results: usize,
    max_summary_length: usize,
    compression_trigger: usize,
    /// Tool messages exceeding this byte threshold attempt micro-tool reference replacement
    compress_tool_result_threshold: usize,
    results: VecDeque<ToolResultEntry>,
}

impl ToolResultCompressor {
    pub fn new(settings: &ToolResultCompressorSettings) -> Self {
        Self {
            enabled: settings.enabled,
            max_full_results: settings.max_full_results,
            max_summary_length: settings.max_summary_length,
            compression_trigger: settings.compression_trigger,
            compress_tool_result_threshold: settings.compress_tool_result_threshold,
            results: VecDeque::new(),
        }
    }

    pub fn add_result(
        &mut self,
        turn: u32,
        tool_name: &str,
        session_id: &str,
        provider_tool_call_id: &str,
        content: &str,
    ) {
        let routing = ResultRoutingIdentity::new(session_id, provider_tool_call_id);
        self.add_result_with_routing(turn, tool_name, &routing, content);
    }

    /// Record an exact composite routing key without claiming that a reader
    /// exists. Production AgentRunner paths use
    /// [`Self::add_result_with_routing_and_reader`] after checking the runtime
    /// registry; callers without that receipt fail closed here.
    pub fn add_result_with_routing(
        &mut self,
        turn: u32,
        tool_name: &str,
        routing: &ResultRoutingIdentity,
        content: &str,
    ) {
        self.add_result_with_routing_and_reader(turn, tool_name, routing, content, false);
    }

    /// Record one result together with the runtime-confirmed availability of
    /// its exact session reader. Callers must never infer this flag from a raw
    /// provider call ID.
    pub fn add_result_with_routing_and_reader(
        &mut self,
        turn: u32,
        tool_name: &str,
        routing: &ResultRoutingIdentity,
        content: &str,
        reader_available: bool,
    ) {
        let entry = ToolResultEntry {
            turn,
            tool_name: tool_name.to_string(),
            provider_tool_call_id: routing.provider_call_id.clone(),
            routing_call_key: routing.routing_call_key.clone(),
            session_scope: routing.session_scope.clone(),
            original_content_sha256: crate::utils::CryptoUtils::sha256_hex(content),
            content: content.to_string(),
            is_compressed: false,
            reader_available,
        };
        self.results.push_back(entry);

        let session_result_count = self
            .results
            .iter()
            .filter(|entry| entry.session_scope == routing.session_scope)
            .count();
        if session_result_count >= self.compression_trigger {
            self.compress_old_results(&routing.session_scope);
        }
    }

    fn compress_old_results(&mut self, session_scope: &str) {
        let session_indices = self
            .results
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| (entry.session_scope == session_scope).then_some(index))
            .collect::<Vec<_>>();
        if session_indices.len() <= self.max_full_results {
            return;
        }

        // `max_full_results` historically counted individual messages. That
        // could retain only two members of a four-file tool batch: after the
        // next provider decision the other siblings were replaced by history
        // markers, and re-reading them evicted the files that had just become
        // visible. Preserve complete recent turns so every provider-visible
        // tool batch remains a coherent working set. Large individual results
        // are still bounded by ResultRouter before they reach this history.
        let mut protected_turns = std::collections::HashSet::new();
        if self.max_full_results > 0 {
            for index in session_indices.iter().rev() {
                protected_turns.insert(self.results[*index].turn);
                if protected_turns.len() == self.max_full_results {
                    break;
                }
            }
        }
        let summaries: Vec<(usize, String)> = self
            .results
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.session_scope == session_scope && !protected_turns.contains(&entry.turn)
            })
            .filter(|(_, entry)| {
                !entry.is_compressed && entry.content.len() > self.max_summary_length
            })
            .map(|(i, entry)| (i, self.summarize_content(entry)))
            .collect();

        for (i, summary) in summaries {
            if let Some(entry) = self.results.get_mut(i) {
                entry.content = summary;
                entry.is_compressed = true;
            }
        }
    }

    /// Drop transcript-local compression state when its L1 execution ends.
    /// The shared ToolExecutor is intentionally not touched: sibling agents
    /// may still be using their own scoped result readers concurrently.
    pub fn remove_session(&mut self, session_id: &str) {
        let session_scope = ResultRoutingIdentity::session_scope_for(session_id);
        self.results
            .retain(|entry| entry.session_scope != session_scope);
    }

    fn summarize_content(&self, entry: &ToolResultEntry) -> String {
        let content = &entry.content;
        if content.len() <= self.max_summary_length {
            return content.clone();
        }

        let routing = entry
            .reader_available
            .then(|| {
                ResultRoutingIdentity::from_routing_call_key(
                    &entry.routing_call_key,
                    &entry.provider_tool_call_id,
                )
            })
            .flatten();
        let full_result_hint = routing.as_ref().map_or_else(String::new, |routing| {
            format!(
                "\nSession reader: `{}`. Call it only while that exact name is advertised in the current turn.",
                routing.reader_name
            )
        });

        // file_read results are JSON with path/total_lines — keep that context so
        // the LLM knows which file this was and how much remains.
        if entry.tool_name == "file_read" {
            if let Some(summary) = compact_file_read_history(content, routing.as_ref()) {
                return summary;
            }
        }

        let lines: Vec<&str> = content.lines().take(5).collect();
        let unbounded_preview = if lines.len() > 3 {
            lines[..3].join("\n")
        } else {
            lines.join("\n")
        };
        let preview =
            crate::utils::text::safe_truncate(&unbounded_preview, self.max_summary_length.max(1));

        format!(
            "[Summary {} bytes] {}... (total {} chars){}",
            self.max_summary_length,
            preview,
            content.len(),
            full_result_hint
        )
    }

    /// Compress tool result content in messages.
    /// Used together with compress_old_results(): the latter compresses entries inside the compressor,
    /// this method writes compressed results back to the corresponding tool messages via tool_call_id matching.
    pub fn compress_tool_messages(&self, messages: &mut Vec<ChatMessage>, session_id: &str) {
        self.compress_tool_messages_with_reader_set(messages, session_id, None);
    }

    /// Production compression variant which revalidates ephemeral reader
    /// capabilities at the moment a historical summary is written back. A
    /// reader may have been live when the result was routed but retired after
    /// complete consumption; a historical boolean must not resurrect it.
    pub fn compress_tool_messages_with_active_readers(
        &self,
        messages: &mut Vec<ChatMessage>,
        session_id: &str,
        active_readers: &std::collections::HashSet<String>,
    ) {
        self.compress_tool_messages_with_reader_set(messages, session_id, Some(active_readers));
    }

    fn compress_tool_messages_with_reader_set(
        &self,
        messages: &mut Vec<ChatMessage>,
        session_id: &str,
        active_readers: Option<&std::collections::HashSet<String>>,
    ) {
        if !self.enabled {
            return;
        }
        // Build compressed entry map: session-scoped routing key -> entry.
        let compressed_map: std::collections::HashMap<&str, &ToolResultEntry> = self
            .results
            .iter()
            .filter(|e| e.is_compressed)
            .map(|e| (e.routing_call_key.as_str(), e))
            .collect();

        if compressed_map.is_empty() {
            return;
        }

        let expected_session_scope = ResultRoutingIdentity::session_scope_for(session_id);

        // Historical ChatMessage has only the raw provider ID, which may be
        // reused by later requests. Prefer the exact IRI/reader embedded by
        // ResultRouter. For legacy inline messages, accept a content-hash
        // match only when it identifies exactly one routing key; ambiguity is
        // deliberately left uncompressed rather than linked to the wrong
        // request.
        for msg in messages.iter_mut() {
            if msg.role != "tool" {
                continue;
            }
            let call_id = match msg.tool_call_id.as_deref() {
                Some(id) if !id.is_empty() => id,
                _ => continue,
            };
            let embedded_key = crate::tools::result_router::routing_identity_from_content(
                &msg.content,
                session_id,
                call_id,
            )
            .map(|routing| routing.routing_call_key);
            let routing_key = embedded_key.or_else(|| {
                let content_sha256 = crate::utils::CryptoUtils::sha256_hex(&msg.content);
                let mut candidates = self
                    .results
                    .iter()
                    .filter(|entry| {
                        entry.is_compressed
                            && entry.session_scope == expected_session_scope
                            && entry.provider_tool_call_id == call_id
                            && entry.original_content_sha256 == content_sha256
                    })
                    .map(|entry| entry.routing_call_key.as_str());
                let first = candidates.next()?;
                if candidates.any(|candidate| candidate != first) {
                    None
                } else {
                    Some(first.to_string())
                }
            });
            if let Some(entry) = routing_key
                .as_deref()
                .and_then(|key| compressed_map.get(key))
            {
                let compressed_content = active_readers.map_or_else(
                    || Cow::Borrowed(entry.content.as_str()),
                    |active| compressed_content_for_active_readers(entry, active),
                );
                msg.content = compressed_content.into_owned();
            }
        }
    }

    pub fn get_results(&self) -> &VecDeque<ToolResultEntry> {
        &self.results
    }

    pub fn clear(&mut self) {
        self.results.clear();
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn max_full_results(&self) -> usize {
        self.max_full_results
    }

    pub fn max_summary_length(&self) -> usize {
        self.max_summary_length
    }

    pub fn compress_tool_result_threshold(&self) -> usize {
        self.compress_tool_result_threshold
    }
}

fn compressed_content_for_active_readers<'a>(
    entry: &'a ToolResultEntry,
    active_readers: &std::collections::HashSet<String>,
) -> Cow<'a, str> {
    if !entry.reader_available {
        let sanitized = omit_inactive_reader_reference(&entry.content);
        return if sanitized == entry.content {
            Cow::Borrowed(&entry.content)
        } else {
            Cow::Owned(sanitized)
        };
    }
    let reader_is_active = ResultRoutingIdentity::from_routing_call_key(
        &entry.routing_call_key,
        &entry.provider_tool_call_id,
    )
    .is_some_and(|routing| active_readers.contains(&routing.reader_name));
    if reader_is_active {
        return Cow::Borrowed(&entry.content);
    }

    Cow::Owned(omit_inactive_reader_reference(&entry.content))
}

/// Remove an ephemeral session-reader capability from historical content once
/// that exact reader is no longer advertisable. Stable file coordinates stay
/// available; typed reader cursors and IRIs do not outlive their L1 authority.
pub(crate) fn omit_inactive_reader_reference(content: &str) -> String {
    if let Ok(serde_json::Value::Object(mut object)) =
        serde_json::from_str::<serde_json::Value>(content)
    {
        object.remove("result_iri");
        object.remove("session_reader");
        object.remove("reader_cursor");
        object.remove("reader_cursors");
        if object.get("tool").and_then(serde_json::Value::as_str) == Some("file_read")
            || object
                .get("history_compacted")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            || (object.get("path").is_some() && object.get("total_lines").is_some())
        {
            let path = object
                .get("path")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            let message = object
                .get("next_offset")
                .and_then(serde_json::Value::as_u64)
                .map_or_else(
                    || {
                        "Historical file content was observed completely; the retired session reader was omitted."
                            .to_string()
                    },
                    |offset| {
                        format!(
                            "The session reader has retired. Continue, if needed, with bounded file_read path={path:?} offset={offset}."
                        )
                    },
                );
            object.insert("message".to_string(), serde_json::Value::String(message));
        } else if object
            .get("message")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|message| {
                message.contains("reader")
                    || message.contains("Reader")
                    || message.contains("advertised")
            })
        {
            object.remove("message");
        }
        return serde_json::to_string(&object)
            .unwrap_or_else(|_| "[Historical tool result summary]".to_string());
    }

    let sanitized = content
        .lines()
        .filter(|line| {
            let line = line.trim();
            !line.starts_with("Session reader:")
                && !line.starts_with("Call it only while that exact name is advertised")
                && !(line.contains("Full result stored") && line.contains("session reader"))
                && !line.starts_with("IRI: iri://tool-result/")
        })
        .collect::<Vec<_>>()
        .join("\n");
    if sanitized.trim().is_empty() {
        "[Historical tool result summary; retired session reader omitted]".to_string()
    } else {
        sanitized
    }
}

/// Reduce an already-observed `file_read` envelope to stable replay metadata.
/// Unlike a generic result summary, this retains the exact source coordinate
/// system needed for the next bounded `file_read`. A session reader is emitted
/// only when the caller has confirmed that exact composite routing identity is
/// currently registered.
pub(crate) fn compact_file_read_history(
    content: &str,
    reader: Option<&ResultRoutingIdentity>,
) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(content).ok()?;
    let object = value.as_object()?;
    let path = object.get("path")?.as_str()?;
    let source_offset = object
        .get("source_offset")
        .or_else(|| object.get("offset"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let returned = object
        .get("returned")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            object
                .get("lines")
                .and_then(serde_json::Value::as_array)
                .map(|lines| lines.len() as u64)
        })
        .unwrap_or(0);
    let total_lines = object
        .get("total_lines")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| source_offset.saturating_add(returned));
    let observed_end = source_offset.saturating_add(returned).min(total_lines);
    let partial_line_preview = object
        .get("partial_line_preview")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let next_offset = object
        .get("next_offset")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| (partial_line_preview || observed_end < total_lines).then_some(observed_end));
    let reader_cursor = object
        .get("reader_cursor")
        .filter(|cursor| cursor.is_object())
        .cloned();

    let mut summary = serde_json::Map::new();
    summary.insert(
        "history_compacted".to_string(),
        serde_json::Value::Bool(true),
    );
    summary.insert(
        "tool".to_string(),
        serde_json::Value::String("file_read".to_string()),
    );
    summary.insert(
        "path".to_string(),
        serde_json::Value::String(path.to_string()),
    );
    summary.insert(
        "content_sha256".to_string(),
        object
            .get("content_sha256")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    );
    summary.insert("source_offset".to_string(), source_offset.into());
    summary.insert("returned".to_string(), returned.into());
    summary.insert("total_lines".to_string(), total_lines.into());
    summary.insert(
        "next_offset".to_string(),
        next_offset
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    summary.insert(
        "partial_line_preview".to_string(),
        serde_json::Value::Bool(partial_line_preview),
    );
    summary.insert(
        "message".to_string(),
        serde_json::Value::String(match (reader, reader_cursor.as_ref(), next_offset) {
            (Some(_), Some(_), _) =>
                "Historical file content was already observed. For duplicate-free archived continuation, copy reader_cursor exactly while session_reader is advertised; otherwise use bounded file_read at next_offset."
                    .to_string(),
            (_, _, Some(offset)) => format!(
                "Historical file content was already observed. Continue with bounded file_read path={path:?} offset={offset}; do not restart from offset 0."
            ),
            (_, _, None) => "Historical file content was already observed completely; do not read it again unless the revision changes."
                .to_string(),
        }),
    );
    if let Some(reader) = reader {
        summary.insert(
            "result_iri".to_string(),
            serde_json::Value::String(reader.storage_iri.clone()),
        );
        summary.insert(
            "session_reader".to_string(),
            serde_json::Value::String(reader.reader_name.clone()),
        );
        if let Some(reader_cursor) = reader_cursor {
            summary.insert("reader_cursor".to_string(), reader_cursor);
        }
    }
    serde_json::to_string(&summary).ok()
}

/// Approximate token count of a single text. Raw `len()/4` counts UTF-8
/// bytes, undervaluing CJK characters (3 bytes each) at 0.75 tokens/char;
/// real tokenizers cost them ~1 token/char. CJK chars get 1 token, all
/// other bytes stay at the 4-bytes-per-token heuristic.
fn estimate_text_tokens(text: &str) -> usize {
    let mut cjk_chars = 0usize;
    let mut other_bytes = 0usize;
    for ch in text.chars() {
        if is_cjk_char(ch) {
            cjk_chars += 1;
        } else {
            other_bytes += ch.len_utf8();
        }
    }
    cjk_chars + other_bytes / 4
}

fn is_cjk_char(c: char) -> bool {
    matches!(c as u32,
        0x2E80..=0x9FFF   // CJK radicals, punctuation, kana, bopomofo, unified ideographs
        | 0xAC00..=0xD7AF // Hangul syllables
        | 0xF900..=0xFAFF // CJK compatibility ideographs
    )
}

/// Context-window sizes (in tokens) for known model families.
/// Keys are lowercase substrings matched against the active model name.
const MODEL_CONTEXT_WINDOWS: &[(&str, usize)] = &[
    ("deepseek", 128_000),
    ("gpt-4o", 128_000),
    ("gpt-4", 32_000),
    ("gpt-3.5", 16_000),
    ("claude", 200_000),
    ("gemini", 1_000_000),
    ("llama-3", 128_000),
    ("qwen", 128_000),
    ("glm", 128_000),
    ("mistral", 32_000),
    ("command-r", 128_000),
];

/// Fraction of the model context window used as the compression budget.
const MODEL_AWARE_BUDGET_RATIO: f32 = 0.8;

/// Best-effort context-window lookup for a model name (lowercased substring match).
pub fn model_context_window(model: &str) -> usize {
    let model_lower = model.to_lowercase();
    for (key, window) in MODEL_CONTEXT_WINDOWS {
        if model_lower.contains(key) {
            return *window;
        }
    }
    64_000
}

pub struct ContextWindowManager {
    max_messages: usize,
    max_tokens: usize,
    compression_ratio: f32,
    preserve_recent: usize,
    model_aware: bool,
}

/// The immutable task prefix cannot be compressed without changing the task
/// the model is being asked to execute.  Callers must reject the dispatch
/// explicitly when that prefix alone cannot fit the configured token budget.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "immutable initial context requires {immutable_tokens} tokens plus {request_reserve_tokens} tool schema reserve tokens ({required_tokens} total), exceeding the {budget_tokens}-token budget for model {model}"
)]
pub struct ImmutableContextBudgetExceeded {
    pub model: String,
    pub immutable_tokens: usize,
    pub request_reserve_tokens: usize,
    pub required_tokens: usize,
    pub budget_tokens: usize,
}

impl ContextWindowManager {
    pub fn new(settings: &ContextWindowSettings) -> Self {
        Self {
            max_messages: settings.max_messages,
            max_tokens: settings.max_tokens,
            compression_ratio: settings.compression_ratio,
            preserve_recent: settings.preserve_recent,
            model_aware: settings.model_aware,
        }
    }

    /// Effective compression budget for the given model.
    /// Model awareness protects smaller model windows, but never expands past
    /// the configured cost ceiling. Expanding 16K to 80% of a 128K model made
    /// cumulative multi-turn prompt usage grow dramatically.
    pub fn budget_for_model(&self, model: &str) -> usize {
        if self.model_aware() {
            self.max_tokens
                .min(((model_context_window(model) as f32) * MODEL_AWARE_BUDGET_RATIO) as usize)
        } else {
            self.max_tokens
        }
    }

    pub fn model_aware(&self) -> bool {
        self.model_aware
    }

    /// Validate the complete immutable initial prompt envelope before any
    /// provider call.  This envelope contains the leading kernel authority,
    /// typed task fragments and the freshly generated agent.md.  Silently
    /// dropping any of it would let a runtime-control tail outlive the
    /// business objective it is meant to govern.
    pub fn validate_immutable_prefix_for_model(
        &self,
        immutable_prefix: &[ChatMessage],
        model: &str,
    ) -> Result<(), ImmutableContextBudgetExceeded> {
        self.validate_immutable_prefix_for_model_with_reserve(immutable_prefix, model, 0)
    }

    /// Validate that the immutable task envelope and request-only token
    /// reserve can coexist in the active model budget. Tool definitions are
    /// sent outside `messages`, so tail compression cannot recover space when
    /// this sum alone is already over budget.
    pub fn validate_immutable_prefix_for_model_with_reserve(
        &self,
        immutable_prefix: &[ChatMessage],
        model: &str,
        request_reserve_tokens: usize,
    ) -> Result<(), ImmutableContextBudgetExceeded> {
        let immutable_tokens = Self::estimate_tokens(immutable_prefix);
        let required_tokens = immutable_tokens.saturating_add(request_reserve_tokens);
        let budget_tokens = self.budget_for_model(model);
        if required_tokens > budget_tokens {
            return Err(ImmutableContextBudgetExceeded {
                model: model.to_string(),
                immutable_tokens,
                request_reserve_tokens,
                required_tokens,
                budget_tokens,
            });
        }
        Ok(())
    }

    /// Estimate token consumption of a message list (4 chars ≈ 1 token, mixed CJK/Latin estimation)
    pub fn estimate_tokens(messages: &[ChatMessage]) -> usize {
        messages
            .iter()
            .map(|m| {
                let mut total = estimate_text_tokens(&m.content) + estimate_text_tokens(&m.role);
                if let Some(ref calls) = m.tool_calls {
                    for call in calls {
                        total += estimate_text_tokens(&call.function.name);
                        total += estimate_text_tokens(&call.function.arguments);
                        // Include tool_call_id (~36 chars per UUID)
                        total += estimate_text_tokens(&call.id);
                    }
                }
                if let Some(ref id) = m.tool_call_id {
                    total += estimate_text_tokens(id);
                }
                total
            })
            .sum()
    }

    /// Estimate the request-side token cost of tool schemas. Providers count
    /// these definitions on every turn even though they are not represented
    /// as chat messages, so omitting them defeats the configured cost ceiling.
    pub fn estimate_tool_schema_tokens(tools: &[serde_json::Value]) -> usize {
        tools
            .iter()
            .map(|tool| estimate_text_tokens(&tool.to_string()))
            .sum()
    }

    /// Determine whether compression is needed. Checks both message count and estimated token count.
    pub fn should_compress(&self, message_count: usize, messages: &[ChatMessage]) -> bool {
        if message_count > self.max_messages {
            return true;
        }
        if Self::estimate_tokens(messages) > self.max_tokens {
            return true;
        }
        false
    }

    /// Model-aware variant: the token budget follows the active model's context
    /// window when `model_aware` is enabled.
    pub fn should_compress_for_model(
        &self,
        message_count: usize,
        messages: &[ChatMessage],
        model: &str,
    ) -> bool {
        if message_count > self.max_messages {
            return true;
        }
        if Self::estimate_tokens(messages) > self.budget_for_model(model) {
            return true;
        }
        false
    }

    pub fn should_compress_for_model_with_reserve(
        &self,
        message_count: usize,
        messages: &[ChatMessage],
        model: &str,
        request_reserve_tokens: usize,
    ) -> bool {
        if message_count > self.max_messages {
            return true;
        }
        Self::estimate_tokens(messages).saturating_add(request_reserve_tokens)
            > self.budget_for_model(model)
    }

    pub fn compress_messages(&self, messages: &[ChatMessage]) -> (Vec<ChatMessage>, String) {
        // `should_compress*` can trigger on either message count or token
        // count.  Do not turn the token branch into a no-op merely because a
        // few very large messages are still below max_messages.  We only need
        // enough history to preserve the system message and recent tool-call
        // group intact.
        if messages.len() <= self.preserve_recent.saturating_add(1) {
            return (messages.to_vec(), String::new());
        }

        let system_msg = messages.first().filter(|m| m.role == "system").cloned();
        let mut recent_start = messages.len().saturating_sub(self.preserve_recent);

        // OpenAI/DeepSeek require every `role: "tool"` message to be preceded
        // by an `assistant` message whose `tool_calls` array contains a
        // matching id.  Adjust the boundary so tool_call groups stay intact.
        recent_start = Self::adjust_boundary_for_tool_calls(messages, recent_start);
        let recent: Vec<_> = messages[recent_start..].to_vec();

        let middle_start = if system_msg.is_some() { 1 } else { 0 };
        let middle: Vec<_> = messages[middle_start..recent_start].to_vec();

        let keep_count = (middle.len() as f32 * self.compression_ratio) as usize;
        let keep_count = keep_count.min(middle.len());
        let empty: &[ChatMessage] = &[];
        let (to_summarize, to_keep) = if keep_count > 0 && keep_count < middle.len() {
            let mut split = middle.len() - keep_count;
            // Adjust split to avoid splitting tool_call groups within middle
            split = Self::adjust_boundary_for_tool_calls(&middle, split);
            (&middle[..split], &middle[split..])
        } else if keep_count >= middle.len() {
            (empty, &middle[..])
        } else {
            (&middle[..], empty)
        };

        let summary = self.summarize_middle_messages(to_summarize);

        let mut compressed = Vec::new();
        if let Some(sys) = system_msg {
            compressed.push(sys);
        }

        if !summary.is_empty() {
            compressed.push(ChatMessage {
                role: "user".to_string(),
                content: format!("[History Summary] {}", summary),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }

        compressed.extend(to_keep.iter().cloned());
        compressed.extend(recent);

        // Safety: remove any orphaned tool messages that slipped through
        let cleaned = Self::remove_orphaned_tool_messages(compressed);
        (cleaned, summary)
    }

    /// Compress only mutable provider history while preserving the complete
    /// initial task envelope byte-for-byte and in order.
    ///
    /// `immutable_prefix_len` is captured once by AgentRunner immediately
    /// after it assembles the leading kernel prompt, typed ContextFragments
    /// and model-generated agent.md, before checkpoint/current-L1 protocol is
    /// appended.  Provider tool-call IDs in the retained tail are cloned
    /// unchanged by `compress_messages`; they are never used as compression
    /// identities or rewritten here.
    pub fn compress_messages_preserving_prefix(
        &self,
        messages: &[ChatMessage],
        immutable_prefix_len: usize,
    ) -> (Vec<ChatMessage>, String) {
        let prefix_len = immutable_prefix_len.min(messages.len());
        if prefix_len == 0 {
            return self.compress_messages(messages);
        }

        let immutable_prefix = &messages[..prefix_len];
        let protocol_tail = &messages[prefix_len..];
        let (compressed_tail, summary) = self.compress_messages(protocol_tail);
        let mut compressed = Vec::with_capacity(immutable_prefix.len() + compressed_tail.len());
        compressed.extend_from_slice(immutable_prefix);
        compressed.extend(compressed_tail);
        (compressed, summary)
    }

    /// OpenAI/DeepSeek require every `role: "tool"` message to be preceded
    /// by an `assistant` with a matching `tool_calls` entry.  Adjust a
    /// message-array boundary so that these groups are never split.
    fn adjust_boundary_for_tool_calls(messages: &[ChatMessage], boundary: usize) -> usize {
        if boundary == 0 || boundary >= messages.len() {
            return boundary;
        }
        if messages[boundary].role != "tool" {
            return boundary;
        }
        let tool_call_id = match messages[boundary].tool_call_id.as_deref() {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => return boundary,
        };
        for j in (0..boundary).rev() {
            if let Some(ref calls) = messages[j].tool_calls {
                if calls.iter().any(|c| c.id == tool_call_id) {
                    return j;
                }
            }
        }
        boundary
    }

    /// Safety net: convert orphaned `role: "tool"` messages (no preceding
    /// assistant with matching `tool_calls`) to `user` messages so the
    /// content is preserved but the API-invalid role is removed.
    pub fn remove_orphaned_tool_messages(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        let mut known_tool_call_ids: Vec<String> = Vec::new();
        let mut result = Vec::with_capacity(messages.len());

        for msg in messages {
            if msg.role == "assistant" {
                if let Some(ref calls) = msg.tool_calls {
                    for call in calls {
                        known_tool_call_ids.push(call.id.clone());
                    }
                }
                result.push(msg);
            } else if msg.role == "tool" {
                let is_orphaned = match msg.tool_call_id.as_deref() {
                    Some(id) if !id.is_empty() => !known_tool_call_ids.iter().any(|kid| kid == id),
                    _ => true,
                };
                if is_orphaned {
                    result.push(ChatMessage {
                        role: "user".to_string(),
                        content: msg.content,
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    });
                } else {
                    result.push(msg);
                }
            } else {
                result.push(msg);
            }
        }
        result
    }

    fn summarize_middle_messages(&self, messages: &[ChatMessage]) -> String {
        let mut tool_calls = Vec::new();
        let mut summaries = Vec::new();
        let mut errors = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "assistant" => {
                    if let Some(ref tool_calls_data) = msg.tool_calls {
                        for tc in tool_calls_data {
                            tool_calls.push(tc.function.name.clone());
                        }
                    }
                    if msg.content.len() > 50 && msg.content.len() < 200 {
                        summaries.push(msg.content.clone());
                    }
                }
                "tool" => {
                    if msg.content.contains("error") || msg.content.contains("Error") {
                        errors.push(msg.content.chars().take(100).collect::<String>());
                    }
                }
                _ => {}
            }
        }

        let mut parts = Vec::new();

        if !tool_calls.is_empty() {
            let unique_tools: std::collections::HashSet<_> = tool_calls.into_iter().collect();
            parts.push(format!(
                "Tools called: {}",
                unique_tools.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }

        if !errors.is_empty() {
            parts.push(format!("Errors: {}", errors.len()));
        }

        if !summaries.is_empty() {
            parts.push(format!("Key content: {}", summaries.join("; ")));
        }

        parts.join(" | ")
    }

    pub fn max_messages(&self) -> usize {
        self.max_messages
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SESSION: &str = "l1-context-compressor-test";

    fn default_settings() -> ToolResultCompressorSettings {
        ToolResultCompressorSettings {
            enabled: true,
            max_full_results: 2,
            max_summary_length: 200,
            compression_trigger: 5,
            compress_tool_result_threshold: 500,
        }
    }

    fn default_context_settings() -> ContextWindowSettings {
        ContextWindowSettings {
            max_messages: 15,
            max_tokens: 16000,
            compression_ratio: 0.3,
            preserve_recent: 4,
            model_aware: false,
        }
    }

    #[test]
    fn test_compressor_add_result() {
        let mut compressor = ToolResultCompressor::new(&default_settings());

        compressor.add_result(1, "file_read", TEST_SESSION, "call_001", "test content");
        assert_eq!(compressor.get_results().len(), 1);
        assert_eq!(
            compressor.get_results()[0].provider_tool_call_id,
            "call_001"
        );
    }

    #[test]
    fn test_compressor_compress() {
        let mut compressor = ToolResultCompressor::new(&default_settings());

        let long_content = "x".repeat(500);

        for i in 1..=6 {
            compressor.add_result(
                i,
                "file_read",
                TEST_SESSION,
                &format!("call_{}", i),
                &long_content,
            );
        }

        let results = compressor.get_results();
        assert!(results.front().unwrap().is_compressed);
        assert!(!results.back().unwrap().is_compressed);
    }

    #[test]
    fn compression_never_splits_a_provider_visible_tool_batch() {
        let mut settings = default_settings();
        settings.compression_trigger = 1;
        settings.max_full_results = 1;
        let mut compressor = ToolResultCompressor::new(&settings);
        let long = "current-batch".repeat(64);

        compressor.add_result(1, "file_read", TEST_SESSION, "old", &long);
        for call in ["design", "implementation", "tests", "readme"] {
            compressor.add_result(2, "file_read", TEST_SESSION, call, &long);
        }

        let old = compressor
            .results
            .iter()
            .find(|entry| entry.provider_tool_call_id == "old")
            .unwrap();
        assert!(old.is_compressed);
        assert!(compressor
            .results
            .iter()
            .filter(|entry| entry.turn == 2)
            .all(|entry| !entry.is_compressed));
    }

    #[test]
    fn compression_keeps_all_siblings_from_each_recent_batch() {
        let mut settings = default_settings();
        settings.compression_trigger = 1;
        settings.max_full_results = 2;
        let mut compressor = ToolResultCompressor::new(&settings);
        let long = "multi-file-evidence".repeat(64);

        for turn in 1..=3 {
            for sibling in 0..3 {
                compressor.add_result(
                    turn,
                    "file_read",
                    TEST_SESSION,
                    &format!("turn_{turn}_file_{sibling}"),
                    &long,
                );
            }
        }

        assert!(compressor
            .results
            .iter()
            .filter(|entry| entry.turn == 1)
            .all(|entry| entry.is_compressed));
        assert!(compressor
            .results
            .iter()
            .filter(|entry| entry.turn >= 2)
            .all(|entry| !entry.is_compressed));
    }

    #[test]
    fn compression_thresholds_and_call_matching_are_isolated_per_l1_session() {
        let mut compressor = ToolResultCompressor::new(&default_settings());
        let long = "session-a".repeat(64);
        let session_a = "l1-pa-independent";
        let session_b = "l1-da-independent";

        // Four PA results plus one DA result reach the old process-global
        // trigger of five, but neither independent L1 has reached it.
        for index in 0..4 {
            compressor.add_result(index, "bash", session_a, &format!("call_{index}"), &long);
        }
        compressor.add_result(1, "bash", session_b, "call_0", &"session-b".repeat(64));
        assert!(compressor.results.iter().all(|entry| !entry.is_compressed));

        // PA now crosses its own threshold. The DA entry reuses provider
        // call_0, yet must neither influence nor receive PA compression.
        compressor.add_result(5, "bash", session_a, "call_4", &long);
        assert!(compressor.results.iter().any(|entry| {
            entry.session_scope == ResultRoutingIdentity::new(session_a, "call_0").session_scope
                && entry.is_compressed
        }));
        assert!(compressor.results.iter().all(|entry| {
            entry.session_scope != ResultRoutingIdentity::new(session_b, "call_0").session_scope
                || !entry.is_compressed
        }));

        let original_b = "session-b-visible".repeat(32);
        let mut b_messages = vec![ChatMessage {
            role: "tool".to_string(),
            content: original_b.clone(),
            name: None,
            tool_calls: None,
            tool_call_id: Some("call_0".to_string()),
            reasoning_content: None,
        }];
        compressor.compress_tool_messages(&mut b_messages, session_b);
        assert_eq!(b_messages[0].content, original_b);

        compressor.remove_session(session_a);
        assert_eq!(compressor.results.len(), 1);
        assert_eq!(compressor.results[0].provider_tool_call_id, "call_0");
        assert_eq!(
            compressor.results[0].session_scope,
            ResultRoutingIdentity::new(session_b, "call_0").session_scope
        );
    }

    #[test]
    fn file_read_history_keeps_revision_and_replay_cursor_without_fake_reader() {
        let mut compressor = ToolResultCompressor::new(&default_settings());
        let file_content = serde_json::json!({
            "path": "/tmp/game.js",
            "content_sha256": "d".repeat(64),
            "total_lines": 800,
            "offset": 200,
            "lines": (200..300).map(|i| format!("line {:04}", i)).collect::<Vec<_>>(),
            "returned": 100,
        })
        .to_string();
        assert!(file_content.len() > 200);

        compressor.add_result(1, "file_read", TEST_SESSION, "call_f1", &file_content);
        compressor.add_result(2, "file_read", TEST_SESSION, "call_f2", &"y".repeat(500));
        compressor.add_result(3, "file_read", TEST_SESSION, "call_f3", &"z".repeat(500));
        compressor.add_result(4, "file_read", TEST_SESSION, "call_f4", &"w".repeat(500));
        compressor.add_result(5, "file_read", TEST_SESSION, "call_f5", &"v".repeat(500));

        let results = compressor.get_results();
        let first = &results[0];
        assert!(first.is_compressed);
        let summary: serde_json::Value =
            serde_json::from_str(&first.content).expect("structured file history");
        assert_eq!(summary["path"], "/tmp/game.js");
        assert_eq!(summary["content_sha256"], "d".repeat(64));
        assert_eq!(summary["source_offset"], 200);
        assert_eq!(summary["returned"], 100);
        assert_eq!(summary["total_lines"], 800);
        assert_eq!(summary["next_offset"], 300);
        assert!(summary.get("session_reader").is_none());
        assert!(summary.get("result_iri").is_none());
        assert!(!first.content.contains("read_full_result_"));
    }

    #[test]
    fn file_read_history_names_reader_only_after_runtime_confirmation() {
        let mut settings = default_settings();
        settings.compression_trigger = 1;
        settings.max_full_results = 0;
        let mut compressor = ToolResultCompressor::new(&settings);
        let routing = ResultRoutingIdentity::new(TEST_SESSION, "call_registered_file");
        let file_content = serde_json::json!({
            "path": "docs/large.md",
            "content_sha256": "e".repeat(64),
            "total_lines": 900,
            "offset": 400,
            "lines": (400..500).map(|i| format!("line {i}" )).collect::<Vec<_>>(),
            "returned": 100,
            "result_iri": routing.storage_iri.clone(),
            "session_reader": routing.reader_name.clone(),
        })
        .to_string();
        compressor.add_result_with_routing_and_reader(
            1,
            "file_read",
            &routing,
            &file_content,
            true,
        );
        let summary: serde_json::Value =
            serde_json::from_str(&compressor.get_results().front().expect("result").content)
                .expect("structured file history");
        assert_eq!(summary["session_reader"], routing.reader_name.as_str());
        assert_eq!(summary["result_iri"], routing.storage_iri.as_str());
    }

    #[test]
    fn test_summarize_plain_text_preserves_old_behavior() {
        let mut compressor = ToolResultCompressor::new(&default_settings());
        // body > 200 chars so compression triggers, but first 3 lines are unique text
        let long_text = "alpha line\nbeta line\ngamma line\n".to_string()
            + &"filler content to exceed the summary length threshold for compression".repeat(4);

        compressor.add_result(1, "bash", TEST_SESSION, "call_t1", &long_text);
        compressor.add_result(2, "bash", TEST_SESSION, "call_t2", &"a".repeat(500));
        compressor.add_result(3, "bash", TEST_SESSION, "call_t3", &"b".repeat(500));
        compressor.add_result(4, "bash", TEST_SESSION, "call_t4", &"c".repeat(500));
        compressor.add_result(5, "bash", TEST_SESSION, "call_t5", &"d".repeat(500));

        let first = &compressor.get_results()[0];
        assert!(first.is_compressed);
        assert!(first.content.starts_with("[Summary"));
        assert!(
            first.content.contains("alpha line"),
            "plain preview keeps text"
        );
        assert!(
            !first.content.contains("read_full_result_"),
            "a routing key without a registered handler must not create a reader hint"
        );
    }

    #[test]
    fn single_line_summary_is_bounded_before_metadata_is_added() {
        let mut settings = default_settings();
        settings.compression_trigger = 1;
        settings.max_full_results = 0;
        settings.max_summary_length = 64;
        let mut compressor = ToolResultCompressor::new(&settings);
        let content = format!("{{\"payload\":\"{}TAIL_SENTINEL\"}}", "x".repeat(20_000));
        compressor.add_result(1, "custom_tool", TEST_SESSION, "call_single_line", &content);
        let compressed = &compressor.get_results()[0];
        assert!(compressed.is_compressed);
        assert!(!compressed.content.contains("TAIL_SENTINEL"));
        assert!(
            compressed.content.len() < 256,
            "metadata overhead may exceed the preview budget, but the source line itself must remain bounded"
        );
    }

    #[test]
    fn test_compress_tool_messages_by_call_id() {
        let mut compressor = ToolResultCompressor::new(&default_settings());

        // Add results and trigger compression
        let long = "y".repeat(500);
        for i in 1..=6 {
            compressor.add_result(i, "file_read", TEST_SESSION, &format!("call_{}", i), &long);
        }
        assert!(compressor.get_results().front().unwrap().is_compressed);

        // Build messages: system + several tool messages
        let mut msgs = vec![ChatMessage {
            role: "system".to_string(),
            content: "sys".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        for i in 1..=4 {
            msgs.push(ChatMessage {
                role: "tool".to_string(),
                content: long.clone(),
                name: None,
                tool_calls: None,
                tool_call_id: Some(format!("call_{}", i)),
                reasoning_content: None,
            });
        }

        compressor.compress_tool_messages(&mut msgs, TEST_SESSION);

        // call_1 and call_2 are compressed (first two entries)
        let compressed_ids: std::collections::HashSet<String> = compressor
            .results
            .iter()
            .filter(|e| e.is_compressed)
            .map(|e| e.provider_tool_call_id.clone())
            .collect();
        for msg in msgs.iter().filter(|m| m.role == "tool") {
            let cid = msg.tool_call_id.as_ref().unwrap();
            if compressed_ids.contains(cid) {
                assert!(
                    msg.content.starts_with("[Summary"),
                    "tool_call_id={} should be compressed",
                    cid
                );
            } else {
                assert_eq!(
                    msg.content.len(),
                    long.len(),
                    "tool_call_id={} should remain full",
                    cid
                );
            }
        }
    }

    #[test]
    fn compression_uses_full_routing_key_when_raw_id_is_reused() {
        let mut settings = default_settings();
        settings.compression_trigger = 1;
        settings.max_full_results = 0;
        settings.max_summary_length = 24;
        let mut compressor = ToolResultCompressor::new(&settings);
        let first =
            ResultRoutingIdentity::for_tool_call("agent", TEST_SESSION, "request-1", "call_0");
        let second =
            ResultRoutingIdentity::for_tool_call("agent", TEST_SESSION, "request-2", "call_0");
        let first_content = format!(
            "first payload {}\nIRI: {}\nSession reader: {}",
            "a".repeat(80),
            first.storage_iri,
            first.reader_name
        );
        let second_content = format!(
            "second payload {}\nIRI: {}\nSession reader: {}",
            "b".repeat(80),
            second.storage_iri,
            second.reader_name
        );
        compressor.add_result_with_routing_and_reader(1, "bash", &first, &first_content, true);
        compressor.add_result_with_routing_and_reader(2, "bash", &second, &second_content, true);

        let mut messages = vec![
            ChatMessage {
                role: "tool".to_string(),
                content: first_content,
                name: None,
                tool_calls: None,
                tool_call_id: Some("call_0".to_string()),
                reasoning_content: None,
            },
            ChatMessage {
                role: "tool".to_string(),
                content: second_content,
                name: None,
                tool_calls: None,
                tool_call_id: Some("call_0".to_string()),
                reasoning_content: None,
            },
        ];
        compressor.compress_tool_messages(&mut messages, TEST_SESSION);

        assert!(messages[0].content.contains(&first.reader_name));
        assert!(!messages[0].content.contains(&second.reader_name));
        assert!(messages[1].content.contains(&second.reader_name));
        assert!(!messages[1].content.contains(&first.reader_name));
        assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_0"));
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("call_0"));
    }

    #[test]
    fn compression_preserves_only_live_exact_file_reader_cursor() {
        let mut settings = default_settings();
        settings.compression_trigger = 1;
        settings.max_full_results = 0;
        let mut compressor = ToolResultCompressor::new(&settings);
        let routing = ResultRoutingIdentity::for_tool_call(
            "agent-file-history",
            TEST_SESSION,
            "request-wide-line",
            "call_0",
        );
        let routed = serde_json::json!({
            "path": "docs/报告.md",
            "content_sha256": "a".repeat(64),
            "total_lines": 10,
            "offset": 0,
            "lines": ["已展示前缀...[line preview truncated]"],
            "returned": 0,
            "next_offset": 0,
            "reader_cursor": {"offset": 0, "limit": 1, "char_offset": 6},
            "partial_line_preview": true,
            "result_iri": routing.storage_iri,
            "session_reader": routing.reader_name,
            "message": format!("Use reader {} while advertised", routing.reader_name),
        })
        .to_string();
        compressor.add_result_with_routing_and_reader(1, "file_read", &routing, &routed, true);

        let message = || ChatMessage {
            role: "tool".to_string(),
            content: routed.clone(),
            name: None,
            tool_calls: None,
            tool_call_id: Some("call_0".to_string()),
            reasoning_content: None,
        };
        let mut active_messages = vec![message()];
        compressor.compress_tool_messages_with_active_readers(
            &mut active_messages,
            TEST_SESSION,
            &std::collections::HashSet::from([routing.reader_name.clone()]),
        );
        let active: serde_json::Value = serde_json::from_str(&active_messages[0].content).unwrap();
        assert_eq!(active["session_reader"], routing.reader_name);
        assert_eq!(active["result_iri"], routing.storage_iri);
        assert_eq!(active["reader_cursor"]["offset"], 0);
        assert_eq!(active["reader_cursor"]["limit"], 1);
        assert_eq!(active["reader_cursor"]["char_offset"], 6);

        let mut retired_messages = vec![message()];
        compressor.compress_tool_messages_with_active_readers(
            &mut retired_messages,
            TEST_SESSION,
            &std::collections::HashSet::new(),
        );
        let retired: serde_json::Value =
            serde_json::from_str(&retired_messages[0].content).unwrap();
        assert!(retired.get("session_reader").is_none());
        assert!(retired.get("result_iri").is_none());
        assert!(retired.get("reader_cursor").is_none());
        assert_eq!(retired["next_offset"], 0);
        assert_eq!(retired_messages[0].tool_call_id.as_deref(), Some("call_0"));
        assert!(!retired_messages[0].content.contains(&routing.reader_name));
    }

    #[test]
    fn compression_fails_closed_for_ambiguous_legacy_inline_reuse() {
        let mut settings = default_settings();
        settings.compression_trigger = 1;
        settings.max_full_results = 0;
        settings.max_summary_length = 20;
        let mut compressor = ToolResultCompressor::new(&settings);
        let first =
            ResultRoutingIdentity::for_tool_call("agent", TEST_SESSION, "request-1", "call_0");
        let second =
            ResultRoutingIdentity::for_tool_call("agent", TEST_SESSION, "request-2", "call_0");
        let content = "same legacy inline payload without a routing reference".repeat(3);
        compressor.add_result_with_routing(1, "bash", &first, &content);
        compressor.add_result_with_routing(2, "bash", &second, &content);
        let mut messages = vec![ChatMessage {
            role: "tool".to_string(),
            content: content.clone(),
            name: None,
            tool_calls: None,
            tool_call_id: Some("call_0".to_string()),
            reasoning_content: None,
        }];

        compressor.compress_tool_messages(&mut messages, TEST_SESSION);
        assert_eq!(messages[0].content, content);
    }

    #[test]
    fn test_context_window_should_compress() {
        let manager = ContextWindowManager::new(&default_context_settings());
        let empty: Vec<ChatMessage> = Vec::new();

        assert!(!manager.should_compress(10, &empty));
        assert!(manager.should_compress(20, &empty));
    }

    #[test]
    fn test_model_context_window_lookup() {
        assert_eq!(model_context_window("deepseek-v4-flash"), 128_000);
        assert_eq!(model_context_window("DeepSeek-V3"), 128_000);
        assert_eq!(model_context_window("gpt-4o-mini"), 128_000);
        assert_eq!(model_context_window("claude-sonnet-4"), 200_000);
        assert_eq!(model_context_window("llama-3.1-70b"), 128_000);
        assert_eq!(model_context_window("unknown-model"), 64_000);
    }

    #[test]
    fn test_model_aware_budget() {
        let mut settings = default_context_settings();
        settings.model_aware = true;
        let manager = ContextWindowManager::new(&settings);

        // A large model window must not override the configured cost ceiling.
        assert_eq!(manager.budget_for_model("deepseek-v4-flash"), 16_000);
        assert_eq!(manager.budget_for_model("mystery-model"), 16_000);
    }

    #[test]
    fn test_model_aware_compression_trigger() {
        // max_messages raised so the token budget is the only trigger criterion.
        let mut settings = default_context_settings();
        settings.model_aware = true;
        settings.max_messages = 5000;
        let manager = ContextWindowManager::new(&settings);

        // Build a message payload that exceeds the configured 16K ceiling.
        let mut msgs = Vec::new();
        // 2000 x 100-char ASCII messages ≈ 50K tokens: above the 16K static
        // budget yet below the 102K model-aware budget.
        for _ in 0..2000 {
            msgs.push(ChatMessage {
                role: "user".to_string(),
                content: "X".repeat(100),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
        assert!(ContextWindowManager::estimate_tokens(&msgs) > 16_000);
        assert!(manager.should_compress_for_model(msgs.len(), &msgs, "deepseek-v4-flash"));

        // Same payload on a legacy manager (model_aware=false) triggers compression.
        let mut legacy_settings = default_context_settings();
        legacy_settings.max_messages = 5000;
        let legacy = ContextWindowManager::new(&legacy_settings);
        assert!(legacy.should_compress_for_model(msgs.len(), &msgs, "deepseek-v4-flash"));
    }

    #[test]
    fn tool_schema_reserve_is_part_of_the_request_budget() {
        let mut settings = default_context_settings();
        settings.max_messages = 100;
        settings.max_tokens = 100;
        let manager = ContextWindowManager::new(&settings);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "small request".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "large_tool",
                "description": "x".repeat(500),
                "parameters": {"type": "object", "properties": {}}
            }
        })];
        let reserve = ContextWindowManager::estimate_tool_schema_tokens(&tools);
        assert!(reserve > 100);
        assert!(!manager.should_compress_for_model(messages.len(), &messages, "deepseek-v4-flash"));
        assert!(manager.should_compress_for_model_with_reserve(
            messages.len(),
            &messages,
            "deepseek-v4-flash",
            reserve,
        ));
    }

    #[test]
    fn token_trigger_compresses_even_below_message_count_limit() {
        let mut settings = default_context_settings();
        settings.max_messages = 100;
        settings.max_tokens = 100;
        settings.preserve_recent = 2;
        let manager = ContextWindowManager::new(&settings);

        let mut messages = vec![ChatMessage {
            role: "system".to_string(),
            content: "system contract".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        for index in 0..7 {
            messages.push(ChatMessage {
                role: if index % 2 == 0 { "assistant" } else { "user" }.to_string(),
                content: format!("history-{index}-{}", "x".repeat(180)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }

        assert!(messages.len() < settings.max_messages);
        assert!(manager.should_compress(messages.len(), &messages));
        let (compressed, _) = manager.compress_messages(&messages);
        assert!(compressed.len() < messages.len());
        assert_eq!(compressed.first().unwrap().role, "system");
        assert_eq!(
            compressed.last().unwrap().content,
            messages.last().unwrap().content
        );
    }

    #[test]
    fn active_session_stays_full_before_compression_boundary() {
        let settings = default_context_settings();
        let manager = ContextWindowManager::new(&settings);
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "system contract".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "CURRENT_SESSION_FULL_ASSISTANT_CONTENT".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: Some("CURRENT_SESSION_REASONING".to_string()),
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "CURRENT_SESSION_FULL_TOOL_RESULT".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: Some("call-current".to_string()),
                reasoning_content: None,
            },
        ];

        assert!(!manager.should_compress_for_model(messages.len(), &messages, "deepseek-v3"));
        assert_eq!(
            messages[1].content,
            "CURRENT_SESSION_FULL_ASSISTANT_CONTENT"
        );
        assert_eq!(messages[2].content, "CURRENT_SESSION_FULL_TOOL_RESULT");
    }

    #[test]
    fn compression_preserves_recent_current_session_messages_verbatim() {
        let mut settings = default_context_settings();
        settings.max_messages = 4;
        settings.max_tokens = usize::MAX;
        settings.preserve_recent = 2;
        let manager = ContextWindowManager::new(&settings);
        let mut messages = vec![ChatMessage {
            role: "system".to_string(),
            content: "system contract".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        for index in 0..6 {
            messages.push(ChatMessage {
                role: if index % 2 == 0 { "assistant" } else { "user" }.to_string(),
                content: format!(
                    "full-current-session-message-{index}-{}",
                    "payload".repeat(8)
                ),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: Some(format!("reasoning-{index}")),
            });
        }
        let expected_recent = messages[messages.len() - 2..].to_vec();

        assert!(manager.should_compress(messages.len(), &messages));
        let (compressed, summary) = manager.compress_messages(&messages);
        assert!(!summary.is_empty());
        let actual_recent = &compressed[compressed.len() - 2..];
        for (actual, expected) in actual_recent.iter().zip(expected_recent.iter()) {
            assert_eq!(actual.role, expected.role);
            assert_eq!(actual.content, expected.content);
            assert_eq!(actual.reasoning_content, expected.reasoning_content);
        }
    }

    #[test]
    fn compression_preserves_complete_typed_task_prefix_and_raw_tool_call_id() {
        use crate::gateway::unified_gateway::{ToolCallFunction, ToolCallPayload};

        let mut settings = default_context_settings();
        settings.max_messages = 6;
        settings.max_tokens = usize::MAX;
        settings.compression_ratio = 0.0;
        settings.preserve_recent = 2;
        let manager = ContextWindowManager::new(&settings);

        // This is the exact shape assembled by AgentRunner before any
        // checkpoint/current-session protocol is appended. The old whole-list
        // compression kept only the first system message and could therefore
        // leave the execution contract visible while silently deleting all
        // three business-task sentinels below.
        let immutable_prefix = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "KERNEL_EXECUTION_CONTRACT_SENTINEL".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "system".to_string(),
                content: "TYPED_EFFECT_POLICY_SENTINEL".to_string(),
                name: Some("context_authoritative_instruction".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "FRESH_DYNAMIC_AGENT_MD_SENTINEL".to_string(),
                name: Some("context_model_generated_plan".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "ORIGINAL_BUSINESS_TASK_SENTINEL".to_string(),
                name: Some("context_user_input".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "CURRENT_CHILD_OBJECTIVE_SENTINEL".to_string(),
                name: Some("context_model_history".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        let prefix_wire = serde_json::to_string(&immutable_prefix).unwrap();
        let immutable_prefix_len = immutable_prefix.len();
        let mut messages = immutable_prefix;
        for index in 0..6 {
            messages.push(ChatMessage {
                role: "user".to_string(),
                content: format!("discardable protocol history {index}"),
                name: Some("context_model_history".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
        let raw_provider_call_id = "provider/call:原样-001";
        let assistant = ChatMessage {
            role: "assistant".to_string(),
            content: String::new(),
            name: None,
            tool_calls: Some(vec![ToolCallPayload {
                id: raw_provider_call_id.to_string(),
                call_type: "function".to_string(),
                function: ToolCallFunction {
                    name: "bash".to_string(),
                    arguments: r#"{"command":"python -m pytest -q"}"#.to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: Some("verify the current child objective".to_string()),
        };
        let tool = ChatMessage {
            role: "tool".to_string(),
            content: "51 passed".to_string(),
            name: Some("bash".to_string()),
            tool_calls: None,
            tool_call_id: Some(raw_provider_call_id.to_string()),
            reasoning_content: None,
        };
        messages.push(assistant.clone());
        messages.push(tool.clone());

        assert!(manager.should_compress(messages.len(), &messages));
        let (compressed, _) =
            manager.compress_messages_preserving_prefix(&messages, immutable_prefix_len);

        assert_eq!(
            serde_json::to_string(&compressed[..immutable_prefix_len]).unwrap(),
            prefix_wire,
            "every immutable typed task/agent.md message must remain byte-for-byte present"
        );
        assert!(compressed.len() < messages.len());
        let retained_assistant = &compressed[compressed.len() - 2];
        let retained_tool = &compressed[compressed.len() - 1];
        assert_eq!(
            serde_json::to_string(retained_assistant).unwrap(),
            serde_json::to_string(&assistant).unwrap()
        );
        assert_eq!(
            serde_json::to_string(retained_tool).unwrap(),
            serde_json::to_string(&tool).unwrap()
        );
        assert_eq!(
            retained_assistant.tool_calls.as_ref().unwrap()[0].id,
            raw_provider_call_id
        );
        assert_eq!(
            retained_tool.tool_call_id.as_deref(),
            Some(raw_provider_call_id)
        );
    }

    #[test]
    fn oversized_immutable_task_prefix_is_rejected_explicitly() {
        let mut settings = default_context_settings();
        settings.max_tokens = 8;
        settings.model_aware = false;
        let manager = ContextWindowManager::new(&settings);
        let prefix = vec![ChatMessage {
            role: "user".to_string(),
            content: "CURRENT_CHILD_OBJECTIVE_MUST_NOT_BE_DROPPED".repeat(8),
            name: Some("context_model_history".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let error = manager
            .validate_immutable_prefix_for_model(&prefix, "test-model")
            .expect_err("an uncompressible task prefix must fail closed");
        assert!(error.required_tokens > error.budget_tokens);
        assert_eq!(error.budget_tokens, 8);
        assert!(error.to_string().contains("immutable initial context"));
    }

    #[test]
    fn tool_schema_reserve_that_cannot_coexist_with_immutable_prefix_is_rejected() {
        let mut settings = default_context_settings();
        settings.max_tokens = 40;
        settings.model_aware = false;
        let manager = ContextWindowManager::new(&settings);
        let prefix = vec![ChatMessage {
            role: "user".to_string(),
            content: "fixed task identity".to_string(),
            name: Some("context_user_input".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let immutable_tokens = ContextWindowManager::estimate_tokens(&prefix);
        assert!(immutable_tokens < settings.max_tokens);
        manager
            .validate_immutable_prefix_for_model(&prefix, "test-model")
            .expect("the immutable prefix alone fits");

        let reserve = settings.max_tokens - immutable_tokens + 1;
        let error = manager
            .validate_immutable_prefix_for_model_with_reserve(&prefix, "test-model", reserve)
            .expect_err("compression cannot remove an out-of-band tool schema reserve");
        assert_eq!(error.immutable_tokens, immutable_tokens);
        assert_eq!(error.request_reserve_tokens, reserve);
        assert_eq!(error.required_tokens, immutable_tokens + reserve);
        assert_eq!(error.budget_tokens, settings.max_tokens);
        assert!(error.to_string().contains(&reserve.to_string()));
        assert!(error.to_string().contains("tool schema reserve"));
    }

    #[test]
    fn test_estimate_text_tokens_cjk_weighting() {
        // UTF-8 Chinese is 3 bytes/char; naive len()/4 undervalued it at
        // 0.75 tokens/char. A 4-char phrase must cost ~4 tokens, one per char.
        assert_eq!(estimate_text_tokens("你好世界"), 4);
        // Plain ASCII still ~4 bytes per token.
        assert_eq!(estimate_text_tokens("Hello, world!"), 3);
        // Mixed content weights each part correctly.
        assert_eq!(estimate_text_tokens("你好 world"), 3);
    }
}
