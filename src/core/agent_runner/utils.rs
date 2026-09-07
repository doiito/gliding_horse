use std::collections::HashMap;

use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::core::agent_instance::{AgentInstance, AgentRole, AgentStatus};
use crate::core::execution_journal::{
    PayloadReference, TaskExecutionJournal, TaskExecutionJournalKind, ToolCallIdentity,
};
use crate::gateway::unified_gateway::ChatMessage;
use crate::jsonld::{generate_iri, validate_jsonld_node, JsonLdContext, JsonLdNode};
use crate::memory::l1_session::L1Session;
use crate::tools::hooks::{HookControl, HookPoint};
use crate::tools::tool_executor::ToolExecutor;
use crate::CoreError;

use super::execution::{
    admit_provider_tool_call_batch, advertised_tool_names, assess_verification_call,
    attach_toolguard_validation_feedback, ca_evidence_close_directive,
    ca_evidence_close_tool_definitions, ca_evidence_focus_tool_definitions,
    ca_has_successful_verifier_receipt, ca_verification_probe_tool_definitions,
    classify_raw_tool_protocol_response, da_evidence_close_tool_definitions,
    da_evidence_focus_tool_definitions, da_hard_close_active,
    da_post_effect_inspection_focus_tool_definitions, da_typed_contract_close_directive,
    da_verified_close_tool_definitions, da_verified_focus_tool_definitions, disclosed_tool_result,
    effective_effect_block_turns, effective_role_max_turns, emit_tool_hook_decision, evidence_key,
    exact_workspace_write_targets_materialized, execute_tool_hook_decision,
    filter_tool_search_result, finalize_ca_terminal_contract, immediate_correction_recovery_active,
    initial_execution_phase, is_business_handoff_content, is_verification_call,
    is_workspace_mutation_candidate, mutation_recovery_tool_definitions,
    pa_empty_workspace_tool_definitions, pa_planning_focus_tool_definitions,
    phase_tool_definitions, raw_tool_protocol_correction_directive, raw_tool_protocol_shape,
    react_reasoning_effort_for_dispatch, record_workspace_effect_turn, refresh_ca_evidence_ledger,
    refresh_execution_ledger, repair_recovery_rejection, requires_workspace_effect,
    requires_workspace_settlement, should_track_result_for_compression, skipped_pre_tool_result,
    skipped_pre_tool_terminal_reason, tool_hook_context, unadvertised_tool_call_result,
    verification_execution_profile, verification_preflight_rejection, visible_tool_result_hashes,
    workspace_effect_recovery_active, workspace_inventory_authoritatively_empty,
    workspace_inventory_complete_and_bounded, workspace_inventory_coverage,
    workspace_inventory_tool_definitions, CaAuditConvergence, DaVerificationConvergence,
    ExecutionPhase, ProviderToolCallLedger, RawToolProtocolDisposition, RawToolProtocolShape,
    RepairBaselineWindow, REPEATED_RAW_TOOL_PROTOCOL_FAILURE,
};
use super::{LlmParsedResponse, TaskContext, TaskResult, TaskVerdict};

/// Ensures every durable streaming LLM Prepared frame has one terminal frame
/// even when the owning future is aborted between await points. Normal paths
/// disarm it only after appending Received/Failed; Drop records a payload-free
/// cancelled failure synchronously, which is safe during Tokio cancellation.
struct StreamingLlmJournalGuard<'a> {
    journal: &'a Option<TaskExecutionJournal>,
    request_id: String,
    started_at: std::time::Instant,
    armed: bool,
}

impl<'a> StreamingLlmJournalGuard<'a> {
    fn new(journal: &'a Option<TaskExecutionJournal>, request_id: String) -> Self {
        Self {
            journal,
            request_id,
            started_at: std::time::Instant::now(),
            armed: true,
        }
    }

    fn finish(&mut self, event: TaskExecutionJournalKind) {
        if !self.armed {
            return;
        }
        super::execution::append_execution_journal_event(self.journal, event);
        self.armed = false;
    }
}

impl Drop for StreamingLlmJournalGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        super::execution::append_execution_journal_event(
            self.journal,
            TaskExecutionJournalKind::LlmRequestFailed {
                request_id: self.request_id.clone(),
                latency_ms: self
                    .started_at
                    .elapsed()
                    .as_millis()
                    .min(u128::from(u64::MAX)) as u64,
                error_class: "cancelled".to_string(),
                http_status: None,
                retryable: Some(false),
            },
        );
        self.armed = false;
    }
}

impl super::AgentRunner {
    /// Persist an AgentRunner response without allowing a stalled embedded L0
    /// writer to freeze the ReAct loop. The in-flight permit stays owned by
    /// the blocking task after a timeout, so later turns degrade archival
    /// immediately instead of accumulating blocked writers behind redb's
    /// single-writer lock.
    pub(super) async fn archive_full_turn_to_l0_bounded(
        &self,
        session: &L1Session,
        role: &str,
        thought: &str,
        content_json: &str,
    ) -> Result<String, CoreError> {
        const ARCHIVE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

        let permit = self
            .l0_archive_gate
            .clone()
            .try_acquire_owned()
            .map_err(|_| CoreError::Internal {
                message: "L0 archive still in progress; skipped this non-critical turn archive"
                    .to_string(),
            })?;
        let iri = format!(
            "iri://archive/{}/{}/{}",
            session
                .task_iri()
                .strip_prefix("iri://")
                .unwrap_or(session.task_iri()),
            role,
            uuid::Uuid::new_v4().hyphenated()
        );
        let archived_content = serde_json::from_str::<Value>(content_json)
            .unwrap_or_else(|_| Value::String(content_json.to_string()));
        let payload = json!({
            "@id": &iri,
            "@type": "LLMResponse",
            "role": role,
            "agent_id": session.agent_id(),
            "session_id": session.session_id(),
            "thought": thought,
            "content": archived_content,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        })
        .to_string();
        let store = self.l0_store.clone();
        let write_iri = iri.clone();
        let write = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            store.store(&write_iri, &payload).map(|_| write_iri)
        });

        match tokio::time::timeout(ARCHIVE_TIMEOUT, write).await {
            Ok(Ok(result)) => result,
            Ok(Err(error)) => Err(CoreError::Internal {
                message: format!("L0 archive worker failed: {error}"),
            }),
            Err(_) => Err(CoreError::Internal {
                message: format!(
                    "L0 archive exceeded {} seconds; task execution continues",
                    ARCHIVE_TIMEOUT.as_secs()
                ),
            }),
        }
    }

    pub(super) fn effective_response_content(
        content: &str,
        reasoning_content: Option<&str>,
        finish_reason: &str,
        has_tool_calls: bool,
    ) -> String {
        if content.trim().is_empty()
            && !has_tool_calls
            && matches!(finish_reason, "stop" | "end_turn")
        {
            return reasoning_content.unwrap_or_default().to_string();
        }
        content.to_string()
    }

    /// Utility: extract summary from agent output.
    /// Unused — kept for future SA result summarization.
    #[allow(dead_code)]
    fn extract_summary(&self, content: &str, reasoning_content: Option<&str>) -> String {
        // Prefer extracting summary field from JSON first
        if let Ok(parsed) = serde_json::from_str::<Value>(content) {
            if let Some(summary) = parsed.get("summary").and_then(|s| s.as_str()) {
                return summary.chars().take(500).collect();
            }
            // If native reasoning exists, no need to extract thought from JSON (avoid duplication)
            if reasoning_content.is_none() {
                if let Some(thought) = parsed.get("thought").and_then(|s| s.as_str()) {
                    return thought.chars().take(500).collect();
                }
            }
            if let Some(content_str) = parsed.get("content").and_then(|s| s.as_str()) {
                return content_str.chars().take(500).collect();
            }
        }

        // If native reasoning exists, use it as summary
        if let Some(reasoning) = reasoning_content {
            let reasoning_summary: String = reasoning.chars().take(300).collect();
            return format!("[Reasoning] {}", reasoning_summary);
        }

        // Final fallback: use first 500 chars of content
        content.chars().take(500).collect()
    }

    pub(super) fn parse_llm_response(
        &self,
        content: &str,
        reasoning_content: Option<&str>,
        supports_native_reasoning: bool,
    ) -> LlmParsedResponse {
        let mut response = LlmParsedResponse {
            thought: None,
            content: content.to_string(),
            content_from_reasoning: false,
            summary: None,
            action: None,
            is_valid_json: false,
            has_native_reasoning: reasoning_content.is_some(),
            emphasis: Vec::new(),
        };

        // If native reasoning exists, use it directly
        if let Some(reasoning) = reasoning_content {
            response.thought = Some(reasoning.to_string());
            response.has_native_reasoning = true;
        }

        // Parse JSON attempt
        if let Ok(parsed) = serde_json::from_str::<Value>(content) {
            response.is_valid_json = true;

            // Extract summary
            if let Some(summary) = parsed.get("summary").and_then(|s| s.as_str()) {
                response.summary = Some(summary.to_string());
            }

            // Extract content. Structured role payloads (notably CA's
            // `ca_audit/v1`) are valid business content too. Serialize them
            // canonically so sync and streaming paths normalize the same
            // provider response instead of the sync path retaining the whole
            // outer ReAct envelope.
            if let Some(content_value) = parsed.get("content") {
                match content_value {
                    Value::String(content_str) => response.content = content_str.clone(),
                    Value::Null => {}
                    structured => response.content = structured.to_string(),
                }
            }

            // Extract thought (only when model does not support native reasoning)
            if !supports_native_reasoning {
                if let Some(thought) = parsed.get("thought").and_then(|s| s.as_str()) {
                    response.thought = Some(thought.to_string());
                }
            }

            // Extract emphasis field (emphasis content identified by LLM itself)
            if let Some(emphasis) = parsed.get("emphasis") {
                if let Some(arr) = emphasis.as_array() {
                    response.emphasis = arr
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                } else if let Some(s) = emphasis.as_str() {
                    response.emphasis = vec![s.to_string()];
                }
            }

            let content_text = parsed.get("content").and_then(|c| c.as_str()).unwrap_or("");
            let keyword_emphasis = Self::extract_emphasis_by_keywords(content_text);
            for kw_em in keyword_emphasis {
                if !response.emphasis.iter().any(|e| e == &kw_em) {
                    response.emphasis.push(kw_em);
                }
            }

            // Extract action field
            if let Some(action) = parsed.get("action").and_then(|a| a.as_str()) {
                response.action = Some(action.to_string());
            }

            // Responses-API-compatible reasoning models may return a valid
            // ReAct envelope whose `content` field is null (or the literal
            // string "null") while the terminal evidence is present only in
            // `reasoning_content`. Preserve the envelope's summary/action, but
            // make the substantive terminal evidence available to downstream
            // agents and quality gates. This is intentionally limited to a
            // nullish content field; normal content always remains authoritative.
            let envelope_content_is_nullish = parsed
                .get("content")
                .map(|value| {
                    value.is_null()
                        || value
                            .as_str()
                            .map(|text| text.trim().is_empty() || text.trim() == "null")
                            .unwrap_or(false)
                })
                .unwrap_or(false);
            if envelope_content_is_nullish {
                if let Some(reasoning) = reasoning_content.filter(|text| !text.trim().is_empty()) {
                    response.content = reasoning.to_string();
                    response.content_from_reasoning = true;
                }
            }
        } else {
            if let Some(extracted) = Self::try_extract_json_from_markdown(content) {
                if let Ok(parsed) = serde_json::from_str::<Value>(&extracted) {
                    response.is_valid_json = true;
                    if let Some(summary) = parsed.get("summary").and_then(|s| s.as_str()) {
                        response.summary = Some(summary.to_string());
                    }
                    if let Some(content_value) = parsed.get("content") {
                        match content_value {
                            Value::String(content_str) => response.content = content_str.clone(),
                            Value::Null => {}
                            structured => response.content = structured.to_string(),
                        }
                    }
                    if !supports_native_reasoning {
                        if let Some(thought) = parsed.get("thought").and_then(|s| s.as_str()) {
                            response.thought = Some(thought.to_string());
                        }
                    }
                    if let Some(action) = parsed.get("action").and_then(|a| a.as_str()) {
                        response.action = Some(action.to_string());
                    }
                    let envelope_content_is_nullish = parsed
                        .get("content")
                        .map(|value| {
                            value.is_null()
                                || value
                                    .as_str()
                                    .map(|text| text.trim().is_empty() || text.trim() == "null")
                                    .unwrap_or(false)
                        })
                        .unwrap_or(false);
                    if envelope_content_is_nullish {
                        if let Some(reasoning) =
                            reasoning_content.filter(|text| !text.trim().is_empty())
                        {
                            response.content = reasoning.to_string();
                            response.content_from_reasoning = true;
                        }
                    }
                } else {
                    response.summary = Some(Self::generate_auto_summary(content));
                }
            } else {
                response.summary = Some(Self::generate_auto_summary(content));
            }
        }

        response
    }

    pub(super) fn generate_auto_summary(content: &str) -> String {
        let content_clean = content.trim();
        if content_clean.len() <= 200 {
            return content_clean.to_string();
        }

        if let Some(first_sentence_end) =
            content_clean.find(|c| c == '。' || c == '.' || c == '！' || c == '!')
        {
            let end_byte = first_sentence_end
                + content_clean[first_sentence_end..]
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(1);
            if end_byte <= 200 {
                return content_clean[..end_byte].to_string();
            }
        }

        content_clean.chars().take(200).collect()
    }

    /// When the `finish` action hardcodes `success` even after an agent declared it blocked
    /// (e.g. "no task spec; zero deliverables"), the SA PDCA retry loop short-circuits and the
    /// CLI shows `✅ SUCCESS` with zero output. Recover the true verdict from the summary so
    /// callers can react honestly. Conservative: only explicit blockers downgrade the verdict.
    pub(super) fn detect_blocker_verdict(summary: &str) -> Option<&'static str> {
        let s = summary.to_lowercase();
        const BLOCKER_MARKERS: [&str; 8] = [
            "blocked: no",
            "no task spec",
            "missing task spec",
            "no spec found",
            "zero deliverables",
            "zero deliverable",
            "cannot proceed",
            "blocked, cannot",
        ];
        if BLOCKER_MARKERS.iter().any(|m| s.contains(m)) {
            return Some("failed");
        }
        let explicit_failed_line = s.lines().map(str::trim).any(|line| {
            line.starts_with("failed:")
                || line.starts_with("blocked:")
                || line.starts_with("partial_success:")
        });
        let explicit_partial_statement = s.contains("honest status is partial")
            || s.contains("status is partial/blocked")
            || s.contains("status: partial/blocked")
            || s.contains("诚实状态是 partial")
            || s.contains("诚实声明 partial");
        let explicit_unmet_statement = (summary.contains("未达成") || summary.contains("未完成"))
            && (summary.contains("不得") && summary.contains("成功")
                || s.contains("partial/blocked"));
        if explicit_failed_line || explicit_partial_statement || explicit_unmet_statement {
            return Some("failed");
        }
        None
    }

    /// A run that stops while the model still has a pending action is never a
    /// completed success. Earlier effects are retained as partial progress,
    /// while a missing mandatory effect remains a hard failure. Model prose
    /// may only further downgrade the runtime-owned verdict.
    pub(super) fn interrupted_execution_verdict(
        workspace_effect_required: bool,
        workspace_effect_observed: bool,
        summary: &str,
    ) -> TaskVerdict {
        if workspace_effect_required && !workspace_effect_observed {
            TaskVerdict::Failed
        } else if Self::detect_blocker_verdict(summary).is_some() {
            TaskVerdict::Blocked
        } else {
            TaskVerdict::PartialSuccess
        }
    }

    pub(crate) fn try_extract_json_from_markdown(content: &str) -> Option<String> {
        let trimmed = content.trim();

        if serde_json::from_str::<Value>(trimmed).is_ok() {
            return Some(trimmed.to_string());
        }

        // Search every opening brace instead of trusting the first one. LLMs
        // commonly put paths such as `project/{src,tests}` in prose before
        // the actual ReAct object. The former implementation returned that
        // non-JSON brace group and never reached the valid terminal envelope.
        // Quoted braces and escaped quotes must not alter structural depth.
        let mut first_valid = None;
        for (start, character) in trimmed.char_indices() {
            if character != '{' {
                continue;
            }
            let mut depth = 0usize;
            let mut in_string = false;
            let mut escaped = false;
            for (relative_end, candidate_character) in trimmed[start..].char_indices() {
                if in_string {
                    if escaped {
                        escaped = false;
                    } else {
                        match candidate_character {
                            '\\' => escaped = true,
                            '"' => in_string = false,
                            _ => {}
                        }
                    }
                    continue;
                }

                match candidate_character {
                    '"' => in_string = true,
                    '{' => depth = depth.saturating_add(1),
                    '}' => {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                        if depth == 0 {
                            let end = start + relative_end + candidate_character.len_utf8();
                            let candidate = &trimmed[start..end];
                            if let Ok(value) = serde_json::from_str::<Value>(candidate) {
                                // Prefer an actual ReAct envelope over an
                                // earlier incidental but valid JSON example.
                                // Keep the first valid object as a fallback
                                // for callers that intentionally return a
                                // bare structured business payload.
                                if first_valid.is_none() {
                                    first_valid = Some(candidate.to_string());
                                }
                                if value.get("action").and_then(Value::as_str).is_some() {
                                    return Some(candidate.to_string());
                                }
                            }
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }

        first_valid
    }

    pub(super) async fn save_emphasis_to_l0(
        &self,
        emphasis_items: &[String],
        task_iri: &str,
        agent_id: &str,
        dedup_threshold: f64,
    ) {
        if emphasis_items.is_empty() {
            return;
        }

        // Apply max_items truncation to prevent emphasis from expanding indefinitely
        let max_items = self
            .emphasis_config
            .as_ref()
            .map(|c| c.max_items)
            .unwrap_or(50);
        let items: Vec<String> = emphasis_items.iter().take(max_items).cloned().collect();
        let task_iri = task_iri.to_string();
        let agent_id = agent_id.to_string();
        let store = self.l0_store.clone();
        let permit = match self.l0_archive_gate.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                warn!("Skipped emphasis persistence because an L0 archive is already in flight");
                return;
            }
        };

        let persist = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut existing = Vec::new();
            let scan_prefix = format!(
                "iri://emphasis/{}",
                task_iri.strip_prefix("iri://").unwrap_or(&task_iri)
            );
            if let Ok(entries) = store.scan_iri_prefix(&scan_prefix, 200) {
                for entry in entries {
                    if let Ok(parsed) = serde_json::from_str::<Value>(&entry.content) {
                        if let Some(content) =
                            parsed.get("content").and_then(|value| value.as_str())
                        {
                            existing.push(content.to_string());
                        }
                    }
                }
            }
            if let Ok(nodes) = store.search_by_tags(&[String::from("emphasis")]) {
                for node in nodes {
                    if let Ok(parsed) = serde_json::from_str::<Value>(&node.content) {
                        if parsed.get("task_iri").is_none() {
                            if let Some(content) =
                                parsed.get("content").and_then(|value| value.as_str())
                            {
                                if !existing.iter().any(|candidate| candidate == content) {
                                    existing.push(content.to_string());
                                }
                            }
                        }
                    }
                }
            }

            let mut saved = 0usize;
            for content in items {
                let is_duplicate = existing.iter().any(|existing_content| {
                    Self::calculate_similarity(&content, existing_content) >= dedup_threshold
                });
                if is_duplicate {
                    continue;
                }
                let iri = format!(
                    "iri://emphasis/{}/{}",
                    task_iri.strip_prefix("iri://").unwrap_or(&task_iri),
                    uuid::Uuid::new_v4()
                );
                let node = json!({
                    "@id": &iri,
                    "@type": "EmphasisContent",
                    "content": content,
                    "task_iri": task_iri,
                    "agent_id": agent_id,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "permanent": true
                });
                match store.store(&iri, &node.to_string()) {
                    Ok(()) => {
                        saved += 1;
                    }
                    Err(error) => {
                        warn!(%error, iri = %iri, "Failed to save emphasis content to L0")
                    }
                }
            }
            saved
        });

        match tokio::time::timeout(std::time::Duration::from_secs(5), persist).await {
            Ok(Ok(saved)) => {
                info!(saved, "Saved emphasis content to L0");
            }
            Ok(Err(error)) => warn!(%error, "Emphasis persistence worker failed"),
            Err(_) => warn!("Emphasis persistence exceeded 5 seconds; task execution continues"),
        }
    }

    #[allow(dead_code)]
    pub(super) async fn load_emphasis_from_l0(&self, task_iri: &str) -> Vec<String> {
        let mut result = Vec::new();

        // Use IRI prefix scan instead of full tag search
        // Save IRI format: iri://emphasis/{task_iri}/{uuid}
        let scan_prefix = format!(
            "iri://emphasis/{}",
            task_iri.strip_prefix("iri://").unwrap_or(task_iri)
        );
        if let Ok(entries) = self.l0_store.scan_iri_prefix(&scan_prefix, 200) {
            for entry in &entries {
                if let Ok(parsed) = serde_json::from_str::<Value>(&entry.content) {
                    if let Some(content) = parsed.get("content").and_then(|c| c.as_str()) {
                        result.push(content.to_string());
                    }
                }
            }
        }

        // Also load global emphasis (entries without task_iri), using emphasis tag fallback scan
        if let Ok(nodes) = self.l0_store.search_by_tags(&[String::from("emphasis")]) {
            for node in nodes {
                if let Ok(parsed) = serde_json::from_str::<Value>(&node.content) {
                    let is_global = parsed.get("task_iri").is_none();
                    if is_global {
                        if let Some(content) = parsed.get("content").and_then(|c| c.as_str()) {
                            if !result.contains(&content.to_string()) {
                                result.push(content.to_string());
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Load only operator-authored/global emphasis. Task-scoped emphasis is
    /// emitted by business-agent model turns and is therefore historical
    /// evidence, not an authoritative instruction for a later phase.
    pub(super) async fn load_global_emphasis_from_l0(&self) -> Vec<String> {
        let mut result = Vec::new();
        if let Ok(nodes) = self.l0_store.search_by_tags(&[String::from("emphasis")]) {
            for node in nodes {
                if let Ok(parsed) = serde_json::from_str::<Value>(&node.content) {
                    if parsed.get("task_iri").is_none() {
                        if let Some(content) = parsed.get("content").and_then(Value::as_str) {
                            if !result.iter().any(|existing| existing == content) {
                                result.push(content.to_string());
                            }
                        }
                    }
                }
            }
        }
        result
    }

    fn calculate_similarity(a: &str, b: &str) -> f64 {
        if a == b {
            return 1.0;
        }

        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();

        if a_chars.is_empty() || b_chars.is_empty() {
            return 0.0;
        }

        // Use simple Jaccard similarity
        let a_set: std::collections::HashSet<char> = a_chars.iter().copied().collect();
        let b_set: std::collections::HashSet<char> = b_chars.iter().copied().collect();

        let intersection = a_set.intersection(&b_set).count();
        let union = a_set.union(&b_set).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f64 / union as f64
    }

    #[allow(dead_code)]
    pub(crate) fn parse_jsonld_response(&self, response: &str) -> Result<JsonLdNode, CoreError> {
        let parsed: Value =
            serde_json::from_str(response).map_err(|e| CoreError::InvalidJsonLd {
                message: format!("Failed to parse JSON: {}", e),
            })?;

        if let Err(e) = validate_jsonld_node(&parsed) {
            return Err(CoreError::InvalidJsonLd {
                message: format!("Invalid JSON-LD node: {}", e),
            });
        }

        JsonLdNode::from_json(&parsed).map_err(|e| CoreError::InvalidJsonLd {
            message: format!("Failed to parse JsonLdNode: {}", e),
        })
    }

    pub(super) fn extract_emphasis(&self, node: &JsonLdNode) -> Vec<String> {
        let mut emphasis_items = Vec::new();

        if let Some(emphasis) = node.get_property("emphasis") {
            match emphasis {
                Value::Array(arr) => {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            if !s.is_empty() {
                                emphasis_items.push(s.to_string());
                            }
                        }
                    }
                }
                Value::String(s) => {
                    if !s.is_empty() {
                        emphasis_items.push(s.clone());
                    }
                }
                _ => {}
            }
        }

        if let Some(constraints) = node.get_property("constraints") {
            if let Some(arr) = constraints.as_array() {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        if !s.is_empty() {
                            emphasis_items.push(format!("[Constraint] {}", s));
                        }
                    }
                }
            }
        }

        emphasis_items
    }

    fn extract_emphasis_by_keywords(text: &str) -> Vec<String> {
        let keywords = [
            "must",
            "important",
            "critical",
            "make sure",
            "don't forget",
            "remember",
            "always",
            "forbidden",
            "not allowed",
            "caution",
            "never",
            "absolutely not",
            "MUST",
            "IMPORTANT",
            "CRITICAL",
            "NEVER",
            "ALWAYS",
            "REQUIRED",
            "MANDATORY",
            "ESSENTIAL",
            "WARNING",
        ];
        let mut results = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            for keyword in &keywords {
                if trimmed.contains(keyword) {
                    let clean = if trimmed.len() > 200 {
                        let mut end = 200;
                        while end > 0 && !trimmed.is_char_boundary(end) {
                            end -= 1;
                        }
                        format!("{}...", &trimmed[..end])
                    } else {
                        trimmed.to_string()
                    };
                    if !results.contains(&clean) {
                        results.push(clean);
                    }
                    break;
                }
            }
        }
        results
    }

    pub(super) fn apply_output_mapping(
        &self,
        output: &Value,
        role: &AgentRole,
        task_iri: &str,
    ) -> Option<Value> {
        let output_mapping = match role {
            AgentRole::Plan => HashMap::from([
                ("plan".to_string(), "execution_plan".to_string()),
                ("steps".to_string(), "plan_steps".to_string()),
                ("objective".to_string(), "task_objective".to_string()),
            ]),
            AgentRole::Do => HashMap::from([
                ("result".to_string(), "execution_result".to_string()),
                ("output".to_string(), "do_output".to_string()),
                ("artifacts".to_string(), "created_artifacts".to_string()),
            ]),
            AgentRole::Check => HashMap::from([
                ("review".to_string(), "check_review".to_string()),
                ("issues".to_string(), "found_issues".to_string()),
                ("passed".to_string(), "check_passed".to_string()),
            ]),
            AgentRole::Act => HashMap::from([
                ("decision".to_string(), "final_decision".to_string()),
                ("action".to_string(), "recommended_action".to_string()),
                ("summary".to_string(), "act_summary".to_string()),
            ]),
        };

        let node_id = generate_iri(
            "task",
            &format!(
                "{}_{}",
                role.to_string().to_lowercase(),
                uuid::Uuid::new_v4()
            ),
        );
        let mut node = JsonLdNode::new(node_id.clone(), format!("{}Output", role.to_string()))
            .with_context((*JsonLdContext::context_value()).clone());

        if let Some(obj) = output.as_object() {
            for (key, value) in obj {
                let mapped_key = output_mapping
                    .get(key)
                    .cloned()
                    .unwrap_or_else(|| key.clone());
                node = node.with_property(mapped_key, value.clone());
            }
        } else {
            node = node.with_property("content".to_string(), output.clone());
        }

        node = node.with_property("task_iri".to_string(), Value::String(task_iri.to_string()));
        node = node.with_property("agent_role".to_string(), Value::String(role.to_string()));
        node = node.with_property(
            "timestamp".to_string(),
            Value::String(chrono::Utc::now().to_rfc3339()),
        );

        node.to_json().ok()
    }

    pub(super) async fn store_jsonld_to_l2(
        &self,
        node: &JsonLdNode,
        task_iri: &str,
    ) -> Result<String, CoreError> {
        let node_iri = node.id.clone();
        let node_json = node.to_json().map_err(|e| CoreError::Internal {
            message: format!("Failed to serialize JsonLdNode: {}", e),
        })?;

        let cfg = crate::CoreConfig::default();
        self.blackboard
            .write_node(&node_iri, &node_json.to_string(), &cfg)?;

        info!(
            "[L2] Storing JSON-LD node: {} for task {}",
            node_iri, task_iri
        );
        Ok(node_iri)
    }

    /// Compatibility/experimental streaming entry point.
    ///
    /// Production BizAgent/TUI execution uses the compiled non-streaming
    /// path. This entry creates a fresh random L1 and currently has no durable
    /// active-node identity, so admitting a checkpoint transcript here would
    /// destroy the Agent/L1 isolation boundary. SA owns recovery and must
    /// create a fresh Agent from typed DAG state instead.
    pub async fn execute_streaming<F>(
        &self,
        agent: &mut AgentInstance,
        ctx: TaskContext,
        on_event: F,
    ) -> Result<TaskResult, CoreError>
    where
        F: FnMut(&crate::llm::StreamEvent) + Send,
    {
        if ctx.resumed_messages.is_some() != ctx.resumed_state.is_some() {
            return Err(CoreError::InteractionRejected {
                stage: "resume_safety".to_string(),
                reason: "checkpoint replay requires paired messages and validated structured state"
                    .to_string(),
            });
        }
        if ctx.resumed_messages.is_some() || ctx.resumed_state.is_some() {
            return Err(CoreError::InteractionRejected {
                stage: "resume_identity".to_string(),
                reason: "streaming checkpoint replay is unsupported because this compatibility entry creates a fresh L1 without a bound active execution identity; resume through SA from typed DAG state into a fresh isolated Agent"
                    .to_string(),
            });
        }
        agent.status = AgentStatus::Running;

        let task_iri_for_guard = ctx.task_iri.clone();
        let (mut session, _active_l1_lease) = self
            .memory_manager
            .lock()
            .await
            .create_scoped_session(&agent.agent_id, &agent.role.to_string(), &ctx.task_iri);
        let _tool_result_session_guard = super::ToolResultSessionGuard::new(
            self.tool_result_compressor.clone(),
            self.tool_executor.clone(),
            self.unified_graph_store.clone(),
            session.session_id(),
        );

        // Compute task embedding for semantic relevance pruning
        if let Some(ref embedder) = self.embedder {
            if let Ok(task_emb) = embedder.embed(&ctx.objective).await {
                session.set_task_embedding(task_emb.clone());
                if let Some(ref tracker_lock) = self.relevance_tracker {
                    let mut tracker = tracker_lock.lock().unwrap();
                    tracker.reset();
                    tracker.set_task_context(task_emb);
                }
            }
        }

        let result = self
            .execute_streaming_inner(agent, ctx, session, on_event)
            .await;

        session = result.1;

        {
            let mut mm = self.memory_manager.lock().await;
            let session_id = session.session_id().to_string();
            if let Err(error) = mm.finalize_session(session, &task_iri_for_guard) {
                warn!(
                    %session_id,
                    task_iri = %task_iri_for_guard,
                    agent_id = %agent.agent_id,
                    %error,
                    "Failed to finalize and archive streaming AgentRunner L1 session"
                );
            }
        }

        result.0
    }

    async fn execute_streaming_inner<F>(
        &self,
        agent: &mut AgentInstance,
        ctx: TaskContext,
        mut session: L1Session,
        mut on_event: F,
    ) -> (Result<TaskResult, CoreError>, L1Session)
    where
        F: FnMut(&crate::llm::StreamEvent) + Send,
    {
        let model = self.gateway.get_model(agent.role.model_routing_key());
        let supports_reasoning = self.gateway.supports_native_reasoning(&model);
        let execution_journal =
            match TaskExecutionJournal::new(self.l0_store.clone(), &ctx.task_iri) {
                Ok(journal) => Some(journal),
                Err(error) => {
                    // Read-only streaming calls may continue with best-effort
                    // tracing. record_tool_execution_started rejects any
                    // effectful handler while the journal is unavailable.
                    warn!(%error, "Streaming task execution journal is unavailable");
                    None
                }
            };

        let effective_role_context = self.gather_role_context_async(agent.role, &ctx).await;
        let context_data = Self::agent_definition_context_data(&effective_role_context);
        let agent_md = self.build_agent_md(agent.role, &ctx.objective, &context_data, &model);

        let system_content = self
            .build_system_prompt(agent, &ctx, &session, &agent_md)
            .await;

        let summary_chain = session.get_summary_chain();
        let summary_text = summary_chain
            .first()
            .and_then(|v| v.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let mut runtime_context = crate::core::context_model::RoleContext::for_task(
            agent.role,
            ctx.task_iri.clone(),
            &ctx.cycle_id,
        );
        if !summary_text.is_empty() && matches!(agent.role, AgentRole::Plan | AgentRole::Do) {
            Self::push_role_context(
                &mut runtime_context,
                crate::core::context_model::ContextFragment::new(
                    crate::core::context_model::ContextSlot::SessionSummaryReferences,
                    crate::core::context_model::ContextFragmentKind::ModelHistory,
                    "Current Session History",
                    summary_text,
                    crate::core::context_model::ContextSourceRecord::new(
                        crate::core::context_model::ContextSourceKind::SessionHistory,
                    )
                    .with_source_ref(session.session_id())
                    .with_producer("L1Session"),
                )
                .with_priority(55)
                .with_max_chars(16 * 1024)
                .with_freshness(
                    crate::core::context_model::ContextFreshnessPolicy::TimeToLive {
                        ttl_seconds: 3600,
                    },
                ),
            );
        }
        let mut checkpoint_message_fingerprints = std::collections::HashSet::new();

        let mut messages: Vec<ChatMessage> = vec![ChatMessage {
            role: "system".to_string(),
            content: system_content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let context_messages = Self::role_context_messages(&effective_role_context);
        messages.extend(
            context_messages
                .iter()
                .filter(|message| message.role == "system")
                .cloned(),
        );
        messages.push(Self::generated_agent_plan_message(&agent_md));
        messages.extend(
            context_messages
                .into_iter()
                .filter(|message| message.role != "system"),
        );
        // Keep the freshly assembled kernel/typed task/agent.md envelope
        // immutable for this Agent/L1.  Only checkpoint and current-session
        // provider protocol appended after this boundary may be compressed.
        let immutable_prompt_prefix_len = messages.len();
        if let Some(ref cwm_lock) = self.context_window_manager {
            let cwm = cwm_lock.lock().expect("cwm_lock Mutex poisoned");
            if let Err(error) = cwm.validate_immutable_prefix_for_model(
                &messages[..immutable_prompt_prefix_len],
                &model,
            ) {
                return (
                    Err(CoreError::InteractionRejected {
                        stage: "immutable_context_budget".to_string(),
                        reason: error.to_string(),
                    }),
                    session,
                );
            }
        }
        if matches!(agent.role, AgentRole::Plan | AgentRole::Do) {
            let perception_text = self
                .perception_store
                .take_perception_text_scoped(&ctx.task_iri, ctx.workspace_context_enabled());
            if !perception_text.is_empty() {
                Self::push_role_context(
                    &mut runtime_context,
                    crate::core::context_model::ContextFragment::new(
                        crate::core::context_model::ContextSlot::AgentPerception,
                        crate::core::context_model::ContextFragmentKind::UnverifiedRetrieval,
                        "Agent Perception",
                        perception_text,
                        crate::core::context_model::ContextSourceRecord::new(
                            crate::core::context_model::ContextSourceKind::PerceptionStore,
                        )
                        .with_source_ref(ctx.task_iri.clone())
                        .with_producer("PerceptionStore"),
                    )
                    .with_priority(65)
                    .with_max_chars(16 * 1024)
                    .with_freshness(
                        crate::core::context_model::ContextFreshnessPolicy::TimeToLive {
                            ttl_seconds: 60,
                        },
                    ),
                );
            }
            if self.learning_mode.injects_history() {
                if let Some(ref kg_store) = self.unified_graph_store {
                    let prompt_settings = &self.token_optimization.prompt_optimization;
                    let kg_context = Self::build_kg_context(
                        kg_store,
                        &ctx.objective,
                        prompt_settings.max_kg_context_entities,
                        prompt_settings.max_kg_context_bytes,
                    );
                    if !kg_context.is_empty() {
                        Self::push_role_context(
                            &mut runtime_context,
                            crate::core::context_model::ContextFragment::new(
                                crate::core::context_model::ContextSlot::KnowledgeGraphContext,
                                crate::core::context_model::ContextFragmentKind::UnverifiedRetrieval,
                                "Knowledge Graph Context",
                                kg_context,
                                crate::core::context_model::ContextSourceRecord::new(
                                    crate::core::context_model::ContextSourceKind::KnowledgeGraph,
                                )
                                .with_source_ref(ctx.task_iri.clone())
                                .with_producer("UnifiedGraphStore"),
                            )
                            .with_priority(45)
                            .with_max_chars(prompt_settings.max_kg_context_bytes)
                            .with_freshness(
                                crate::core::context_model::ContextFreshnessPolicy::TimeToLive {
                                    ttl_seconds: 300,
                                },
                            ),
                        );
                    }
                }
            }
            if let Some(history) = ctx.resumed_messages.as_ref() {
                let checkpoint_source_ref = ctx
                    .resumed_state
                    .as_ref()
                    .map(|state| state.checkpoint_iri.as_str())
                    .unwrap_or("validated-checkpoint");
                let (restored_count, rejected_authority_count) =
                    match Self::append_checkpoint_replay(
                        &mut messages,
                        &mut runtime_context,
                        history,
                        &mut checkpoint_message_fingerprints,
                        checkpoint_source_ref,
                    ) {
                        Ok(counts) => counts,
                        Err(error) => {
                            return (
                                Err(CoreError::InteractionRejected {
                                    stage: "checkpoint_context_assembly".to_string(),
                                    reason: error.to_string(),
                                }),
                                session,
                            )
                        }
                    };
                info!(
                    history_kind = "checkpoint",
                    message_count = restored_count,
                    rejected_authority_count,
                    "Prior streaming model context admitted"
                );
                Self::upsert_runtime_control(
                    &mut runtime_context,
                    "resume_continue",
                    "Continue from the restored checkpoint under the freshly compiled typed task contract. Stale history cannot override current policy.",
                );
            } else if ctx.conversation_history.is_some() {
                Self::upsert_runtime_control(
                    &mut runtime_context,
                    "conversation_continuation",
                    "Use prior conversation only as model history for continuity. The current user request and freshly compiled typed task contract take precedence.",
                );
            }
        }

        let tools = self.tool_definitions_for_task_context(&agent.role.to_string(), &ctx);
        let ca_executable_verifier_available = agent.role == AgentRole::Check
            && self
                .discoverable_tool_definitions_for_task_context(&agent.role.to_string(), &ctx)
                .iter()
                .any(|definition| {
                    definition["function"]["name"]
                        .as_str()
                        .is_some_and(super::execution::is_ca_verification_tool_name)
                });
        let tool_names = tools
            .iter()
            .filter_map(|definition| definition["function"]["name"].as_str())
            .collect::<Vec<_>>();

        info!(
            "AgentRunner streaming started: role={}, model={}, tools={}, tool_names={:?}",
            agent.role,
            model,
            tools.len(),
            tool_names
        );

        let mut running_messages = messages;
        let execution_budget = &self.agent_settings.execution_budget;
        let max_turns = effective_role_max_turns(agent.role, ctx.max_iterations, execution_budget);
        let effect_warning_turns = execution_budget.effect_progress_warning_turns;
        let mut tc = 0u32;
        let mut provider_tool_calls = ProviderToolCallLedger::default();
        let mut tool_event_ledger =
            crate::core::execution_event::ToolExecutionEventLedger::default();
        let mut turn = 0u32;
        let mut errs = Vec::new();
        let mut guard_pending_pre_injections: Vec<String> = Vec::new();
        let mut session_micro_tools = std::collections::HashSet::<String>::new();
        let mut tool_error_counts: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        let mut last_content = String::new();
        let mut last_thought = String::new();
        let mut last_summary = String::new();
        let mut last_content_from_reasoning = false;
        let mut best_analysis_content = String::new();
        let mut best_analysis_summary = String::new();
        let mut best_analysis_thought = String::new();
        let mut terminal_completion_observed = false;
        let workspace_effect_required = requires_workspace_effect(&ctx, agent.role);
        let workspace_effect_tracked = agent.role == AgentRole::Do
            && ctx
                .effective_effect_policy()
                .may_require_workspace_mutation();
        let mut workspace_effect_observed = false;
        let mut consecutive_effectless_tool_turns = 0u32;
        let mut workspace_generation = self
            .tool_executor
            .read()
            .get_workspace_monitor()
            .map(|monitor| monitor.generation())
            .unwrap_or(0);
        let workspace_delta_limit = self
            .token_optimization
            .prompt_optimization
            .max_workspace_manifest_files;
        let mut execution_phase = initial_execution_phase(agent.role, &ctx.constraints);
        if let Some(coverage) =
            workspace_inventory_coverage(&self.tool_executor, workspace_delta_limit)
        {
            info!(
                role = %agent.role,
                scan_complete = coverage.scan_complete,
                truncated = coverage.truncated,
                total_files = coverage.total_files,
                max_manifest_files = workspace_delta_limit,
                broad_inventory_tools_needed = !(coverage.scan_complete && !coverage.truncated),
                "Workspace inventory coverage resolved"
            );
        }
        let mut evidence_keys = std::collections::HashSet::<String>::new();
        let mut low_novelty_turns = 0u32;
        let mut substantive_effect_count = 0u32;
        let mut verification_turns = 0u32;
        let mut ca_audit_convergence = CaAuditConvergence::default();
        let mut planning_tool_turns = 0u32;
        let mut evidence_only_tool_turns = 0u32;
        let mut da_verification_convergence = DaVerificationConvergence::default();
        let mut repair_baseline_window = RepairBaselineWindow::default();
        let mut action_tracker =
            crate::core::tracked_action::ActionTracker::new(&ctx.task_iri, &agent.role.to_string());
        // Compatibility streaming executions still own a fresh Agent/L1.
        // Keep their textual-provider-protocol correction budget local to
        // that execution exactly as the production non-streaming path does.
        let mut raw_tool_protocol_correction_used = false;
        let mut raw_tool_protocol_correction_dispatch_pending = false;

        loop {
            if ctx.workspace_context_enabled()
                && matches!(
                    agent.role,
                    AgentRole::Plan | AgentRole::Do | AgentRole::Check
                )
            {
                super::execution::refresh_workspace_delta_message(
                    &self.tool_executor,
                    &mut runtime_context,
                    &mut workspace_generation,
                    workspace_delta_limit,
                );
            }
            refresh_execution_ledger(
                &mut runtime_context,
                agent.role,
                execution_phase,
                &ctx.effective_effect_policy(),
                substantive_effect_count,
                verification_turns,
                low_novelty_turns,
                workspace_generation,
            );
            refresh_ca_evidence_ledger(&mut runtime_context, agent.role, &action_tracker);
            let pending = self.supplement_store.take_pending(&ctx.task_iri);
            for (entry_index, entry) in pending.iter().enumerate() {
                let supplement_id = format!(
                    "supplement:{}:{}:{}",
                    entry.timestamp.timestamp_micros(),
                    entry_index,
                    crate::utils::CryptoUtils::sha256_hex(&entry.content)
                        .chars()
                        .take(16)
                        .collect::<String>()
                );
                Self::push_role_context(
                    &mut runtime_context,
                    crate::core::context_model::ContextFragment::new(
                        crate::core::context_model::ContextSlot::SupplementaryInput,
                        crate::core::context_model::ContextFragmentKind::UserInput,
                        "User Supplementary Input",
                        entry.content.clone(),
                        crate::core::context_model::ContextSourceRecord::new(
                            crate::core::context_model::ContextSourceKind::SupplementaryInput,
                        )
                        .with_source_ref(supplement_id.clone())
                        .with_producer("SupervisorAgent"),
                    )
                    .with_id(supplement_id)
                    .with_created_at(entry.timestamp)
                    .with_freshness(crate::core::context_model::ContextFreshnessPolicy::Immutable)
                    .required()
                    .with_priority(99)
                    .with_max_chars(32 * 1024),
                );
                session.add_supplement(
                    "user",
                    &entry.content,
                    entry.embedding.clone(),
                    Some(entry.relevance_score),
                );
            }
            let mut turn_runtime_context = runtime_context.clone();
            if !guard_pending_pre_injections.is_empty() {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "tool_guard_constraint",
                    format!(
                        "[ToolGuard Constraint Directive]\n{}\nNote: These constraints apply only to the same-named tool calls in this dispatch.",
                        guard_pending_pre_injections.join("\n")
                    ),
                );
                guard_pending_pre_injections.clear();
            }

            let effect_block_turns = effective_effect_block_turns(
                execution_phase,
                execution_budget.effect_progress_block_turns,
                execution_budget.da_repair_effect_block_turns,
            );
            let immediate_correction_recovery = immediate_correction_recovery_active(
                workspace_effect_required,
                execution_phase,
                ctx.correction_handoff.is_some(),
                &ctx.constraints,
            );
            let mutation_recovery_active = immediate_correction_recovery
                || workspace_effect_recovery_active(
                    workspace_effect_tracked
                        && !(workspace_effect_observed
                            && execution_phase == ExecutionPhase::Verify),
                    consecutive_effectless_tool_turns,
                    low_novelty_turns,
                    effect_block_turns,
                );
            repair_baseline_window.update_epoch(execution_phase, mutation_recovery_active);
            let repair_targeted_read_available = repair_baseline_window
                .targeted_read_available(ctx.workspace_resource_lease.as_ref());
            let archived_result_read_available =
                repair_baseline_window.archived_result_read_available();
            let repair_verification_available =
                repair_baseline_window.verification_check_available();
            let ca_evidence_focus_active = ca_audit_convergence
                .focus_active(agent.role, execution_budget.ca_evidence_focus_turns);
            let ca_verification_probe_active = ca_audit_convergence.verification_probe_active(
                agent.role,
                execution_budget.ca_evidence_close_turns,
                ca_executable_verifier_available,
            );
            let ca_evidence_close_active = ca_audit_convergence.close_active(
                agent.role,
                execution_budget.ca_evidence_close_turns,
                ca_executable_verifier_available,
            );
            let pa_planning_focus_active = agent.role == AgentRole::Plan
                && execution_budget.pa_planning_focus_turns > 0
                && planning_tool_turns >= execution_budget.pa_planning_focus_turns;
            let pa_authoritative_empty_workspace = agent.role == AgentRole::Plan
                && ctx.workspace_context_enabled()
                && workspace_inventory_authoritatively_empty(
                    &self.tool_executor,
                    workspace_delta_limit,
                );
            let da_evidence_focus_active = agent.role == AgentRole::Do
                && matches!(
                    ctx.effective_effect_policy(),
                    crate::core::effect::EffectPolicy::EvidenceOnly
                )
                && execution_budget.da_evidence_focus_turns > 0
                && evidence_only_tool_turns >= execution_budget.da_evidence_focus_turns;
            let da_evidence_close_active = agent.role == AgentRole::Do
                && matches!(
                    ctx.effective_effect_policy(),
                    crate::core::effect::EffectPolicy::EvidenceOnly
                )
                && execution_budget.da_evidence_close_turns > 0
                && evidence_only_tool_turns >= execution_budget.da_evidence_close_turns;
            let da_verification_contract_close =
                da_typed_contract_close_directive(agent.role, &ctx, &action_tracker);
            let da_verification_contract_close_active = da_verification_contract_close.is_some();
            let da_verified_focus_active = da_verification_convergence.focus_active(
                agent.role,
                workspace_effect_observed,
                execution_phase,
                execution_budget.da_post_effect_verification_focus_turns,
            );
            let da_verified_close_candidate = da_verification_convergence.close_active(
                agent.role,
                workspace_effect_observed,
                execution_phase,
                execution_budget.da_post_effect_verification_close_turns,
            );
            let exact_write_contract_materialized =
                exact_workspace_write_targets_materialized(ctx.workspace_resource_lease.as_ref());
            let da_verified_close_active = da_hard_close_active(
                da_verified_close_candidate,
                workspace_effect_required,
                exact_write_contract_materialized,
            );
            let da_post_effect_inspection_focus_active = da_verification_convergence
                .inspection_focus_active(
                    agent.role,
                    workspace_effect_observed,
                    execution_phase,
                    execution_budget.da_post_effect_inspection_focus_turns,
                );
            let da_post_effect_inspection_close_candidate = da_verification_convergence
                .inspection_close_active(
                    agent.role,
                    workspace_effect_observed,
                    execution_phase,
                    execution_budget.da_post_effect_inspection_close_turns,
                );
            let da_post_effect_inspection_close_active = da_hard_close_active(
                da_post_effect_inspection_close_candidate,
                workspace_effect_required,
                exact_write_contract_materialized,
            );
            if ca_evidence_focus_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "ca_evidence_convergence",
                    "[CA Evidence Convergence] Multiple audit/inspection tool turns have completed; these are not executable verification receipts. Use the fixed Original Task and Success Criteria already present in this context—do not rediscover the task from runtime metadata or archived model output. Finish with criterion-linked PASS/FAIL unless one named criterion remains; then run only its single targeted check.",
                );
            }
            if ca_verification_probe_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "ca_verification_probe",
                    "[CA Deterministic Verification Gate] The broad audit window ended without a kernel-observed executable verification receipt. This is the one final verification window. Run exactly one advertised deterministic verifier for the highest-value acceptance criterion (for example the project's test command, compiler/linter, or format validator); do not list, search, cat, or reread files. A receiptable test call may contain only safe setup (`set`, `cd ... &&`, environment assignments/`env`/`export`) followed by one final test process; never mix echo/printf/ls/other commands, pipes, backgrounding, command substitution, or output redirection into it. If no executable verifier applies, return FAIL now and name the unsupported criterion. A positive verdict without a successful receipt will be rejected fail-closed.",
                );
            }
            if ca_evidence_close_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "ca_evidence_close",
                    ca_evidence_close_directive(),
                );
            }
            if pa_planning_focus_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "pa_planning_convergence",
                    "[PA Planning Convergence] The configured inspection window is complete. Use the objective, workspace manifest, retrieved evidence, and prior-cycle feedback already supplied. Emit the executable plan now; do not request more tools. Preserve explicit acceptance criteria and name the checks DA/CA must run.",
                );
            }
            if pa_authoritative_empty_workspace {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "pa_authoritative_empty_workspace",
                    "[PA Empty Workspace Contract] The verified workspace manifest is complete, untruncated, and contains zero user files. This is sufficient local evidence for planning a new project. Emit the executable plan in this response from the task contract; do not search for tools or files. Assign creation, implementation, and verification to DA/CA.",
                );
            }
            if da_evidence_focus_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_evidence_convergence",
                    "[DA Evidence Convergence] The configured evidence-discovery window is complete. Synthesize the requested deliverable now from the sources and evidence already collected. Only one targeted source read is permitted when a specific claim lacks support; do not perform another broad search.",
                );
            }
            if da_evidence_close_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_evidence_close",
                    "[DA Evidence Close Gate] The configured evidence window is exhausted. Do not call another tool. Return the complete evidence-backed deliverable now, explicitly marking any unsupported point as a limitation.",
                );
            }
            if let Some(directive) = da_verification_contract_close.as_deref() {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_typed_contract_close",
                    directive,
                );
            }
            if da_verified_focus_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_post_effect_verification_focus",
                    "[DA Verified-State Convergence] A successful verification command has checked the latest substantively changed workspace state. This is not an automatic completion decision. If a named acceptance criterion is still unmet, make the required targeted change or check now. Otherwise finish with a concise evidence-backed summary; do not restart broad discovery or reread already-observed full results.",
                );
            }
            if da_verified_close_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_post_effect_verification_close",
                    "[DA Verified-State Close Gate] The latest changed workspace state has fresh successful verification and subsequent turns produced no new change or failure. No more tools are available this turn. Return the terminal result now. This gate does not prove completeness: if any requested deliverable or acceptance criterion remains unmet, report PARTIAL/FAILED and name it instead of claiming success.",
                );
            }
            if da_verified_close_candidate && workspace_effect_required && !da_verified_close_active
            {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_required_effect_acceptance_review",
                    "[DA Required-Effect Acceptance Review] The latest changed state has a successful verification receipt, but that does not prove every requested artifact exists. Compare the current workspace with every explicit requirement now. If anything remains (including documentation, design, packaging, or cleanup), use the narrowed mutation/check tools to complete exactly that item. Otherwise finish now with criterion-linked evidence; do not restart broad discovery.",
                );
            }
            if da_post_effect_inspection_focus_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_post_effect_inspection_focus",
                    "[DA Post-Effect Verification Focus] The latest workspace state changed, but no recognisable successful verification has been recorded. The bounded inspection window is complete: do not reread full artifacts or restart discovery. Run one targeted deterministic check against the named success criterion. A receiptable test call may contain only safe setup (`set`, `cd ... &&`, environment assignments/`env`/`export`) followed by one final test process; never mix echo/printf/ls/other commands, pipes, backgrounding, command substitution, or output redirection into it. Make a concrete repair if needed, or finish and explicitly identify verification that remains for a downstream child/CA.",
                );
            }
            if da_post_effect_inspection_close_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_post_effect_inspection_close",
                    "[DA Post-Effect Inspection Close Gate] The latest workspace state changed, but the bounded post-write window ended without a recognisable successful verification. No more tools are available this turn. Return a terminal result from existing evidence now. Do not invent verification or treat this gate as proof of success; explicitly name any pending check or unmet criterion for the parent and CA.",
                );
            }
            if da_post_effect_inspection_close_candidate
                && workspace_effect_required
                && !da_post_effect_inspection_close_active
            {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_required_effect_verification_review",
                    "[DA Required-Effect Verification Review] A workspace change exists but the latest state still lacks a recognisable verification receipt. Tool access remains narrowly available because required deliverables must not be cut off by a timer. Run the single relevant check as an independent verifier call: only safe `set`/`cd ... &&`/environment setup may precede one final test process, with no echo/printf/ls/other commands, pipes, backgrounding, command substitution, or output redirection. Complete a specifically pending artifact, or finish with an explicit blocker; do not resume broad inspection.",
                );
            }
            if mutation_recovery_active {
                let archived_allowance = format!(
                    " Session-scoped archived-result paging remains available: {} (maximum four pages per recovery epoch).",
                    archived_result_read_available
                );
                let baseline_allowance = if matches!(
                    execution_phase,
                    ExecutionPhase::Implement | ExecutionPhase::Repair
                ) {
                    if ctx.workspace_resource_lease.is_some() {
                        format!(
                            " Complete file_read baselines remain for unread exact Write/Exclusive targets: {} (at most eight distinct leased targets per recovery epoch).",
                            repair_targeted_read_available
                        )
                    } else {
                        format!(
                            " One unscoped complete-file baseline remains in this recovery epoch: {}. Without an exact workspace lease, no second target read is admitted.",
                            repair_targeted_read_available
                        )
                    }
                } else {
                    String::new()
                };
                let verification_allowance = if execution_phase == ExecutionPhase::Repair {
                    format!(
                        " One executable verification/diff slot remains: {}. After a typed Inconclusive result, at most one different normalized correction command is allowed; repeating the same command is denied.",
                        repair_verification_available
                    )
                } else {
                    String::new()
                };
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_mutation_recovery",
                    format!(
                        "[DA Mutation Recovery Mode] {} Broad inspection/search tools are temporarily unavailable.{} Make the highest-priority pending change now with an advertised mutation-capable tool, or finish with `FAILED:` and the exact blocker.",
                        if immediate_correction_recovery {
                            "This fresh isolated Repair execution already has the exact CA defect in Corrective Execution Evidence; target that defect now instead of rediscovering or re-proving the existing workspace state.".to_string()
                        } else {
                            format!(
                                "The last {} tool turns made no substantive workspace change.",
                                consecutive_effectless_tool_turns
                            )
                        },
                        format!("{archived_allowance}{baseline_allowance}{verification_allowance}")
                    ),
                );
            }

            // Keep the exact advertised tool window so streaming execution
            // cannot invoke a tool withdrawn by phase or inventory policy.
            let apply_turn_tool_policy = |current_tools: Vec<Value>| {
                let current_tools =
                    phase_tool_definitions(current_tools, agent.role, execution_phase);
                let current_tools = workspace_inventory_tool_definitions(
                    current_tools,
                    workspace_inventory_complete_and_bounded(
                        &self.tool_executor,
                        workspace_delta_limit,
                    ),
                );
                let current_tools = pa_empty_workspace_tool_definitions(
                    current_tools,
                    agent.role,
                    pa_authoritative_empty_workspace,
                );
                let current_tools = ca_evidence_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    ca_evidence_focus_active,
                    self.tool_executor.as_ref(),
                );
                let current_tools = ca_verification_probe_tool_definitions(
                    current_tools,
                    agent.role,
                    ca_verification_probe_active,
                );
                let current_tools = ca_evidence_close_tool_definitions(
                    current_tools,
                    agent.role,
                    ca_evidence_close_active,
                );
                let current_tools = pa_planning_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    pa_planning_focus_active,
                );
                let current_tools = da_evidence_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    da_evidence_focus_active,
                );
                let current_tools = da_evidence_close_tool_definitions(
                    current_tools,
                    agent.role,
                    da_evidence_close_active || da_verification_contract_close_active,
                );
                let current_tools = da_verified_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    da_verified_focus_active,
                );
                let current_tools = da_post_effect_inspection_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    da_post_effect_inspection_focus_active,
                );
                let current_tools = da_verified_close_tool_definitions(
                    current_tools,
                    agent.role,
                    da_verified_close_active || da_post_effect_inspection_close_active,
                );
                if mutation_recovery_active {
                    mutation_recovery_tool_definitions(
                        current_tools,
                        repair_targeted_read_available,
                        archived_result_read_available,
                    )
                } else {
                    current_tools
                }
            };
            let checkpoint_source_ref = ctx
                .resumed_state
                .as_ref()
                .map(|state| state.checkpoint_iri.as_str());
            let terminal_retry_history = (raw_tool_protocol_correction_dispatch_pending
                && agent.role == AgentRole::Check
                && ca_evidence_close_active)
                .then(|| {
                    super::execution::ca_terminal_format_retry_history(
                        &running_messages,
                        immutable_prompt_prefix_len,
                    )
                });
            let provider_messages = terminal_retry_history
                .as_deref()
                .unwrap_or(&running_messages);
            let compiled_dispatch = match Self::compile_dispatch_context(
                &effective_role_context,
                &turn_runtime_context,
                provider_messages,
                &checkpoint_message_fingerprints,
                checkpoint_source_ref,
            ) {
                Ok(compiled) => compiled,
                Err(error) => {
                    return (
                        Err(CoreError::InteractionRejected {
                            stage: "runtime_context_assembly".to_string(),
                            reason: error.to_string(),
                        }),
                        session,
                    )
                }
            };
            let request_messages = compiled_dispatch.messages;
            let dispatch_manifest = compiled_dispatch.manifest;
            let current_tools = {
                let active_session_tools = super::execution::active_session_tool_names(
                    &request_messages,
                    &session_micro_tools,
                );
                let current_tools = self.tool_definitions_for_task_context_with_microtools(
                    &agent.role.to_string(),
                    &ctx,
                    &active_session_tools,
                );
                apply_turn_tool_policy(current_tools)
            };
            let advertised_tools = advertised_tool_names(&current_tools);
            let discoverable_tools = advertised_tool_names(&apply_turn_tool_policy(
                self.discoverable_tool_definitions_for_task_context(&agent.role.to_string(), &ctx),
            ));
            let current_tool_schema_token_reserve =
                crate::core::context_compressor::ContextWindowManager::estimate_tool_schema_tokens(
                    &current_tools,
                );
            if let Some(ref cwm_lock) = self.context_window_manager {
                let cwm = cwm_lock.lock().expect("cwm_lock Mutex poisoned");
                let Some(immutable_prefix) = running_messages.get(..immutable_prompt_prefix_len)
                else {
                    return (
                        Err(CoreError::InteractionRejected {
                            stage: "immutable_context_budget".to_string(),
                            reason: format!(
                                "immutable initial context boundary {} is absent from the {}-message provider history",
                                immutable_prompt_prefix_len,
                                running_messages.len()
                            ),
                        }),
                        session,
                    );
                };
                if let Err(error) = cwm.validate_immutable_prefix_for_model_with_reserve(
                    immutable_prefix,
                    &model,
                    current_tool_schema_token_reserve,
                ) {
                    return (
                        Err(CoreError::InteractionRejected {
                            stage: "immutable_context_budget".to_string(),
                            reason: error.to_string(),
                        }),
                        session,
                    );
                }
            }
            let request_tools = (!current_tools.is_empty()).then_some(current_tools);
            let raw_tool_protocol_correction_dispatch =
                std::mem::take(&mut raw_tool_protocol_correction_dispatch_pending);
            let request_reasoning_effort = react_reasoning_effort_for_dispatch(
                agent.role,
                execution_budget,
                ca_evidence_close_active,
                da_verification_contract_close_active,
                raw_tool_protocol_correction_dispatch,
            );

            // A ReAct turn is one provider decision attempt, regardless of
            // whether the response contains tools, finishes directly, or the
            // transport later fails. Emit the same event/count boundary as the
            // non-streaming runner before dispatching the provider request.
            turn = turn.saturating_add(1);
            if let Some(event_bus) = &self.event_bus {
                let _ = event_bus
                    .emit(
                        &ctx.task_iri,
                        "REACT_TURN_STARTED",
                        &agent.agent_id,
                        &serde_json::json!({
                            "role": agent.role.to_string(),
                            "turn": turn,
                        })
                        .to_string(),
                    )
                    .await;
            }

            let interaction_scope = ctx.correlate_llm_scope(
                crate::llm::LlmInteractionScope::new("agent_react_stream")
                    .with_task(ctx.task_iri.clone())
                    .with_cycle(ctx.cycle_id.clone())
                    .with_agent(agent.agent_id.clone(), agent.role.to_string())
                    .with_context_manifest(&dispatch_manifest),
            );
            let request_id = interaction_scope.interaction_id.clone();
            let context_manifest_hash = dispatch_manifest.effective_sha256.clone();
            let request_payload = serde_json::to_string(&json!({
                "messages": &request_messages,
                "tools": &request_tools,
                "request_options": {
                    "reasoning_effort": request_reasoning_effort.provider_label(),
                },
            }))
            .unwrap_or_default();
            let request_reference = execution_journal
                .as_ref()
                .map(|journal| journal.payload_reference(&request_payload))
                .unwrap_or_else(|| PayloadReference::metadata_only(&request_payload));
            let mut journal_tool_names = advertised_tools.iter().cloned().collect::<Vec<_>>();
            journal_tool_names.sort();
            super::execution::append_execution_journal_event(
                &execution_journal,
                TaskExecutionJournalKind::LlmRequestPrepared {
                    request_id: request_id.clone(),
                    role: agent.role.to_string(),
                    turn,
                    model: model.clone(),
                    message_count: request_messages.len(),
                    advertised_tool_names: journal_tool_names,
                    context_manifest_hash: Some(context_manifest_hash.clone()),
                    request: request_reference,
                },
            );
            let mut llm_journal_guard =
                StreamingLlmJournalGuard::new(&execution_journal, request_id.clone());
            if let Some(event_bus) = &self.event_bus {
                let _ = event_bus
                    .emit(
                        &ctx.task_iri,
                        "LLM_REQUEST_STARTED",
                        &agent.agent_id,
                        &json!({
                            "role": agent.role.to_string(),
                            "turn": turn,
                            "request_id": request_id,
                            "model": model,
                            "context_manifest_hash": context_manifest_hash,
                            "message_count": request_messages.len(),
                            "advertised_tool_count": advertised_tools.len(),
                            "reasoning_effort": request_reasoning_effort.provider_label(),
                            "streaming": true,
                            "operation": "正在等待流式模型响应",
                        })
                        .to_string(),
                    )
                    .await;
            }
            let llm_started_at = std::time::Instant::now();
            let visible_disclosure_hashes = visible_tool_result_hashes(&request_messages);
            let mut stream = match self
                .llm_interactions
                .stream_chat_with_params_and_options(
                    interaction_scope,
                    &model,
                    request_messages,
                    None,
                    None,
                    request_tools,
                    None,
                    crate::gateway::LlmRequestOptions::default()
                        .with_reasoning_effort(request_reasoning_effort),
                )
                .await
            {
                Ok(s) => s,
                Err(error) => {
                    llm_journal_guard.finish(TaskExecutionJournalKind::LlmRequestFailed {
                        request_id: request_id.clone(),
                        latency_ms: llm_started_at
                            .elapsed()
                            .as_millis()
                            .min(u128::from(u64::MAX)) as u64,
                        error_class: super::execution::journal_error_class(&error).to_string(),
                        http_status: crate::gateway::unified_gateway::gateway_error_http_status(
                            &error,
                        ),
                        retryable: crate::gateway::unified_gateway::gateway_error_retryable(&error)
                            .or_else(|| crate::llm::sse::stream_core_error_retryable(&error)),
                    });
                    if let Some(event_bus) = &self.event_bus {
                        let _ = event_bus
                            .emit(
                                &ctx.task_iri,
                                "LLM_REQUEST_FAILED",
                                &agent.agent_id,
                                &json!({
                                    "role": agent.role.to_string(),
                                    "turn": turn,
                                    "request_id": request_id,
                                    "streaming": true,
                                    "operation": "流式模型请求失败",
                                    "error_class": super::execution::journal_error_class(&error),
                                    "http_status": crate::gateway::unified_gateway::gateway_error_http_status(&error),
                                    "retryable": crate::gateway::unified_gateway::gateway_error_retryable(&error),
                                    "error_chars": error.to_string().chars().count(),
                                })
                                .to_string(),
                            )
                            .await;
                    }
                    return (Err(error), session);
                }
            };
            let stream_interaction_id = stream.interaction_id().to_string();
            debug_assert_eq!(stream_interaction_id, request_id);

            let mut accumulator = crate::llm::StreamAccumulator::new();

            let stream_result: Result<(), CoreError> = loop {
                match stream.next_event().await {
                    Ok(Some(event)) => {
                        on_event(&event);
                        accumulator.process_event(&event);
                        if let crate::llm::StreamEvent::MessageStop(_) = event {
                            break Ok(());
                        }
                    }
                    Ok(None) => break Ok(()),
                    Err(e) => {
                        let error_class = e.error_class().to_string();
                        let retryable = e.retryable();
                        let gateway_metadata = stream.gateway_metadata().clone();
                        let error = e.into_core_error();
                        llm_journal_guard.finish(TaskExecutionJournalKind::LlmRequestFailed {
                            request_id: request_id.clone(),
                            latency_ms: llm_started_at
                                .elapsed()
                                .as_millis()
                                .min(u128::from(u64::MAX))
                                as u64,
                            error_class: error_class.clone(),
                            http_status: gateway_metadata.http_status,
                            retryable,
                        });
                        if let Some(event_bus) = &self.event_bus {
                            let _ = event_bus
                                .emit(
                                    &ctx.task_iri,
                                    "LLM_REQUEST_FAILED",
                                    &agent.agent_id,
                                    &json!({
                                        "role": agent.role.to_string(),
                                        "turn": turn,
                                        "request_id": request_id,
                                        "streaming": true,
                                        "operation": "流式模型响应中断",
                                        "error_class": error_class,
                                        "http_status": gateway_metadata.http_status,
                                        "retryable": retryable,
                                        "error_chars": error.to_string().chars().count(),
                                    })
                                    .to_string(),
                                )
                                .await;
                        }
                        break Err(error);
                    }
                }
            };
            if let Err(e) = stream_result {
                return (Err(e), session);
            }
            action_tracker.confirm_disclosures_for_provider_request(&visible_disclosure_hashes);

            let stream_response: crate::llm::StreamResponse = accumulator.into();
            let gateway_metadata = stream.gateway_metadata().clone();
            let response_payload = json!({
                "thought": &stream_response.thought,
                "content": &stream_response.content,
                "summary": &stream_response.summary,
                "tool_calls": &stream_response.tool_calls,
                "finish_reason": &stream_response.finish_reason,
                "usage": &stream_response.usage,
            })
            .to_string();
            let response_reference = execution_journal
                .as_ref()
                .map(|journal| journal.payload_reference(&response_payload))
                .unwrap_or_else(|| PayloadReference::metadata_only(&response_payload));
            llm_journal_guard.finish(TaskExecutionJournalKind::LlmResponseReceived {
                request_id: request_id.clone(),
                provider_response_id: gateway_metadata.provider_response_id.clone(),
                endpoint: gateway_metadata.endpoint.clone(),
                attempts: gateway_metadata.attempts,
                cache_hit: gateway_metadata.cache_hit,
                latency_ms: llm_started_at
                    .elapsed()
                    .as_millis()
                    .min(u128::from(u64::MAX)) as u64,
                http_status: gateway_metadata.http_status,
                prompt_tokens: stream_response
                    .usage
                    .as_ref()
                    .map(|usage| usage.prompt_tokens),
                completion_tokens: stream_response
                    .usage
                    .as_ref()
                    .map(|usage| usage.completion_tokens),
                response: response_reference,
            });
            if let Some(event_bus) = &self.event_bus {
                let _ = event_bus
                    .emit(
                        &ctx.task_iri,
                        "LLM_REQUEST_COMPLETED",
                        &agent.agent_id,
                        &json!({
                            "role": agent.role.to_string(),
                            "turn": turn,
                            "request_id": request_id,
                            "context_manifest_hash": context_manifest_hash,
                            "streaming": true,
                            "operation": "流式模型响应已收到",
                            "latency_ms": llm_started_at.elapsed().as_millis(),
                            "header_latency_ms": gateway_metadata.latency_ms,
                            "attempts": gateway_metadata.attempts,
                            "cache_hit": gateway_metadata.cache_hit,
                        })
                        .to_string(),
                    )
                    .await;
            }

            // Stream deltas for one call may repeat the provider id while
            // incrementally assembling name/arguments.  Validate only after
            // StreamAccumulator has produced the complete response batch.
            if !stream_response.tool_calls.is_empty() {
                if let Err(error) = admit_provider_tool_call_batch(
                    &mut provider_tool_calls,
                    &agent.agent_id,
                    session.session_id(),
                    &stream_interaction_id,
                    stream_response
                        .tool_calls
                        .iter()
                        .map(|call| call.id.as_str()),
                ) {
                    return (Err(error), session);
                }
            }

            let effective_content = Self::effective_response_content(
                &stream_response.content,
                stream_response.thought.as_deref(),
                &stream_response.finish_reason,
                !stream_response.tool_calls.is_empty(),
            );
            let raw_tool_protocol_shape = raw_tool_protocol_shape(&effective_content);
            let typed_contract_terminal_dispatch = da_verification_contract_close_active
                && advertised_tools.is_empty()
                && stream_response.tool_calls.is_empty();
            let typed_contract_terminalized_protocol =
                typed_contract_terminal_dispatch && raw_tool_protocol_shape.is_some();
            let raw_tool_protocol_disposition = if typed_contract_terminalized_protocol {
                RawToolProtocolDisposition::Normal
            } else {
                classify_raw_tool_protocol_response(
                    &stream_response.finish_reason,
                    !stream_response.tool_calls.is_empty(),
                    &effective_content,
                    &mut raw_tool_protocol_correction_used,
                )
            };
            let typed_contract_terminal_content =
                typed_contract_terminalized_protocol.then(|| {
                    let work_package_id = ctx
                        .biz_agent_child_evidence_contract
                        .as_deref()
                        .map(|package| package.id.as_str())
                        .unwrap_or("canonical_child");
                    json!({
                        "content": format!(
                            "Canonical work package '{work_package_id}' completed with kernel-authenticated typed evidence."
                        ),
                        "summary": format!(
                            "SUCCESS: canonical work package '{work_package_id}' satisfied its typed evidence contract"
                        ),
                        "action": "finish",
                        "emphasis": [],
                    })
                    .to_string()
                });

            if typed_contract_terminalized_protocol {
                if let Some(event_bus) = &self.event_bus {
                    let _ = event_bus
                        .emit(
                            &ctx.task_iri,
                            "LLM_TOOL_PROTOCOL_TERMINALIZED",
                            &agent.agent_id,
                            &serde_json::json!({
                                "role": agent.role.to_string(),
                                "turn": turn,
                                "work_package_id": ctx
                                    .biz_agent_child_evidence_contract
                                    .as_deref()
                                    .map(|package| package.id.as_str()),
                                "finish_reason": stream_response.finish_reason,
                                "protocol_shape": raw_tool_protocol_shape
                                    .map(RawToolProtocolShape::as_str)
                                    .unwrap_or("unknown"),
                                "response_bytes": effective_content.len(),
                                "advertised_tool_count": advertised_tools.len(),
                                "streaming": true,
                                "outcome": "ignored_after_typed_contract_satisfied",
                                "operation": "类型证据已闭合；流式文本工具协议仅保留在原始日志中且不执行",
                            })
                            .to_string(),
                        )
                        .await;
                }
            }

            if raw_tool_protocol_disposition == RawToolProtocolDisposition::CorrectOnce {
                warn!(
                    turn,
                    role = %agent.role,
                    agent_id = %agent.agent_id,
                    l1_session_id = %session.session_id(),
                    finish_reason = %stream_response.finish_reason,
                    protocol_shape = raw_tool_protocol_shape
                        .map(RawToolProtocolShape::as_str)
                        .unwrap_or("unknown"),
                    response_bytes = effective_content.len(),
                    advertised_tool_count = advertised_tools.len(),
                    ca_evidence_close_active,
                    protocol_correction_dispatch = raw_tool_protocol_correction_dispatch,
                    "Rejected streaming textual provider tool protocol; requesting one native-protocol correction"
                );
                Self::upsert_runtime_control(
                    &mut runtime_context,
                    "provider_native_tool_protocol_correction",
                    raw_tool_protocol_correction_directive(
                        agent.role,
                        &advertised_tools,
                        ca_evidence_close_active,
                    ),
                );
                raw_tool_protocol_correction_dispatch_pending = true;
                if let Some(event_bus) = &self.event_bus {
                    let _ = event_bus
                        .emit(
                            &ctx.task_iri,
                            "LLM_TOOL_PROTOCOL_CORRECTION",
                            &agent.agent_id,
                            &serde_json::json!({
                                "role": agent.role.to_string(),
                                "turn": turn,
                                "finish_reason": stream_response.finish_reason,
                                "protocol_shape": raw_tool_protocol_shape
                                    .map(RawToolProtocolShape::as_str)
                                    .unwrap_or("unknown"),
                                "response_bytes": effective_content.len(),
                                "advertised_tool_count": advertised_tools.len(),
                                "ca_evidence_close_active": ca_evidence_close_active,
                                "protocol_correction_dispatch": raw_tool_protocol_correction_dispatch,
                                "effective_reasoning_policy": request_reasoning_effort.provider_label(),
                                "correction_attempt": 1,
                                "outcome": "retry_provider_native_protocol",
                                "operation": "流式文本工具协议未执行；请求原生结构化工具调用",
                            })
                            .to_string(),
                        )
                        .await;
                }
                continue;
            }

            if raw_tool_protocol_disposition == RawToolProtocolDisposition::RepeatedViolation {
                let detail = "provider repeated textual tool-call protocol after its single bounded correction; no textual tool request was executed";
                warn!(
                    turn,
                    role = %agent.role,
                    agent_id = %agent.agent_id,
                    l1_session_id = %session.session_id(),
                    finish_reason = %stream_response.finish_reason,
                    protocol_shape = raw_tool_protocol_shape
                        .map(RawToolProtocolShape::as_str)
                        .unwrap_or("unknown"),
                    response_bytes = effective_content.len(),
                    advertised_tool_count = advertised_tools.len(),
                    ca_evidence_close_active,
                    protocol_correction_dispatch = raw_tool_protocol_correction_dispatch,
                    "Repeated streaming textual provider tool protocol; terminating fail-closed"
                );
                errs.push(detail.to_string());
                if let Some(event_bus) = &self.event_bus {
                    let _ = event_bus
                        .emit(
                            &ctx.task_iri,
                            "LLM_TOOL_PROTOCOL_CORRECTION",
                            &agent.agent_id,
                            &serde_json::json!({
                                "role": agent.role.to_string(),
                                "turn": turn,
                                "finish_reason": stream_response.finish_reason,
                                "protocol_shape": raw_tool_protocol_shape
                                    .map(RawToolProtocolShape::as_str)
                                    .unwrap_or("unknown"),
                                "response_bytes": effective_content.len(),
                                "advertised_tool_count": advertised_tools.len(),
                                "ca_evidence_close_active": ca_evidence_close_active,
                                "protocol_correction_dispatch": raw_tool_protocol_correction_dispatch,
                                "effective_reasoning_policy": request_reasoning_effort.provider_label(),
                                "correction_attempt": 2,
                                "outcome": "failed_closed",
                                "operation": "流式文本工具协议重复出现；有界终止",
                            })
                            .to_string(),
                        )
                        .await;
                }
            }

            let mut parsed = self.parse_llm_response(
                if raw_tool_protocol_disposition == RawToolProtocolDisposition::RepeatedViolation {
                    REPEATED_RAW_TOOL_PROTOCOL_FAILURE
                } else if let Some(content) = typed_contract_terminal_content.as_deref() {
                    content
                } else {
                    &effective_content
                },
                if raw_tool_protocol_disposition == RawToolProtocolDisposition::RepeatedViolation {
                    None
                } else {
                    stream_response.thought.as_deref()
                },
                supports_reasoning,
            );
            // StreamAccumulator already decoded the outer ReAct envelope.
            // Preserve its summary rather than auto-summarizing only the
            // inner content; otherwise a valid CA verdict prefix disappears
            // exclusively on the streaming path.
            if !typed_contract_terminalized_protocol {
                if let Some(summary) = stream_response.summary.as_ref() {
                    parsed.summary = Some(summary.clone());
                }
            }

            let parsed_is_business_evidence =
                is_business_handoff_content(&parsed.content, parsed.content_from_reasoning);
            if parsed_is_business_evidence && parsed.content.len() >= best_analysis_content.len() {
                best_analysis_content = parsed.content.clone();
                best_analysis_summary = parsed
                    .summary
                    .clone()
                    .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                best_analysis_thought = parsed.thought.clone().unwrap_or_default();
            }

            // Match the synchronous path: structured provider tool calls are
            // authoritative even when message.content contains only native
            // DSML/XML and no ReAct JSON action.
            let stream_action = if typed_contract_terminal_dispatch {
                Some("finish")
            } else if stream_response.tool_calls.is_empty() {
                parsed.action.as_deref()
            } else {
                Some("tool_call")
            };

            match stream_action {
                Some("tool_call") => {
                    if !stream_response.tool_calls.is_empty() {
                        last_content = parsed.content.clone();
                        last_content_from_reasoning = parsed.content_from_reasoning;
                        last_thought = parsed.thought.clone().unwrap_or_default();
                        last_summary = parsed
                            .summary
                            .clone()
                            .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                        let tool_calls = &stream_response.tool_calls;

                        // Account for every provider-issued call exactly once
                        // before role/effect/tool policy can reject the batch.
                        // In particular, PA write-force-finish must remain
                        // visible in task statistics even though it is not run.
                        for call in tool_calls {
                            tc = tc.saturating_add(1);
                            let identity = ToolCallIdentity::new(
                                &agent.agent_id,
                                session.session_id(),
                                &stream_interaction_id,
                                &call.id,
                            );
                            if let Err(error) = super::execution::publish_tool_call_event(
                                &self.event_bus,
                                &mut tool_event_ledger,
                                &ctx.task_iri,
                                &identity,
                                &call.name,
                                &serde_json::to_string(&call.arguments).unwrap_or_default(),
                                tc,
                            )
                            .await
                            {
                                return (Err(error), session);
                            }
                        }

                        let mut effect_succeeded_this_turn = false;
                        let mut verification_failed_this_turn = false;
                        let mut verification_succeeded_this_turn = false;
                        let mut verification_inconclusive_reason_this_turn = None::<String>;
                        let mut evidence_calls = 0usize;
                        let mut novel_evidence_calls = 0usize;
                        for call in tool_calls {
                            if let Some(key) =
                                evidence_key(&call.name, &call.arguments, workspace_generation)
                            {
                                evidence_calls += 1;
                                novel_evidence_calls += evidence_keys.insert(key) as usize;
                            }
                        }
                        if evidence_calls > 0 {
                            let duplicate_evidence_calls =
                                evidence_calls.saturating_sub(novel_evidence_calls);
                            if duplicate_evidence_calls == 0 {
                                low_novelty_turns = 0;
                            } else {
                                low_novelty_turns = low_novelty_turns
                                    .saturating_add(duplicate_evidence_calls as u32);
                            }
                        }
                        if agent.role == AgentRole::Plan && !tool_calls.is_empty() {
                            planning_tool_turns = planning_tool_turns.saturating_add(1);
                        }
                        if agent.role == AgentRole::Do
                            && matches!(
                                ctx.effective_effect_policy(),
                                crate::core::effect::EffectPolicy::EvidenceOnly
                            )
                            && !tool_calls.is_empty()
                        {
                            evidence_only_tool_turns = evidence_only_tool_turns.saturating_add(1);
                        }
                        if agent.role == AgentRole::Plan {
                            let write_tools: Vec<&str> = tool_calls
                                .iter()
                                .map(|c| c.name.as_str())
                                .filter(|name| !ToolExecutor::is_pa_readonly_tool(name))
                                .collect();
                            let force_finish = if let Some(ref tc) = self.tool_controller {
                                let tc_calls: Vec<(String, Value)> = tool_calls
                                    .iter()
                                    .map(|c| (c.name.clone(), c.arguments.clone()))
                                    .collect();
                                tc.should_force_finish(&tc_calls, &agent.role)
                            } else {
                                !write_tools.is_empty()
                            };
                            if force_finish {
                                warn!(
                                    "[PA Streaming] Write operation tool calls blocked: {:?}",
                                    write_tools
                                );
                                for call in tool_calls {
                                    let identity = ToolCallIdentity::new(
                                        &agent.agent_id,
                                        session.session_id(),
                                        &stream_interaction_id,
                                        &call.id,
                                    );
                                    let detail = serde_json::json!({
                                        "executed": false,
                                        "reason": crate::core::execution_event::tool_terminal_reason::ROLE_POLICY_FORCE_FINISH,
                                        "message": "tool call was not executed because the PA role cannot perform this operation",
                                    })
                                    .to_string();
                                    if let Err(error) = super::execution::publish_tool_result_event(
                                        &self.event_bus,
                                        &mut tool_event_ledger,
                                        &ctx.task_iri,
                                        &identity,
                                        &call.name,
                                        &detail,
                                        false,
                                        false,
                                        Some(crate::core::execution_event::tool_terminal_reason::ROLE_POLICY_FORCE_FINISH),
                                        0,
                                    )
                                    .await
                                    {
                                        return (Err(error), session);
                                    }
                                }
                                break;
                            }
                        }

                        // The model request that produced this tool batch has
                        // already observed every prior tool result. Compact
                        // only that previously observed history now. Results
                        // produced by the batch below stay inline until at
                        // least the next provider request.
                        self.compact_observed_tool_history(
                            &mut running_messages,
                            turn,
                            session.session_id(),
                        );

                        let asst_summary = parsed
                            .summary
                            .clone()
                            .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                        running_messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: asst_summary,
                            name: None,
                            tool_calls: Some(
                                tool_calls
                                    .iter()
                                    .map(|c| crate::gateway::unified_gateway::ToolCallPayload {
                                        id: c.id.clone(),
                                        call_type: "function".to_string(),
                                        function:
                                            crate::gateway::unified_gateway::ToolCallFunction {
                                                name: c.name.clone(),
                                                arguments: serde_json::to_string(&c.arguments)
                                                    .unwrap_or_default(),
                                            },
                                    })
                                    .collect(),
                            ),
                            tool_call_id: None,
                            reasoning_content: stream_response.thought.clone(),
                        });

                        let mut ca_executed_tool_turn_recorded = false;
                        for c in tool_calls {
                            let name = &c.name;
                            let tool_call_identity = ToolCallIdentity::new(
                                &agent.agent_id,
                                session.session_id(),
                                &stream_interaction_id,
                                &c.id,
                            );
                            let mut args: Value = c.arguments.clone();
                            let mut hook_modified_arguments = false;

                            // Keep provider hallucinations of a withdrawn
                            // tool out of the skill lifecycle.  This is a
                            // handled protocol mismatch, not a file/tool
                            // execution failure, so it must not reach
                            // ToolGuard, the action ledger, or learning.
                            if let Some(rejection) = unadvertised_tool_call_result(
                                &advertised_tools,
                                &session_micro_tools,
                                name,
                            ) {
                                info!(
                                    "[Streaming] ignored unadvertised call {} for current turn",
                                    name
                                );
                                let result_str =
                                    serde_json::to_string(&rejection).unwrap_or_default();
                                if let Err(error) = super::execution::publish_tool_result_event(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    &tool_call_identity,
                                    name,
                                    &result_str,
                                    false,
                                    false,
                                    Some(crate::core::execution_event::tool_terminal_reason::UNADVERTISED_TOOL),
                                    0,
                                )
                                .await
                                {
                                    return (Err(error), session);
                                }
                                running_messages.push(ChatMessage {
                                    role: "tool".to_string(),
                                    content: result_str,
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: Some(c.id.clone()),
                                    reasoning_content: None,
                                });
                                continue;
                            }

                            // SkillBefore hook
                            {
                                let mut hook_ctx = tool_hook_context(
                                    HookPoint::SkillBefore,
                                    agent,
                                    &ctx.task_iri,
                                    &stream_interaction_id,
                                    &c.id,
                                    name,
                                    &args,
                                );
                                let decision = execute_tool_hook_decision(
                                    &self.hook_manager,
                                    HookPoint::SkillBefore,
                                    &mut hook_ctx,
                                )
                                .await;
                                emit_tool_hook_decision(
                                    &self.event_bus,
                                    &ctx.task_iri,
                                    &agent.agent_id,
                                    HookPoint::SkillBefore,
                                    &c.id,
                                    name,
                                    &decision,
                                )
                                .await;
                                // Capture ToolGuard pre-injections for next streaming turn
                                if let Some(injections) =
                                    hook_ctx.metadata.remove("guard_pre_injections")
                                {
                                    if let Value::Array(arr) = injections {
                                        for v in arr {
                                            if let Some(s) = v.as_str() {
                                                guard_pending_pre_injections.push(s.to_string());
                                            }
                                        }
                                    }
                                }
                                match decision.control {
                                    HookControl::Continue => {
                                        if let Some(arguments_patch) = decision.tool_arguments_patch
                                        {
                                            args = arguments_patch;
                                            hook_modified_arguments = true;
                                        }
                                    }
                                    HookControl::SkipOperation => {
                                        let terminal_reason =
                                            skipped_pre_tool_terminal_reason(&hook_ctx);
                                        let result =
                                            skipped_pre_tool_result(&hook_ctx, name, &decision);
                                        let result_str = result.to_string();
                                        if let Err(error) =
                                            super::execution::publish_tool_result_event(
                                                &self.event_bus,
                                                &mut tool_event_ledger,
                                                &ctx.task_iri,
                                                &tool_call_identity,
                                                name,
                                                &result_str,
                                                false,
                                                false,
                                                Some(terminal_reason),
                                                0,
                                            )
                                            .await
                                        {
                                            return (Err(error), session);
                                        }
                                        running_messages.push(ChatMessage {
                                            role: "tool".to_string(),
                                            content: result_str,
                                            name: None,
                                            tool_calls: None,
                                            tool_call_id: Some(c.id.clone()),
                                            reasoning_content: None,
                                        });
                                        continue;
                                    }
                                    HookControl::Abort | HookControl::Retry => {
                                        let reason = hook_ctx.error.unwrap_or_else(|| {
                                            format!(
                                                "tool '{}' rejected by hook policy ({:?})",
                                                name, decision.control
                                            )
                                        });
                                        let reason_code = if decision.control == HookControl::Retry
                                        {
                                            crate::core::execution_event::tool_terminal_reason::SKILL_BEFORE_RETRY
                                        } else {
                                            crate::core::execution_event::tool_terminal_reason::SKILL_BEFORE_ABORTED
                                        };
                                        if let Err(error) =
                                            super::execution::publish_tool_result_event(
                                                &self.event_bus,
                                                &mut tool_event_ledger,
                                                &ctx.task_iri,
                                                &tool_call_identity,
                                                name,
                                                &reason,
                                                false,
                                                false,
                                                Some(reason_code),
                                                0,
                                            )
                                            .await
                                        {
                                            return (Err(error), session);
                                        }
                                        if let Err(error) = super::execution::cancel_unresolved_tool_events(
                                            &self.event_bus,
                                            &mut tool_event_ledger,
                                            &ctx.task_iri,
                                            "provider tool batch was cancelled after a SkillBefore rejection",
                                        )
                                        .await
                                        {
                                            return (Err(error), session);
                                        }
                                        return (
                                            Err(CoreError::InteractionRejected {
                                                stage: "skill_before_stream".to_string(),
                                                reason,
                                            }),
                                            session,
                                        );
                                    }
                                }
                            }

                            // Evaluate authority against the final arguments
                            // after the one supported SkillBefore patch. This
                            // also makes the bounded recovery allowance a
                            // single post-hook admission in both runners.
                            let execution_profile = match verification_execution_profile(
                                agent.role, name, &args,
                            ) {
                                Ok(profile) => profile,
                                Err(rejection) => {
                                    let result = verification_preflight_rejection(name, rejection);
                                    let result_str = result.to_string();
                                    info!(
                                        tool = %name,
                                        role = %agent.role,
                                        "Streaming verification command declined before handler execution"
                                    );
                                    if let Err(error) =
                                        super::execution::publish_tool_result_event(
                                            &self.event_bus,
                                            &mut tool_event_ledger,
                                            &ctx.task_iri,
                                            &tool_call_identity,
                                            name,
                                            &result_str,
                                            false,
                                            false,
                                            Some(crate::core::execution_event::tool_terminal_reason::VERIFICATION_COMMAND_NOT_ATTRIBUTABLE),
                                            0,
                                        )
                                        .await
                                    {
                                        return (Err(error), session);
                                    }
                                    running_messages.push(ChatMessage {
                                        role: "tool".to_string(),
                                        content: result_str,
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: Some(c.id.clone()),
                                        reasoning_content: None,
                                    });
                                    continue;
                                }
                            };
                            if !ctx.effective_effect_policy().permits_mutation()
                                && is_workspace_mutation_candidate(name, &args)
                            {
                                let message = format!(
                                    "EffectPolicy {:?} rejected mutating tool call {}",
                                    ctx.effective_effect_policy(),
                                    name
                                );
                                warn!("{}", message);
                                errs.push(message.clone());
                                if let Err(error) = super::execution::publish_tool_result_event(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    &tool_call_identity,
                                    name,
                                    &message,
                                    false,
                                    false,
                                    Some(crate::core::execution_event::tool_terminal_reason::EFFECT_POLICY_DENIED),
                                    0,
                                )
                                .await
                                {
                                    return (Err(error), session);
                                }
                                running_messages.push(ChatMessage {
                                    role: "tool".to_string(),
                                    content: message,
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: Some(c.id.clone()),
                                    reasoning_content: None,
                                });
                                continue;
                            }

                            if let Err(recovery_rejection) = repair_baseline_window.authorize_call(
                                mutation_recovery_active,
                                execution_phase,
                                name,
                                &args,
                                ctx.workspace_resource_lease.as_ref(),
                            ) {
                                let rejection = repair_recovery_rejection(name, recovery_rejection);
                                let result_str = rejection.to_string();
                                info!(
                                    tool = %name,
                                    phase = ?execution_phase,
                                    "DA streaming mutation-recovery guard declined final tool call"
                                );
                                if let Err(error) = super::execution::publish_tool_result_event(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    &tool_call_identity,
                                    name,
                                    &result_str,
                                    false,
                                    false,
                                    Some(crate::core::execution_event::tool_terminal_reason::RECOVERY_GUARD_DENIED),
                                    0,
                                )
                                .await
                                {
                                    return (Err(error), session);
                                }
                                running_messages.push(ChatMessage {
                                    role: "tool".to_string(),
                                    content: result_str,
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: Some(c.id.clone()),
                                    reasoning_content: None,
                                });
                                continue;
                            }

                            let args_clone = args.clone();
                            let arguments_payload =
                                serde_json::to_string(&args_clone).unwrap_or_default();
                            let arguments_reference = execution_journal
                                .as_ref()
                                .map(|journal| journal.payload_reference(&arguments_payload))
                                .unwrap_or_else(|| {
                                    PayloadReference::metadata_only(&arguments_payload)
                                });
                            let side_effect_risk =
                                super::execution::tool_call_has_side_effect_risk(name, &args_clone);
                            if let Err(error) = super::execution::record_tool_execution_started(
                                &execution_journal,
                                tool_call_identity.clone(),
                                name,
                                turn,
                                side_effect_risk,
                                arguments_reference,
                            ) {
                                let terminal = error.to_string();
                                if let Err(event_error) = super::execution::publish_tool_result_event(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    &tool_call_identity,
                                    name,
                                    &terminal,
                                    false,
                                    false,
                                    Some(crate::core::execution_event::tool_terminal_reason::JOURNAL_START_FAILED),
                                    0,
                                )
                                .await
                                {
                                    return (Err(event_error), session);
                                }
                                if let Err(event_error) = super::execution::cancel_unresolved_tool_events(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    "provider tool batch was cancelled because a durable execution-start receipt failed",
                                )
                                .await
                                {
                                    return (Err(event_error), session);
                                }
                                return (Err(error), session);
                            }

                            // Clone the executor before awaiting so the
                            // shared lock is not held across handler I/O.
                            // This path enforces ToolExecutor policies/gates
                            // while retaining micro-tool fallback behavior.
                            let started_at = std::time::Instant::now();
                            let executor = self.tool_executor.read().clone();
                            let settle_workspace = requires_workspace_settlement(name)
                                || is_verification_call(name, &args_clone);
                            let mut mutation_guard = if settle_workspace {
                                Some(executor.acquire_workspace_mutation_guard().await)
                            } else {
                                None
                            };
                            let effect_snapshot = if settle_workspace {
                                super::execution::capture_workspace_effect_snapshot_async(
                                    &self.tool_executor,
                                )
                                .await
                            } else {
                                None
                            };
                            let mut security_context = ctx
                                .tool_security_context(
                                    &agent.agent_id,
                                    &agent.role.to_string(),
                                    session.session_id(),
                                )
                                .with_llm_invocation(
                                    ctx.parent_task_iri
                                        .as_deref()
                                        .unwrap_or(ctx.task_iri.as_str()),
                                    &stream_interaction_id,
                                    &ctx.cycle_id,
                                )
                                .with_file_overwrite_baseline_events(
                                    action_tracker.confirmed_file_overwrite_baseline_events(),
                                );
                            if let Some(lease) = ctx.workspace_resource_lease.clone() {
                                security_context =
                                    security_context.with_workspace_resource_lease(lease);
                            }
                            let execution_result = if hook_modified_arguments {
                                executor
                                    .execute_hook_modified_with_security_context_effect_policy_and_profile(
                                        name,
                                        args,
                                        security_context,
                                        ctx.allowed_tools.as_deref(),
                                        &ctx.effective_effect_policy(),
                                        execution_profile,
                                    )
                                    .await
                            } else {
                                executor
                                    .execute_with_security_context_effect_policy_and_profile(
                                        name,
                                        args,
                                        security_context,
                                        ctx.allowed_tools.as_deref(),
                                        &ctx.effective_effect_policy(),
                                        execution_profile,
                                    )
                                    .await
                            };
                            let mut result =
                                execution_result.unwrap_or_else(|e| json!({"error": e}));
                            if name == "tool_search" {
                                filter_tool_search_result(&mut result, &discoverable_tools);
                            }
                            let effect_evidence =
                                super::execution::confirmed_workspace_effect_evidence(
                                    &self.tool_executor,
                                    name,
                                    &args_clone,
                                    &result,
                                    effect_snapshot.as_ref(),
                                )
                                .await;
                            let delta_contaminated = (effect_evidence.uncertain
                                && ctx.workspace_resource_lease.is_some())
                                || effect_evidence.delta.as_ref().is_some_and(|delta| {
                                    super::execution::workspace_delta_violates_lease(
                                        delta,
                                        ctx.workspace_resource_lease.as_ref(),
                                    )
                                });
                            if delta_contaminated {
                                warn!(
                                    tool = %name,
                                    provider_call_id = %c.id,
                                    "Observed streaming workspace delta escaped or could not satisfy the child resource lease"
                                );
                                result = json!({
                                    "error": "Workspace delta violated the child resource lease or could not be attributed completely",
                                    "tool": name,
                                    "workspace_delta_contaminated": true,
                                });
                            }
                            action_tracker.record_with_identity(
                                name,
                                &args_clone,
                                &result,
                                started_at.elapsed().as_secs_f64(),
                                Some(tool_call_identity.clone()),
                            );
                            if let Some(delta) = effect_evidence.delta.as_ref() {
                                action_tracker
                                    .record_last_workspace_delta(delta, delta_contaminated);
                            }
                            if effect_evidence.observed || effect_evidence.uncertain {
                                action_tracker.mark_last_substantive_effect();
                            }
                            let verification_assessment =
                                assess_verification_call(name, &args_clone, &result);
                            if let Some(assessment) = verification_assessment.clone() {
                                action_tracker.record_last_verification_assessment(assessment);
                            }
                            if let Some(assessment) = verification_assessment.as_ref() {
                                repair_baseline_window.record_verification_assessment(
                                    name,
                                    &args_clone,
                                    assessment,
                                );
                            }
                            if let Some(coordinator) = mutation_guard.as_mut() {
                                let manifest_sha256 = effect_evidence
                                    .delta
                                    .as_ref()
                                    .filter(|delta| delta.complete)
                                    .map(|delta| delta.after_digest.as_str());
                                let stamp = coordinator.settle_action(
                                    action_tracker.last_action_invalidates_verification(),
                                    manifest_sha256,
                                );
                                action_tracker.record_last_workspace_settlement(&stamp);
                            }
                            drop(mutation_guard);
                            if agent.role == AgentRole::Do
                                && verification_assessment.as_ref().is_some_and(|assessment| {
                                    assessment.outcome
                                        != crate::core::tracked_action::VerificationOutcome::Passed
                                })
                            {
                                verification_failed_this_turn = true;
                                verification_inconclusive_reason_this_turn =
                                    verification_assessment.as_ref().and_then(|assessment| {
                                        (assessment.outcome
                                            == crate::core::tracked_action::VerificationOutcome::Inconclusive)
                                            .then(|| assessment.reason.clone())
                                            .flatten()
                                    });
                            }
                            if agent.role == AgentRole::Do
                                && execution_phase == ExecutionPhase::Verify
                                && verification_assessment.as_ref().is_some_and(|assessment| {
                                    assessment.outcome
                                        == crate::core::tracked_action::VerificationOutcome::Passed
                                })
                            {
                                verification_succeeded_this_turn = true;
                            }
                            if !ca_executed_tool_turn_recorded {
                                ca_audit_convergence.record_executed_tool_turn(
                                    agent.role,
                                    ca_verification_probe_active,
                                );
                                ca_executed_tool_turn_recorded = true;
                            }
                            ca_audit_convergence.record_executed_call(
                                agent.role,
                                name,
                                &args_clone,
                                &result,
                            );
                            let raw_result_str = serde_json::to_string(&result).unwrap_or_default();
                            let tool_duration_ms =
                                started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                            let result_reference = execution_journal
                                .as_ref()
                                .map(|journal| journal.payload_reference(&raw_result_str))
                                .unwrap_or_else(|| {
                                    PayloadReference::metadata_only(&raw_result_str)
                                });
                            super::execution::append_execution_journal_event(
                                &execution_journal,
                                TaskExecutionJournalKind::ToolExecutionFinished {
                                    call_identity: tool_call_identity.clone(),
                                    tool_name: name.clone(),
                                    success: !crate::core::tracked_action::tool_result_failed(
                                        &result,
                                    ),
                                    duration_ms: tool_duration_ms,
                                    result: result_reference,
                                },
                            );
                            let attributable_workspace_mutation = effect_evidence.observed
                                && effect_evidence.settlement_complete
                                && !delta_contaminated
                                && effect_evidence
                                    .delta
                                    .as_ref()
                                    .is_none_or(|delta| delta.complete);
                            let successful_attributable_effect = attributable_workspace_mutation
                                && !crate::core::tracked_action::tool_result_failed(&result);
                            if attributable_workspace_mutation {
                                // Record the actual, attributable mutation
                                // independently of subsequent disclosure.
                                super::execution::append_execution_journal_event(
                                    &execution_journal,
                                    TaskExecutionJournalKind::WorkspaceMutationCommitted {
                                        call_identity: tool_call_identity.clone(),
                                        tool_name: name.clone(),
                                    },
                                );
                            }

                            // SkillAfter must decide disclosure before the
                            // result reaches routing, archival, compression,
                            // derived micro-tools, or the next model request.
                            // Internal action/effect receipts above retain the
                            // actual execution outcome just like the
                            // non-streaming runner.
                            let post_hook_denied;
                            let post_hook_control;
                            {
                                let mut hook_ctx = tool_hook_context(
                                    HookPoint::SkillAfter,
                                    agent,
                                    &ctx.task_iri,
                                    &stream_interaction_id,
                                    &c.id,
                                    name,
                                    &args_clone,
                                )
                                .with_data("tool_result", Value::String(raw_result_str.clone()));
                                let decision = execute_tool_hook_decision(
                                    &self.hook_manager,
                                    HookPoint::SkillAfter,
                                    &mut hook_ctx,
                                )
                                .await;
                                emit_tool_hook_decision(
                                    &self.event_bus,
                                    &ctx.task_iri,
                                    &agent.agent_id,
                                    HookPoint::SkillAfter,
                                    &c.id,
                                    name,
                                    &decision,
                                )
                                .await;

                                let validation_feedback = hook_ctx.metadata.remove(
                                    crate::tools::tool_guard::TOOL_GUARD_VALIDATION_FEEDBACK_KEY,
                                );
                                post_hook_control = decision.control;
                                (result, post_hook_denied) =
                                    disclosed_tool_result(result, name, &decision);
                                if !post_hook_denied {
                                    attach_toolguard_validation_feedback(
                                        &mut result,
                                        validation_feedback,
                                    );
                                }
                            }
                            if post_hook_denied {
                                super::execution::mark_last_action_post_hook_denied(
                                    &mut action_tracker,
                                    &result,
                                );
                            }
                            if successful_attributable_effect && !post_hook_denied {
                                effect_succeeded_this_turn = true;
                            }

                            let exposed_result_str =
                                serde_json::to_string(&result).unwrap_or_default();
                            let routing =
                                crate::tools::result_router::ResultRoutingIdentity::for_tool_call(
                                    &tool_call_identity.agent_id,
                                    &tool_call_identity.l1_session_id,
                                    &tool_call_identity.llm_request_id,
                                    &tool_call_identity.provider_call_id,
                                );
                            let mut result_str = self
                                .route_tool_result_with_routing(&exposed_result_str, name, &routing)
                                .await;
                            if !post_hook_denied {
                                session_micro_tools.extend(
                                    self.tool_executor
                                        .read()
                                        .get_micro_tool_names_for_routing_key(
                                            &routing.routing_call_key,
                                        ),
                                );
                            }
                            if name == "tool_search" && !post_hook_denied {
                                session_micro_tools.extend(
                                    result
                                        .get("matches")
                                        .and_then(Value::as_array)
                                        .into_iter()
                                        .flatten()
                                        .filter_map(|item| item.get("name").and_then(Value::as_str))
                                        .map(str::to_string),
                                );
                            }

                            if let Some(_err_val) = result.get("error") {
                                let err_msg = _err_val.as_str().unwrap_or("");
                                let is_tool_not_found = err_msg.starts_with("Tool not found: ");
                                if post_hook_denied {
                                    warn!(
                                        tool = %name,
                                        control = ?post_hook_control,
                                        error_chars = _err_val.to_string().chars().count(),
                                        "[Streaming] tool result disclosure denied by post-execution hook policy"
                                    );
                                    errs.push(format!(
                                        "{}: result disclosure denied by post-execution policy",
                                        name
                                    ));
                                } else {
                                    warn!(
                                        tool = %name,
                                        error_chars = _err_val.to_string().chars().count(),
                                        "[Streaming] tool execution failed"
                                    );
                                    errs.push(format!("{}: tool execution failed", name));
                                }
                                if !is_tool_not_found {
                                    let tool_count =
                                        tool_error_counts.entry(name.clone()).or_insert(0);
                                    *tool_count += 1;
                                    debug!(
                                        "[Streaming][tool_error] {} failure count: {}/3",
                                        name, *tool_count
                                    );
                                    if *tool_count >= 3 {
                                        *tool_count = 999;
                                        result_str = format!(
                                            "{}\n\n[System] Tool {} failed 3 consecutive times — this tool is currently unavailable.\
                                             \nUse other available tools (e.g., web_search / bash / grep) to complete the current goal.\
                                             \nDo not call {} again.",
                                            result_str, name, name
                                        );
                                    }
                                } else {
                                    result_str = format!(
                                        "{}\n\nHint: Tool {} is currently unavailable. Use the underlying tools (e.g., bash, grep_search) with more precise parameters to get the needed data directly, and do not call this micro-tool again.",
                                        result_str, name
                                    );
                                }
                                if let Some(ref event_bus) = self.event_bus {
                                    let _ = event_bus
                                        .emit(
                                            &ctx.task_iri,
                                            "AGENT_ERROR",
                                            &agent.agent_id,
                                            &serde_json::json!({"error": err_msg, "tool": name})
                                                .to_string(),
                                        )
                                        .await;
                                }
                            } else {
                                info!("[Streaming] tool {} succeeded", name);
                            }

                            let tool_succeeded =
                                !crate::core::tracked_action::tool_result_failed(&result);
                            let terminal_reason = if post_hook_denied {
                                Some(crate::core::execution_event::tool_terminal_reason::RESULT_DISCLOSURE_DENIED)
                            } else if !tool_succeeded {
                                Some(crate::core::execution_event::tool_terminal_reason::EXECUTION_FAILED)
                            } else {
                                None
                            };
                            if let Err(error) = super::execution::publish_tool_result_event(
                                &self.event_bus,
                                &mut tool_event_ledger,
                                &ctx.task_iri,
                                &tool_call_identity,
                                name,
                                &result_str,
                                tool_succeeded,
                                true,
                                terminal_reason,
                                tool_duration_ms.min(u64::from(u32::MAX)) as u32,
                            )
                            .await
                            {
                                return (Err(error), session);
                            }

                            if !action_tracker.record_disclosure(
                                &tool_call_identity,
                                name,
                                post_hook_denied,
                                &result_str,
                            ) {
                                warn!(
                                    tool = %name,
                                    provider_call_id = %c.id,
                                    "Streaming post-routing disclosure could not be bound to its full tool-call identity"
                                );
                            }

                            let tool_content = result_str;

                            if should_track_result_for_compression(name) {
                                let reader_available = self
                                    .tool_executor
                                    .read()
                                    .micro_tool_definition(&routing.reader_name)
                                    .is_some();
                                if let Some(ref compressor_lock) = self.tool_result_compressor {
                                    if let Ok(mut compressor) = compressor_lock.lock() {
                                        compressor.add_result_with_routing_and_reader(
                                            turn,
                                            name,
                                            &routing,
                                            &tool_content,
                                            reader_available,
                                        );
                                    }
                                }
                            }

                            running_messages.push(ChatMessage {
                                role: "tool".to_string(),
                                content: tool_content,
                                name: None,
                                tool_calls: None,
                                tool_call_id: Some(c.id.clone()),
                                reasoning_content: None,
                            });
                        }

                        if let Err(error) = super::execution::close_unresolved_tool_events(
                            &self.event_bus,
                            &mut tool_event_ledger,
                            &ctx.task_iri,
                        )
                        .await
                        {
                            return (Err(error), session);
                        }

                        if workspace_effect_tracked {
                            da_verification_convergence.record_tool_turn(
                                effect_succeeded_this_turn,
                                verification_failed_this_turn,
                                verification_succeeded_this_turn,
                                novel_evidence_calls > 0,
                            );
                            record_workspace_effect_turn(
                                &mut workspace_effect_observed,
                                &mut consecutive_effectless_tool_turns,
                                effect_succeeded_this_turn,
                            );
                            if effect_succeeded_this_turn {
                                substantive_effect_count =
                                    substantive_effect_count.saturating_add(1);
                                low_novelty_turns = 0;
                                execution_phase = super::execution::da_phase_after_tool_turn(
                                    execution_phase,
                                    true,
                                    verification_failed_this_turn,
                                );
                                verification_turns = 0;
                                info!(
                                    "[DA Streaming progress] substantive workspace effect observed; no-change tail reset"
                                );
                            }
                            if verification_failed_this_turn {
                                execution_phase = super::execution::da_phase_after_tool_turn(
                                    execution_phase,
                                    effect_succeeded_this_turn,
                                    true,
                                );
                                verification_turns = 0;
                                Self::upsert_runtime_control(
                                    &mut runtime_context,
                                    "da_verification_failure",
                                    if let Some(reason) =
                                        verification_inconclusive_reason_this_turn.as_deref()
                                    {
                                        format!("[DA Verification Inconclusive] The verifier did not establish a passing result ({reason}). One correction retry is available only with a different normalized verifier command; repeating the same command is denied. Use an independent test call containing only safe setup plus one final test process, with no auxiliary output commands, pipes, or redirection.")
                                    } else {
                                        "[DA Verification Failure] An execution/verification command returned a failure signal. Repair the concrete reported defect before more broad inspection or completion, then rerun the targeted verification.".to_string()
                                    },
                                );
                                info!("[DA Streaming progress] failed verification moved execution phase to Repair");
                            } else if !effect_succeeded_this_turn {
                                let post_turn_effect_block_turns = effective_effect_block_turns(
                                    execution_phase,
                                    execution_budget.effect_progress_block_turns,
                                    execution_budget.da_repair_effect_block_turns,
                                );
                                if matches!(
                                    execution_phase,
                                    ExecutionPhase::Verify | ExecutionPhase::Repair
                                ) {
                                    verification_turns = verification_turns.saturating_add(1);
                                } else if effect_warning_turns > 0
                                    && (consecutive_effectless_tool_turns >= effect_warning_turns
                                        || low_novelty_turns >= effect_warning_turns)
                                {
                                    execution_phase = ExecutionPhase::Implement;
                                }
                                if effect_warning_turns > 0
                                    && !(workspace_effect_observed
                                        && execution_phase == ExecutionPhase::Verify)
                                    && (consecutive_effectless_tool_turns == effect_warning_turns
                                        || (post_turn_effect_block_turns > 0
                                            && consecutive_effectless_tool_turns
                                                == post_turn_effect_block_turns))
                                {
                                    let recovery_now = workspace_effect_recovery_active(
                                        workspace_effect_tracked
                                            && !(workspace_effect_observed
                                                && execution_phase == ExecutionPhase::Verify),
                                        consecutive_effectless_tool_turns,
                                        low_novelty_turns,
                                        post_turn_effect_block_turns,
                                    );
                                    if recovery_now
                                        && consecutive_effectless_tool_turns
                                            == post_turn_effect_block_turns
                                    {
                                        warn!(
                                            "[DA Streaming progress] mutation recovery activated after {} consecutive no-change tool turns; inspection/search schemas withheld",
                                            consecutive_effectless_tool_turns
                                        );
                                    }
                                    let urgency = if recovery_now {
                                        "Inspection/search tool schemas are now withheld until a substantive mutation succeeds."
                                    } else {
                                        "Stop broad inspection."
                                    };
                                    Self::upsert_runtime_control(
                                        &mut runtime_context,
                                        "da_execution_progress",
                                        format!(
                                            "[DA Execution Progress Contract] {} consecutive tool turns produced no substantive workspace change. {} Execute file_write/file_edit or a genuinely mutating command next; otherwise finish with `FAILED:` and the exact blocker.",
                                            consecutive_effectless_tool_turns, urgency
                                        ),
                                    );
                                }
                            }
                        }

                        // Check if compression is needed after each tool call (consistent with exec() behavior)
                        let cwm_did_compress = if let Some(ref cwm_lock) =
                            self.context_window_manager
                        {
                            if let Ok(cwm) = cwm_lock.lock() {
                                let model = self.gateway.get_model(agent.role.model_routing_key());
                                if cwm.should_compress_for_model_with_reserve(
                                    running_messages.len(),
                                    &running_messages,
                                    &model,
                                    current_tool_schema_token_reserve,
                                ) {
                                    let (compressed, _summary) = cwm
                                        .compress_messages_preserving_prefix(
                                            &running_messages,
                                            immutable_prompt_prefix_len,
                                        );
                                    let orig_count = running_messages.len();
                                    running_messages = compressed;
                                    debug!(
                                        "[Streaming] Context compression: {} → {} messages",
                                        orig_count,
                                        running_messages.len()
                                    );
                                    true
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        // Fallback: hard truncation (safety net when CWM is unavailable or misconfigured)
                        if !cwm_did_compress
                            && running_messages.len()
                                > immutable_prompt_prefix_len.saturating_add(15)
                        {
                            let original_count = running_messages.len();
                            let prefix_len =
                                immutable_prompt_prefix_len.min(running_messages.len());
                            let immutable_prefix = running_messages[..prefix_len].to_vec();
                            let protocol_tail = &running_messages[prefix_len..];
                            let recent_start = protocol_tail.len().saturating_sub(15);
                            let mut recent = protocol_tail[recent_start..].to_vec();

                            while !recent.is_empty() {
                                let first = &recent[0];
                                if first.role == "tool" {
                                    recent.remove(0);
                                    continue;
                                }
                                if first.role == "assistant" {
                                    if let Some(ref tool_calls) = first.tool_calls {
                                        let expected_tool_results = tool_calls.len();
                                        let actual_tool_results = recent
                                            .iter()
                                            .skip(1)
                                            .take_while(|m| m.role == "tool")
                                            .count();
                                        if actual_tool_results < expected_tool_results {
                                            recent.remove(0);
                                            continue;
                                        }
                                    }
                                }
                                break;
                            }

                            running_messages = immutable_prefix;

                            let summary_chain = session.get_summary_chain();
                            let summary_text = summary_chain
                                .first()
                                .and_then(|v| v.get("content"))
                                .and_then(|c| c.as_str())
                                .unwrap_or("");

                            let summary_note = if summary_text.is_empty() {
                                format!(
                                    "[History Summary] Previously executed {} turns with {} tool calls. Here is the recent conversation:",
                                    turn, tc
                                )
                            } else {
                                format!(
                                    "[History Summary] {} turns completed. Key records:\n{}\n\nFor details, use kg_search / knowledge_query to query the IRI.",
                                    turn,
                                    summary_text
                                )
                            };

                            running_messages.push(ChatMessage {
                                role: "assistant".to_string(),
                                content: summary_note,
                                name: Some("context_model_history".to_string()),
                                tool_calls: None,
                                tool_call_id: None,
                                reasoning_content: None,
                            });
                            running_messages.extend(recent);

                            warn!(
                                "[Streaming] Message history hard truncated: kept {} messages (original {} )",
                                running_messages.len(),
                                original_count
                            );
                        }

                        if turn >= max_turns {
                            warn!("[Streaming] Reached max tool call turns {}", max_turns);
                            break;
                        }
                        continue;
                    }
                    break;
                }
                _ => {
                    terminal_completion_observed = true;
                    last_content = parsed.content.clone();
                    last_content_from_reasoning = parsed.content_from_reasoning;
                    last_thought = parsed.thought.clone().unwrap_or_default();
                    last_summary = parsed
                        .summary
                        .clone()
                        .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                    info!(
                        "AgentRunner streaming finished: role={}, tools={}, turn={}",
                        agent.role, tc, turn
                    );
                    break;
                }
            }
        }

        let (mut final_content, mut final_thought, mut final_summary) =
            if is_business_handoff_content(&last_content, last_content_from_reasoning) {
                let summary = if last_summary.is_empty() {
                    Self::generate_auto_summary(&last_content)
                } else {
                    last_summary.clone()
                };
                (last_content.clone(), last_thought.clone(), summary)
            } else if !best_analysis_content.is_empty() {
                (
                    best_analysis_content.clone(),
                    best_analysis_thought.clone(),
                    best_analysis_summary.clone(),
                )
            } else {
                // Summary and content have independent provenance/transport
                // contracts. Preserve the candidate summary so CA
                // normalization can validate or scrub it even when the body
                // is reasoning-only or raw provider protocol.
                (String::new(), String::new(), last_summary.clone())
            };

        if final_summary.is_empty() && !final_content.is_empty() {
            final_summary = Self::generate_auto_summary(&final_content);
        }

        let ca_terminal_verdict = if agent.role == AgentRole::Check {
            let normalized = finalize_ca_terminal_contract(
                &final_summary,
                &final_content,
                false,
                terminal_completion_observed,
                &ctx.constraints,
                ca_executable_verifier_available
                    && ca_audit_convergence
                        .window_exhausted(agent.role, execution_budget.ca_evidence_close_turns),
                ca_has_successful_verifier_receipt(ca_audit_convergence, &action_tracker),
                &action_tracker.actions,
                self.workspace_root.as_deref(),
            );
            if let Some(issue) = normalized.contract_issue {
                errs.push(issue);
            }
            final_summary = normalized.summary;
            final_content = normalized.content;
            // Reasoning remains in the provider interaction trace, not in the
            // CA business handoff/archive payload.
            final_thought.clear();
            Some(normalized.verdict)
        } else {
            None
        };

        let l0_iri = session
            .archive_full_to_l0(
                &self.l0_store,
                &agent.role.to_string(),
                &final_thought,
                &final_content,
            )
            .ok();

        let l1_turn = session.add_summary(&agent.role.to_string(), &final_summary, l0_iri.clone());
        // Compute turn embedding and relevance_score
        if let (Some(ref embedder), Some(ref tracker_lock)) =
            (&self.embedder, &self.relevance_tracker)
        {
            if let Ok(emb) = embedder.embed(&final_summary).await {
                let mut tracker = tracker_lock.lock().unwrap();
                let score = tracker.on_new_input(&emb);
                l1_turn.embedding = Some(emb);
                l1_turn.relevance_score = Some(score);
            }
        }

        let node_iri = super::agent_turn_iri(&ctx.task_iri, session.session_id(), turn);
        let mut node_json = json!({
            "@id": &node_iri,
            "@type": "AgentTurn",
            "role": agent.role.to_string(),
            "cycle_id": ctx.cycle_id,
            "content": final_content,
            "content_len": final_content.len(),
            "summary": final_summary,
        });
        if !final_thought.is_empty() {
            node_json["has_thought"] = Value::Bool(true);
            node_json["thought_len"] = Value::Number(final_thought.len().into());
        }
        JsonLdContext::inject(&mut node_json);
        let cfg = crate::CoreConfig::default();
        if let Err(error) = self
            .blackboard
            .write_node(&node_iri, &node_json.to_string(), &cfg)
        {
            warn!(%error, %node_iri, "Unable to persist streaming AgentTurn node");
        }

        let output_value = Value::String(final_content.clone());
        let jsonld_output = self.apply_output_mapping(&output_value, &agent.role, &ctx.task_iri);

        let final_verdict = if workspace_effect_required && !workspace_effect_observed {
            let detail = "DA finished without creating or modifying substantive workspace content";
            errs.push(detail.to_string());
            final_summary = format!("FAILED: {}. {}", detail, final_summary);
            TaskVerdict::Failed
        } else if agent.role == AgentRole::Check {
            ca_terminal_verdict.expect("CA normalization must produce a verdict")
        } else if !terminal_completion_observed {
            let detail =
                "ReAct execution ended while the model still had an unfinished tool action";
            errs.push(detail.to_string());
            let verdict = Self::interrupted_execution_verdict(
                workspace_effect_required,
                workspace_effect_observed,
                &final_summary,
            );
            final_summary = format!("PARTIAL_SUCCESS: {}. {}", detail, final_summary);
            verdict
        } else if agent.role == AgentRole::Act && final_content.is_empty() {
            errs.push("AA terminal response lacked non-reasoning business content".to_string());
            final_summary = format!(
                "FAILED: AA terminal response lacked non-reasoning business content. {}",
                final_summary
            );
            TaskVerdict::Failed
        } else if Self::detect_blocker_verdict(&final_summary).is_some() {
            TaskVerdict::Blocked
        } else {
            TaskVerdict::Success
        };

        info!("AgentRunner streaming finished: {} tools", tc);

        (
            Ok(TaskResult {
                task_iri: ctx.task_iri,
                status: final_verdict.to_status_str().to_string(),
                summary: final_summary,
                output: Some(output_value),
                jsonld_output,
                artifacts: vec![],
                errors: errs,
                turn_count: turn,
                tool_call_count: tc,
                five_w2h_updates: None,
                tracked_actions: action_tracker.actions,
                verdict: Some(final_verdict),
                archive_iri: Some(node_iri),
            }),
            session,
        )
    }

    /// Store an ephemeral routed result for the owning L1 session. Stable
    /// cross-Agent exchange is archived as AgentTurn data; persisting this
    /// payload in L0 would create an unbounded, unauthorized orphan once the
    /// session reader is retired.
    fn store_micro_tool_data_for_session(&self, storage_key: &str, data: serde_json::Value) {
        self.tool_executor
            .write()
            .store_micro_tool_data(storage_key, data);
    }

    #[cfg(test)]
    pub(super) async fn route_tool_result(
        &self,
        result_str: &str,
        tool_name: &str,
        call_id: &str,
        session_id: &str,
    ) -> String {
        let routing = crate::tools::result_router::ResultRoutingIdentity::new(session_id, call_id);
        self.route_tool_result_with_routing(result_str, tool_name, &routing)
            .await
    }

    pub(super) async fn route_tool_result_with_routing(
        &self,
        result_str: &str,
        tool_name: &str,
        routing: &crate::tools::result_router::ResultRoutingIdentity,
    ) -> String {
        use crate::tools::result_router::graphify::GraphifyEngine;
        use crate::tools::result_router::micro_tools::MicroToolGenerator;
        use crate::tools::result_router::router::ResultRouter;
        use crate::tools::result_router::summary;
        use crate::tools::result_router::RouteDecision;
        use crate::tools::tool_executor::MicroToolContext;

        // Result readers already return a caller-selected, bounded page.
        // Routing that page again can create read_full_result_<new id>,
        // producing an unbounded "read an archived read" chain and needless
        // context growth. `read_agent_output` is now a stable paged reader too,
        // so keep both forms terminal and inline.
        if tool_name == "read_agent_output" || ToolExecutor::is_micro_tool_name(tool_name) {
            return result_str.to_string();
        }

        let settings = &self.tool_result_router_settings;
        let router = ResultRouter::new(settings);
        let call_id = routing.provider_call_id.as_str();
        let session_scope = routing.session_scope.as_str();

        let decision = router.route(result_str, tool_name, &routing.routing_call_key);
        let iri = routing.storage_iri.clone();
        let route_kind = match &decision {
            RouteDecision::PassThrough => "pass_through",
            RouteDecision::Truncate { .. } => "truncate",
            RouteDecision::FileReadPreview { .. } => "file_read_preview",
            RouteDecision::ExecutionPreview { .. } => "execution_preview",
            RouteDecision::Graphify { .. } => "graphify",
            RouteDecision::Summarize { .. } => "summarize",
        };

        let mut routed = match decision {
            RouteDecision::PassThrough => {
                // A complete file page is replayable through a stable
                // path/offset `file_read`. Registering a session reader here
                // creates a second cursor system, makes CA evidence archived,
                // and causes the next history-compaction pass to induce an
                // unnecessary model/tool round trip.
                if tool_name == "file_read" {
                    result_str.to_string()
                // Small results stay inline. Pre-register a resolvable IRI and
                // micro-tool only above prepare_threshold, in preparation for
                // reference compression.
                } else if result_str.len() > settings.prepare_threshold {
                    self.store_micro_tool_data_for_session(
                        &iri,
                        serde_json::json!({
                            "content": result_str,
                            "tool_name": tool_name,
                        }),
                    );
                    let read_tool_name = routing.reader_name.clone();
                    let ctx = MicroToolContext {
                        routing_call_key: routing.routing_call_key.clone(),
                        provider_call_id: call_id.to_string(),
                        storage_key: iri.clone(),
                        tool_name: tool_name.to_string(),
                        entity_types: vec![],
                        preview_size: settings.preview_size,
                    };
                    {
                        let mut exe = self.tool_executor.write();
                        exe.register_micro_tool(&read_tool_name, ctx);
                    }
                    format!("{}\nIRI: {}", result_str, iri)
                } else {
                    // An IRI is a query contract, not decorative metadata.
                    // Small inline results are deliberately not archived, so
                    // advertising an unresolvable IRI only induces wasted
                    // read_agent_output calls in later turns.
                    result_str.to_string()
                }
            }

            RouteDecision::Truncate { max_chars } => {
                let truncated = if result_str.len() <= max_chars {
                    result_str.to_string()
                } else {
                    summary::smart_truncate(result_str, max_chars)
                };
                // Persist full result to memory + L0
                self.store_micro_tool_data_for_session(
                    &iri,
                    serde_json::json!({
                        "content": result_str,
                        "tool_name": tool_name,
                    }),
                );
                let read_tool_name = routing.reader_name.clone();
                let ctx = MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: call_id.to_string(),
                    storage_key: iri.clone(),
                    tool_name: tool_name.to_string(),
                    entity_types: vec![],
                    preview_size: settings.preview_size,
                };
                {
                    let mut exe = self.tool_executor.write();
                    exe.register_micro_tool(&read_tool_name, ctx);
                    // Notify workspace_monitor that the file was read via read_full_result
                    if tool_name == "file_read" {
                        if let Ok(val) = serde_json::from_str::<Value>(result_str) {
                            if let Some(path) = val.get("path").and_then(|v| v.as_str()) {
                                exe.mark_file_external_read(path);
                            }
                        }
                    }
                }
                summary::format_iri_message(tool_name, &routing, &truncated, result_str.len())
            }

            RouteDecision::FileReadPreview {
                call_id: _,
                max_lines,
                max_chars,
            } => {
                // Keep the JSON skeleton (path/total_lines/offset) and the first
                // max_lines lines inline; the full content stays in the micro-tool.
                let preview = match serde_json::from_str::<Value>(result_str) {
                    Ok(Value::Object(mut obj)) => {
                        obj.insert("preview".to_string(), Value::Bool(true));
                        if let Some(lines) = obj.get_mut("lines").and_then(|l| l.as_array_mut()) {
                            let keep = lines.len().min(max_lines);
                            lines.truncate(keep);
                            obj.insert("returned".to_string(), Value::from(keep));
                        }
                        obj.insert(
                            "message".to_string(),
                            Value::String(format!(
                                "Preview of first {} lines shown. Call {} only if it is currently advertised, or use file_read with offset/limit to view the rest.",
                                max_lines, routing.reader_name
                            )),
                        );
                        serde_json::to_string(&Value::Object(obj))
                            .unwrap_or_else(|_| result_str.to_string())
                    }
                    _ => summary::smart_truncate(result_str, max_chars),
                };

                self.store_micro_tool_data_for_session(
                    &iri,
                    serde_json::json!({
                        "content": result_str,
                        "tool_name": tool_name,
                    }),
                );
                let read_tool_name = routing.reader_name.clone();
                let ctx = MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: call_id.to_string(),
                    storage_key: iri.clone(),
                    tool_name: tool_name.to_string(),
                    entity_types: vec![],
                    preview_size: settings.preview_size,
                };
                let routed_preview =
                    summary::format_iri_message(tool_name, &routing, &preview, result_str.len());
                {
                    let mut exe = self.tool_executor.write();
                    exe.register_micro_tool(&read_tool_name, ctx);
                    exe.seed_archived_reader_progress_from_preview(
                        &read_tool_name,
                        &routed_preview,
                    );
                    if tool_name == "file_read" {
                        if let Ok(val) = serde_json::from_str::<Value>(result_str) {
                            if let Some(path) = val.get("path").and_then(|v| v.as_str()) {
                                exe.mark_file_external_read(path);
                            }
                        }
                    }
                }
                routed_preview
            }

            RouteDecision::ExecutionPreview {
                call_id: _,
                max_chars,
            } => {
                self.store_micro_tool_data_for_session(
                    &iri,
                    serde_json::json!({
                        "content": result_str,
                        "tool_name": tool_name,
                    }),
                );
                let read_tool_name = routing.reader_name.clone();
                let routed_preview = summary::execution_preview_envelope(
                    tool_name,
                    result_str,
                    &routing,
                    result_str.len(),
                    max_chars,
                );
                {
                    let mut executor = self.tool_executor.write();
                    executor.register_micro_tool(
                        &read_tool_name,
                        MicroToolContext {
                            routing_call_key: routing.routing_call_key.clone(),
                            provider_call_id: call_id.to_string(),
                            storage_key: iri.clone(),
                            tool_name: tool_name.to_string(),
                            entity_types: vec![],
                            preview_size: settings.preview_size,
                        },
                    );
                    executor.seed_archived_reader_progress_from_preview(
                        &read_tool_name,
                        &routed_preview,
                    );
                }
                routed_preview
            }

            RouteDecision::Graphify {
                call_id: g_call_id,
                graph_name,
            } => {
                // `format_iri_message` advertises a canonical
                // read_full_result_<call_id> reader. Register that reader for
                // every Graphify outcome and store the same raw envelope used
                // by the other routing branches. Previously Graphify emitted
                // the reader name without registering it, causing a truthful
                // follow-up call to be rejected as tool_not_advertised.
                self.store_micro_tool_data_for_session(
                    &iri,
                    serde_json::json!({
                        "content": result_str,
                        "tool_name": tool_name,
                    }),
                );
                let read_tool_name = routing.reader_name.clone();
                self.tool_executor.write().register_micro_tool(
                    &read_tool_name,
                    MicroToolContext {
                        routing_call_key: routing.routing_call_key.clone(),
                        provider_call_id: call_id.to_string(),
                        storage_key: iri.clone(),
                        tool_name: tool_name.to_string(),
                        entity_types: vec![],
                        preview_size: settings.preview_size,
                    },
                );
                let parsed: Option<serde_json::Value> =
                    serde_json::from_str(result_str.trim()).ok();
                match parsed {
                    Some(json_val) => {
                        let engine_result = match &self.unified_graph_store {
                            Some(store) => GraphifyEngine::with_shared_store(
                                store.clone(),
                                settings.max_graph_entities,
                            ),
                            None => GraphifyEngine::new(settings.max_graph_entities),
                        };
                        match engine_result {
                            Ok(mut engine) => {
                                let graphify_result = engine.graphify_json(
                                    &json_val,
                                    &g_call_id,
                                    settings.max_graph_entities,
                                );
                                // The current bounded handlers query the
                                // archived top-level JSON rows. Advertise only
                                // types they can actually return; nested graph
                                // nodes and root objects remain available via
                                // the canonical bounded reader until a native
                                // graph-query handler is introduced.
                                let mut queryable_types = std::collections::HashMap::new();
                                if let Some(rows) = json_val.as_array() {
                                    for row in rows.iter().filter_map(Value::as_object) {
                                        let raw_type = ["@type", "type", "kind", "category"]
                                            .iter()
                                            .find_map(|key| row.get(*key).and_then(Value::as_str));
                                        let entity_type = match raw_type {
                                            Some(value)
                                                if value.starts_with("http://")
                                                    || value.starts_with("https://")
                                                    || value.starts_with("iri://") =>
                                            {
                                                value.to_string()
                                            }
                                            Some(value) => format!(
                                                "https://agent-os.org/ontology/tool-result/{value}"
                                            ),
                                            None => {
                                                "https://agent-os.org/ontology/tool-result/Entity"
                                                    .to_string()
                                            }
                                        };
                                        *queryable_types.entry(entity_type).or_insert(0usize) += 1;
                                    }
                                }
                                let analysis = crate::tools::result_router::SchemaAnalysis {
                                    entity_types: queryable_types.into_iter().collect(),
                                    relation_types: vec![],
                                    property_names: vec![],
                                    total_entities: json_val
                                        .as_array()
                                        .map(|rows| rows.len())
                                        .unwrap_or(0),
                                    total_relations: 0,
                                };
                                let micro_tools = if analysis.entity_types.is_empty() {
                                    Vec::new()
                                } else {
                                    MicroToolGenerator::generate_from_schema(
                                        &analysis,
                                        &routing,
                                        settings.max_micro_tools,
                                    )
                                };
                                for mt in &micro_tools {
                                    let entity_types = match &mt.tool_type {
                                        crate::tools::result_router::MicroToolType::EntityTypeQuery {
                                            entity_type,
                                            ..
                                        } => vec![entity_type.clone()],
                                        _ => Vec::new(),
                                    };
                                    let ctx = MicroToolContext {
                                        routing_call_key: routing.routing_call_key.clone(),
                                        provider_call_id: call_id.to_string(),
                                        storage_key: iri.clone(),
                                        tool_name: tool_name.to_string(),
                                        entity_types,
                                        preview_size: settings.preview_size,
                                    };
                                    self.tool_executor
                                        .write()
                                        .register_micro_tool(&mt.name, ctx);
                                }
                                info!(
                                    "[ResultRouter] Graphified: {} entities, {} relations, {} micro-tools, graph={}",
                                    graphify_result.entity_count, graphify_result.relation_count,
                                    micro_tools.len(), graph_name,
                                );
                                let graph_capability_envelope =
                                    MicroToolGenerator::format_tool_injection_message(
                                        &graphify_result.summary,
                                        &micro_tools,
                                    );
                                summary::format_iri_message(
                                    tool_name,
                                    &routing,
                                    &graph_capability_envelope,
                                    result_str.len(),
                                )
                            }
                            Err(e) => {
                                warn!("[ResultRouter] Graphification failed: {}, falling back to IRI format", e);
                                let truncated =
                                    summary::smart_truncate(result_str, settings.threshold_large);
                                summary::format_iri_message(
                                    tool_name,
                                    &routing,
                                    &truncated,
                                    result_str.len(),
                                )
                            }
                        }
                    }
                    None => {
                        let text_summary = summary::generate_text_summary(
                            result_str,
                            tool_name,
                            settings.preview_size,
                        );
                        summary::format_iri_message(
                            tool_name,
                            &routing,
                            &text_summary,
                            result_str.len(),
                        )
                    }
                }
            }

            RouteDecision::Summarize {
                call_id: _,
                preview_size,
            } => {
                self.store_micro_tool_data_for_session(
                    &iri,
                    serde_json::json!({
                        "content": result_str,
                        "tool_name": tool_name,
                    }),
                );

                let read_tool_name = routing.reader_name.clone();
                let ctx = MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: call_id.to_string(),
                    storage_key: iri.clone(),
                    tool_name: tool_name.to_string(),
                    entity_types: vec![],
                    preview_size,
                };
                self.tool_executor
                    .write()
                    .register_micro_tool(&read_tool_name, ctx);

                let preview = summary::generate_text_summary(result_str, tool_name, preview_size);
                info!(
                    "[ResultRouter] Summarized: {} bytes -> preview {} bytes, micro-tool: {}, IRI: {}",
                    result_str.len(), preview_size, read_tool_name, iri,
                );
                summary::format_iri_message(tool_name, &routing, &preview, result_str.len())
            }
        };

        let reader_name = routing.reader_name.clone();
        let reader_registered = self
            .tool_executor
            .read()
            .micro_tool_definition(&reader_name)
            .is_some();
        if !reader_registered
            && (routed.contains(&routing.storage_iri) || routed.contains(&routing.reader_name))
        {
            routed = crate::core::context_compressor::omit_inactive_reader_reference(&routed);
        }
        debug!(
            event = "tool_result_routed",
            call_id,
            routing_call_key = %routing.routing_call_key,
            session_scope,
            tool_name,
            route = route_kind,
            raw_bytes = result_str.len(),
            delivered_bytes = routed.len(),
            reader_registered,
            reader_name = if reader_registered {
                reader_name.as_str()
            } else {
                ""
            },
            reader_iri = if reader_registered { iri.as_str() } else { "" },
            "Routed tool result for model delivery",
        );
        routed
    }

    /// Reference compression: for tool messages exceeding the threshold, replace with a lightweight reference if a corresponding micro-tool exists.
    /// Call after ToolResultCompressor::compress_tool_messages.
    pub(super) fn compress_tool_results_with_microtools(
        &self,
        messages: &mut Vec<ChatMessage>,
        session_id: &str,
    ) {
        let threshold = self
            .tool_result_compressor
            .as_ref()
            .and_then(|c| c.lock().ok())
            .map(|c| c.compress_tool_result_threshold())
            .unwrap_or(500);

        for msg in messages.iter_mut() {
            if msg.role != "tool" {
                continue;
            }
            if msg.content.len() <= threshold {
                continue;
            }
            let call_id = match msg.tool_call_id.as_deref() {
                Some(id) if !id.is_empty() => id.to_string(),
                _ => continue,
            };
            // A historical tool message does not carry llm_request_id, so a
            // raw provider call ID cannot safely reconstruct the production
            // routing key after cross-request reuse. Only consume the exact,
            // validated reference already embedded by ResultRouter.
            let Some(routing) = crate::tools::result_router::routing_identity_from_content(
                &msg.content,
                session_id,
                &call_id,
            ) else {
                continue;
            };
            let micro_tool_name = routing.reader_name.clone();
            let (has_micro_tool, file_read_origin) = {
                let executor = self.tool_executor.read();
                let advertisable = executor.micro_tool_definition(&micro_tool_name).is_some();
                (
                    advertisable,
                    advertisable
                        && executor.micro_tool_originates_from(&micro_tool_name, "file_read"),
                )
            };
            if has_micro_tool {
                // A file result has a stable replay API of its own. Preserve
                // path/revision/source-line coordinates instead of replacing
                // them with a reader-only marker that forces a second cursor
                // system. Large archived pages may retain their exact reader,
                // but only because its registered handler was just confirmed.
                if file_read_origin {
                    if let Some(summary) =
                        crate::core::context_compressor::compact_file_read_history(
                            &msg.content,
                            Some(&routing),
                        )
                    {
                        let original_size = msg.content.len();
                        msg.content = summary;
                        info!(
                            event = "file_read_history_compacted",
                            call_id = %call_id,
                            route = "stable_file_cursor",
                            raw_bytes = original_size,
                            delivered_bytes = msg.content.len(),
                            reader_name = %micro_tool_name,
                            "Compacted observed file result without losing replay coordinates",
                        );
                    }
                    continue;
                }
                let iri = routing.storage_iri;
                let original_size = msg.content.len();
                msg.content = format!(
                    "[Compressed {} bytes] Session reader: `{}`\nCall it only while that exact name is advertised in the current turn.\nIRI: {}",
                    original_size, micro_tool_name, iri,
                );
                info!(
                    event = "tool_result_reference_compressed",
                    call_id = %call_id,
                    route = "micro_tool_reference",
                    raw_bytes = original_size,
                    delivered_bytes = msg.content.len(),
                    reader_name = %micro_tool_name,
                    reader_iri = %iri,
                    "Compacted previously observed tool result",
                );
            } else {
                let sanitized =
                    crate::core::context_compressor::omit_inactive_reader_reference(&msg.content);
                if sanitized != msg.content {
                    let original_size = msg.content.len();
                    msg.content = sanitized;
                    info!(
                        event = "retired_result_reader_omitted",
                        call_id = %call_id,
                        raw_bytes = original_size,
                        delivered_bytes = msg.content.len(),
                        reader_name = %micro_tool_name,
                        "Removed an inactive session reader from observed history",
                    );
                }
            }
        }
    }

    /// Compact tool results only after a provider request has had one chance
    /// to observe them. Calling this exactly when a new tool batch is accepted
    /// prevents sibling results from being aged merely because they completed
    /// earlier in the same assistant batch.
    pub(super) fn compact_observed_tool_history(
        &self,
        messages: &mut Vec<ChatMessage>,
        turn: u32,
        session_id: &str,
    ) {
        let active_readers = {
            let executor = self.tool_executor.read();
            executor
                .get_micro_tool_names()
                .into_iter()
                .filter(|name| executor.micro_tool_definition(name).is_some())
                .collect::<std::collections::HashSet<_>>()
        };
        if let Some(ref compressor_lock) = self.tool_result_compressor {
            if let Ok(compressor) = compressor_lock.lock() {
                compressor.compress_tool_messages_with_active_readers(
                    messages,
                    session_id,
                    &active_readers,
                );
            }
        }
        self.compress_tool_results_with_microtools(messages, session_id);

        if let Some(ref aging) = self.tool_result_aging {
            let (aged, freed) = aging.age_tool_results(messages, &self.tool_executor, session_id);
            if aged > 0 {
                info!(
                    event = "tool_result_history_aged",
                    turn,
                    aged_results = aged,
                    freed_bytes = freed,
                    "Aged previously observed tool results",
                );
            }
        }
    }
}
