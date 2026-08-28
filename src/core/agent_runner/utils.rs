use std::collections::HashMap;
use std::sync::atomic::Ordering;

use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::core::agent_instance::{AgentInstance, AgentRole, AgentStatus};
use crate::gateway::unified_gateway::ChatMessage;
use crate::jsonld::{generate_iri, validate_jsonld_node, JsonLdContext, JsonLdNode};
use crate::memory::l1_session::L1Session;
use crate::tools::hooks::{HookContext, HookPoint, HookResult};
use crate::tools::tool_executor::ToolExecutor;
use crate::CoreError;

use super::execution::{
    advertised_tool_names, ca_evidence_close_tool_definitions, ca_evidence_focus_tool_definitions,
    da_evidence_close_tool_definitions, da_evidence_focus_tool_definitions,
    effective_effect_block_turns, effective_role_max_turns, evidence_key,
    filter_tool_search_result, initial_execution_phase, is_substantive_workspace_effect,
    is_workspace_mutation_candidate, mutation_recovery_tool_definitions,
    pa_planning_focus_tool_definitions, phase_tool_definitions, record_workspace_effect_turn,
    refresh_execution_ledger, requires_workspace_effect, unadvertised_tool_call_result,
    workspace_effect_recovery_active, workspace_inventory_complete_and_bounded,
    workspace_inventory_coverage, workspace_inventory_tool_definitions, ExecutionPhase,
};
use super::{LlmParsedResponse, TaskContext, TaskResult};

impl super::AgentRunner {
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

            // Extract content
            if let Some(content_str) = parsed.get("content").and_then(|s| s.as_str()) {
                response.content = content_str.to_string();
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
                }
            }
        } else {
            if let Some(extracted) = Self::try_extract_json_from_markdown(content) {
                if let Ok(parsed) = serde_json::from_str::<Value>(&extracted) {
                    response.is_valid_json = true;
                    if let Some(summary) = parsed.get("summary").and_then(|s| s.as_str()) {
                        response.summary = Some(summary.to_string());
                    }
                    if let Some(content_str) = parsed.get("content").and_then(|s| s.as_str()) {
                        response.content = content_str.to_string();
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

    pub(crate) fn try_extract_json_from_markdown(content: &str) -> Option<String> {
        let trimmed = content.trim();

        if trimmed.starts_with("```json") {
            let without_start = trimmed.trim_start_matches("```json").trim();
            if let Some(pos) = without_start.rfind("```") {
                return Some(without_start[..pos].trim().to_string());
            }
            return Some(without_start.trim().to_string());
        }

        if trimmed.starts_with("```") {
            let without_start = trimmed.trim_start_matches("```").trim();
            if let Some(pos) = without_start.rfind("```") {
                let candidate = without_start[..pos].trim();
                if candidate.starts_with('{') && candidate.ends_with('}') {
                    return Some(candidate.to_string());
                }
            }
            return Some(without_start.trim().to_string());
        }

        if let Some(start) = trimmed.find('{') {
            let mut depth = 0i32;
            for (i, c) in trimmed[start..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(trimmed[start..start + i + 1].to_string());
                        }
                    }
                    _ => {}
                }
            }
        }

        None
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
        let items: Vec<&String> = emphasis_items.iter().take(max_items).collect();

        // Load existing emphasis content for deduplication
        let existing = self.load_emphasis_from_l0(task_iri).await;

        for content in items {
            // Deduplication check
            let is_duplicate = existing.iter().any(|existing_content| {
                let similarity = Self::calculate_similarity(content, existing_content);
                similarity >= dedup_threshold
            });

            if is_duplicate {
                debug!(
                    "[L0] Skipping duplicate emphasis content: {}",
                    content.chars().take(50).collect::<String>()
                );
                continue;
            }

            let iri = format!(
                "iri://emphasis/{}/{}",
                task_iri.strip_prefix("iri://").unwrap_or(task_iri),
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

            if let Err(e) = self.l0_store.store(&iri, &node.to_string()) {
                warn!("Failed to save emphasis content to L0: {}", e);
            } else {
                info!("[L0] Saved emphasis content: {} -> {}", agent_id, &iri);
            }
        }
    }

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

    pub async fn execute_streaming<F>(
        &self,
        agent: &mut AgentInstance,
        ctx: TaskContext,
        on_event: F,
    ) -> Result<TaskResult, CoreError>
    where
        F: FnMut(&crate::llm::StreamEvent) + Send,
    {
        agent.status = AgentStatus::Running;

        let task_iri_for_guard = ctx.task_iri.clone();
        let mut session = self.memory_manager.lock().await.create_session(
            &agent.agent_id,
            &agent.role.to_string(),
            &ctx.task_iri,
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
            let _ = mm.finalize_session(session, &task_iri_for_guard);
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
        let model = self
            .gateway
            .get_model(&agent.role.to_string().to_lowercase());
        let supports_reasoning = self.gateway.supports_native_reasoning(&model);

        let context_data = self.gather_context_data_async(agent.role, &ctx).await;
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

        let context_msg = if summary_text.is_empty() {
            format!(
                "## Current Task\n{}\n\n## Available Tools\nUse the tools as needed to complete the task.",
                ctx.objective
            )
        } else {
            format!(
                "## Current Task\n{}\n\n## History Summary\n{}\n\n## Available Tools\nUse the tools as needed to complete the task.",
                ctx.objective, summary_text
            )
        };

        let messages: Vec<ChatMessage> = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_content,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: context_msg,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let tools = self.tool_definitions_for_task_context(&agent.role.to_string(), &ctx);
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
        let mut turn = 0u32;
        let mut errs = Vec::new();
        let mut guard_pending_pre_injections: Vec<String> = Vec::new();
        let mut session_micro_tools = std::collections::HashSet::<String>::new();
        let mut tool_error_counts: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        let mut last_content = String::new();
        let mut last_thought = String::new();
        let mut last_summary = String::new();
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
        let effect_block_turns = effective_effect_block_turns(
            execution_phase,
            execution_budget.effect_progress_block_turns,
            execution_budget.da_repair_effect_block_turns,
        );
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
        let mut planning_tool_turns = 0u32;
        let mut evidence_only_tool_turns = 0u32;
        let mut action_tracker =
            crate::core::tracked_action::ActionTracker::new(&ctx.task_iri, &agent.role.to_string());

        loop {
            if ctx.workspace_context_enabled() {
                super::execution::refresh_workspace_delta_message(
                    &self.tool_executor,
                    &mut running_messages,
                    &mut workspace_generation,
                    workspace_delta_limit,
                );
            }
            refresh_execution_ledger(
                &mut running_messages,
                agent.role,
                execution_phase,
                &ctx.effective_effect_policy(),
                substantive_effect_count,
                verification_turns,
                low_novelty_turns,
                workspace_generation,
            );
            if !guard_pending_pre_injections.is_empty() {
                let prompt = format!(
                    "\n\n[ToolGuard Constraint Directive]\n{}\nNote: The above constraints only apply to the upcoming tool call with the same name. Strictly comply.",
                    guard_pending_pre_injections.join("\n")
                );
                if let Some(sys_msg) = running_messages.first_mut() {
                    if sys_msg.role == "system" {
                        sys_msg.content.push_str(&prompt);
                    }
                }
                guard_pending_pre_injections.clear();
            }

            let mutation_recovery_active = workspace_effect_recovery_active(
                workspace_effect_tracked,
                consecutive_effectless_tool_turns,
                low_novelty_turns,
                effect_block_turns,
            );
            let mut request_messages = running_messages.clone();
            let ca_evidence_focus_active = agent.role == AgentRole::Check
                && execution_budget.ca_evidence_focus_turns > 0
                && verification_turns >= execution_budget.ca_evidence_focus_turns;
            let ca_evidence_close_active = agent.role == AgentRole::Check
                && execution_budget.ca_evidence_close_turns > 0
                && verification_turns >= execution_budget.ca_evidence_close_turns;
            let pa_planning_focus_active = agent.role == AgentRole::Plan
                && execution_budget.pa_planning_focus_turns > 0
                && planning_tool_turns >= execution_budget.pa_planning_focus_turns;
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
            if ca_evidence_focus_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "[CA Evidence Convergence] Multiple verification tool turns have already completed. Finish now with PASS/FAIL and criterion-linked evidence unless one named acceptance criterion is still unverified. If one remains, perform only the single targeted check needed for that criterion; do not repeat broad discovery or already-passing checks.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            if ca_evidence_close_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "[CA Evidence Close Gate] The configured evidence window is exhausted. Do not call another tool. Return the final criterion-linked PASS/FAIL audit now. Any criterion lacking evidence must be marked FAIL; uncertainty is not a reason for more discovery.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            if pa_planning_focus_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "[PA Planning Convergence] The configured inspection window is complete. Use the objective, workspace manifest, retrieved evidence, and prior-cycle feedback already supplied. Emit the executable plan now; do not request more tools. Preserve explicit acceptance criteria and name the checks DA/CA must run.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            if da_evidence_focus_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "[DA Evidence Convergence] The configured evidence-discovery window is complete. Synthesize the requested deliverable now from the sources and evidence already collected. Only one targeted source read is permitted when a specific claim lacks support; do not perform another broad search.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            if da_evidence_close_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "[DA Evidence Close Gate] The configured evidence window is exhausted. Do not call another tool. Return the complete evidence-backed deliverable now, explicitly marking any unsupported point as a limitation.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            if mutation_recovery_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: format!(
                        "[DA Mutation Recovery Mode] The last {} tool turns made no substantive workspace change. Inspection/search tools are temporarily unavailable. Make the highest-priority pending change now with an advertised mutation-capable tool, or finish with `FAILED:` and the exact blocker.",
                        consecutive_effectless_tool_turns
                    ),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
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
                let current_tools = ca_evidence_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    ca_evidence_focus_active,
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
                    da_evidence_close_active,
                );
                if mutation_recovery_active {
                    mutation_recovery_tool_definitions(current_tools)
                } else {
                    current_tools
                }
            };
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
            let request_tools = (!current_tools.is_empty()).then_some(current_tools);

            let mut stream = match self
                .gateway
                .stream_chat_with_params(&model, request_messages, None, None, request_tools, None)
                .await
            {
                Ok(s) => s,
                Err(e) => return (Err(e), session),
            };

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
                        break Err(CoreError::Internal {
                            message: e.to_string(),
                        })
                    }
                }
            };
            if let Err(e) = stream_result {
                return (Err(e), session);
            }

            let stream_response: crate::llm::StreamResponse = accumulator.into();

            // Accumulate token usage from streaming response
            if let Some(ref usage) = stream_response.usage {
                self.total_prompt_tokens
                    .fetch_add(usage.prompt_tokens as u64, Ordering::Relaxed);
                self.total_completion_tokens
                    .fetch_add(usage.completion_tokens as u64, Ordering::Relaxed);
                self.last_prompt_tokens
                    .store(usage.prompt_tokens as u64, Ordering::Relaxed);
                self.last_completion_tokens
                    .store(usage.completion_tokens as u64, Ordering::Relaxed);
            }

            let parsed = self.parse_llm_response(
                &stream_response.content,
                stream_response.thought.as_deref(),
                supports_reasoning,
            );

            match parsed.action.as_deref() {
                Some("tool_call") => {
                    if !stream_response.tool_calls.is_empty() {
                        let tool_calls = &stream_response.tool_calls;
                        let has_effect_candidate = tool_calls.iter().any(|call| {
                            is_substantive_workspace_effect(&call.name, &call.arguments)
                        });
                        let block_effectless_calls =
                            mutation_recovery_active && !has_effect_candidate;
                        let mut effect_succeeded_this_turn = false;
                        let mut verification_failed_this_turn = false;
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
                        if agent.role == AgentRole::Check && !tool_calls.is_empty() {
                            verification_turns = verification_turns.saturating_add(1);
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
                                break;
                            }
                        }

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

                        for c in tool_calls {
                            tc += 1;
                            let name = &c.name;
                            let args: Value = c.arguments.clone();

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
                                running_messages.push(ChatMessage {
                                    role: "tool".to_string(),
                                    content: serde_json::to_string(&rejection).unwrap_or_default(),
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: Some(c.id.clone()),
                                    reasoning_content: None,
                                });
                                continue;
                            }

                            // SkillBefore hook
                            {
                                let mut hook_ctx = HookContext::new(
                                    HookPoint::SkillBefore,
                                    &agent.agent_id,
                                    &agent.role.to_string(),
                                )
                                .with_task(&ctx.task_iri, &ctx.task_iri)
                                .with_data("tool_name", Value::String(name.clone()));
                                self.hook_manager
                                    .execute(HookPoint::SkillBefore, &mut hook_ctx)
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
                            }

                            // Clone the executor before awaiting so the
                            // shared lock is not held across handler I/O.
                            // This path enforces ToolExecutor policies/gates
                            // while retaining micro-tool fallback behavior.
                            let effect_snapshot = (is_substantive_workspace_effect(name, &args)
                                && !matches!(name.as_str(), "file_write" | "file_edit"))
                            .then(|| {
                                super::execution::capture_workspace_effect_snapshot(
                                    &self.tool_executor,
                                )
                            })
                            .flatten();
                            let started_at = std::time::Instant::now();
                            let mut result = if block_effectless_calls {
                                json!({
                                    "error": "DA execution-progress guard blocked another inspection-only turn",
                                    "required_next_action": "Create or modify a substantive artifact; otherwise finish with FAILED and the exact blocker."
                                })
                            } else {
                                let executor = self.tool_executor.read().clone();
                                executor
                                    .execute_with_security_context(
                                        name,
                                        args,
                                        crate::skill_graph::security::SecurityContext::new(
                                            &agent.agent_id,
                                            &agent.role.to_string(),
                                        )
                                        .with_task(&ctx.task_iri),
                                        ctx.allowed_tools.as_deref(),
                                    )
                                    .await
                                    .unwrap_or_else(|e| json!({"error": e}))
                            };
                            if name == "tool_search" {
                                filter_tool_search_result(&mut result, &discoverable_tools);
                            }
                            action_tracker.record(
                                name,
                                &c.arguments,
                                &result,
                                started_at.elapsed().as_secs_f64(),
                            );
                            if agent.role == AgentRole::Do
                                && matches!(name.as_str(), "bash" | "powershell" | "code_execute")
                                && crate::core::tracked_action::tool_result_failed(&result)
                            {
                                verification_failed_this_turn = true;
                            }
                            if !block_effectless_calls
                                && super::execution::confirmed_workspace_effect(
                                    &self.tool_executor,
                                    name,
                                    &c.arguments,
                                    &result,
                                    effect_snapshot.as_ref(),
                                )
                                .await
                            {
                                effect_succeeded_this_turn = true;
                                action_tracker.mark_last_substantive_effect();
                            }
                            let raw_result_str = serde_json::to_string(&result).unwrap_or_default();
                            let mut result_str =
                                self.route_tool_result(&raw_result_str, name, &c.id).await;
                            session_micro_tools.extend(
                                self.tool_executor
                                    .read()
                                    .get_micro_tool_names_for_call(&c.id),
                            );
                            if name == "tool_search" {
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

                            // SkillAfter hook
                            let guard_aborted = {
                                let mut hook_ctx = HookContext::new(
                                    HookPoint::SkillAfter,
                                    &agent.agent_id,
                                    &agent.role.to_string(),
                                )
                                .with_task(&ctx.task_iri, &ctx.task_iri)
                                .with_data("tool_name", Value::String(name.clone()))
                                .with_data("tool_result", Value::String(raw_result_str.clone()));
                                let hook_result = self
                                    .hook_manager
                                    .execute(HookPoint::SkillAfter, &mut hook_ctx)
                                    .await;

                                if hook_result == HookResult::Abort {
                                    Some(hook_ctx.error.unwrap_or_else(|| {
                                        "Tool result rejected by guard".to_string()
                                    }))
                                } else {
                                    None
                                }
                            };

                            if let Some(guard_msg) = &guard_aborted {
                                warn!("[Streaming] {} ToolGuard intercepted: {}", name, guard_msg);
                            } else if let Some(_err_val) = result.get("error") {
                                let err_msg = _err_val.as_str().unwrap_or("");
                                let is_tool_not_found = err_msg.starts_with("Tool not found: ");
                                warn!("[Streaming] tool {} failed: {}", name, err_msg);
                                errs.push(format!("{}: {}", name, err_msg));
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

                            let tool_content = if let Some(guard_msg) = &guard_aborted {
                                format!("[ToolGuard Intercepted] Result for tool {} was rejected by the security system. {}", name, guard_msg)
                            } else {
                                result_str
                            };

                            if let Some(ref compressor_lock) = self.tool_result_compressor {
                                if let Ok(mut compressor) = compressor_lock.lock() {
                                    compressor.add_result(turn, name, &c.id, &tool_content);
                                    compressor.compress_tool_messages(&mut running_messages);
                                }
                            }
                            self.compress_tool_results_with_microtools(&mut running_messages);

                            // Cross-turn aging: compress old tool results by staleness
                            if let Some(ref aging) = self.tool_result_aging {
                                let (aged, freed) = aging
                                    .age_tool_results(&mut running_messages, &self.tool_executor);
                                if aged > 0 {
                                    info!(
                                        "[turn {}] ToolResultAging aged {} results, freed {} bytes",
                                        turn, aged, freed
                                    );
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

                        if workspace_effect_tracked {
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
                                running_messages.push(ChatMessage {
                                    role: "user".to_string(),
                                    content: "[DA Verification Failure] An execution/verification command returned a failure signal. Repair the concrete reported defect before more broad inspection or completion, then rerun the targeted verification.".to_string(),
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: None,
                                    reasoning_content: None,
                                });
                                info!("[DA Streaming progress] failed verification moved execution phase to Repair");
                            } else if !effect_succeeded_this_turn {
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
                                    && (consecutive_effectless_tool_turns == effect_warning_turns
                                        || (effect_block_turns > 0
                                            && consecutive_effectless_tool_turns
                                                == effect_block_turns))
                                {
                                    let recovery_now = workspace_effect_recovery_active(
                                        workspace_effect_tracked,
                                        consecutive_effectless_tool_turns,
                                        low_novelty_turns,
                                        effect_block_turns,
                                    );
                                    if recovery_now
                                        && consecutive_effectless_tool_turns == effect_block_turns
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
                                    running_messages.push(ChatMessage {
                                        role: "user".to_string(),
                                        content: format!(
                                            "[DA Execution Progress Contract] {} consecutive tool turns produced no substantive workspace change. {} Execute file_write/file_edit or a genuinely mutating command next; otherwise finish with `FAILED:` and the exact blocker.",
                                            consecutive_effectless_tool_turns, urgency
                                        ),
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: None,
                                        reasoning_content: None,
                                    });
                                }
                            }
                        }

                        turn += 1;

                        // Check if compression is needed after each tool call (consistent with exec() behavior)
                        let cwm_did_compress =
                            if let Some(ref cwm_lock) = self.context_window_manager {
                                if let Ok(cwm) = cwm_lock.lock() {
                                    let model = self
                                        .gateway
                                        .get_model(&agent.role.to_string().to_lowercase());
                                    if cwm.should_compress_for_model_with_reserve(
                                        running_messages.len(),
                                        &running_messages,
                                        &model,
                                        current_tool_schema_token_reserve,
                                    ) {
                                        let (compressed, _summary) =
                                            cwm.compress_messages(&running_messages);
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
                        if !cwm_did_compress && running_messages.len() > 40 {
                            let system_msg = running_messages.first().cloned();
                            let kept_recent = running_messages.len().saturating_sub(15);

                            let mut recent: Vec<_> =
                                running_messages.drain(kept_recent..).collect();

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

                            running_messages.clear();
                            if let Some(sys) = system_msg {
                                running_messages.push(sys);
                            }

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
                                role: "user".to_string(),
                                content: summary_note,
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                                reasoning_content: None,
                            });
                            running_messages.extend(recent);

                            warn!(
                                "[Streaming] Message history hard truncated: kept {} messages (original {} )",
                                running_messages.len(),
                                kept_recent + 17
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
                    last_content = parsed.content.clone();
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

        let mut final_summary = if last_summary.is_empty() {
            Self::generate_auto_summary(&last_content)
        } else {
            last_summary.clone()
        };

        let l0_iri = session
            .archive_full_to_l0(
                &self.l0_store,
                &agent.role.to_string(),
                &last_thought,
                &last_content,
            )
            .ok();

        let l1_turn = session.add_summary(&agent.role.to_string(), &last_summary, l0_iri.clone());
        // Compute turn embedding and relevance_score
        if let (Some(ref embedder), Some(ref tracker_lock)) =
            (&self.embedder, &self.relevance_tracker)
        {
            if let Ok(emb) = embedder.embed(&last_summary).await {
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
            "content": last_content,
            "content_len": last_content.len(),
            "summary": final_summary,
        });
        if !last_thought.is_empty() {
            node_json["has_thought"] = Value::Bool(true);
            node_json["thought_len"] = Value::Number(last_thought.len().into());
        }
        JsonLdContext::inject(&mut node_json);
        let cfg = crate::CoreConfig::default();
        if let Err(error) = self
            .blackboard
            .write_node(&node_iri, &node_json.to_string(), &cfg)
        {
            warn!(%error, %node_iri, "Unable to persist streaming AgentTurn node");
        }

        let output_value = Value::String(last_content.clone());
        let jsonld_output = self.apply_output_mapping(&output_value, &agent.role, &ctx.task_iri);

        let final_status = if workspace_effect_required && !workspace_effect_observed {
            let detail = "DA finished without creating or modifying substantive workspace content";
            errs.push(detail.to_string());
            final_summary = format!("FAILED: {}. {}", detail, final_summary);
            "failed"
        } else {
            "success"
        };

        info!("AgentRunner streaming finished: {} tools", tc);

        (
            Ok(TaskResult {
                task_iri: ctx.task_iri,
                status: final_status.to_string(),
                summary: final_summary,
                output: Some(output_value),
                jsonld_output,
                artifacts: vec![],
                errors: errs,
                turn_count: turn,
                tool_call_count: tc,
                five_w2h_updates: None,
                tracked_actions: action_tracker.actions,
                verdict: None,
                archive_iri: Some(node_iri),
            }),
            session,
        )
    }

    /// Store micro-tool data to both memory and L0 persistent storage
    fn store_micro_tool_data_persistent(&self, storage_key: &str, data: serde_json::Value) {
        self.tool_executor
            .write()
            .store_micro_tool_data(storage_key, data.clone());
        // L0 persistence for cross-session availability
        if let Ok(data_str) = serde_json::to_string(&data) {
            let _ = self.l0_store.store(storage_key, &data_str);
        }
    }

    pub(super) async fn route_tool_result(
        &self,
        result_str: &str,
        tool_name: &str,
        call_id: &str,
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

        let decision = router.route(result_str, tool_name, call_id);
        let iri = format!("iri://tool-result/{}", call_id);

        match decision {
            RouteDecision::PassThrough => {
                // Small results stay inline. Pre-register a resolvable IRI and
                // micro-tool only above prepare_threshold, in preparation for
                // reference compression.
                if result_str.len() > settings.prepare_threshold {
                    self.store_micro_tool_data_persistent(
                        &iri,
                        serde_json::json!({
                            "content": result_str,
                            "tool_name": tool_name,
                        }),
                    );
                    let read_tool_name = format!("read_full_result_{}", call_id);
                    let ctx = MicroToolContext {
                        call_id: call_id.to_string(),
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
                self.store_micro_tool_data_persistent(
                    &iri,
                    serde_json::json!({
                        "content": result_str,
                        "tool_name": tool_name,
                    }),
                );
                let read_tool_name = format!("read_full_result_{}", call_id);
                let ctx = MicroToolContext {
                    call_id: call_id.to_string(),
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
                summary::format_iri_message(tool_name, call_id, &truncated, result_str.len())
            }

            RouteDecision::FileReadPreview {
                call_id: p_call_id,
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
                                "Preview of first {} lines shown. Call read_full_result_{} or file_read with offset/limit to view the rest.",
                                max_lines, p_call_id
                            )),
                        );
                        serde_json::to_string(&Value::Object(obj))
                            .unwrap_or_else(|_| result_str.to_string())
                    }
                    _ => summary::smart_truncate(result_str, max_chars),
                };

                self.store_micro_tool_data_persistent(
                    &iri,
                    serde_json::json!({
                        "content": result_str,
                        "tool_name": tool_name,
                    }),
                );
                let read_tool_name = format!("read_full_result_{}", p_call_id);
                let ctx = MicroToolContext {
                    call_id: p_call_id.to_string(),
                    storage_key: iri.clone(),
                    tool_name: tool_name.to_string(),
                    entity_types: vec![],
                    preview_size: settings.preview_size,
                };
                {
                    let mut exe = self.tool_executor.write();
                    exe.register_micro_tool(&read_tool_name, ctx);
                    if tool_name == "file_read" {
                        if let Ok(val) = serde_json::from_str::<Value>(result_str) {
                            if let Some(path) = val.get("path").and_then(|v| v.as_str()) {
                                exe.mark_file_external_read(path);
                            }
                        }
                    }
                }
                summary::format_iri_message(tool_name, call_id, &preview, result_str.len())
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
                self.store_micro_tool_data_persistent(
                    &iri,
                    serde_json::json!({
                        "content": result_str,
                        "tool_name": tool_name,
                    }),
                );
                let read_tool_name = format!("read_full_result_{}", call_id);
                self.tool_executor.write().register_micro_tool(
                    &read_tool_name,
                    MicroToolContext {
                        call_id: call_id.to_string(),
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
                                let analysis = crate::tools::result_router::SchemaAnalysis {
                                    entity_types: graphify_result
                                        .entity_types
                                        .iter()
                                        .map(|t| (t.clone(), 0))
                                        .collect(),
                                    relation_types: vec![],
                                    property_names: vec![],
                                    total_entities: graphify_result.entity_count,
                                    total_relations: graphify_result.relation_count,
                                };
                                let micro_tools = MicroToolGenerator::generate_from_schema(
                                    &analysis,
                                    &g_call_id,
                                    settings.max_micro_tools,
                                );
                                for mt in &micro_tools {
                                    let ctx = MicroToolContext {
                                        call_id: g_call_id.clone(),
                                        storage_key: iri.clone(),
                                        tool_name: tool_name.to_string(),
                                        entity_types: vec![],
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
                                summary::format_iri_message(
                                    tool_name,
                                    call_id,
                                    &graphify_result.summary,
                                    result_str.len(),
                                )
                            }
                            Err(e) => {
                                warn!("[ResultRouter] Graphification failed: {}, falling back to IRI format", e);
                                let truncated =
                                    summary::smart_truncate(result_str, settings.threshold_large);
                                summary::format_iri_message(
                                    tool_name,
                                    call_id,
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
                            call_id,
                            &text_summary,
                            result_str.len(),
                        )
                    }
                }
            }

            RouteDecision::Summarize {
                call_id: s_call_id,
                preview_size,
            } => {
                self.store_micro_tool_data_persistent(
                    &iri,
                    serde_json::json!({
                        "content": result_str,
                        "tool_name": tool_name,
                    }),
                );

                let read_tool_name = format!("read_full_result_{}", s_call_id);
                let ctx = MicroToolContext {
                    call_id: s_call_id.to_string(),
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
                summary::format_iri_message(tool_name, call_id, &preview, result_str.len())
            }
        }
    }

    /// Reference compression: for tool messages exceeding the threshold, replace with a lightweight reference if a corresponding micro-tool exists.
    /// Call after ToolResultCompressor::compress_tool_messages.
    pub(super) fn compress_tool_results_with_microtools(&self, messages: &mut Vec<ChatMessage>) {
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
            let micro_tool_name = format!("read_full_result_{}", call_id);
            let has_micro_tool = self
                .tool_executor
                .read()
                .try_get_handler(&micro_tool_name)
                .is_some();
            if has_micro_tool {
                let iri = format!("iri://tool-result/{}", call_id);
                let original_size = msg.content.len();
                msg.content = format!(
                    "[Compressed {} bytes] Call the `{}` tool for the full result\nIRI: {}",
                    original_size, micro_tool_name, iri,
                );
                debug!(
                    "[tool_compress] Reference compression: {} ({} bytes -> {} bytes)",
                    micro_tool_name,
                    original_size,
                    msg.content.len(),
                );
            }
        }
    }
}
