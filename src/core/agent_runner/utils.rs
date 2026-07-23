use std::collections::HashMap;
use std::sync::atomic::Ordering;

use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::core::agent_instance::{AgentInstance, AgentRole, AgentStatus};
use crate::core::system_prompt::{
    build_constitution_prompt, build_time_awareness_text, SystemPromptBuilder, SystemPromptRegion,
};
use crate::gateway::unified_gateway::ChatMessage;
use crate::jsonld::{generate_iri, validate_jsonld_node, JsonLdContext, JsonLdNode};
use crate::memory::l1_session::L1Session;
use crate::methodology::integration::MethodologyPromptInjector;
use crate::tools::hooks::{HookContext, HookPoint, HookResult};
use crate::tools::tool_executor::ToolExecutor;
use crate::CoreError;

use super::{
    LlmParsedResponse, TaskContext, TaskResult, LLM_RESPONSE_FORMAT_NO_THOUGHT,
    LLM_RESPONSE_FORMAT_WITH_THOUGHT,
};

impl super::AgentRunner {
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

        let mut prompt_builder = SystemPromptBuilder::new();
        prompt_builder.set_region(SystemPromptRegion::RoleDefinition, agent_md.clone());

        // Region 2: Time awareness — current time and session context
        {
            let session_start = session
                .created_at()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string();
            let time_text = build_time_awareness_text(Some(&session_start));
            prompt_builder.set_region(SystemPromptRegion::TimeAwareness, time_text);
        }

        // Region 3: Workspace environment info
        if let Some(ref ws_root) = self.workspace_root {
            let env_info = format!(
                "## Workspace\n\n- Workspace path: {}\n\
                 - All your file operations (read, write, search, command execution) are limited to the workspace\n\
                 - Files outside the workspace are not relevant to the current task and should not be accessed\n\
                 - The workspace root may contain other directories and files unrelated to the current task — please distinguish carefully",
                ws_root.display()
            );
            prompt_builder.set_region(SystemPromptRegion::EnvironmentInfo, env_info);
        }

        // Region 2: Code of conduct (constitution + methodology)
        {
            let mut policy_text = build_constitution_prompt(agent.role);

            policy_text.push_str("\n\n### 🔴 Task Focus Principle (Must Follow)\n");
            policy_text.push_str("- Your only task is the currently assigned \"Current Task\". Other directories/files in the workspace are not relevant to your task\n");
            policy_text.push_str("- For unrelated files or directories (e.g., other projects, test outputs, irrelevant codebases), you must ignore them directly — do not explore or process them\n");
            policy_text.push_str("- When using glob_search, file_list, or similar tools, if results contain irrelevant content, you must automatically filter it out — do not let it distract you\n");
            policy_text.push_str("- If you encounter any files/directories that do not belong to the current task, skip them and continue executing the current task — do not change direction due to irrelevant content\n");
            policy_text.push_str("- Check Agent (CA) special note: Your audit report may only contain content related to the current task. If you discover irrelevant files, ignore them and do not include them in the report\n");
            policy_text.push_str("- Decision Agent (AA) special note: Do not actively explore files. Your decisions must be based solely on CA audit results — ignore any irrelevant content in the audit results\n");

            // Inject methodology discipline (PA/CA/AA specific)
            if let Some(methodology_addendum) =
                MethodologyPromptInjector::build_for_role(agent.role)
            {
                policy_text.push_str(&methodology_addendum);
            }
            // Inject persuasive directives for active methodologies
            if let Some(ref gate) = self.methodology_gate {
                let directives = gate.inner().read().persuasive_directives();
                if !directives.is_empty() {
                    policy_text.push_str("\n\n### Methodology Execution Requirements\n");
                    for d in &directives {
                        policy_text.push_str(&format!("- {}\n", d));
                    }
                }
            }
            // AA-specific: inject methodology evolution briefing
            if agent.role == AgentRole::Act {
                if let Some(ref gate) = self.methodology_gate {
                    if let Some(ref evo) = gate.evolution_handle() {
                        let briefing = evo.inner().read().aa_evolution_briefing();
                        if !briefing.is_empty() {
                            policy_text.push_str("\n\n");
                            policy_text.push_str(&briefing);
                        }
                    }
                }
            }
            prompt_builder.set_region(SystemPromptRegion::BehavioralPolicy, policy_text);
        }

        let emphasis_items = self.load_emphasis_from_l0(&ctx.task_iri).await;
        if !emphasis_items.is_empty() {
            let emphasis_content = emphasis_items
                .iter()
                .map(|e| format!("- {}", e))
                .collect::<Vec<_>>()
                .join("\n");
            prompt_builder.set_region(SystemPromptRegion::EmphasizedConstraints, emphasis_content);
        }

        let format_constraint = if supports_reasoning {
            LLM_RESPONSE_FORMAT_NO_THOUGHT.to_string()
        } else {
            LLM_RESPONSE_FORMAT_WITH_THOUGHT.to_string()
        };
        prompt_builder.set_region(SystemPromptRegion::OutputFormat, format_constraint);

        // Region: Output management area
        prompt_builder.set_region(
            SystemPromptRegion::OutputManagement,
            crate::core::system_prompt::OUTPUT_MANAGEMENT.to_string(),
        );

        let tool_menu = self.build_readable_tool_menu(&agent.role);
        if !tool_menu.is_empty() {
            prompt_builder.set_region(SystemPromptRegion::Tools, tool_menu);
        }

        // Region: Extraction prompt area (loaded from config)
        if let Some(ref config) = self.emphasis_config {
            if config.enabled {
                prompt_builder.set_region(
                    SystemPromptRegion::ExtractionPrompt,
                    config.extraction_prompt.clone(),
                );
            }
        }

        let system_content = prompt_builder.build();

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

        let tools = self
            .tool_executor
            .read()
            .tool_definitions_for_role(&agent.role.to_string());

        info!(
            "AgentRunner streaming started: role={}, model={}, tools={}",
            agent.role,
            model,
            tools.len()
        );

        let mut running_messages = messages;
        let max_turns = 10u32;
        let mut tc = 0u32;
        let mut turn = 0u32;
        let mut errs = Vec::new();
        let mut guard_pending_pre_injections: Vec<String> = Vec::new();
        let mut tool_error_counts: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        let mut last_content = String::new();
        let mut last_thought = String::new();
        let mut last_summary = String::new();

        loop {
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

            let mut stream = match self
                .gateway
                .stream_chat_with_params(
                    &model,
                    running_messages.clone(),
                    None,
                    None,
                    {
                        // Refresh the tools list before each call to include newly registered micro-tools
                        let current_tools = self
                            .tool_executor
                            .read()
                            .tool_definitions_for_role(&agent.role.to_string());
                        if current_tools.is_empty() {
                            None
                        } else {
                            Some(current_tools)
                        }
                    },
                    None,
                )
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
                            let executor = self.tool_executor.read().clone();
                            let result = executor
                                .execute_with_security_context(
                                    name,
                                    args,
                                    crate::skill_graph::security::SecurityContext::new(
                                        &agent.agent_id,
                                        &agent.role.to_string(),
                                    )
                                    .with_task(&ctx.task_iri),
                                )
                                .await
                                .unwrap_or_else(|e| json!({"error": e}));
                            let raw_result_str = serde_json::to_string(&result).unwrap_or_default();
                            let mut result_str =
                                self.route_tool_result(&raw_result_str, name, &c.id).await;

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
                                aging.age_tool_results(&mut running_messages, &self.tool_executor);
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

                        turn += 1;

                        // Check if compression is needed after each tool call (consistent with exec() behavior)
                        let cwm_did_compress = if let Some(ref cwm_lock) =
                            self.context_window_manager
                        {
                            if let Ok(cwm) = cwm_lock.lock() {
                                if cwm.should_compress(running_messages.len(), &running_messages) {
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

        let final_summary = if last_summary.is_empty() {
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

        let task_id = ctx
            .task_iri
            .strip_prefix("iri://task/")
            .unwrap_or_else(|| ctx.task_iri.strip_prefix("iri://").unwrap_or(&ctx.task_iri));
        let node_iri = format!("iri://task/{}/turn_{}", task_id, turn);
        let mut node_json = json!({
            "@id": &node_iri,
            "@type": "AgentTurn",
            "role": agent.role.to_string(),
            "content_len": last_content.len(),
        });
        if !last_thought.is_empty() {
            node_json["has_thought"] = Value::Bool(true);
            node_json["thought_len"] = Value::Number(last_thought.len().into());
        }

        let output_value = Value::String(last_content.clone());
        let jsonld_output = self.apply_output_mapping(&output_value, &agent.role, &ctx.task_iri);

        info!("AgentRunner streaming finished: {} tools", tc);

        (
            Ok(TaskResult {
                task_iri: ctx.task_iri,
                status: "success".to_string(),
                summary: final_summary,
                output: Some(output_value),
                jsonld_output,
                artifacts: vec![],
                errors: errs,
                turn_count: turn,
                tool_call_count: tc,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                archive_iri: None,
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

        let settings = crate::config::settings::ToolResultRouterSettings::default();
        let router = ResultRouter::new(&settings);

        let decision = router.route(result_str, tool_name, call_id);
        let iri = format!("iri://tool-result/{}", call_id);

        match decision {
            RouteDecision::PassThrough => {
                // Small result: pass through but attach IRI metadata
                // Pre-register micro-tool for results exceeding prepare_threshold, in preparation for reference compression
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
                }
                format!("{}\nIRI: {}", result_str, iri)
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

            RouteDecision::Graphify {
                call_id: g_call_id,
                graph_name,
            } => {
                let parsed: Option<serde_json::Value> =
                    serde_json::from_str(result_str.trim()).ok();
                match parsed {
                    Some(json_val) => {
                        self.store_micro_tool_data_persistent(&iri, json_val.clone());
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
