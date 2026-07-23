use std::collections::HashMap;
use tracing::debug;

use crate::core::agent_instance::{AgentInstance, AgentRole};
use crate::core::sa::PlanStep;
use crate::memory::l1_session::L1Session;

use super::{TaskContext, LLM_RESPONSE_FORMAT_NO_THOUGHT, LLM_RESPONSE_FORMAT_WITH_THOUGHT};

impl super::AgentRunner {
    pub(super) fn build_agent_md_from_step(
        &self,
        role: AgentRole,
        step: &PlanStep,
        context_data: &HashMap<String, String>,
    ) -> String {
        let role_name = match role {
            AgentRole::Plan => "Plan",
            AgentRole::Do => "Do",
            AgentRole::Check => "Check",
            AgentRole::Act => "Act",
        };

        let tools_list = if step.tools_allowed.is_empty() {
            self.tool_executor.read().list_tools(&role.to_string())
        } else {
            step.tools_allowed.clone()
        };

        let model = self.gateway.get_model(&role.to_string().to_lowercase());
        let supports_reasoning = self.gateway.supports_native_reasoning(&model);
        let format_constraint = if supports_reasoning {
            LLM_RESPONSE_FORMAT_NO_THOUGHT
        } else {
            LLM_RESPONSE_FORMAT_WITH_THOUGHT
        };

        let context_section = if context_data.is_empty() {
            String::new()
        } else {
            let mut sections = Vec::new();
            if let Some(original) = context_data.get("original_task") {
                sections.push(format!("## Original Task Requirements\n{}\n\n⚠️ Important: You must verify that all the above requirements have been completed.", original));
            }
            if let Some(plan) = context_data.get("plan_content") {
                sections.push(format!("## Superior Plan\n{}", plan));
            }
            if let Some(result) = context_data.get("execution_result") {
                sections.push(format!("## Execution Result\n{}", result));
            }
            if let Some(check) = context_data.get("check_result") {
                sections.push(format!("## Check Conclusion\n{}", check));
            }
            if let Some(ctx_summary) = context_data.get("context_summary") {
                sections.push(format!("## Related Context\n{}", ctx_summary));
            }
            if let Some(completed) = context_data.get("completed_steps") {
                sections.push(format!("## Completed Steps\n{}", completed));
            }
            if let Some(pending) = context_data.get("pending_steps") {
                sections.push(format!("## Pending Steps\n{}", pending));
            }
            let has_w2h = context_data.contains_key("five_w2h_what");
            if has_w2h {
                let mut w2h_lines = Vec::new();
                if let Some(v) = context_data.get("five_w2h_what") {
                    w2h_lines.push(format!("- What: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_why") {
                    w2h_lines.push(format!("- Why: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_success_criteria") {
                    w2h_lines.push(format!("- Success Criteria: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_deadline") {
                    w2h_lines.push(format!("- Deadline: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_execution_env") {
                    w2h_lines.push(format!("- Execution Environment: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_required_steps") {
                    w2h_lines.push(format!("- Required Steps: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_forbidden_tools") {
                    w2h_lines.push(format!("- Forbidden Tools: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_token_budget") {
                    w2h_lines.push(format!("- Token Budget: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_max_cycles") {
                    w2h_lines.push(format!("- Max Cycles: {}", v));
                }
                if !w2h_lines.is_empty() {
                    sections.push(format!("## Task Metadata (5W2H)\n{}", w2h_lines.join("\n")));
                }
            }
            sections.join("\n\n")
        };

        let mut agent_md = format!(
            r#"# {} Agent

## Current Task Objective
{}

## Expected Output
{}

## Success Criteria
{}

## Available Tools
{}

## Output Format Requirements
{}
"#,
            role_name,
            step.objective,
            step.expected_output,
            step.success_criteria,
            tools_list.join(", "),
            format_constraint
        );

        if !context_section.is_empty() {
            agent_md.push_str("\n\n");
            agent_md.push_str(&context_section);
        }

        agent_md
    }

    /// Create an L1 session for the given agent/task.
    /// Planned for SA integration — currently unused.
    #[allow(dead_code)]
    pub(super) async fn create_session(
        &self,
        agent: &AgentInstance,
        ctx: &TaskContext,
    ) -> L1Session {
        self.memory_manager.lock().await.create_session(
            &agent.agent_id,
            &agent.role.to_string(),
            &ctx.task_iri,
        )
    }

    fn gather_context_data(&self, role: AgentRole, ctx: &TaskContext) -> HashMap<String, String> {
        let mut context_data = HashMap::new();

        if let Some(ref original_task) = ctx.original_task {
            context_data.insert("original_task".to_string(), original_task.clone());
        }

        if let Some(ref summary) = ctx.prev_agent_summary {
            match role {
                AgentRole::Do => {
                    context_data.insert("plan_content".to_string(), summary.clone());
                }
                AgentRole::Check => {
                    context_data.insert("execution_result".to_string(), summary.clone());
                }
                AgentRole::Act => {
                    context_data.insert("check_result".to_string(), summary.clone());
                }
                _ => {}
            }
        }

        if !ctx.completed_steps.is_empty() {
            context_data.insert(
                "completed_steps".to_string(),
                ctx.completed_steps.join(", "),
            );
        }
        if !ctx.pending_steps.is_empty() {
            context_data.insert("pending_steps".to_string(), ctx.pending_steps.join(", "));
        }

        for (k, v) in &ctx.constraints {
            context_data.insert(k.clone(), v.clone());
        }

        if let Some(ref snapshot) = ctx.five_w2h_snapshot {
            // Inject 5W2H data by role to avoid redundancy
            // PA: what, why, success_criteria, deadline, env
            // DA: what, required_steps, forbidden_tools
            // CA: full 7 dimensions
            // AA: what + why (minimal reference set)
            match role {
                AgentRole::Plan => {
                    context_data.insert("five_w2h_what".to_string(), snapshot.what.clone());
                    context_data
                        .insert("five_w2h_why".to_string(), snapshot.why.description.clone());
                    if !snapshot.why.success_criteria.is_empty() {
                        context_data.insert(
                            "five_w2h_success_criteria".to_string(),
                            snapshot.why.success_criteria.join(", "),
                        );
                    }
                    if let Some(ref when) = snapshot.when {
                        if let Some(ref deadline) = when.deadline {
                            context_data
                                .insert("five_w2h_deadline".to_string(), deadline.to_rfc3339());
                        }
                    }
                    if let Some(ref where_) = snapshot.where_ {
                        if let Some(ref env) = where_.execution_environment {
                            context_data.insert("five_w2h_execution_env".to_string(), env.clone());
                        }
                    }
                }
                AgentRole::Do => {
                    context_data.insert("five_w2h_what".to_string(), snapshot.what.clone());
                    if let Some(ref how) = snapshot.how {
                        if let Some(ref steps) = how.required_steps {
                            context_data
                                .insert("five_w2h_required_steps".to_string(), steps.clone());
                        }
                        if !how.forbidden_tools.is_empty() {
                            context_data.insert(
                                "five_w2h_forbidden_tools".to_string(),
                                how.forbidden_tools.join(", "),
                            );
                        }
                    }
                }
                AgentRole::Check => {
                    context_data.insert("five_w2h_what".to_string(), snapshot.what.clone());
                    context_data
                        .insert("five_w2h_why".to_string(), snapshot.why.description.clone());
                    if !snapshot.why.success_criteria.is_empty() {
                        context_data.insert(
                            "five_w2h_success_criteria".to_string(),
                            snapshot.why.success_criteria.join(", "),
                        );
                    }
                    if let Some(ref when) = snapshot.when {
                        if let Some(ref deadline) = when.deadline {
                            context_data
                                .insert("five_w2h_deadline".to_string(), deadline.to_rfc3339());
                        }
                    }
                    if let Some(ref where_) = snapshot.where_ {
                        if let Some(ref env) = where_.execution_environment {
                            context_data.insert("five_w2h_execution_env".to_string(), env.clone());
                        }
                    }
                    if let Some(ref how) = snapshot.how {
                        if let Some(ref steps) = how.required_steps {
                            context_data
                                .insert("five_w2h_required_steps".to_string(), steps.clone());
                        }
                        if !how.forbidden_tools.is_empty() {
                            context_data.insert(
                                "five_w2h_forbidden_tools".to_string(),
                                how.forbidden_tools.join(", "),
                            );
                        }
                    }
                    if let Some(ref how_much) = snapshot.how_much {
                        if let Some(budget) = how_much.token_budget {
                            context_data
                                .insert("five_w2h_token_budget".to_string(), budget.to_string());
                        }
                        if let Some(cycles) = how_much.max_pdca_cycles {
                            context_data
                                .insert("five_w2h_max_cycles".to_string(), cycles.to_string());
                        }
                    }
                }
                AgentRole::Act => {
                    context_data.insert("five_w2h_what".to_string(), snapshot.what.clone());
                    context_data
                        .insert("five_w2h_why".to_string(), snapshot.why.description.clone());
                }
            }
        }

        context_data
    }

    pub(super) async fn gather_context_data_async(
        &self,
        role: AgentRole,
        ctx: &TaskContext,
    ) -> HashMap<String, String> {
        let mut context_data = self.gather_context_data(role, ctx);

        let frame_name = match role {
            AgentRole::Plan => "pa_init",
            AgentRole::Do => "da_input",
            AgentRole::Check => "ca_review",
            AgentRole::Act => "aa_decision",
        };

        if let Ok(projection_str) = self
            .projection
            .project(&ctx.task_iri, frame_name, HashMap::new())
            .await
        {
            if !projection_str.is_empty() {
                context_data.insert("context_summary".to_string(), projection_str);
            }
        }

        context_data
    }

    pub(super) fn build_agent_md(
        &self,
        role: AgentRole,
        objective: &str,
        context_data: &HashMap<String, String>,
        model: &str,
    ) -> String {
        let role_name = role.to_string();
        let role_lower = role_name.to_lowercase();
        let tools_list = self.tool_executor.read().list_tools(&role_name);

        let supports_reasoning = self.gateway.supports_native_reasoning(model);
        let _format_constraint = if supports_reasoning {
            LLM_RESPONSE_FORMAT_NO_THOUGHT
        } else {
            LLM_RESPONSE_FORMAT_WITH_THOUGHT
        };

        let mut vars: HashMap<String, serde_json::Value> = HashMap::new();
        vars.insert(
            "task_description".to_string(),
            serde_json::Value::String(objective.to_string()),
        );
        vars.insert(
            "available_skills".to_string(),
            serde_json::Value::String(tools_list.join(", ")),
        );
        vars.insert(
            "context_summary".to_string(),
            serde_json::Value::String(
                context_data
                    .get("context_summary")
                    .cloned()
                    .unwrap_or_default(),
            ),
        );
        vars.insert(
            "task_specific_constraints".to_string(),
            serde_json::Value::String(context_data.get("constraints").cloned().unwrap_or_default()),
        );
        vars.insert(
            "plan_content".to_string(),
            serde_json::Value::String(
                context_data
                    .get("plan_content")
                    .cloned()
                    .unwrap_or_else(|| "(filled by SA)".to_string()),
            ),
        );
        vars.insert(
            "execution_result".to_string(),
            serde_json::Value::String(
                context_data
                    .get("execution_result")
                    .cloned()
                    .unwrap_or_else(|| "(generated by DA)".to_string()),
            ),
        );
        vars.insert(
            "check_result".to_string(),
            serde_json::Value::String(
                context_data
                    .get("check_result")
                    .cloned()
                    .unwrap_or_else(|| "(generated by CA)".to_string()),
            ),
        );

        if let Some(ref loader) = self.prompt_loader {
            let result = loader.load(&role_lower, "skeleton", &vars);
            if !result.is_empty() {
                let md = format!("# {} Agent.md\n\n{}", role_name, result);
                debug!(role = %role_name, "=== agent.md (from PromptLoader) ===\n{}", md);
                return md;
            }
        }

        if let Ok(rendered) =
            self.templates
                .render_prompt(&role_lower, "skeleton", &vars, false, None)
        {
            let md = format!("# {} Agent.md\n\n{}\n", role_name, rendered,);
            debug!(role = %role_name, supports_reasoning = supports_reasoning, "=== agent.md (from template) ===\n{}", md);
            return md;
        }

        let role_prompt = match role {
            AgentRole::Plan => {
                let w2h_what = context_data.get("five_w2h_what").cloned().unwrap_or_else(|| "(not specified)".to_string());
                let w2h_why = context_data.get("five_w2h_why").cloned().unwrap_or_else(|| "(not specified)".to_string());
                let w2h_success = context_data.get("five_w2h_success_criteria").cloned().unwrap_or_else(|| "(not specified)".to_string());
                let w2h_deadline = context_data.get("five_w2h_deadline").cloned().unwrap_or_else(|| "(not specified)".to_string());
                let w2h_env = context_data.get("five_w2h_execution_env").cloned().unwrap_or_else(|| "(not specified)".to_string());
                format!("You are the Plan Agent (PA). Your responsibility is to analyze user tasks and create execution plans.\n\n🔴 Strictly Prohibited:\n1. Do not call write-operation tools (file_write, file_edit, etc.)\n2. Do not perform concrete work (create files, modify code, etc.)\n3. Do not use bash for write operations (e.g., writing files, installing packages, deleting)\n\n✅ Allowed Operations:\n1. You may call read-only tools to gather information (file_read, file_list, grep_search, etc.)\n2. You may use bash for read-only commands (e.g., ls, cat, grep, find, which, pwd, echo) to explore the environment\n3. Analyze user task requirements\n4. Create clear execution steps\n5. Output a JSON-formatted plan\n\n📋 Task Metadata (5W2H — Must Reference):\n- What: {}\n- Why: {}\n- Success Criteria: {}\n- Deadline: {}\n- Execution Environment: {}\n\nCreate a plan under the above metadata constraints. If you find information that needs to be supplemented, explain it in the plan.\n\nAfter planning, it is recommended to backfill the How and Where dimensions (optional):\n{{\"five_w2h_updates\": {{\"how\": {{\"planIRI\": \"Plan IRI\", \"preferredSkills\": [...], \"requiredSteps\": \"...\"}}, \"where\": {{\"dataSources\": [...], \"executionEnvironment\": \"...\"}}}}}}", w2h_what, w2h_why, w2h_success, w2h_deadline, w2h_env)
            }
            AgentRole::Do => "You are the Do Agent (DA). Your responsibility is to execute tasks concretely.\n\n🔴 Strictly Prohibited:\n1. Do not execute recursive searches in the current directory (e.g., grep -r, find /) — this will cause timeout\n2. Do not use relative paths; you must use the absolute paths specified in the task\n3. Do not perform operations unrelated to the task\n\n✅ Execution Requirements:\n1. Create/modify files strictly according to the paths specified in the task\n2. If the task requires creating a directory, create the directory first, then create the file\n3. Verify the result after every step\n4. Call finish immediately after completing the task\n5. For research tasks requiring the latest information, prioritize using web_search to fetch data. If the network tool still fails after multiple attempts, answer based on your own knowledge\n\n📋 Output Management Rules (Must Follow):\n1. When executing commands that may return large output (ls, find, grep, cat large files, etc.), use | head -N to limit output lines\n2. Prefer precise searches (grep + path restriction, glob filtering), avoid scanning entire directories\n3. When you only need to confirm a command result, use | grep keyword or | tail to filter key information — do not view the full output\n4. The system will automatically truncate output exceeding 16KB, and results over 2KB will be summarized — actively control output volume to avoid information loss\n5. If a tool returns results showing an \"output truncated\" or \"archived\" indicator, the output is too large — re-run with a more precise command\n\nExample Flow:\n1. Task requires creating /tmp/test/file.txt → First use Bash to create the directory, then use file_write to write\n2. Task requires modifying a file → Use file_read to read, process, then use file_write to write\n3. Task requires verification → Use file_read to read and check the content\n4. Search tool fails → After 1 attempt, if still failing, answer based on your own knowledge".to_string(),
            AgentRole::Check => {
                let w2h_what = context_data.get("five_w2h_what").cloned().unwrap_or_else(|| "(not specified)".to_string());
                let w2h_why = context_data.get("five_w2h_why").cloned().unwrap_or_else(|| "(not specified)".to_string());
                let w2h_deadline = context_data.get("five_w2h_deadline").cloned().unwrap_or_else(|| "(not specified)".to_string());
                let w2h_env = context_data.get("five_w2h_execution_env").cloned().unwrap_or_else(|| "(not specified)".to_string());
                let w2h_steps = context_data.get("five_w2h_required_steps").cloned().unwrap_or_else(|| "(not specified)".to_string());
                let w2h_budget = context_data.get("five_w2h_token_budget").cloned().unwrap_or_else(|| "(not specified)".to_string());
                format!("You are the Check Agent (CA). Your duty is to review execution results and ensure task objectives are met.\n\n🔴 Strictly Prohibited:\n1. Do not check or report any files/directories unrelated to the current task — even if other projects are found in the workspace, they must be ignored\n2. Do not include irrelevant content in audit reports — reports must focus solely on the current task objectives\n3. Do not explore directories that do not belong to the current task\n\n✅ Inspection Scope Limits:\n1. Only inspect files explicitly required to be created or modified by the current task\n2. If DA created unexpected files, only inspect them if they are relevant to the task\n3. Other projects/directories in the workspace (e.g., previous test outputs) are irrelevant to the task and must be ignored\n\n📋 Mandatory Verification Steps (MUST execute in order):\n1. Read the `## Original Task Requirements` section — this defines what the task ACTUALLY requires\n2. Read the `## Task Metadata (5W2H)` section — What/Why define the task objective\n3. Compare the execution results against the original task requirements — does the work done match what was requested?\n4. If the completed work addresses a DIFFERENT task or misses core requirements, return FAIL with specific evidence\n\n📋 Recommended Audit Reference (5W2H Dimensions — one of the critical dimensions to focus on):\n- What: {} — Has the task objective been achieved?\n- Why: {} — Does it satisfy the original intent?\n- When: {} — Is the deadline met?\n- Where: {} — Is it operating in the correct environment?\n- How: {} — Were the steps executed as planned?\n- HowMuch: {} — Are resources overspent?\n\nNote: 5W2H is one of the important analysis dimensions. You can add other audit perspectives based on the task nature (e.g., security, maintainability, performance, etc.).\n\n📋 Output Format:\nPlease output structured audit results including:\n1. Original task alignment: PASS/FAIL (is the work done matching what was requested?)\n2. Inspection conclusions per audit perspective (PASS/FAIL/CONDITIONAL + evidence)\n3. Overall conclusion (PASS/CONDITIONAL_PASS/FAIL)\n4. Issues found and recommendations", w2h_what, w2h_why, w2h_deadline, w2h_env, w2h_steps, w2h_budget)
            }
            AgentRole::Act => "You are the Decision Agent (AA), not an Execution Agent. Your sole duty is to make decisions based on the CA's audit results and provide disposition recommendations.\n\n🔴 Strictly Prohibited (must comply):\n1. Do not call file exploration tools such as glob_search, file_list, file_read, grep_search — your input comes only from CA audit results and task context\n2. Do not execute bash commands\n3. Do not proactively collect additional information — you are already the final decision layer and should not explore files on your own\n4. Do not process any files/directories mentioned in the CA audit results that are unrelated to the current task\n\n✅ Allowed Operations:\n1. Make decisions solely based on CA audit results and task context\n2. Output decision conclusion (task status + disposition recommendation + final summary)\n\n📋 Mandatory Verification Steps (MUST execute in order BEFORE making a decision):\n1. Read the `## Original Task Requirements` section — this is the ACTUAL task goal\n2. Read the `## Task Metadata (5W2H)` section — What/Why dimensions define the task objective\n3. Compare the execution results against the original task requirements:\n   - Does the completed work satisfy ALL requirements listed in Original Task Requirements?\n   - Are there any requirements that were NOT addressed?\n   - Does the completed work address a DIFFERENT task by mistake?\n4. If the work does NOT match the original task requirements, return status \"failed\" with a clear explanation of which requirements were missed or misaligned\n5. ONLY if ALL requirements are met, return status \"success\"\n\n📋 Decision Reference:\n- CA audit conclusion (already cross-checked against original task)\n- Task constraints (5W2H dimensions: What/Why/When/Where/How/HowMuch)\n- Task actual situation\n\n📋 Common Decision Paths (for reference only):\n- All audits passed AND original task requirements satisfied → Archive task, capture experience\n- Objective/intent not met → Return failed with specific gap description\n- Execution method/environment issue → Suggest plan correction\n- Time/resource overspent → Evaluate reasonableness, then decide to approve or downgrade\n\n📋 Output Format:\n1. Original task verification: PASS/FAIL (with evidence from requirements)\n2. CA audit verification: PASS/FAIL\n3. Task status: success / failed / partial_success\n4. Disposition recommendation: Specific action suggestion\n5. Final conclusion: Concise summary".to_string(),
        };

        let context_section = if context_data.is_empty() {
            String::new()
        } else {
            let mut sections = Vec::new();
            if let Some(original) = context_data.get("original_task") {
                sections.push(format!("## Original Task Requirements\n{}\n\n⚠️ Important: You must verify that all the above requirements have been completed.", original));
            }
            if let Some(plan) = context_data.get("plan_content") {
                sections.push(format!("## Superior Plan\n{}", plan));
            }
            if let Some(result) = context_data.get("execution_result") {
                sections.push(format!("## Execution Result\n{}", result));
            }
            if let Some(check) = context_data.get("check_result") {
                sections.push(format!("## Check Conclusion\n{}", check));
            }
            if let Some(ctx_summary) = context_data.get("context_summary") {
                sections.push(format!("## Related Context\n{}", ctx_summary));
            }
            if let Some(completed) = context_data.get("completed_steps") {
                sections.push(format!("## Completed Steps\n{}", completed));
            }
            if let Some(pending) = context_data.get("pending_steps") {
                sections.push(format!("## Pending Steps\n{}", pending));
            }
            let has_w2h = context_data.contains_key("five_w2h_what");
            if has_w2h {
                let mut w2h_lines = Vec::new();
                if let Some(v) = context_data.get("five_w2h_what") {
                    w2h_lines.push(format!("- What: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_why") {
                    w2h_lines.push(format!("- Why: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_success_criteria") {
                    w2h_lines.push(format!("- Success Criteria: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_deadline") {
                    w2h_lines.push(format!("- Deadline: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_execution_env") {
                    w2h_lines.push(format!("- Execution Environment: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_required_steps") {
                    w2h_lines.push(format!("- Required Steps: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_forbidden_tools") {
                    w2h_lines.push(format!("- Forbidden Tools: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_token_budget") {
                    w2h_lines.push(format!("- Token Budget: {}", v));
                }
                if let Some(v) = context_data.get("five_w2h_max_cycles") {
                    w2h_lines.push(format!("- Max Cycles: {}", v));
                }
                if !w2h_lines.is_empty() {
                    sections.push(format!("## Task Metadata (5W2H)\n{}", w2h_lines.join("\n")));
                }
            }
            sections.join("\n\n")
        };

        let md = format!(
            "# {} Agent.md\n\nRole: {}\nTask: {}\nWork Mode: {}\n\n{}\n\nImportant: After fulfilling your responsibility, directly output the final result without calling additional tools. Your response should include the complete conclusion or result.",
            role_name, role_name, objective, role_prompt, context_section
        );
        debug!(role = %role_name, "=== agent.md (fallback) ===\n{}", md);
        md
    }

    pub(super) fn build_readable_tool_menu(&self, role: &AgentRole) -> String {
        let role_str = role.to_string();
        let tool_defs = self
            .tool_executor
            .read()
            .tool_definitions_for_role(&role_str);

        if tool_defs.is_empty() {
            return String::new();
        }

        let os_hint = if cfg!(target_os = "windows") {
            "[Platform: Windows | bash tool actually uses PowerShell]"
        } else if cfg!(target_os = "macos") {
            "[Platform: macOS]"
        } else {
            "[Platform: Linux]"
        };
        let mut lines = vec![os_hint.to_string(), "Available tools list:".to_string()];
        for tool_def in &tool_defs {
            let name = tool_def["function"]["name"].as_str().unwrap_or("");
            let desc = tool_def["function"]["description"].as_str().unwrap_or("");
            if desc.is_empty() {
                lines.push(format!("- ID: {}", name));
            } else {
                lines.push(format!("- ID: {} | Purpose: {}", name, desc));
            }
        }
        lines.join("\n")
    }
}
