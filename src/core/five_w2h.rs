use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

// ── Dimension Audit types ──────────────────────────────────────────────

/// Per-dimension audit status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditStatus {
    Pass,
    Warning(String),
    Fail(String),
}

/// Result of auditing a single 5W2H dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionAuditResult {
    pub dimension: String,
    pub status: AuditStatus,
    pub evidence: String,
    pub details: Vec<String>,
}

/// Run a dimension-level audit of task results against the 5W2H specification.
///
/// Evaluates each of the 7 dimensions (what, why, who, when, where, how, how_much)
/// against the task result. Failed dimensions are recorded as `CausalObservation`s
/// on the optional `causal_engine` for automated root-cause analysis.
///
/// Returns a `Vec<DimensionAuditResult>` — one entry per dimension, each with
/// pass/fail/warning status and supporting evidence.
pub fn audit_dimensions(
    five_w2h: &Task5W2H,
    result: &crate::core::agent_runner::TaskResult,
    task_iri: &str,
    causal_engine: Option<&crate::causal::CausalEngine>,
) -> Vec<DimensionAuditResult> {
    use crate::causal::CausalObservation;
    let now = Utc::now();

    let mut results: Vec<DimensionAuditResult> = Vec::new();

    // ── What dimension: was the objective achieved? ──
    {
        let finished = result.status == "success" || result.status == "partial_success";
        let has_content = result.output.is_some() || !result.summary.is_empty();
        let detail = if finished && has_content {
            AuditStatus::Pass
        } else if result.status == "partial_success" {
            AuditStatus::Warning("Objective partially achieved".to_string())
        } else {
            if let Some(ce) = causal_engine {
                ce.record_observation(CausalObservation::new(
                    &format!("obs-{}-{}-what", task_iri, now.timestamp_millis()),
                    "iri://system/5w2h/what",
                    "TaskNotCompleted",
                    result.status.as_str(),
                ));
            }
            AuditStatus::Fail("Task was not completed successfully".to_string())
        };
        results.push(DimensionAuditResult {
            dimension: "what".to_string(),
            status: detail,
            evidence: format!("status={}, has_output={}, summary_len={}", result.status, has_content, result.summary.len()),
            details: vec![format!("Task status: {}", result.status)],
        });
    }

    // ── Why dimension: did we meet the success criteria? ──
    {
        let criteria = &five_w2h.why.success_criteria;
        if criteria.is_empty() {
            results.push(DimensionAuditResult {
                dimension: "why".to_string(),
                status: AuditStatus::Warning("No explicit success criteria".to_string()),
                evidence: "success_criteria is empty".to_string(),
                details: vec![],
            });
        } else {
            let matched: Vec<&String> = criteria.iter().filter(|c| result.summary.contains(c.as_str())).collect();
            let ratio = matched.len() as f64 / criteria.len().max(1) as f64;
            let detail = if ratio >= 0.8 {
                AuditStatus::Pass
            } else if ratio >= 0.5 {
                AuditStatus::Warning(format!("{:.0}% of success criteria met", ratio * 100.0))
            } else {
                if let Some(ce) = causal_engine {
                    for c in criteria.iter().filter(|c| !matched.contains(c)) {
                        let obs_id = format!("obs-{}-{}-why-{}", task_iri, now.timestamp_millis(), c.chars().take(16).collect::<String>());
                        ce.record_observation(CausalObservation::new(
                            &obs_id,
                            "iri://system/5w2h/why",
                            "SuccessCriteriaNotMet",
                            c,
                        ));
                    }
                }
                AuditStatus::Fail(format!("Only {:.0}% of success criteria met", ratio * 100.0))
            };
            results.push(DimensionAuditResult {
                dimension: "why".to_string(),
                status: detail,
                evidence: format!("{}/{} criteria matched", matched.len(), criteria.len()),
                details: criteria.iter().map(|c| format!("Criterion: {}", c)).collect(),
            });
        }
    }

    // ── Who dimension: was the execution role appropriate? ──
    {
        let detail = if let Some(ref who) = five_w2h.who {
            if let Some(ref required_role) = who.required_role {
                if result.status == "success" || result.status == "partial_success" {
                    AuditStatus::Pass
                } else {
                    AuditStatus::Warning(format!("Required role '{}' but execution had issues", required_role))
                }
            } else {
                AuditStatus::Pass
            }
        } else {
            AuditStatus::Pass
        };
        results.push(DimensionAuditResult {
            dimension: "who".to_string(),
            status: detail,
            evidence: format!("required_role={:?}", five_w2h.who.as_ref().and_then(|w| w.required_role.as_ref())),
            details: vec![],
        });
    }

    // ── When dimension: was the task completed on time? ──
    {
        let detail = if let Some(ref when) = five_w2h.when {
            if let Some(ref deadline) = when.deadline {
                if now > *deadline {
                    AuditStatus::Warning(format!("Deadline exceeded: deadline={}, now={}", deadline, now))
                } else {
                    AuditStatus::Pass
                }
            } else {
                AuditStatus::Pass
            }
        } else {
            AuditStatus::Pass
        };
        results.push(DimensionAuditResult {
            dimension: "when".to_string(),
            status: detail,
            evidence: format!("deadline={:?}", five_w2h.when.as_ref().and_then(|w| w.deadline)),
            details: vec![],
        });
    }

    // ── Where dimension: was the execution environment correct? ──
    {
        let detail = if let Some(ref where_detail) = five_w2h.where_ {
            if let Some(ref env) = where_detail.execution_environment {
                if let Ok(cwd) = std::env::current_dir() {
                    let cwd_str = cwd.to_string_lossy().to_string();
                    if !cwd_str.contains(env.trim_end_matches('/')) {
                        AuditStatus::Warning(format!("Execution environment mismatch: expected={}, actual={}", env, cwd_str))
                    } else {
                        AuditStatus::Pass
                    }
                } else {
                    AuditStatus::Pass
                }
            } else {
                AuditStatus::Pass
            }
        } else {
            AuditStatus::Pass
        };
        results.push(DimensionAuditResult {
            dimension: "where".to_string(),
            status: detail,
            evidence: format!("execution_environment={:?}", five_w2h.where_.as_ref().and_then(|w| w.execution_environment.as_ref())),
            details: vec![],
        });
    }

    // ── How dimension: was the approach quality acceptable? ──
    {
        let detail = match result.status.as_str() {
            "success" => AuditStatus::Pass,
            "partial_success" => AuditStatus::Warning("Task partially succeeded — approach had gaps".to_string()),
            s => {
                if let Some(ce) = causal_engine {
                    ce.record_observation(CausalObservation::new(
                        &format!("obs-{}-{}-how", task_iri, now.timestamp_millis()),
                        "iri://system/5w2h/how",
                        "ExecutionFailed",
                        s,
                    ));
                }
                AuditStatus::Fail(format!("Execution failed with status: {}", s))
            }
        };
        results.push(DimensionAuditResult {
            dimension: "how".to_string(),
            status: detail,
            evidence: format!("status={}, errors={}", result.status, result.errors.len()),
            details: result.errors.clone(),
        });
    }

    // ── HowMuch dimension: was the task within budget? ──
    {
        let detail = if let Some(ref hm) = five_w2h.how_much {
            let mut warnings = Vec::new();
            if let Some(ref budget) = hm.token_budget {
                let usage = result.tool_call_count as u64 * 1000; // rough estimate
                if usage > *budget {
                    warnings.push(format!("Token budget exceeded: budget={}, estimated={}", budget, usage));
                }
            }
            if let Some(ref max_cycles) = hm.max_pdca_cycles {
                if result.turn_count > *max_cycles {
                    warnings.push(format!("Turn count exceeded: max={}, actual={}", max_cycles, result.turn_count));
                }
            }
            if warnings.is_empty() {
                AuditStatus::Pass
            } else {
                AuditStatus::Warning(warnings.join("; "))
            }
        } else {
            AuditStatus::Pass
        };
        results.push(DimensionAuditResult {
            dimension: "how_much".to_string(),
            status: detail,
            evidence: format!("turns={}, tool_calls={}, errors={}", result.turn_count, result.tool_call_count, result.errors.len()),
            details: vec![],
        });
    }

    results
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Priority {
    High,
    Medium,
    Low,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum AccessLevel {
    Read,
    Write,
    Admin,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WhyDetail {
    pub description: String,
    pub success_criteria: Vec<String>,
    pub priority: Priority,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WhoDetail {
    pub requestor: Option<String>,
    pub assignees: Vec<String>,
    pub stakeholders: Vec<String>,
    pub required_role: Option<String>,
    pub access_level: Option<AccessLevel>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WhenDetail {
    pub deadline: Option<DateTime<Utc>>,
    pub start_after: Option<DateTime<Utc>>,
    pub estimated_duration: Option<String>,
    pub timezone: Option<String>,
    pub reminder_before: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WhereDetail {
    pub data_sources: Vec<String>,
    pub execution_environment: Option<String>,
    pub target_repository: Option<String>,
    pub target_branch: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct HowDetail {
    pub plan_iri: Option<String>,
    pub preferred_skills: Vec<String>,
    pub forbidden_tools: Vec<String>,
    pub required_steps: Option<String>,
    pub dependencies: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct HowMuchDetail {
    pub token_budget: Option<u64>,
    pub max_sub_agents: Option<u32>,
    pub max_pdca_cycles: Option<u32>,
    pub expected_quality: Option<f32>,
    pub actual_cost: Option<ActualCost>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ActualCost {
    pub tokens_used: u64,
    pub cycles_used: u32,
    pub duration_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FillStage {
    Create,
    Plan,
    Do,
    Check,
    Act,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionMeta {
    pub fill_stage: FillStage,
    pub filled_by: Option<String>,
    pub filled_at: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Task5W2H {
    pub what: String,
    pub why: WhyDetail,
    pub who: Option<WhoDetail>,
    pub when: Option<WhenDetail>,
    #[serde(rename = "where")]
    pub where_: Option<WhereDetail>,
    pub how: Option<HowDetail>,
    pub how_much: Option<HowMuchDetail>,
    pub dimension_meta: HashMap<String, DimensionMeta>,
    pub frozen: bool,
}

impl Task5W2H {
    pub fn new(what: &str, why_description: &str) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        let mut dimension_meta = HashMap::new();
        dimension_meta.insert("what".to_string(), DimensionMeta {
            fill_stage: FillStage::Create,
            filled_by: Some("SA".to_string()),
            filled_at: Some(now.clone()),
        });
        dimension_meta.insert("why".to_string(), DimensionMeta {
            fill_stage: FillStage::Create,
            filled_by: Some("SA".to_string()),
            filled_at: Some(now),
        });
        Self {
            what: what.to_string(),
            why: WhyDetail {
                description: why_description.to_string(),
                success_criteria: Vec::new(),
                priority: Priority::Medium,
            },
            who: None,
            when: None,
            where_: None,
            how: None,
            how_much: None,
            dimension_meta,
            frozen: false,
        }
    }

    pub fn with_who(mut self, who: WhoDetail) -> Self {
        self.who = Some(who);
        self
    }

    pub fn with_when(mut self, when: WhenDetail) -> Self {
        self.when = Some(when);
        self
    }

    pub fn with_where(mut self, where_: WhereDetail) -> Self {
        self.where_ = Some(where_);
        self
    }

    pub fn with_how(mut self, how: HowDetail) -> Self {
        self.how = Some(how);
        self
    }

    pub fn with_how_much(mut self, how_much: HowMuchDetail) -> Self {
        self.how_much = Some(how_much);
        self
    }

    pub fn record_fill(&mut self, dimension: &str, stage: FillStage, filled_by: &str) {
        self.dimension_meta.insert(dimension.to_string(), DimensionMeta {
            fill_stage: stage,
            filled_by: Some(filled_by.to_string()),
            filled_at: Some(chrono::Utc::now().to_rfc3339()),
        });
    }

    /// Merge structured five_w2h_updates from agent (typically PA) output into this Task5W2H.
    /// The JSON value is expected to have optional keys: "who", "when", "where", "how", "how_much".
    /// Each key's value follows the camelCase field names of the corresponding detail struct.
    /// Only non-null, non-empty values are applied.
    pub fn merge_updates(&mut self, updates: &serde_json::Value) {
        let obj = match updates.as_object() {
            Some(o) => o,
            None => return,
        };

        // ── who ──
        if let Some(who_val) = obj.get("who").and_then(|v| v.as_object()) {
            let requestor = who_val.get("requestor").and_then(|v| v.as_str()).map(String::from);
            let assignees = who_val.get("assignees").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let stakeholders = who_val.get("stakeholders").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let required_role = who_val.get("requiredRole").and_then(|v| v.as_str()).map(String::from);
            let access_level = who_val.get("accessLevel").and_then(|v| v.as_str())
                .and_then(|s| match s {
                    "Read" => Some(AccessLevel::Read),
                    "Write" => Some(AccessLevel::Write),
                    "Admin" => Some(AccessLevel::Admin),
                    _ => None,
                });
            self.who = Some(WhoDetail { requestor, assignees, stakeholders, required_role, access_level });
            self.record_fill("who", FillStage::Plan, "PA");
        }

        // ── when ──
        if let Some(when_val) = obj.get("when").and_then(|v| v.as_object()) {
            let deadline = when_val.get("deadline").and_then(|v| v.as_str())
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc));
            let start_after = when_val.get("startAfter").and_then(|v| v.as_str())
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc));
            let estimated_duration = when_val.get("estimatedDuration").and_then(|v| v.as_str()).map(String::from);
            let timezone = when_val.get("timezone").and_then(|v| v.as_str()).map(String::from);
            let reminder_before = when_val.get("reminderBefore").and_then(|v| v.as_str()).map(String::from);
            if deadline.is_some() || estimated_duration.is_some() {
                self.when = Some(WhenDetail { deadline, start_after, estimated_duration, timezone, reminder_before });
                self.record_fill("when", FillStage::Plan, "PA");
            }
        }

        // ── where ──
        if let Some(where_val) = obj.get("where").and_then(|v| v.as_object()) {
            let data_sources: Vec<String> = where_val.get("dataSources").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let execution_environment = where_val.get("executionEnvironment").and_then(|v| v.as_str()).map(String::from);
            let target_repository = where_val.get("targetRepository").and_then(|v| v.as_str()).map(String::from);
            let target_branch = where_val.get("targetBranch").and_then(|v| v.as_str()).map(String::from);
            if !data_sources.is_empty() || execution_environment.is_some() {
                self.where_ = Some(WhereDetail { data_sources, execution_environment, target_repository, target_branch });
                self.record_fill("where", FillStage::Plan, "PA");
            }
        }

        // ── how ──
        if let Some(how_val) = obj.get("how").and_then(|v| v.as_object()) {
            let plan_iri = how_val.get("planIRI").and_then(|v| v.as_str()).map(String::from);
            let preferred_skills = how_val.get("preferredSkills").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let forbidden_tools = how_val.get("forbiddenTools").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let required_steps = how_val.get("requiredSteps").and_then(|v| v.as_str()).map(String::from);
            let dependencies = how_val.get("dependencies").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            self.how = Some(HowDetail { plan_iri, preferred_skills, forbidden_tools, required_steps, dependencies });
            self.record_fill("how", FillStage::Plan, "PA");
        }

        // ── how_much ──
        if let Some(hm_val) = obj.get("howMuch").or_else(|| obj.get("how_much")).and_then(|v| v.as_object()) {
            let token_budget = hm_val.get("tokenBudget").and_then(|v| v.as_u64());
            let max_sub_agents = hm_val.get("maxSubAgents").and_then(|v| v.as_u64()).map(|n| n as u32);
            let max_pdca_cycles = hm_val.get("maxPdcaCycles").and_then(|v| v.as_u64()).map(|n| n as u32);
            let expected_quality = hm_val.get("expectedQuality").and_then(|v| v.as_f64()).map(|n| n as f32);
            self.how_much = Some(HowMuchDetail {
                token_budget, max_sub_agents, max_pdca_cycles,
                expected_quality, actual_cost: None,
            });
            self.record_fill("how_much", FillStage::Plan, "PA");
        }
    }

    /// Fill remaining missing dimensions with SA-level defaults.
    /// This is called AFTER merge_updates(), so only truly unknown dimensions get defaults.
    pub fn derive_defaults(&mut self, _max_iterations: u32, max_pdca_cycles: u32) {
        if self.who.is_none() {
            self.who = Some(WhoDetail {
                requestor: None,
                assignees: vec![],
                stakeholders: vec![],
                required_role: None,
                access_level: None,
            });
            self.record_fill("who", FillStage::Plan, "SA-Default");
        }
        if self.where_.is_none() {
            if let Ok(cwd) = std::env::current_dir() {
                self.where_ = Some(WhereDetail {
                    data_sources: vec![],
                    execution_environment: Some(cwd.to_string_lossy().to_string()),
                    target_repository: None,
                    target_branch: None,
                });
                self.record_fill("where", FillStage::Plan, "SA-Default");
            }
        }
        if self.how.is_none() {
            self.how = Some(HowDetail {
                plan_iri: None,
                preferred_skills: vec![],
                forbidden_tools: vec![],
                required_steps: None,
                dependencies: vec![],
            });
            self.record_fill("how", FillStage::Plan, "SA-Default");
        }
        if self.how_much.is_none() {
            self.how_much = Some(HowMuchDetail {
                token_budget: None,
                max_sub_agents: None,
                max_pdca_cycles: Some(max_pdca_cycles),
                expected_quality: None,
                actual_cost: None,
            });
            self.record_fill("how_much", FillStage::Plan, "SA-Default");
        }
    }

    /// Check which required dimensions are missing for the given task level.
    /// Only checks dimension_meta (whether dimension has been filled/recorded).
    pub fn check_completeness(&self, task_level: &str) -> Vec<String> {
        let all_dims = vec!["what", "why", "who", "when", "where", "how", "how_much"];
        let required: Vec<&str> = match task_level {
            "Instant" => vec!["what"],
            "Simple" => vec!["what", "why"],
            "Standard" | "Complex" => all_dims.clone(),
            _ => vec!["what", "why"],
        };
        required.into_iter().filter(|d| !self.dimension_meta.contains_key(*d)).map(String::from).collect()
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    pub fn is_minimal_set_ready(&self) -> bool {
        !self.what.is_empty() && !self.why.description.is_empty()
    }

    pub fn derive_objective(&self) -> String {
        self.what.clone()
    }

    pub fn to_json_ld(&self, task_iri: &str) -> Result<Value, String> {
        let stripped = task_iri.trim_start_matches("iri://task/");
        let id = format!("iri://task/{}/5w2h", stripped);

        let mut result = json!({
            "@context": {
                "task": "https://pdca-agent.org/ontology/task#"
            },
            "@id": id,
            "@type": "task:5W2H",
            "task:what": self.what,
            "task:why": {
                "task:description": self.why.description,
                "task:successCriteria": self.why.success_criteria,
                "task:priority": format!("{:?}", self.why.priority)
            }
        });

        let map = result.as_object_mut()
            .expect("json!({}) always creates an object");

        if let Some(ref who) = self.who {
            map.insert(
                "task:who".to_string(),
                json!({
                    "task:requestor": who.requestor,
                    "task:assignees": who.assignees,
                    "task:stakeholders": who.stakeholders,
                    "task:requiredRole": who.required_role,
                    "task:accessLevel": who.access_level.as_ref().map(|al| format!("{:?}", al))
                }),
            );
        }

        if let Some(ref when) = self.when {
            map.insert(
                "task:when".to_string(),
                json!({
                    "task:deadline": when.deadline.map(|dt| dt.to_rfc3339()),
                    "task:startAfter": when.start_after.map(|dt| dt.to_rfc3339()),
                    "task:estimatedDuration": when.estimated_duration,
                    "task:timezone": when.timezone,
                    "task:reminderBefore": when.reminder_before
                }),
            );
        }

        if let Some(ref where_) = self.where_ {
            map.insert(
                "task:where".to_string(),
                json!({
                    "task:dataSources": where_.data_sources,
                    "task:executionEnvironment": where_.execution_environment,
                    "task:targetRepository": where_.target_repository,
                    "task:targetBranch": where_.target_branch
                }),
            );
        }

        if let Some(ref how) = self.how {
            map.insert(
                "task:how".to_string(),
                json!({
                    "task:planIri": how.plan_iri,
                    "task:preferredSkills": how.preferred_skills,
                    "task:forbiddenTools": how.forbidden_tools,
                    "task:requiredSteps": how.required_steps,
                    "task:dependencies": how.dependencies
                }),
            );
        }

        if let Some(ref how_much) = self.how_much {
            let mut hm = json!({
                "task:tokenBudget": how_much.token_budget,
                "task:maxSubAgents": how_much.max_sub_agents,
                "task:maxPdcaCycles": how_much.max_pdca_cycles,
                "task:expectedQuality": how_much.expected_quality
            });
            if let Some(ref cost) = how_much.actual_cost {
                hm.as_object_mut()
                    .expect("json!({}) always creates an object")
                    .insert(
                    "task:actualCost".to_string(),
                    json!({
                        "task:tokensUsed": cost.tokens_used,
                        "task:cyclesUsed": cost.cycles_used,
                        "task:durationSecs": cost.duration_secs
                    }),
                );
            }
            map.insert("task:howMuch".to_string(), hm);
        }

        map.insert(
            "task:frozen".to_string(),
            json!(self.frozen),
        );

        let mut meta_map = serde_json::Map::new();
        for (dim, meta) in &self.dimension_meta {
            meta_map.insert(dim.clone(), json!({
                "task:fillStage": format!("{:?}", meta.fill_stage),
                "task:filledBy": meta.filled_by,
                "task:filledAt": meta.filled_at
            }));
        }
        map.insert(
            "task:dimensionMeta".to_string(),
            Value::Object(meta_map),
        );

        Ok(result)
    }

    pub fn from_json_ld(value: &Value) -> Result<Self, String> {
        let obj = value
            .as_object()
            .ok_or("JSON-LD value is not an object")?;

        let what = obj
            .get("task:what")
            .and_then(|v| v.as_str())
            .ok_or("Missing task:what")?
            .to_string();

        let why_obj = obj
            .get("task:why")
            .and_then(|v| v.as_object())
            .ok_or("Missing task:why")?;

        let why = WhyDetail {
            description: why_obj
                .get("task:description")
                .and_then(|v| v.as_str())
                .ok_or("Missing task:description in why")?
                .to_string(),
            success_criteria: why_obj
                .get("task:successCriteria")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            priority: why_obj
                .get("task:priority")
                .and_then(|v| v.as_str())
                .and_then(|s| match s {
                    "High" => Some(Priority::High),
                    "Medium" => Some(Priority::Medium),
                    "Low" => Some(Priority::Low),
                    _ => None,
                })
                .ok_or("Invalid task:priority")?,
        };

        let who = obj
            .get("task:who")
            .and_then(|v| v.as_object())
            .map(|who_obj| WhoDetail {
                requestor: who_obj
                    .get("task:requestor")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                assignees: who_obj
                    .get("task:assignees")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
                stakeholders: who_obj
                    .get("task:stakeholders")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
                required_role: who_obj
                    .get("task:requiredRole")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                access_level: who_obj
                    .get("task:accessLevel")
                    .and_then(|v| v.as_str())
                    .and_then(|s| match s {
                        "Read" => Some(AccessLevel::Read),
                        "Write" => Some(AccessLevel::Write),
                        "Admin" => Some(AccessLevel::Admin),
                        _ => None,
                    }),
            });

        let when = obj
            .get("task:when")
            .and_then(|v| v.as_object())
            .map(|when_obj| WhenDetail {
                deadline: when_obj
                    .get("task:deadline")
                    .and_then(|v| v.as_str())
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
                start_after: when_obj
                    .get("task:startAfter")
                    .and_then(|v| v.as_str())
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
                estimated_duration: when_obj
                    .get("task:estimatedDuration")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                timezone: when_obj
                    .get("task:timezone")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                reminder_before: when_obj
                    .get("task:reminderBefore")
                    .and_then(|v| v.as_str())
                    .map(String::from),
            });

        let where_ = obj
            .get("task:where")
            .and_then(|v| v.as_object())
            .map(|where_obj| WhereDetail {
                data_sources: where_obj
                    .get("task:dataSources")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
                execution_environment: where_obj
                    .get("task:executionEnvironment")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                target_repository: where_obj
                    .get("task:targetRepository")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                target_branch: where_obj
                    .get("task:targetBranch")
                    .and_then(|v| v.as_str())
                    .map(String::from),
            });

        let how = obj
            .get("task:how")
            .and_then(|v| v.as_object())
            .map(|how_obj| HowDetail {
                plan_iri: how_obj
                    .get("task:planIri")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                preferred_skills: how_obj
                    .get("task:preferredSkills")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
                forbidden_tools: how_obj
                    .get("task:forbiddenTools")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
                required_steps: how_obj
                    .get("task:requiredSteps")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                dependencies: how_obj
                    .get("task:dependencies")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
            });

        let how_much = obj
            .get("task:howMuch")
            .and_then(|v| v.as_object())
            .map(|hm_obj| HowMuchDetail {
                token_budget: hm_obj
                    .get("task:tokenBudget")
                    .and_then(|v| v.as_u64()),
                max_sub_agents: hm_obj
                    .get("task:maxSubAgents")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32),
                max_pdca_cycles: hm_obj
                    .get("task:maxPdcaCycles")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32),
                expected_quality: hm_obj
                    .get("task:expectedQuality")
                    .and_then(|v| v.as_f64())
                    .map(|n| n as f32),
                actual_cost: hm_obj
                    .get("task:actualCost")
                    .and_then(|v| v.as_object())
                    .map(|cost_obj| ActualCost {
                        tokens_used: cost_obj
                            .get("task:tokensUsed")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                        cycles_used: cost_obj
                            .get("task:cyclesUsed")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32,
                        duration_secs: cost_obj
                            .get("task:durationSecs")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0),
                    }),
            });

        let frozen = obj
            .get("task:frozen")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let dimension_meta = obj
            .get("task:dimensionMeta")
            .and_then(|v| v.as_object())
            .map(|meta_obj| {
                let mut map = HashMap::new();
                for (dim, val) in meta_obj {
                    if let Some(meta_val) = val.as_object() {
                        let fill_stage = meta_val
                            .get("task:fillStage")
                            .and_then(|v| v.as_str())
                            .and_then(|s| match s {
                                "Create" => Some(FillStage::Create),
                                "Plan" => Some(FillStage::Plan),
                                "Do" => Some(FillStage::Do),
                                "Check" => Some(FillStage::Check),
                                "Act" => Some(FillStage::Act),
                                _ => None,
                            })
                            .unwrap_or(FillStage::Create);
                        let filled_by = meta_val
                            .get("task:filledBy")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        let filled_at = meta_val
                            .get("task:filledAt")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        map.insert(dim.clone(), DimensionMeta {
                            fill_stage,
                            filled_by,
                            filled_at,
                        });
                    }
                }
                map
            })
            .unwrap_or_default();

        Ok(Self {
            what,
            why,
            who,
            when,
            where_,
            how,
            how_much,
            dimension_meta,
            frozen,
        })
    }
}

impl Default for Task5W2H {
    fn default() -> Self {
        Self {
            what: String::new(),
            why: WhyDetail {
                description: String::new(),
                success_criteria: Vec::new(),
                priority: Priority::Medium,
            },
            who: None,
            when: None,
            where_: None,
            how: None,
            how_much: None,
            dimension_meta: HashMap::new(),
            frozen: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_5w2h_lifecycle_progressive_filling() {
        let mut w2h = Task5W2H::new("Create Web service", "Provide REST API");
        assert!(w2h.is_minimal_set_ready());
        assert!(w2h.how.is_none());
        assert!(w2h.where_.is_none());

        w2h = w2h.with_how(HowDetail {
            plan_iri: Some("iri://plan/web-service".to_string()),
            preferred_skills: vec!["file_read".to_string(), "file_write".to_string()],
            forbidden_tools: vec!["bash".to_string()],
            required_steps: Some("1. Design API 2. Implement routes 3. Test".to_string()),
            dependencies: vec![],
        }).with_where(WhereDetail {
            data_sources: vec!["src/api/".to_string()],
            execution_environment: Some("sandbox".to_string()),
            target_repository: None,
            target_branch: None,
        });
        assert!(w2h.how.is_some());
        assert!(w2h.where_.is_some());

        w2h = w2h.with_how_much(HowMuchDetail {
            token_budget: Some(100000),
            max_sub_agents: None,
            max_pdca_cycles: Some(3),
            expected_quality: Some(0.85),
            actual_cost: Some(ActualCost {
                tokens_used: 45000,
                cycles_used: 1,
                duration_secs: 120.0,
            }),
        });
        assert!(w2h.how_much.is_some());
        let actual = w2h.how_much.as_ref().unwrap().actual_cost.as_ref().unwrap();
        assert_eq!(actual.tokens_used, 45000);
    }

    #[test]
    fn test_5w2h_json_ld_with_all_dimensions() {
        let w2h = Task5W2H::new("Full test", "Verify full 5W2H")
            .with_who(WhoDetail {
                requestor: Some("user:test".to_string()),
                assignees: vec!["agent:pa".to_string(), "agent:da".to_string()],
                stakeholders: vec!["user:pm".to_string()],
                required_role: Some("Do".to_string()),
                access_level: Some(AccessLevel::Write),
            })
            .with_when(WhenDetail {
                deadline: Some("2026-12-31T23:59:59Z".parse().unwrap()),
                start_after: Some("2026-06-01T00:00:00Z".parse().unwrap()),
                estimated_duration: Some("30d".to_string()),
                timezone: Some("Asia/Shanghai".to_string()),
                reminder_before: None,
            })
            .with_where(WhereDetail {
                data_sources: vec!["db/main".to_string(), "api/v1".to_string()],
                execution_environment: Some("production".to_string()),
                target_repository: Some("https://github.com/test/repo".to_string()),
                target_branch: Some("main".to_string()),
            })
            .with_how(HowDetail {
                plan_iri: Some("iri://plan/full-test".to_string()),
                preferred_skills: vec!["file_read".to_string(), "code_execute".to_string()],
                forbidden_tools: vec!["bash".to_string()],
                required_steps: Some("Step 1 Step 2 Step 3".to_string()),
                dependencies: vec!["dep1".to_string()],
            })
            .with_how_much(HowMuchDetail {
                token_budget: Some(200000),
                max_sub_agents: Some(5),
                max_pdca_cycles: Some(4),
                expected_quality: Some(0.95),
                actual_cost: Some(ActualCost {
                    tokens_used: 80000,
                    cycles_used: 2,
                    duration_secs: 360.0,
                }),
            });

        let json_ld = w2h.to_json_ld("full-test-001").unwrap();
        assert_eq!(json_ld["@type"], "task:5W2H");
        assert_eq!(json_ld["@id"], "iri://task/full-test-001/5w2h");
        assert!(json_ld.get("task:who").is_some());
        assert!(json_ld.get("task:when").is_some());
        assert!(json_ld.get("task:where").is_some());
        assert!(json_ld.get("task:how").is_some());
        assert!(json_ld.get("task:howMuch").is_some());
        assert!(json_ld.get("task:frozen").is_some());
        assert!(json_ld.get("task:dimensionMeta").is_some());

        let restored = Task5W2H::from_json_ld(&json_ld).unwrap();
        assert_eq!(restored.what, "Full test");
        assert_eq!(restored.who.unwrap().access_level, Some(AccessLevel::Write));
        assert_eq!(restored.how_much.unwrap().actual_cost.unwrap().tokens_used, 80000);
        assert!(!restored.frozen);
        assert!(restored.dimension_meta.contains_key("what"));
        assert!(restored.dimension_meta.contains_key("why"));
    }

    #[test]
    fn test_task5w2h_new_minimal_set() {
        let w2h = Task5W2H::new("Create Rust project", "Learn Rust language");
        assert_eq!(w2h.what, "Create Rust project");
        assert_eq!(w2h.why.description, "Learn Rust language");
        assert!(w2h.is_minimal_set_ready());
        assert_eq!(w2h.derive_objective(), "Create Rust project");
        assert!(w2h.who.is_none());
        assert!(w2h.when.is_none());
        assert!(w2h.where_.is_none());
        assert!(w2h.how.is_none());
        assert!(w2h.how_much.is_none());
        assert!(!w2h.frozen);
        assert!(w2h.dimension_meta.contains_key("what"));
        assert!(w2h.dimension_meta.contains_key("why"));
        assert_eq!(w2h.dimension_meta.get("what").unwrap().fill_stage, FillStage::Create);
    }

    #[test]
    fn test_task5w2h_builder_pattern() {
        let w2h = Task5W2H::new("Refactor module", "Improve code quality")
            .with_who(WhoDetail {
                requestor: Some("user:1".to_string()),
                assignees: vec!["agent:1".to_string()],
                stakeholders: vec![],
                required_role: Some("Do".to_string()),
                access_level: Some(AccessLevel::Write),
            })
            .with_when(WhenDetail {
                deadline: Some("2026-06-01T00:00:00Z".parse().unwrap()),
                start_after: None,
                estimated_duration: Some("2h".to_string()),
                timezone: Some("UTC".to_string()),
                reminder_before: None,
            })
            .with_where(WhereDetail {
                data_sources: vec!["src/".to_string()],
                execution_environment: Some("sandbox".to_string()),
                target_repository: None,
                target_branch: None,
            })
            .with_how(HowDetail {
                plan_iri: Some("iri://plan/1".to_string()),
                preferred_skills: vec!["file_read".to_string()],
                forbidden_tools: vec!["bash".to_string()],
                required_steps: Some("1. Read 2. Analyze 3. Refactor".to_string()),
                dependencies: vec![],
            })
            .with_how_much(HowMuchDetail {
                token_budget: Some(50000),
                max_sub_agents: Some(3),
                max_pdca_cycles: Some(2),
                expected_quality: Some(0.9),
                actual_cost: None,
            });
        assert!(w2h.who.is_some());
        assert!(w2h.when.is_some());
        assert!(w2h.where_.is_some());
        assert!(w2h.how.is_some());
        assert!(w2h.how_much.is_some());
        assert_eq!(w2h.how.as_ref().unwrap().forbidden_tools, vec!["bash"]);
        assert_eq!(w2h.how_much.as_ref().unwrap().token_budget, Some(50000));
    }

    #[test]
    fn test_task5w2h_json_ld_roundtrip() {
        let original = Task5W2H::new("Test task", "Verify JSON-LD roundtrip")
            .with_when(WhenDetail {
                deadline: Some("2026-06-01T00:00:00Z".parse().unwrap()),
                start_after: None,
                estimated_duration: None,
                timezone: None,
                reminder_before: None,
            })
            .with_how_much(HowMuchDetail {
                token_budget: Some(100000),
                max_sub_agents: None,
                max_pdca_cycles: Some(3),
                expected_quality: None,
                actual_cost: None,
            });
        let json_ld = original.to_json_ld("test-123").unwrap();
        assert_eq!(json_ld["@type"], "task:5W2H");
        assert_eq!(json_ld["task:what"], "Test task");
        assert_eq!(json_ld["@id"], "iri://task/test-123/5w2h");
        let restored = Task5W2H::from_json_ld(&json_ld).unwrap();
        assert_eq!(restored.what, original.what);
        assert_eq!(restored.why.description, original.why.description);
        assert!(restored.when.is_some());
        assert!(restored.how_much.is_some());
    }

    #[test]
    fn test_task5w2h_minimal_set_validation() {
        let valid = Task5W2H::new("Task", "Purpose");
        assert!(valid.is_minimal_set_ready());
        let mut empty_what = valid.clone();
        empty_what.what = String::new();
        assert!(!empty_what.is_minimal_set_ready());
        let mut empty_why = valid.clone();
        empty_why.why.description = String::new();
        assert!(!empty_why.is_minimal_set_ready());
    }

    #[test]
    fn test_priority_access_level_serialization() {
        assert_eq!(serde_json::to_string(&Priority::High).unwrap(), "\"High\"");
        assert_eq!(serde_json::to_string(&Priority::Medium).unwrap(), "\"Medium\"");
        assert_eq!(serde_json::to_string(&Priority::Low).unwrap(), "\"Low\"");
        assert_eq!(serde_json::to_string(&AccessLevel::Read).unwrap(), "\"Read\"");
        assert_eq!(serde_json::to_string(&AccessLevel::Write).unwrap(), "\"Write\"");
        assert_eq!(serde_json::to_string(&AccessLevel::Admin).unwrap(), "\"Admin\"");
    }

    #[test]
    fn test_actual_cost_structure() {
        let cost = ActualCost {
            tokens_used: 15000,
            cycles_used: 2,
            duration_secs: 45.5,
        };
        let json = serde_json::to_string(&cost).unwrap();
        let restored: ActualCost = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.tokens_used, 15000);
        assert_eq!(restored.cycles_used, 2);
        assert!((restored.duration_secs - 45.5).abs() < 0.001);
    }

    #[test]
    fn test_record_fill() {
        let mut w2h = Task5W2H::new("Task", "Reason");
        assert_eq!(w2h.dimension_meta.len(), 2);
        w2h.record_fill("who", FillStage::Plan, "PA");
        assert!(w2h.dimension_meta.contains_key("who"));
        assert_eq!(w2h.dimension_meta.get("who").unwrap().fill_stage, FillStage::Plan);
        assert_eq!(w2h.dimension_meta.get("who").unwrap().filled_by, Some("PA".to_string()));
        assert!(w2h.dimension_meta.get("who").unwrap().filled_at.is_some());
    }

    #[test]
    fn test_check_completeness() {
        let w2h = Task5W2H::new("Task", "Reason");
        let missing_instant = w2h.check_completeness("Instant");
        assert!(missing_instant.is_empty());
        let missing_simple = w2h.check_completeness("Simple");
        assert!(missing_simple.is_empty());
        let missing_standard = w2h.check_completeness("Standard");
        assert!(missing_standard.contains(&"who".to_string()));
        assert!(missing_standard.contains(&"when".to_string()));
        assert!(missing_standard.contains(&"where".to_string()));
        assert!(missing_standard.contains(&"how".to_string()));
        assert!(missing_standard.contains(&"how_much".to_string()));
    }

    #[test]
    fn test_freeze() {
        let mut w2h = Task5W2H::new("Task", "Reason");
        assert!(!w2h.frozen);
        w2h.freeze();
        assert!(w2h.frozen);
    }

    #[test]
    fn test_reminder_before_json_ld_roundtrip() {
        let w2h = Task5W2H::new("Reminder test", "Verify reminder")
            .with_when(WhenDetail {
                deadline: Some("2026-12-31T23:59:59Z".parse().unwrap()),
                start_after: None,
                estimated_duration: None,
                timezone: None,
                reminder_before: Some("1h".to_string()),
            });
        let json_ld = w2h.to_json_ld("reminder-001").unwrap();
        let when_obj = json_ld.get("task:when").unwrap().as_object().unwrap();
        assert_eq!(when_obj.get("task:reminderBefore").unwrap().as_str(), Some("1h"));
        let restored = Task5W2H::from_json_ld(&json_ld).unwrap();
        assert_eq!(restored.when.unwrap().reminder_before, Some("1h".to_string()));
    }
}
