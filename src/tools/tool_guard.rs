use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::tools::hooks::{FunctionHook, HookContext, HookManager, HookPoint, HookResult};

/// Global shared audit log accessible to HTTP endpoints.
pub static GUARD_AUDIT_LOG: Lazy<Arc<RwLock<Vec<GuardAuditEntry>>>> =
    Lazy::new(|| Arc::new(RwLock::new(Vec::new())));

// ─── Tool Category ───

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolCategory {
    FileRead,
    FileWrite,
    Search,
    CodeExecution,
    KnowledgeGraph,
    KnowledgeExtract,
    HttpRequest,
    Meta,
}

// ─── Enforcement Level ───

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnforcementLevel {
    Must,
    Should,
    Info,
}

// ─── Rule Structs ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreInjectionRule {
    pub enforcement: EnforcementLevel,
    pub instruction: String,
    pub tool_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub validator: String,
    pub params: HashMap<String, Value>,
    pub fix_instruction: String,
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardAuditEntry {
    pub timestamp: i64,
    pub tool_name: String,
    pub agent_id: String,
    pub pre_injected: bool,
    pub validation_passed: bool,
    pub retry_count: u32,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardStats {
    pub total_checks: usize,
    pub passed_checks: usize,
    pub failed_checks: usize,
    pub pass_rate: f64,
}

impl Default for GuardStats {
    fn default() -> Self {
        Self {
            total_checks: 0,
            passed_checks: 0,
            failed_checks: 0,
            pass_rate: 1.0,
        }
    }
}

// ─── External Config (guard_rules.json) ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryRules {
    #[serde(default)]
    pub pre_injections: Vec<PreInjectionRule>,
    #[serde(default)]
    pub validations: Vec<ValidationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardRulesConfig {
    pub categories: HashMap<String, CategoryRules>,
}

impl GuardRulesConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: GuardRulesConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
}

// ─── Validation Outcome ───

#[derive(Debug, Clone)]
pub enum ValidationOutcome {
    Pass,
    Warn(String),
    Fail(String),
}

// ─── ToolGuard ───

/// State per file for cumulative read tracking.
struct FileCoverage {
    /// Merged non-overlapping line ranges that have been read so far.
    ranges: Vec<(usize, usize)>,
    /// Number of file_read attempts for this file. Reset when coverage >= 95%.
    attempt_count: u32,
    /// Total lines in the file (captured on first read).
    total_lines: usize,
}

#[derive(Clone)]
pub struct ToolGuard {
    pre_injections: Arc<RwLock<HashMap<ToolCategory, Vec<PreInjectionRule>>>>,
    validations: Arc<RwLock<HashMap<ToolCategory, Vec<ValidationRule>>>>,
    tool_categories: HashMap<String, ToolCategory>,
    audit_log: Arc<RwLock<Vec<GuardAuditEntry>>>,
    config_path: Arc<RwLock<Option<String>>>,
    /// Per-file cumulative read tracking with attempt limit (max 3 per file).
    file_coverage: Arc<Mutex<HashMap<String, FileCoverage>>>,
    /// Optional stale file check callback: returns Some(warning) if file is stale.
    stale_check: Arc<RwLock<Option<Arc<dyn Fn(&str) -> Option<String> + Send + Sync>>>>,
}

impl ToolGuard {
    pub fn new() -> Self {
        let guard = Self {
            pre_injections: Arc::new(RwLock::new(HashMap::new())),
            validations: Arc::new(RwLock::new(HashMap::new())),
            tool_categories: Self::default_tool_categories(),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            config_path: Arc::new(RwLock::new(None)),
            file_coverage: Arc::new(Mutex::new(HashMap::new())),
            stale_check: Arc::new(RwLock::new(None)),
        };
        guard.load_default_rules();
        guard
    }

    /// Create ToolGuard with rules loaded from a JSON config file.
    /// Categories present in the JSON file replace defaults; absent categories keep defaults.
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let config = GuardRulesConfig::from_file(path.as_ref())?;
        let guard = Self {
            pre_injections: Arc::new(RwLock::new(HashMap::new())),
            validations: Arc::new(RwLock::new(HashMap::new())),
            tool_categories: Self::default_tool_categories(),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            config_path: Arc::new(RwLock::new(Some(
                path.as_ref().to_string_lossy().to_string(),
            ))),
            file_coverage: Arc::new(Mutex::new(HashMap::new())),
            stale_check: Arc::new(RwLock::new(None)),
        };
        {
            let mut pre = guard.pre_injections.write();
            let mut val = guard.validations.write();
            for (cat_str, rules) in &config.categories {
                if let Ok(category) = serde_json::from_value::<ToolCategory>(json!(cat_str)) {
                    if !rules.pre_injections.is_empty() {
                        pre.insert(category.clone(), rules.pre_injections.clone());
                    }
                    if !rules.validations.is_empty() {
                        val.insert(category, rules.validations.clone());
                    }
                }
            }
        }
        Ok(guard)
    }

    fn default_tool_categories() -> HashMap<String, ToolCategory> {
        let mut map = HashMap::new();
        map.insert("file_read".to_string(), ToolCategory::FileRead);
        map.insert("file_list".to_string(), ToolCategory::FileRead);
        map.insert("file_write".to_string(), ToolCategory::FileWrite);
        map.insert("file_edit".to_string(), ToolCategory::FileWrite);
        map.insert("grep_search".to_string(), ToolCategory::Search);
        map.insert("glob_search".to_string(), ToolCategory::Search);
        map.insert("bash".to_string(), ToolCategory::CodeExecution);
        map.insert("powershell".to_string(), ToolCategory::CodeExecution);
        map.insert("code_execute".to_string(), ToolCategory::CodeExecution);
        map.insert("knowledge_query".to_string(), ToolCategory::KnowledgeGraph);
        map.insert(
            "knowledge_neighbors".to_string(),
            ToolCategory::KnowledgeGraph,
        );
        map.insert("kg_search".to_string(), ToolCategory::KnowledgeGraph);
        map.insert(
            "knowledge_extract".to_string(),
            ToolCategory::KnowledgeExtract,
        );
        map.insert("web_fetch".to_string(), ToolCategory::HttpRequest);
        map.insert("web_search".to_string(), ToolCategory::HttpRequest);
        map.insert("http_request".to_string(), ToolCategory::HttpRequest);
        map.insert("create_skill".to_string(), ToolCategory::Meta);
        map.insert("convert_skill".to_string(), ToolCategory::Meta);
        map
    }

    fn load_default_rules(&self) {
        let mut pre = self.pre_injections.write();
        let mut val = self.validations.write();
        // ── FileRead ──
        pre.insert(
            ToolCategory::FileRead,
            vec![
                PreInjectionRule {
                    enforcement: EnforcementLevel::Must,
                    instruction: "When reading files, only read content directly relevant to the current task. \
                        If file content has already been obtained through other means (e.g. read_full_result micro-tool, \
                        previous file_read calls, etc.), do NOT re-read it. \
                        Do NOT recursively read all referenced files based on import/use/include declarations — \
                        only read referenced files when you are certain they are directly relevant to the current task."
                        .to_string(),
                    tool_names: vec![],
                },
                PreInjectionRule {
                    enforcement: EnforcementLevel::Must,
                    instruction: "Only read files relevant to the current task. \
                        Do NOT read project source code, node_modules, target, .git or other directories and files unrelated to the task. \
                        If file_list results contain irrelevant content, ignore it."
                        .to_string(),
                    tool_names: vec![],
                },
            ],
        );
        val.insert(
            ToolCategory::FileRead,
            vec![ValidationRule {
                validator: "file_length_check".to_string(),
                params: [(
                    "min_ratio".to_string(),
                    Value::Number(serde_json::Number::from_f64(0.80).expect("0.80 is a valid f64")),
                )]
                .into(),
                fix_instruction: "File read incomplete. If current content is sufficient to understand the file and complete the task, proceed; otherwise use offset/limit to read remaining lines."
                    .to_string(),
                max_retries: 2,
            }],
        );

        // ── Search ──
        pre.insert(
            ToolCategory::Search,
            vec![
                PreInjectionRule {
                    enforcement: EnforcementLevel::Must,
                    instruction: "Search operations must retrieve all matching results. \
                        Check the num_matches / num_files fields in the response \
                        and confirm they match the actual number of returned results. \
                        If there are too many results, use more precise search criteria to narrow the scope."
                        .to_string(),
                    tool_names: vec![],
                },
                PreInjectionRule {
                    enforcement: EnforcementLevel::Must,
                    instruction: "Search scope must be limited to within the current workspace. \
                        Do NOT search project source code, node_modules, target, .git or other directories unrelated to the current task. \
                        If search results contain irrelevant files, ignore them and focus on task-related files."
                        .to_string(),
                    tool_names: vec![],
                },
            ],
        );
        val.insert(
            ToolCategory::Search,
            vec![ValidationRule {
                validator: "search_count_check".to_string(),
                params: HashMap::new(),
                fix_instruction: "Search returned incomplete results. Please increase head_limit or use more precise keywords."
                    .to_string(),
                max_retries: 1,
            }],
        );

        // ── CodeExecution ──
        pre.insert(
            ToolCategory::CodeExecution,
            vec![
                PreInjectionRule {
                    enforcement: EnforcementLevel::Must,
                    instruction: "After executing a command, you MUST check the exit_code. \
                        If exit_code ≠ 0, analyze the full stderr content. \
                        Do NOT use results with non-zero exit codes as valid output."
                        .to_string(),
                    tool_names: vec!["bash".to_string(), "powershell".to_string()],
                },
                PreInjectionRule {
                    enforcement: EnforcementLevel::Must,
                    instruction: "All commands must be executed within the current working directory (workspace). \
                        Do NOT access directories outside the workspace. When using cd, do not go beyond the workspace boundary. \
                        Your workspace boundary is managed by the system; directories outside are unrelated to the current task."
                        .to_string(),
                    tool_names: vec!["bash".to_string(), "powershell".to_string()],
                },
            ],
        );
        val.insert(
            ToolCategory::CodeExecution,
            vec![ValidationRule {
                validator: "exit_code_check".to_string(),
                params: HashMap::new(),
                fix_instruction: "Command exited with non-zero code. Please analyze stderr for error information, fix the issue, and retry."
                    .to_string(),
                max_retries: 2,
            }],
        );

        // ── KnowledgeGraph ──
        pre.insert(
            ToolCategory::KnowledgeGraph,
            vec![
                PreInjectionRule {
                    enforcement: EnforcementLevel::Should,
                    instruction: "When executing SPARQL queries, if results return 0, \
                        try a more relaxed query (e.g., remove FILTER)."
                        .to_string(),
                    tool_names: vec!["knowledge_query".to_string()],
                },
                PreInjectionRule {
                    enforcement: EnforcementLevel::Should,
                    instruction:
                        "When traversing neighbors, the depth parameter should be at least 2 \
                        to obtain meaningful context."
                            .to_string(),
                    tool_names: vec!["knowledge_neighbors".to_string()],
                },
            ],
        );
        val.insert(
            ToolCategory::KnowledgeGraph,
            vec![
                ValidationRule {
                    validator: "knowledge_empty_check".to_string(),
                    params: HashMap::new(),
                    fix_instruction: "Query returned no results. Try removing FILTER conditions or using more relaxed matching."
                        .to_string(),
                    max_retries: 1,
                },
                ValidationRule {
                    validator: "knowledge_depth_check".to_string(),
                    params: HashMap::new(),
                    fix_instruction: "Traversal depth is insufficient. Set the depth parameter to at least 2."
                        .to_string(),
                    max_retries: 1,
                },
            ],
        );

        // ── KnowledgeExtract ──
        pre.insert(
            ToolCategory::KnowledgeExtract,
            vec![PreInjectionRule {
                enforcement: EnforcementLevel::Must,
                instruction: "Extraction results must include entities and relations arrays. \
                    All entities must have a unique id. \
                    If the text is too long (over 4000 characters), extract in segments and then merge with deduplication."
                    .to_string(),
                tool_names: vec![],
            }],
        );
        val.insert(
            ToolCategory::KnowledgeExtract,
            vec![ValidationRule {
                validator: "extract_empty_check".to_string(),
                params: HashMap::new(),
                fix_instruction: "No entities extracted. Ensure the text contains extractable structured information and try again."
                    .to_string(),
                max_retries: 1,
            }],
        );

        // ── HttpRequest ──
        pre.insert(
            ToolCategory::HttpRequest,
            vec![PreInjectionRule {
                enforcement: EnforcementLevel::Must,
                instruction: "Check the HTTP status code. \
                    If status_code ≥ 400, analyze the error cause and report it — do NOT ignore."
                    .to_string(),
                tool_names: vec!["web_fetch".to_string(), "http_request".to_string()],
            }],
        );
        val.insert(
            ToolCategory::HttpRequest,
            vec![ValidationRule {
                validator: "http_status_check".to_string(),
                params: HashMap::new(),
                fix_instruction: "HTTP request returned an error status code. Check if the URL is correct or if authentication is required."
                    .to_string(),
                max_retries: 1,
            }],
        );
    }

    // ─── Hook Registration ───

    /// Register ToolGuard into the HookManager:
    /// - SkillBefore → Pre-Injection (injects constraints into context metadata)
    /// - SkillAfter  → Post-Validation (validates tool result)
    pub fn register_hooks(&self, hook_manager: &HookManager) {
        // ── Pre-Injection Hook (SkillBefore) ──
        let pre_guard = self.clone();
        let pre_hook = FunctionHook::new(
            "toolguard::pre_inject",
            vec![HookPoint::SkillBefore],
            80,
            move |ctx: &mut HookContext| {
                let tool_name = match ctx.data.get("tool_name").and_then(|v| v.as_str()) {
                    Some(name) => name.to_string(),
                    None => return HookResult::Continue,
                };

                if let Some(category) = pre_guard.tool_categories.get(&tool_name) {
                    if let Some(rules) = pre_guard.pre_injections.read().get(category) {
                        let instructions: Vec<String> = rules
                            .iter()
                            .map(|r| {
                                let tag = match r.enforcement {
                                    EnforcementLevel::Must => "MUST",
                                    EnforcementLevel::Should => "SHOULD",
                                    EnforcementLevel::Info => "INFO",
                                };
                                format!("[ToolGuard-{}] {}", tag, r.instruction)
                            })
                            .collect();
                        ctx.metadata.insert(
                            "guard_pre_injections".to_string(),
                            Value::Array(instructions.into_iter().map(Value::String).collect()),
                        );
                        debug!(
                            tool = %tool_name,
                            "ToolGuard: Pre-injection applied ({} rules)",
                            rules.len()
                        );
                    }
                }

                // Stale file check for write operations
                if tool_name == "file_write" || tool_name == "file_edit" {
                    if let Some(ref stale_fn) = *pre_guard.stale_check.read() {
                        if let Some(path) = ctx.data.get("path").and_then(|v| v.as_str()) {
                            if let Some(warning) = stale_fn(path) {
                                warn!(tool = %tool_name, path = %path, warning = %warning, "ToolGuard: stale file detected");
                                ctx.metadata.insert(
                                    "stale_file_warning".to_string(),
                                    Value::String(warning),
                                );
                            }
                        }
                    }
                }

                HookResult::Continue
            },
        );
        hook_manager.register_arc(Arc::new(pre_hook));

        // ── Post-Validation Hook (SkillAfter) ──
        let post_guard = self.clone();
        let post_hook = FunctionHook::new(
            "toolguard::post_validate",
            vec![HookPoint::SkillAfter],
            80,
            move |ctx: &mut HookContext| {
                let tool_name = match ctx.data.get("tool_name").and_then(|v| v.as_str()) {
                    Some(name) => name.to_string(),
                    None => return HookResult::Continue,
                };
                let result_str = match ctx.data.get("tool_result").and_then(|v| v.as_str()) {
                    Some(s) => s.to_string(),
                    None => return HookResult::Continue,
                };

                let result: Value = match serde_json::from_str(&result_str) {
                    Ok(v) => v,
                    Err(_) => return HookResult::Continue,
                };

                if let Some(category) = post_guard.tool_categories.get(&tool_name) {
                    if let Some(rules) = post_guard.validations.read().get(category) {
                        for rule in rules {
                            let outcome = post_guard.run_validator(&rule.validator, &result);
                            match outcome {
                                ValidationOutcome::Fail(msg) => {
                                    if rule.validator == "file_length_check" {
                                        let path_opt = result.get("path").and_then(|v| v.as_str());
                                        let total_opt =
                                            result.get("total_lines").and_then(|v| v.as_u64());
                                        let offset_opt =
                                            result.get("offset").and_then(|v| v.as_u64());
                                        let returned_opt =
                                            result.get("returned").and_then(|v| v.as_u64());

                                        if let (
                                            Some(path),
                                            Some(total),
                                            Some(offset),
                                            Some(returned),
                                        ) = (path_opt, total_opt, offset_opt, returned_opt)
                                        {
                                            match post_guard.check_file_coverage(
                                                path,
                                                offset as usize,
                                                returned as usize,
                                                total as usize,
                                            ) {
                                                None => {
                                                    // >= 95% covered, or retries exhausted → pass through
                                                    continue;
                                                }
                                                Some(ratio) => {
                                                    let a = post_guard
                                                        .file_coverage
                                                        .lock()
                                                        .get(path)
                                                        .map_or(0, |s| s.attempt_count);
                                                    debug!(
                                                        tool = %tool_name,
                                                        ratio = ratio,
                                                        attempt = a,
                                                        "ToolGuard: file read cumulative {:.1}% (attempt {}/3)",
                                                        ratio * 100.0, a,
                                                    );
                                                    continue;
                                                }
                                            }
                                        }
                                    }

                                    let error_msg = format!(
                                        "ToolGuard: {} - {}. Fix suggestion: {}",
                                        tool_name, msg, rule.fix_instruction
                                    );
                                    warn!(
                                        tool = %tool_name,
                                        reason = %msg,
                                        "ToolGuard: Validation failed"
                                    );
                                    ctx.error = Some(error_msg);

                                    post_guard.audit_log.write().push(GuardAuditEntry {
                                        timestamp: chrono::Utc::now().timestamp(),
                                        tool_name: tool_name.clone(),
                                        agent_id: ctx.agent_id.clone(),
                                        pre_injected: true,
                                        validation_passed: false,
                                        retry_count: 0,
                                        error: Some(msg.clone()),
                                    });
                                    GUARD_AUDIT_LOG.write().push(GuardAuditEntry {
                                        timestamp: chrono::Utc::now().timestamp(),
                                        tool_name: tool_name.clone(),
                                        agent_id: ctx.agent_id.clone(),
                                        pre_injected: true,
                                        validation_passed: false,
                                        retry_count: 0,
                                        error: Some(msg),
                                    });

                                    return HookResult::Abort;
                                }
                                ValidationOutcome::Warn(msg) => {
                                    warn!(
                                        tool = %tool_name,
                                        warning = %msg,
                                        "ToolGuard: Validation warning"
                                    );
                                }
                                ValidationOutcome::Pass => {}
                            }
                        }

                        post_guard.audit_log.write().push(GuardAuditEntry {
                            timestamp: chrono::Utc::now().timestamp(),
                            tool_name: tool_name.clone(),
                            agent_id: ctx.agent_id.clone(),
                            pre_injected: true,
                            validation_passed: true,
                            retry_count: 0,
                            error: None,
                        });
                        GUARD_AUDIT_LOG.write().push(GuardAuditEntry {
                            timestamp: chrono::Utc::now().timestamp(),
                            tool_name: tool_name.clone(),
                            agent_id: ctx.agent_id.clone(),
                            pre_injected: true,
                            validation_passed: true,
                            retry_count: 0,
                            error: None,
                        });
                    }
                }

                HookResult::Continue
            },
        );
        hook_manager.register_arc(Arc::new(post_hook));
    }

    // ─── Validators ───

    fn run_validator(&self, validator: &str, result: &Value) -> ValidationOutcome {
        match validator {
            "file_length_check" => self::validators::file_length_check(result),
            "search_count_check" => self::validators::search_count_check(result),
            "exit_code_check" => self::validators::exit_code_check(result),
            "knowledge_empty_check" => self::validators::knowledge_empty_check(result),
            "knowledge_depth_check" => self::validators::knowledge_depth_check(result),
            "extract_empty_check" => self::validators::extract_empty_check(result),
            "http_status_check" => self::validators::http_status_check(result),
            _ => {
                warn!(validator = %validator, "ToolGuard: Unknown validator");
                ValidationOutcome::Pass
            }
        }
    }

    // ─── Audit ───

    pub fn get_audit_log(&self) -> Vec<GuardAuditEntry> {
        self.audit_log.read().clone()
    }

    pub fn get_audit_stats(&self) -> GuardStats {
        let log = self.audit_log.read();
        let total = log.len();
        if total == 0 {
            return GuardStats::default();
        }
        let passed = log.iter().filter(|e| e.validation_passed).count();
        GuardStats {
            total_checks: total,
            passed_checks: passed,
            failed_checks: total - passed,
            pass_rate: passed as f64 / total as f64,
        }
    }

    // ─── Hot-Reload ───

    /// Reload rules from a JSON config file at runtime.
    /// Overrides rules for categories present in the file, keeps others unchanged.
    pub fn reload_from_json<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_ref = path.as_ref().to_path_buf();
        let config = GuardRulesConfig::from_file(&path_ref)?;
        {
            let mut pre = self.pre_injections.write();
            let mut val = self.validations.write();
            for (cat_str, rules) in &config.categories {
                if let Ok(category) = serde_json::from_value::<ToolCategory>(json!(cat_str)) {
                    if !rules.pre_injections.is_empty() {
                        pre.insert(category.clone(), rules.pre_injections.clone());
                    }
                    if !rules.validations.is_empty() {
                        val.insert(category, rules.validations.clone());
                    }
                }
            }
        }
        info!(
            "ToolGuard: Hot-reloaded {} categories from {:?}",
            config.categories.len(),
            path_ref
        );
        Ok(())
    }

    /// Start a background task that polls the config file for changes.
    /// When the file's mtime changes, rules are reloaded automatically.
    pub fn start_hot_reload(self: &Arc<Self>, path: impl Into<String>, interval_secs: u64) {
        let path_str: String = path.into();
        let watch_path = path_str.clone();
        {
            let mut cp = self.config_path.write();
            *cp = Some(path_str);
        }
        let guard = self.clone();
        tokio::spawn(async move {
            let mut last_mtime = std::time::SystemTime::UNIX_EPOCH;
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));
            interval.tick().await; // skip immediate tick
            loop {
                interval.tick().await;
                if let Ok(meta) = std::fs::metadata(&watch_path) {
                    if let Ok(mtime) = meta.modified() {
                        if mtime != last_mtime {
                            last_mtime = mtime;
                            match guard.reload_from_json(&watch_path) {
                                Ok(()) => {
                                    debug!("ToolGuard: Config hot-reloaded from {}", watch_path)
                                }
                                Err(e) => {
                                    warn!("ToolGuard: Hot-reload failed for {}: {}", watch_path, e)
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}

impl Default for ToolGuard {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Cumulative File Coverage Tracking ───

const MAX_FILE_READ_ATTEMPTS: u32 = 3;

impl ToolGuard {
    /// Update cumulative coverage for a file read.
    /// Returns `Some(cumulative_ratio)` if coverage < 95% and within retry limit,
    /// or `None` if coverage >= 95% or retry limit exceeded.
    /// The caller should treat the result as Warn (don't block) regardless.
    fn check_file_coverage(
        &self,
        path: &str,
        offset: usize,
        returned: usize,
        total: usize,
    ) -> Option<f64> {
        if total == 0 {
            return None;
        }
        let end = offset + returned;
        let mut cov_guard = self.file_coverage.lock();
        let state = cov_guard
            .entry(path.to_string())
            .or_insert_with(|| FileCoverage {
                ranges: Vec::new(),
                attempt_count: 0,
                total_lines: total,
            });
        state.attempt_count += 1;

        let mut new_ranges: Vec<(usize, usize)> = Vec::with_capacity(state.ranges.len() + 1);
        new_ranges.push((offset, end));
        for &r in state.ranges.iter() {
            new_ranges.push(r);
        }
        new_ranges.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for r in new_ranges {
            if let Some(last) = merged.last_mut() {
                if r.0 <= last.1 {
                    last.1 = last.1.max(r.1);
                    continue;
                }
            }
            merged.push(r);
        }
        state.ranges = merged;

        let covered: usize = state.ranges.iter().map(|&(s, e)| e - s).sum();
        let ratio = (covered as f64) / (state.total_lines as f64);
        if ratio >= 0.95 {
            cov_guard.remove(path);
            return None;
        }
        if state.attempt_count >= MAX_FILE_READ_ATTEMPTS {
            cov_guard.remove(path);
            return None;
        }
        Some(ratio.min(1.0))
    }

    #[allow(dead_code)]
    /// Set a stale file check callback. Called for file_write/file_edit tools
    /// before execution. If the callback returns Some(warning), the warning is
    /// injected into the tool's context metadata.
    pub fn set_stale_check<F>(&self, check: F)
    where
        F: Fn(&str) -> Option<String> + Send + Sync + 'static,
    {
        *self.stale_check.write() = Some(Arc::new(check));
    }

    pub fn reset_file_coverage(&self, path: &str) {
        self.file_coverage.lock().remove(path);
    }
}

// ─── Validator Implementations ───

mod validators {
    use super::ValidationOutcome;
    use serde_json::Value;

    pub fn file_length_check(result: &Value) -> ValidationOutcome {
        // Cache hit: content already provided in a previous read, skip length check
        if result
            .get("from_cache")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return ValidationOutcome::Pass;
        }
        // file_list output: {"entries": [...]} → skip
        if result.get("entries").is_some() {
            return ValidationOutcome::Pass;
        }
        // file_read output: {"path": "...", "lines": [...], "total_lines": N}
        // file_read result is in `lines` array, not `content`
        if result.get("lines").is_some() || result.get("total_lines").is_some() {
            if result.get("error").is_some() {
                return ValidationOutcome::Fail("File read returned error".to_string());
            }
            // total_lines == 0 means the file exists but is empty → valid state
            let total = result["total_lines"].as_u64().unwrap_or(0);
            if total > 0 {
                let returned = result["returned"].as_u64().unwrap_or(0);
                if returned == 0 {
                    return ValidationOutcome::Warn(
                        "File read result empty (total_lines > 0 but returned is 0)".to_string(),
                    );
                }
                // Check if read is complete: returned >= total_lines * 0.80
                let min_expected = (total as f64 * 0.80).ceil() as u64;
                if returned < min_expected {
                    return ValidationOutcome::Fail(format!(
                        "File read incomplete: {} total lines, only {} returned ({:.1}%)",
                        total,
                        returned,
                        (returned as f64 / total as f64) * 100.0
                    ));
                }
            }
            return ValidationOutcome::Pass;
        }
        // Other tools with `content` field (e.g. web_fetch, bash stdout)
        let content = result["content"].as_str().unwrap_or("");
        if content.is_empty() {
            if result.get("error").is_some() {
                return ValidationOutcome::Fail("File read returned error".to_string());
            }
            return ValidationOutcome::Warn("File content is empty".to_string());
        }
        ValidationOutcome::Pass
    }

    pub fn search_count_check(result: &Value) -> ValidationOutcome {
        if let Some(num_files) = result["num_files"].as_u64() {
            let returned = result["filenames"]
                .as_array()
                .map(|a| a.len() as u64)
                .unwrap_or(0);
            if returned < num_files && returned > 0 {
                return ValidationOutcome::Fail(format!(
                    "Search results incomplete: {} matching files, only {} returned (head_limit restriction)",
                    num_files, returned
                ));
            }
        }
        if let Some(num_matches) = result["num_matches"].as_u64() {
            let returned_count = result["counts"]
                .as_array()
                .map(|a| a.len() as u64)
                .or_else(|| result["filenames"].as_array().map(|a| a.len() as u64))
                .unwrap_or(0);
            if returned_count > 0 && returned_count < num_matches {
                return ValidationOutcome::Warn(format!(
                    "{} matches found, only {} returned (may be limited by limit parameter)",
                    num_matches, returned_count
                ));
            }
        }
        ValidationOutcome::Pass
    }

    pub fn exit_code_check(result: &Value) -> ValidationOutcome {
        if let Some(ec) = result["exit_code"].as_i64() {
            if ec != 0 {
                let stderr = result["stderr"].as_str().unwrap_or("");
                return ValidationOutcome::Fail(format!(
                    "Non-zero exit code: {}, stderr: {}",
                    ec,
                    &stderr.chars().take(200).collect::<String>()
                ));
            }
        } else if result.get("error").is_some() {
            return ValidationOutcome::Fail("Command execution returned error".to_string());
        }
        ValidationOutcome::Pass
    }

    pub fn knowledge_empty_check(result: &Value) -> ValidationOutcome {
        let bindings = result["bindings"].as_array();
        let results_arr = result["results"].as_array();
        let count = bindings
            .map(|a| a.len())
            .or_else(|| results_arr.map(|a| a.len()))
            .unwrap_or(0);
        if count == 0 {
            return ValidationOutcome::Warn(
                "Query returned no results, try more relaxed query conditions".to_string(),
            );
        }
        ValidationOutcome::Pass
    }

    pub fn knowledge_depth_check(result: &Value) -> ValidationOutcome {
        if let Some(depth) = result["depth"].as_u64() {
            if depth < 2 {
                return ValidationOutcome::Warn(format!(
                    "Traversal depth is {}, recommend increasing to at least 2",
                    depth
                ));
            }
        }
        ValidationOutcome::Pass
    }

    pub fn extract_empty_check(result: &Value) -> ValidationOutcome {
        let entities = result["entities"].as_array();
        let extracted = result["extracted"].as_array();
        let count = entities
            .map(|a| a.len())
            .or_else(|| extracted.map(|a| a.len()))
            .unwrap_or(0);
        if count == 0 {
            return ValidationOutcome::Fail("No entities extracted".to_string());
        }
        ValidationOutcome::Pass
    }

    pub fn http_status_check(result: &Value) -> ValidationOutcome {
        if let Some(code) = result["status_code"]
            .as_u64()
            .or_else(|| result["status"].as_u64())
        {
            if code >= 400 {
                let body = result["body"]
                    .as_str()
                    .or_else(|| result["content"].as_str())
                    .unwrap_or("");
                return ValidationOutcome::Fail(format!(
                    "HTTP {} error: {}",
                    code,
                    &body.chars().take(100).collect::<String>()
                ));
            }
        }
        ValidationOutcome::Pass
    }
}

// ════════════════════════════════════════════════════════════════════════
// Methodology Gate Integration
// ════════════════════════════════════════════════════════════════════════

use crate::methodology::gate::{ActivatedMethodology, AntiPatternGateResult};

impl ToolGuard {
    /// Format red flags from active methodologies as pre-injection rules.
    ///
    /// Returns a list of instruction strings suitable for system prompt injection.
    pub fn format_methodology_red_flags(
        &self,
        red_flags: &[(&ActivatedMethodology, &crate::methodology::RedFlagEntry)],
    ) -> Vec<String> {
        red_flags
            .iter()
            .map(|(activated, flag)| {
                let tag = match flag.severity {
                    crate::methodology::RedFlagSeverity::Critical => "🔴 Red Flag",
                    crate::methodology::RedFlagSeverity::Warning => "🟡 Warning",
                    crate::methodology::RedFlagSeverity::Info => "🔵 Info",
                };
                format!(
                    "[Methodology-{}] {}: {}",
                    activated.methodology_id, tag, flag.pattern
                )
            })
            .collect()
    }

    /// Format rationalization checks from active methodologies as pre-injection rules.
    pub fn format_methodology_rationalizations(
        &self,
        rationalizations: &[(
            &ActivatedMethodology,
            &crate::methodology::RedFlagEntry,
            &str,
        )],
    ) -> Vec<String> {
        rationalizations
            .iter()
            .map(|(activated, flag, check)| {
                format!(
                    "[Methodology-{}] ⚠️ 『{}』→ self-check: {}",
                    activated.methodology_id, flag.pattern, check
                )
            })
            .collect()
    }

    /// Format anti-pattern gate warnings as pre-injection rules.
    pub fn format_anti_pattern_gates(
        &self,
        anti_patterns: &[AntiPatternGateResult],
    ) -> Vec<String> {
        anti_patterns.iter().map(|ap| ap.message.clone()).collect()
    }

    /// Format persuasion directives as pre-injection rules.
    pub fn format_methodology_persuasion(&self, directives: &[String]) -> Vec<String> {
        directives
            .iter()
            .map(|d| format!("[Methodology-Persuasion] {}", d))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_default_tool_categories() {
        let guard = ToolGuard::new();
        assert_eq!(
            guard.tool_categories.get("file_read"),
            Some(&ToolCategory::FileRead)
        );
        assert_eq!(
            guard.tool_categories.get("bash"),
            Some(&ToolCategory::CodeExecution)
        );
        assert_eq!(
            guard.tool_categories.get("knowledge_query"),
            Some(&ToolCategory::KnowledgeGraph)
        );
    }

    #[test]
    fn test_file_length_check_pass() {
        let result = json!({"content": "full file content here"});
        let outcome = validators::file_length_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Pass));
    }

    #[test]
    fn test_file_length_check_empty_warn() {
        let result = json!({"content": ""});
        let outcome = validators::file_length_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Warn(_)));
    }

    #[test]
    fn test_file_length_check_error() {
        let result = json!({"error": "file not found"});
        let outcome = validators::file_length_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Fail(_)));
    }

    #[test]
    fn test_exit_code_check_pass() {
        let result = json!({"exit_code": 0, "stdout": "ok"});
        let outcome = validators::exit_code_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Pass));
    }

    #[test]
    fn test_exit_code_check_fail() {
        let result = json!({"exit_code": 1, "stderr": "error occurred"});
        let outcome = validators::exit_code_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Fail(_)));
    }

    #[test]
    fn test_search_count_check_incomplete() {
        let result = json!({
            "num_files": 10,
            "filenames": ["a.rs", "b.rs"],
            "mode": "files_with_matches"
        });
        let outcome = validators::search_count_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Fail(_)));
    }

    #[test]
    fn test_search_count_check_complete() {
        let result = json!({
            "num_files": 2,
            "filenames": ["a.rs", "b.rs"],
            "mode": "files_with_matches"
        });
        let outcome = validators::search_count_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Pass));
    }

    #[test]
    fn test_extract_empty_check_fail() {
        let result = json!({"entities": []});
        let outcome = validators::extract_empty_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Fail(_)));
    }

    #[test]
    fn test_extract_empty_check_pass() {
        let result = json!({"entities": [{"id": "e1", "type": "Person"}]});
        let outcome = validators::extract_empty_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Pass));
    }

    #[test]
    fn test_http_status_check_fail() {
        let result = json!({"status_code": 404, "body": "Not Found"});
        let outcome = validators::http_status_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Fail(_)));
    }

    #[test]
    fn test_http_status_check_pass() {
        let result = json!({"status_code": 200, "content": "ok"});
        let outcome = validators::http_status_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Pass));
    }

    #[test]
    fn test_knowledge_empty_warn() {
        let result = json!({"bindings": []});
        let outcome = validators::knowledge_empty_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Warn(_)));
    }

    #[test]
    fn test_knowledge_depth_warn() {
        let result = json!({"depth": 1});
        let outcome = validators::knowledge_depth_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Warn(_)));
    }

    #[test]
    fn test_knowledge_depth_pass() {
        let result = json!({"depth": 2});
        let outcome = validators::knowledge_depth_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Pass));
    }

    #[test]
    fn test_audit_stats() {
        let guard = ToolGuard::new();
        let stats = guard.get_audit_stats();
        assert_eq!(stats.total_checks, 0);
        assert_eq!(stats.pass_rate, 1.0);
    }

    #[test]
    fn test_register_hooks_no_panic() {
        let guard = ToolGuard::new();
        let manager = HookManager::new();
        // Should not panic
        guard.register_hooks(&manager);
        let hooks = manager.get_hooks(HookPoint::SkillBefore);
        assert!(hooks.contains(&"toolguard::pre_inject".to_string()));
        let hooks = manager.get_hooks(HookPoint::SkillAfter);
        assert!(hooks.contains(&"toolguard::post_validate".to_string()));
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let json_str = r#"{
  "categories": {
    "FileRead": {
      "pre_injections": [
        {
          "enforcement": "Must",
          "instruction": "Must read full content",
          "tool_names": []
        }
      ],
      "validations": [
        {
          "validator": "file_length_check",
          "params": { "min_ratio": 0.80 },
          "fix_instruction": "Read the whole file",
          "max_retries": 2
        }
      ]
    }
  }
}"#;
        let config: GuardRulesConfig = serde_json::from_str(json_str).unwrap();
        assert!(config.categories.contains_key("FileRead"));
        let rules = &config.categories["FileRead"];
        assert_eq!(rules.pre_injections.len(), 1);
        assert_eq!(rules.validations.len(), 1);
        assert_eq!(rules.pre_injections[0].enforcement, EnforcementLevel::Must);
    }

    #[test]
    fn test_reload_from_json() {
        let dir = std::env::temp_dir().join("tool_guard_reload_test");
        let _ = std::fs::create_dir_all(&dir);
        let config_path = dir.join("test_rules.json");

        let initial = r#"{"categories":{"CodeExecution":{"pre_injections":[{"enforcement":"Must","instruction":"Always check exit code","tool_names":[]}],"validations":[{"validator":"exit_code_check","params":{},"fix_instruction":"Fix errors","max_retries":1}]}}}"#;
        std::fs::write(&config_path, initial).unwrap();

        let guard = ToolGuard::from_json(&config_path).unwrap();
        {
            let pre = guard.pre_injections.read();
            let code_rules = pre.get(&ToolCategory::CodeExecution);
            assert!(code_rules.is_some());
            assert_eq!(code_rules.unwrap()[0].instruction, "Always check exit code");
        }

        // Reload with different rules
        let updated = r#"{"categories":{"CodeExecution":{"pre_injections":[{"enforcement":"Should","instruction":"Check exit code carefully","tool_names":[]}],"validations":[]}}}"#;
        std::fs::write(&config_path, updated).unwrap();
        guard.reload_from_json(&config_path).unwrap();

        {
            let pre = guard.pre_injections.read();
            let code_rules = pre.get(&ToolCategory::CodeExecution);
            assert!(code_rules.is_some());
            assert_eq!(
                code_rules.unwrap()[0].instruction,
                "Check exit code carefully"
            );
            assert_eq!(code_rules.unwrap()[0].enforcement, EnforcementLevel::Should);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_global_audit_log() {
        // Verify the global log is accessible and writable
        let initial_len = GUARD_AUDIT_LOG.read().len();
        GUARD_AUDIT_LOG.write().push(GuardAuditEntry {
            timestamp: 0,
            tool_name: "test".to_string(),
            agent_id: "test-agent".to_string(),
            pre_injected: true,
            validation_passed: false,
            retry_count: 1,
            error: Some("test error".to_string()),
        });
        assert_eq!(GUARD_AUDIT_LOG.read().len(), initial_len + 1);
        // Cleanup
        GUARD_AUDIT_LOG.write().pop();
    }

    #[test]
    fn test_from_json_missing_file() {
        let result = ToolGuard::from_json("/nonexistent/path/guard_rules.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_guard_stats_calculation() {
        let guard = ToolGuard::new();
        let stats = guard.get_audit_stats();
        assert_eq!(stats.total_checks, 0);
        assert_eq!(stats.pass_rate, 1.0);

        // Add entries via global log (internal log is only pushed during hook execution)
        // Verify the stats method handles empty log
    }

    #[test]
    fn test_set_stale_check_callback_invoked() {
        let guard = ToolGuard::new();
        let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = call_count.clone();

        guard.set_stale_check(move |path| {
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if path == "stale.rs" {
                Some("File is stale: stale.rs".to_string())
            } else {
                None
            }
        });

        let hm = HookManager::new();
        guard.register_hooks(&hm);

        let mut ctx = HookContext::new(HookPoint::SkillBefore, "test_agent", "DA");
        ctx.data.insert(
            "tool_name".to_string(),
            Value::String("file_write".to_string()),
        );
        ctx.data
            .insert("path".to_string(), Value::String("stale.rs".to_string()));

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(hm.execute(HookPoint::SkillBefore, &mut ctx));

        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        let warning = ctx
            .metadata
            .get("stale_file_warning")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            warning.contains("stale"),
            "Expected stale warning, got: {}",
            warning
        );
    }

    #[test]
    fn test_stale_check_not_invoked_for_non_write_tool() {
        let guard = ToolGuard::new();
        let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = call_count.clone();

        guard.set_stale_check(move |_path| {
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            None
        });

        let hm = HookManager::new();
        guard.register_hooks(&hm);

        let mut ctx = HookContext::new(HookPoint::SkillBefore, "test_agent", "DA");
        ctx.data.insert(
            "tool_name".to_string(),
            Value::String("file_read".to_string()),
        );
        ctx.data
            .insert("path".to_string(), Value::String("ok.rs".to_string()));

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(hm.execute(HookPoint::SkillBefore, &mut ctx));

        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}
