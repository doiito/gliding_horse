use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use crate::tools::hooks::{FunctionHook, HookContext, HookManager, HookPoint, HookResult};

/// Maximum number of retained entries in each ToolGuard audit window.
///
/// Both the per-instance log and the process-wide HTTP log use this bound, so
/// an agent that repeatedly invokes a failing tool cannot grow process memory
/// without limit. Entries are exposed oldest-to-newest within the retained
/// window.
pub const GUARD_AUDIT_LOG_CAPACITY: usize = 4_096;

/// Fixed-capacity FIFO audit window backed by a ring buffer.
///
/// `push` is the only insertion API: once full, it evicts the oldest entry.
/// The inherent `clone` method intentionally returns a `Vec` snapshot to keep
/// the existing HTTP read-side contract source-compatible.
#[derive(Debug)]
pub struct GuardAuditBuffer {
    entries: VecDeque<GuardAuditEntry>,
    capacity: usize,
    dropped_entries: u64,
}

impl GuardAuditBuffer {
    fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
            dropped_entries: 0,
        }
    }

    pub fn push(&mut self, mut entry: GuardAuditEntry) {
        entry.sanitize_for_storage();
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
            self.dropped_entries = self.dropped_entries.saturating_add(1);
        }
        self.entries.push_back(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &GuardAuditEntry> {
        self.entries.iter()
    }

    pub fn retain<F>(&mut self, predicate: F)
    where
        F: FnMut(&GuardAuditEntry) -> bool,
    {
        self.entries.retain(predicate);
    }

    pub fn dropped_entries(&self) -> u64 {
        self.dropped_entries
    }

    pub fn snapshot(&self) -> Vec<GuardAuditEntry> {
        self.entries.iter().cloned().collect()
    }

    /// Compatibility snapshot for the existing `GUARD_AUDIT_LOG` reader.
    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Vec<GuardAuditEntry> {
        self.snapshot()
    }
}

impl Default for GuardAuditBuffer {
    fn default() -> Self {
        Self::with_capacity(GUARD_AUDIT_LOG_CAPACITY)
    }
}

/// Global bounded audit window accessible to HTTP endpoints.
pub static GUARD_AUDIT_LOG: Lazy<Arc<RwLock<GuardAuditBuffer>>> =
    Lazy::new(|| Arc::new(RwLock::new(GuardAuditBuffer::default())));

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
    /// Backwards-compatible failure field. It contains a stable category, not
    /// raw stderr, response bodies, commands, or validator messages.
    pub error: Option<String>,
    /// Byte length of the sensitive failure detail before redaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_length: Option<usize>,
    /// SHA-256 of the sensitive failure detail before redaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_sha256: Option<String>,
    /// Non-sensitive process exit status, when the failure was a command exit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
}

impl GuardAuditEntry {
    fn sanitize_for_storage(&mut self) {
        self.tool_name = bounded_audit_label(&self.tool_name);
        self.agent_id = bounded_audit_label(&self.agent_id);
        if !self
            .error_sha256
            .as_deref()
            .is_some_and(is_sha256_hex_digest)
        {
            self.error_sha256 = None;
        }

        let Some(raw_error) = self.error.as_deref() else {
            self.error_length = None;
            self.error_sha256 = None;
            self.exit_code = None;
            return;
        };
        if matches!(
            raw_error,
            "non_zero_exit"
                | "structured_tool_error"
                | "command_validation_failure"
                | "validation_failure"
                | "external_failure"
        ) {
            return;
        }

        // The buffer is a security boundary even for direct public callers:
        // legacy/free-form errors are reduced to the same redacted shape.
        self.error_length = Some(raw_error.len());
        self.error_sha256 = Some(hex::encode(Sha256::digest(raw_error.as_bytes())));
        self.error = Some("external_failure".to_string());
        self.exit_code = None;
    }
}

fn is_sha256_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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

/// Structured, non-policy feedback emitted by ToolGuard post-validation.
///
/// A failed or incomplete tool result is useful model-visible evidence, not a
/// disclosure-policy decision. Keeping the feedback on the hook context lets
/// observability and later hooks inspect it without overloading
/// `HookResult::Abort`.
pub const TOOL_GUARD_VALIDATION_FEEDBACK_KEY: &str = "toolguard_validation_feedback";

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
    audit_log: Arc<RwLock<GuardAuditBuffer>>,
    config_path: Arc<RwLock<Option<String>>>,
    /// Per-file cumulative read tracking with attempt limit (max 3 per file).
    file_coverage: Arc<Mutex<HashMap<String, FileCoverage>>>,
    /// Optional stale file check callback: returns Some(warning) if file is stale.
    stale_check: Arc<RwLock<Option<Arc<dyn Fn(&str) -> Option<String> + Send + Sync>>>>,
}

#[derive(Debug)]
struct AuditFailure {
    category: String,
    detail_length: usize,
    detail_sha256: String,
    exit_code: Option<i64>,
}

impl AuditFailure {
    fn from_validation(validator: &str, result: &Value, fallback_message: &str) -> Self {
        let exit_code = result.get("exit_code").and_then(Value::as_i64);
        let (category, detail) = if validator == "exit_code_check" {
            if exit_code.is_some_and(|code| code != 0) {
                (
                    "non_zero_exit".to_string(),
                    result
                        .get("stderr")
                        .and_then(Value::as_str)
                        .map(str::as_bytes)
                        .unwrap_or_else(|| fallback_message.as_bytes()),
                )
            } else if let Some(error) = result.get("error") {
                (
                    "structured_tool_error".to_string(),
                    error
                        .as_str()
                        .map(str::as_bytes)
                        .unwrap_or_else(|| fallback_message.as_bytes()),
                )
            } else {
                (
                    "command_validation_failure".to_string(),
                    fallback_message.as_bytes(),
                )
            }
        } else {
            (
                "validation_failure".to_string(),
                fallback_message.as_bytes(),
            )
        };

        Self {
            category,
            detail_length: detail.len(),
            detail_sha256: hex::encode(Sha256::digest(detail)),
            exit_code: exit_code.filter(|code| *code != 0),
        }
    }
}

const MAX_AUDIT_LABEL_BYTES: usize = 256;

/// Bound attacker-influenced labels even though the number of entries itself
/// is bounded. The digest keeps a stable identity when truncation is needed.
fn bounded_audit_label(value: &str) -> String {
    if value.len() <= MAX_AUDIT_LABEL_BYTES {
        return value.to_string();
    }

    let mut prefix_end = MAX_AUDIT_LABEL_BYTES / 2;
    while !value.is_char_boundary(prefix_end) {
        prefix_end -= 1;
    }
    format!(
        "{}...[bytes={},sha256={}]",
        &value[..prefix_end],
        value.len(),
        hex::encode(Sha256::digest(value.as_bytes()))
    )
}

impl ToolGuard {
    /// Command tools normally express an expected process failure with only a
    /// non-zero exit code. Results carrying an `error` field are logged by the
    /// execution layer, so ToolGuard must not emit a second warning for them.
    fn nonzero_exit_without_error_field(result: &Value) -> Option<i64> {
        result
            .get("exit_code")
            .and_then(Value::as_i64)
            .filter(|exit_code| *exit_code != 0 && result.get("error").is_none())
    }

    fn record_audit(
        &self,
        ctx: &HookContext,
        tool_name: &str,
        validation_passed: bool,
        failure: Option<AuditFailure>,
    ) {
        let (error, error_length, error_sha256, exit_code) = match failure {
            Some(failure) => (
                Some(failure.category),
                Some(failure.detail_length),
                Some(failure.detail_sha256),
                failure.exit_code,
            ),
            None => (None, None, None, None),
        };
        let entry = GuardAuditEntry {
            timestamp: chrono::Utc::now().timestamp(),
            tool_name: bounded_audit_label(tool_name),
            agent_id: bounded_audit_label(&ctx.agent_id),
            pre_injected: true,
            validation_passed,
            retry_count: 0,
            error,
            error_length,
            error_sha256,
            exit_code,
        };
        self.audit_log.write().push(entry.clone());
        GUARD_AUDIT_LOG.write().push(entry);
    }

    fn attach_validation_feedback(
        ctx: &mut HookContext,
        tool_name: &str,
        rule: &ValidationRule,
        message: &str,
    ) {
        let diagnostic = format!(
            "ToolGuard: {} - {}. Fix suggestion: {}",
            tool_name, message, rule.fix_instruction
        );
        ctx.error = Some(diagnostic);
        let feedback = json!({
            "tool_name": tool_name,
            "validator": rule.validator,
            "classification": if rule.validator == "exit_code_check" {
                "tool_execution_failure"
            } else {
                "tool_result_quality_failure"
            },
            "message": message,
            "fix_instruction": rule.fix_instruction,
            "blocks_disclosure": false,
        });
        match ctx
            .metadata
            .entry(TOOL_GUARD_VALIDATION_FEEDBACK_KEY.to_string())
            .or_insert_with(|| Value::Array(Vec::new()))
        {
            Value::Array(items) => items.push(feedback),
            slot => *slot = Value::Array(vec![feedback]),
        }
    }

    pub fn new() -> Self {
        let guard = Self {
            pre_injections: Arc::new(RwLock::new(HashMap::new())),
            validations: Arc::new(RwLock::new(HashMap::new())),
            tool_categories: Self::default_tool_categories(),
            audit_log: Arc::new(RwLock::new(GuardAuditBuffer::default())),
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
            audit_log: Arc::new(RwLock::new(GuardAuditBuffer::default())),
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
                    instruction: "For an expected-negative scenario (invalid input, rejection, or error-path test), \
                        prefer a project test-framework assertion. If an ad-hoc shell check is necessary, wrap it so \
                        the overall command exits zero only when the exact expected exit class and observable condition \
                        are both satisfied, and exits non-zero on unexpected acceptance, the wrong failure, timeout, \
                        setup failure, or shell error. Never use `|| true` or unconditional exit-code suppression."
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
                fix_instruction: "Command exited with non-zero code. Analyze stderr, fix the issue, and retry. If the child failure was intentionally exercised as a negative scenario, rerun it as an exact assertion wrapper that exits zero only for the intended rejection; do not suppress arbitrary failures."
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
                            .filter(|rule| {
                                rule.tool_names.is_empty()
                                    || rule.tool_names.iter().any(|name| name == &tool_name)
                            })
                            .map(|r| {
                                let tag = match r.enforcement {
                                    EnforcementLevel::Must => "MUST",
                                    EnforcementLevel::Should => "SHOULD",
                                    EnforcementLevel::Info => "INFO",
                                };
                                format!("[ToolGuard-{}] {}", tag, r.instruction)
                            })
                            .collect();
                        if !instructions.is_empty() {
                            let applied_count = instructions.len();
                            ctx.metadata.insert(
                                "guard_pre_injections".to_string(),
                                Value::Array(instructions.into_iter().map(Value::String).collect()),
                            );
                            debug!(
                                tool = %tool_name,
                                "ToolGuard: Pre-injection applied ({} rules)",
                                applied_count
                            );
                        }
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

                                    // ToolGuard validators assess result
                                    // quality/completeness; they are not
                                    // confidentiality policies. Preserve the
                                    // actual result so the model can diagnose
                                    // an empty read, HTTP failure, incomplete
                                    // search, or command error. A separate
                                    // policy hook may still Abort later in the
                                    // chain and withhold disclosure.
                                    ToolGuard::attach_validation_feedback(
                                        ctx, &tool_name, rule, &msg,
                                    );
                                    let failure = AuditFailure::from_validation(
                                        &rule.validator,
                                        &result,
                                        &msg,
                                    );
                                    post_guard.record_audit(ctx, &tool_name, false, Some(failure));
                                    if rule.validator == "exit_code_check" {
                                        if let Some(exit_code) =
                                            ToolGuard::nonzero_exit_without_error_field(&result)
                                        {
                                            // One metadata-only root-cause
                                            // line for normal command
                                            // failures. stderr and command
                                            // content remain in the
                                            // model-visible tool result and
                                            // are intentionally not logged.
                                            warn!(
                                                tool = %tool_name,
                                                exit_code,
                                                "Tool execution returned non-zero exit status"
                                            );
                                        } else {
                                            debug!(
                                                tool = %tool_name,
                                                validator = %rule.validator,
                                                "ToolGuard validation retained as non-blocking result feedback"
                                            );
                                        }
                                    } else {
                                        debug!(
                                            tool = %tool_name,
                                            validator = %rule.validator,
                                            "ToolGuard validation retained as non-blocking result feedback"
                                        );
                                    }
                                    return HookResult::Continue;
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

                        post_guard.record_audit(ctx, &tool_name, true, None);
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
        self.audit_log.read().snapshot()
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
    use sha2::{Digest, Sha256};

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
                    "Non-zero exit code: {}; stderr_length: {}; stderr_sha256: {}",
                    ec,
                    stderr.len(),
                    hex::encode(Sha256::digest(stderr.as_bytes()))
                ));
            }
        }
        let has_structured_error = match result.get("error") {
            None | Some(Value::Null) | Some(Value::Bool(false)) => false,
            Some(Value::String(message)) => !message.trim().is_empty(),
            Some(Value::Array(items)) => !items.is_empty(),
            Some(Value::Object(fields)) => !fields.is_empty(),
            Some(_) => true,
        };
        if has_structured_error {
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

    #[tokio::test]
    async fn shell_pre_injection_requires_exact_negative_assertions_and_honors_tool_scope() {
        let guard = ToolGuard::new();
        let manager = HookManager::new();
        guard.register_hooks(&manager);

        let mut bash_ctx = HookContext::new(HookPoint::SkillBefore, "test_agent", "CA")
            .with_data("tool_name", Value::String("bash".to_string()));
        let decision = manager
            .execute_decision(HookPoint::SkillBefore, &mut bash_ctx)
            .await;
        assert_eq!(decision.control, crate::tools::hooks::HookControl::Continue);
        let injections = bash_ctx
            .metadata
            .get("guard_pre_injections")
            .and_then(Value::as_array)
            .expect("bash must receive its CodeExecution guard rules");
        let combined = injections
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(combined.contains("expected-negative scenario"));
        assert!(combined.contains("exact expected exit class and observable condition"));
        assert!(combined.contains("Never use `|| true`"));

        let mut code_ctx = HookContext::new(HookPoint::SkillBefore, "test_agent", "CA")
            .with_data("tool_name", Value::String("code_execute".to_string()));
        manager
            .execute_decision(HookPoint::SkillBefore, &mut code_ctx)
            .await;
        assert!(
            !code_ctx.metadata.contains_key("guard_pre_injections"),
            "shell-scoped rules must not leak to another tool in the same category"
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
    fn test_exit_code_check_does_not_ignore_structured_error_with_zero_exit() {
        let result = json!({"exit_code": 0, "error": {"kind": "runtime"}});
        let outcome = validators::exit_code_check(&result);
        assert!(matches!(outcome, ValidationOutcome::Fail(_)));
    }

    #[test]
    fn only_plain_nonzero_exit_owns_the_toolguard_root_cause_log() {
        assert_eq!(
            ToolGuard::nonzero_exit_without_error_field(
                &json!({"exit_code": 2, "stderr": "syntax error"})
            ),
            Some(2)
        );
        assert_eq!(
            ToolGuard::nonzero_exit_without_error_field(
                &json!({"exit_code": 2, "error": "spawn wrapper failed"})
            ),
            None,
            "the execution layer owns results with an error field"
        );
        assert_eq!(
            ToolGuard::nonzero_exit_without_error_field(&json!({"exit_code": 0})),
            None
        );
    }

    fn skill_after_context(tool_name: &str, result: Value) -> HookContext {
        HookContext::new(HookPoint::SkillAfter, "test_agent", "DA")
            .with_data("tool_name", Value::String(tool_name.to_string()))
            .with_data("tool_result", Value::String(result.to_string()))
    }

    #[tokio::test]
    async fn bash_non_zero_exit_is_diagnostic_not_disclosure_abort() {
        let guard = ToolGuard::new();
        let manager = HookManager::new();
        guard.register_hooks(&manager);
        let mut ctx = skill_after_context(
            "bash",
            json!({"exit_code": 2, "stdout": "", "stderr": "syntax error"}),
        );

        let decision = manager
            .execute_decision(HookPoint::SkillAfter, &mut ctx)
            .await;

        assert_eq!(decision.control, crate::tools::hooks::HookControl::Continue);
        assert_eq!(decision.terminal_hook, None);
        assert!(ctx
            .error
            .as_deref()
            .is_some_and(|error| error.contains("Non-zero exit code: 2")));
        let feedback = ctx
            .metadata
            .get(TOOL_GUARD_VALIDATION_FEEDBACK_KEY)
            .and_then(Value::as_array)
            .expect("failed command must retain structured validation feedback");
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0]["classification"], "tool_execution_failure");
        assert_eq!(feedback[0]["blocks_disclosure"], false);
        assert!(feedback[0]["fix_instruction"]
            .as_str()
            .is_some_and(|instruction| instruction.contains("exact assertion wrapper")));
        let audit = guard.get_audit_log();
        assert_eq!(audit.len(), 1);
        assert!(!audit[0].validation_passed);
        assert_eq!(audit[0].error.as_deref(), Some("non_zero_exit"));
        assert_eq!(audit[0].error_length, Some("syntax error".len()));
        assert_eq!(
            audit[0].error_sha256.as_deref(),
            Some(hex::encode(Sha256::digest(b"syntax error")).as_str())
        );
        assert_eq!(audit[0].exit_code, Some(2));
        assert!(!serde_json::to_string(&audit)
            .unwrap()
            .contains("syntax error"));
        assert!(!ctx
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("syntax error"));
        assert!(!ctx.metadata[TOOL_GUARD_VALIDATION_FEEDBACK_KEY]
            .to_string()
            .contains("syntax error"));
    }

    #[test]
    fn audit_ring_buffer_is_bounded_and_evicts_oldest_first() {
        let mut log = GuardAuditBuffer::with_capacity(3);
        for sequence in 0..5 {
            log.push(GuardAuditEntry {
                timestamp: sequence,
                tool_name: "bash".to_string(),
                agent_id: format!("agent-{sequence}"),
                pre_injected: true,
                validation_passed: true,
                retry_count: 0,
                error: None,
                error_length: None,
                error_sha256: None,
                exit_code: None,
            });
        }

        assert_eq!(log.len(), 3);
        assert_eq!(log.dropped_entries(), 2);
        let retained = log.snapshot();
        assert_eq!(retained[0].agent_id, "agent-2");
        assert_eq!(retained[1].agent_id, "agent-3");
        assert_eq!(retained[2].agent_id, "agent-4");
    }

    #[test]
    fn audit_ring_buffer_redacts_legacy_free_form_errors_at_insertion_boundary() {
        let mut log = GuardAuditBuffer::with_capacity(1);
        let secret = "password=hunter2";
        log.push(GuardAuditEntry {
            timestamp: 1,
            tool_name: "bash".to_string(),
            agent_id: "external-writer".to_string(),
            pre_injected: true,
            validation_passed: false,
            retry_count: 0,
            error: Some(secret.to_string()),
            error_length: None,
            error_sha256: None,
            exit_code: Some(99),
        });

        let retained = log.snapshot();
        assert_eq!(retained[0].error.as_deref(), Some("external_failure"));
        assert_eq!(retained[0].error_length, Some(secret.len()));
        assert_eq!(
            retained[0].error_sha256.as_deref(),
            Some(hex::encode(Sha256::digest(secret.as_bytes())).as_str())
        );
        assert_eq!(retained[0].exit_code, None);
        assert!(!serde_json::to_string(&retained).unwrap().contains(secret));
    }

    #[tokio::test]
    async fn nonzero_exit_audit_retains_only_category_length_and_sha256() {
        let guard = ToolGuard::new();
        let manager = HookManager::new();
        guard.register_hooks(&manager);
        let agent_id = format!("audit-redaction-{}", uuid::Uuid::new_v4().hyphenated());
        let sensitive_stderr = "SECRET_TOKEN=do-not-retain\ninvalid calculation";
        let mut ctx = HookContext::new(HookPoint::SkillAfter, &agent_id, "DA")
            .with_data("tool_name", Value::String("bash".to_string()))
            .with_data(
                "tool_result",
                Value::String(
                    json!({
                        "exit_code": 17,
                        "stdout": "",
                        "stderr": sensitive_stderr,
                    })
                    .to_string(),
                ),
            );

        manager
            .execute_decision(HookPoint::SkillAfter, &mut ctx)
            .await;

        let audit = guard.get_audit_log();
        assert_eq!(audit.len(), 1);
        let entry = &audit[0];
        assert_eq!(entry.error.as_deref(), Some("non_zero_exit"));
        assert_eq!(entry.error_length, Some(sensitive_stderr.len()));
        assert_eq!(
            entry.error_sha256.as_deref(),
            Some(hex::encode(Sha256::digest(sensitive_stderr.as_bytes())).as_str())
        );
        assert_eq!(entry.exit_code, Some(17));

        let serialized = serde_json::to_string(entry).unwrap();
        assert!(!serialized.contains("SECRET_TOKEN"));
        assert!(!serialized.contains("do-not-retain"));
        assert!(!ctx
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("SECRET_TOKEN"));
        assert!(!ctx.metadata[TOOL_GUARD_VALIDATION_FEEDBACK_KEY]
            .to_string()
            .contains("SECRET_TOKEN"));

        GUARD_AUDIT_LOG
            .write()
            .retain(|entry| entry.agent_id != agent_id);
    }

    #[test]
    fn audit_entry_deserializes_legacy_shape_without_digest_fields() {
        let entry: GuardAuditEntry = serde_json::from_value(json!({
            "timestamp": 1,
            "tool_name": "bash",
            "agent_id": "legacy-agent",
            "pre_injected": true,
            "validation_passed": false,
            "retry_count": 0,
            "error": "legacy error"
        }))
        .unwrap();

        assert_eq!(entry.error.as_deref(), Some("legacy error"));
        assert_eq!(entry.error_length, None);
        assert_eq!(entry.error_sha256, None);
        assert_eq!(entry.exit_code, None);
    }

    #[tokio::test]
    async fn code_execute_structured_error_is_diagnostic_not_disclosure_abort() {
        let guard = ToolGuard::new();
        let manager = HookManager::new();
        guard.register_hooks(&manager);
        let mut ctx = skill_after_context(
            "code_execute",
            json!({"error": "python interpreter unavailable", "language": "python"}),
        );

        let decision = manager
            .execute_decision(HookPoint::SkillAfter, &mut ctx)
            .await;

        assert_eq!(decision.control, crate::tools::hooks::HookControl::Continue);
        assert!(ctx
            .error
            .as_deref()
            .is_some_and(|error| error.contains("Command execution returned error")));
        assert_eq!(
            ctx.metadata[TOOL_GUARD_VALIDATION_FEEDBACK_KEY][0]["tool_name"],
            "code_execute"
        );
        assert_eq!(
            decision.records[0].result,
            crate::tools::hooks::HookResult::Continue
        );
    }

    #[tokio::test]
    async fn file_read_error_is_visible_quality_feedback_not_policy_abort() {
        let guard = ToolGuard::new();
        let manager = HookManager::new();
        guard.register_hooks(&manager);
        let mut ctx = skill_after_context(
            "file_read",
            json!({
                "path": "/workspace",
                "total_lines": 0,
                "returned": 0,
                "lines": [],
                "error": "path is a directory"
            }),
        );

        let decision = manager
            .execute_decision(HookPoint::SkillAfter, &mut ctx)
            .await;

        assert_eq!(decision.control, crate::tools::hooks::HookControl::Continue);
        let feedback = ctx.metadata[TOOL_GUARD_VALIDATION_FEEDBACK_KEY]
            .as_array()
            .expect("file read failure feedback");
        assert_eq!(feedback[0]["classification"], "tool_result_quality_failure");
        assert_eq!(feedback[0]["blocks_disclosure"], false);
        assert!(ctx
            .error
            .as_deref()
            .is_some_and(|error| error.contains("File read returned error")));
    }

    #[tokio::test]
    async fn genuine_post_execution_policy_abort_remains_terminal() {
        let guard = ToolGuard::new();
        let manager = HookManager::new();
        guard.register_hooks(&manager);
        manager.register(Box::new(FunctionHook::new(
            "deny_tool_result_disclosure",
            vec![HookPoint::SkillAfter],
            90,
            |_| HookResult::Abort,
        )));
        let mut ctx = skill_after_context(
            "bash",
            json!({"exit_code": 9, "stdout": "sensitive", "stderr": "failed"}),
        );

        let decision = manager
            .execute_decision(HookPoint::SkillAfter, &mut ctx)
            .await;

        assert_eq!(decision.control, crate::tools::hooks::HookControl::Abort);
        assert_eq!(
            decision.terminal_hook.as_deref(),
            Some("deny_tool_result_disclosure")
        );
        assert_eq!(decision.records.len(), 2);
        assert_eq!(decision.records[0].hook_name, "toolguard::post_validate");
        assert_eq!(decision.records[0].result, HookResult::Continue);
        assert_eq!(decision.records[1].result, HookResult::Abort);
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
        // Verify the global log by identity rather than a racy length delta:
        // SkillAfter tests legitimately append to this process-wide sink in
        // parallel.
        let marker = format!("test-agent-{}", uuid::Uuid::new_v4().hyphenated());
        GUARD_AUDIT_LOG.write().push(GuardAuditEntry {
            timestamp: 0,
            tool_name: "test".to_string(),
            agent_id: marker.clone(),
            pre_injected: true,
            validation_passed: false,
            retry_count: 1,
            error: Some("test_failure".to_string()),
            error_length: Some(10),
            error_sha256: Some(hex::encode(Sha256::digest(b"test error"))),
            exit_code: None,
        });
        assert!(GUARD_AUDIT_LOG
            .read()
            .iter()
            .any(|entry| entry.agent_id == marker));
        GUARD_AUDIT_LOG
            .write()
            .retain(|entry| entry.agent_id != marker);
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
