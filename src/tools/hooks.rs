use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HookPoint {
    AgentInit,
    AgentStart,
    AgentEnd,
    AgentError,
    TaskStart,
    TaskEnd,
    TaskError,
    LlmRequest,
    LlmResponse,
    MemoryWrite,
    MemoryRead,
    SkillBefore,
    SkillAfter,
    BlackboardWrite,
    BlackboardRead,
    PhaseStart,
    PhaseEnd,
    CycleStart,
    CycleEnd,
    McpToolCall,
    McpToolResult,
}

impl HookPoint {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AgentInit => "agent_init",
            Self::AgentStart => "agent_start",
            Self::AgentEnd => "agent_end",
            Self::AgentError => "agent_error",
            Self::TaskStart => "task_start",
            Self::TaskEnd => "task_end",
            Self::TaskError => "task_error",
            Self::LlmRequest => "llm_request",
            Self::LlmResponse => "llm_response",
            Self::MemoryWrite => "memory_write",
            Self::MemoryRead => "memory_read",
            Self::SkillBefore => "skill_before",
            Self::SkillAfter => "skill_after",
            Self::BlackboardWrite => "blackboard_write",
            Self::BlackboardRead => "blackboard_read",
            Self::PhaseStart => "phase_start",
            Self::PhaseEnd => "phase_end",
            Self::CycleStart => "cycle_start",
            Self::CycleEnd => "cycle_end",
            Self::McpToolCall => "mcp_tool_call",
            Self::McpToolResult => "mcp_tool_result",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HookResult {
    Continue,
    Skip,
    Abort,
    Retry,
    Modify,
    SkipRemaining,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookContext {
    pub hook_point: HookPoint,
    pub agent_id: String,
    pub agent_role: String,
    pub task_id: Option<String>,
    pub task_iri: Option<String>,
    pub data: HashMap<String, Value>,
    pub metadata: HashMap<String, Value>,
    pub timestamp: u64,
    pub error: Option<String>,
}

impl HookContext {
    pub fn new(hook_point: HookPoint, agent_id: &str, agent_role: &str) -> Self {
        Self {
            hook_point,
            agent_id: agent_id.to_string(),
            agent_role: agent_role.to_string(),
            task_id: None,
            task_iri: None,
            data: HashMap::new(),
            metadata: HashMap::new(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            error: None,
        }
    }

    pub fn with_task(mut self, task_id: &str, task_iri: &str) -> Self {
        self.task_id = Some(task_id.to_string());
        self.task_iri = Some(task_iri.to_string());
        self
    }

    pub fn with_data(mut self, key: &str, value: Value) -> Self {
        self.data.insert(key.to_string(), value);
        self
    }

    pub fn with_error(mut self, error: &str) -> Self {
        self.error = Some(error.to_string());
        self
    }
}

#[async_trait]
pub trait Hook: Send + Sync {
    fn name(&self) -> &str;
    fn hook_points(&self) -> Vec<HookPoint>;
    fn priority(&self) -> i32 {
        100
    }

    async fn execute(&self, context: &mut HookContext) -> HookResult;
}

#[derive(Clone)]
pub struct FunctionHook {
    name: String,
    hook_points: Vec<HookPoint>,
    priority: i32,
    handler: Arc<dyn Fn(&mut HookContext) -> HookResult + Send + Sync>,
}

impl FunctionHook {
    pub fn new<F>(name: &str, hook_points: Vec<HookPoint>, priority: i32, handler: F) -> Self
    where
        F: Fn(&mut HookContext) -> HookResult + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            hook_points,
            priority,
            handler: Arc::new(handler),
        }
    }
}

#[async_trait]
impl Hook for FunctionHook {
    fn name(&self) -> &str {
        &self.name
    }
    fn hook_points(&self) -> Vec<HookPoint> {
        self.hook_points.clone()
    }
    fn priority(&self) -> i32 {
        self.priority
    }

    async fn execute(&self, context: &mut HookContext) -> HookResult {
        (self.handler)(context)
    }
}

pub struct AsyncFunctionHook {
    name: String,
    hook_points: Vec<HookPoint>,
    priority: i32,
    handler: Arc<
        dyn Fn(
                &mut HookContext,
            )
                -> std::pin::Pin<Box<dyn std::future::Future<Output = HookResult> + Send>>
            + Send
            + Sync,
    >,
}

impl AsyncFunctionHook {
    pub fn new<F, Fut>(name: &str, hook_points: Vec<HookPoint>, priority: i32, handler: F) -> Self
    where
        F: Fn(&mut HookContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = HookResult> + Send + 'static,
    {
        Self {
            name: name.to_string(),
            hook_points,
            priority,
            handler: Arc::new(move |ctx| Box::pin(handler(ctx))),
        }
    }
}

#[async_trait]
impl Hook for AsyncFunctionHook {
    fn name(&self) -> &str {
        &self.name
    }
    fn hook_points(&self) -> Vec<HookPoint> {
        self.hook_points.clone()
    }
    fn priority(&self) -> i32 {
        self.priority
    }

    async fn execute(&self, context: &mut HookContext) -> HookResult {
        (self.handler)(context).await
    }
}

pub struct LoggingHook;

impl LoggingHook {
    pub fn new() -> Box<dyn Hook> {
        Box::new(FunctionHook::new(
            "logging",
            vec![
                HookPoint::AgentStart,
                HookPoint::AgentEnd,
                HookPoint::TaskStart,
                HookPoint::TaskEnd,
                HookPoint::PhaseStart,
                HookPoint::PhaseEnd,
            ],
            1000,
            |ctx| {
                tracing::info!(
                    "[{}] [{}] {}",
                    ctx.timestamp,
                    ctx.agent_id,
                    ctx.hook_point.as_str()
                );
                HookResult::Continue
            },
        ))
    }
}

pub struct TimingHook {
    #[allow(dead_code)]
    timings: Arc<RwLock<HashMap<String, u64>>>,
}

impl TimingHook {
    pub fn new() -> Box<dyn Hook> {
        let timings = Arc::new(RwLock::new(HashMap::new()));
        let timings_clone = timings.clone();

        Box::new(FunctionHook::new(
            "timing",
            vec![
                HookPoint::TaskStart,
                HookPoint::TaskEnd,
                HookPoint::SkillBefore,
                HookPoint::SkillAfter,
                HookPoint::LlmRequest,
                HookPoint::LlmResponse,
            ],
            0,
            move |ctx| {
                let key = format!(
                    "{}:{}:{}",
                    ctx.agent_id,
                    ctx.task_id.as_deref().unwrap_or("none"),
                    ctx.hook_point.as_str()
                );

                match ctx.hook_point {
                    HookPoint::TaskStart | HookPoint::SkillBefore | HookPoint::LlmRequest => {
                        let mut timings = timings_clone.write();
                        timings.insert(key, ctx.timestamp);
                    }
                    _ => {
                        let start_key = key
                            .replace("_end", "_start")
                            .replace("_after", "_before")
                            .replace("_response", "_request");
                        let mut timings = timings_clone.write();
                        if let Some(start) = timings.remove(&start_key) {
                            let duration = ctx.timestamp.saturating_sub(start);
                            ctx.metadata.insert(
                                "duration_seconds".to_string(),
                                Value::Number(duration.into()),
                            );
                        }
                    }
                }
                HookResult::Continue
            },
        ))
    }
}

pub struct RateLimitHook {
    #[allow(dead_code)]
    max_calls: usize,
    #[allow(dead_code)]
    window_seconds: u64,
    #[allow(dead_code)]
    calls: Arc<RwLock<HashMap<String, Vec<u64>>>>,
}

impl RateLimitHook {
    pub fn new(max_calls: usize, window_seconds: u64) -> Box<dyn Hook> {
        let calls = Arc::new(RwLock::new(HashMap::new()));
        let calls_clone = calls.clone();

        Box::new(FunctionHook::new(
            "rate_limit",
            vec![HookPoint::LlmRequest],
            10,
            move |ctx| {
                let agent_id = ctx.agent_id.clone();
                let now = ctx.timestamp;

                let mut calls = calls_clone.write();
                let entry: &mut Vec<u64> = calls.entry(agent_id.clone()).or_default();

                entry.retain(|&t| now.saturating_sub(t) < window_seconds);

                if entry.len() >= max_calls {
                    ctx.error = Some("Rate limit exceeded".to_string());
                    return HookResult::Abort;
                }

                entry.push(now);
                HookResult::Continue
            },
        ))
    }
}

pub struct MetricsHook {
    #[allow(dead_code)]
    metrics: Arc<RwLock<HashMap<String, Vec<Value>>>>,
}

impl MetricsHook {
    pub fn new() -> Box<dyn Hook> {
        let metrics = Arc::new(RwLock::new(HashMap::new()));
        let metrics_clone = metrics.clone();

        Box::new(FunctionHook::new(
            "metrics",
            vec![
                HookPoint::TaskEnd,
                HookPoint::SkillAfter,
                HookPoint::LlmResponse,
                HookPoint::CycleEnd,
            ],
            500,
            move |ctx| {
                let metric_name = ctx.hook_point.as_str().to_string();
                let mut metrics = metrics_clone.write();

                let entry: &mut Vec<Value> = metrics.entry(metric_name).or_default();
                entry.push(serde_json::json!({
                    "agent_id": ctx.agent_id,
                    "task_id": ctx.task_id,
                    "timestamp": ctx.timestamp,
                    "metadata": ctx.metadata,
                }));

                HookResult::Continue
            },
        ))
    }
}

pub struct HookManager {
    hooks: RwLock<HashMap<HookPoint, Vec<Arc<dyn Hook>>>>,
}

impl HookManager {
    pub fn new() -> Self {
        Self {
            hooks: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_default_hooks() -> Self {
        let manager = Self::new();
        manager.register(LoggingHook::new());
        manager.register(TimingHook::new());
        manager.register(RateLimitHook::new(100, 60));
        manager.register(MetricsHook::new());
        manager
    }

    pub fn register(&self, hook: Box<dyn Hook>) {
        let hook: Arc<dyn Hook> = hook.into();
        let mut hooks = self.hooks.write();
        for point in hook.hook_points() {
            let entry = hooks.entry(point).or_default();
            entry.push(hook.clone());
            entry.sort_by_key(|h| h.priority());
        }
    }

    pub fn register_arc(&self, hook: Arc<dyn Hook>) {
        let mut hooks = self.hooks.write();
        for point in hook.hook_points() {
            let entry = hooks.entry(point).or_default();
            entry.push(hook.clone());
            entry.sort_by_key(|h| h.priority());
        }
    }

    /// Replace every registration with the same stable hook name, then
    /// register this hook for its declared points. This is intentionally
    /// opt-in: ordinary hook registration remains additive, while components
    /// that upgrade an implementation (for example RootCause → fused
    /// RootCause) can avoid executing both implementations.
    pub fn replace_arc(&self, hook: Arc<dyn Hook>) {
        let hook_name = hook.name().to_string();
        let hook_points = hook.hook_points();
        let mut hooks = self.hooks.write();
        for registered in hooks.values_mut() {
            registered.retain(|existing| existing.name() != hook_name);
        }
        for point in hook_points {
            let entry = hooks.entry(point).or_default();
            entry.push(hook.clone());
            entry.sort_by_key(|registered| registered.priority());
        }
    }

    pub async fn execute(&self, hook_point: HookPoint, context: &mut HookContext) -> HookResult {
        let hooks: Vec<Arc<dyn Hook>> = {
            let guard = self.hooks.read();
            guard
                .get(&hook_point)
                .map(|v| v.clone())
                .unwrap_or_default()
        };

        if hooks.is_empty() {
            return HookResult::Continue;
        }

        let mut result = HookResult::Continue;

        for hook in &hooks {
            match hook.execute(context).await {
                HookResult::Continue => {}
                HookResult::Abort => {
                    result = HookResult::Abort;
                    break;
                }
                HookResult::Modify => {
                    result = HookResult::Continue;
                }
                HookResult::SkipRemaining => {
                    result = HookResult::Continue;
                    break;
                }
                other => {
                    result = other;
                }
            }
        }

        result
    }

    pub fn get_hooks(&self, hook_point: HookPoint) -> Vec<String> {
        let hooks = self.hooks.read();
        hooks
            .get(&hook_point)
            .map(|h| h.iter().map(|hook| hook.name().to_string()).collect())
            .unwrap_or_default()
    }
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Human Approval Hook structures
// ============================================================

use chrono::{DateTime, Utc};

/// Approval condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalCondition {
    /// Always requires approval
    Always,
    /// Approve on failure
    OnFailure,
    /// Approve on stage completion
    OnStageComplete,
    /// Custom condition (LLM judgement)
    Custom(String),
}

/// Timeout default behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DefaultAction {
    /// Approve on timeout
    Approve,
    /// Reject on timeout
    Reject,
    /// Retry on timeout
    Retry,
    /// Abort on timeout
    Abort,
}

impl Default for DefaultAction {
    fn default() -> Self {
        Self::Approve
    }
}

/// Approval point configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalPoint {
    /// Hook point to trigger on
    pub hook_point: HookPoint,
    /// Trigger condition
    pub condition: ApprovalCondition,
    /// Message template
    pub message_template: String,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// Default action on timeout
    pub default_action: DefaultAction,
    /// Applicable stages (empty means all stages)
    pub stages: Vec<String>,
}

impl Default for ApprovalPoint {
    fn default() -> Self {
        Self {
            hook_point: HookPoint::PhaseEnd,
            condition: ApprovalCondition::OnStageComplete,
            message_template: "Stage {stage} completed, please confirm whether to continue"
                .to_string(),
            timeout_seconds: 3600,
            default_action: DefaultAction::Approve,
            stages: Vec::new(),
        }
    }
}

/// Approval request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Request ID
    pub request_id: String,
    /// Task IRI
    pub task_iri: String,
    /// Stage ID
    pub stage_id: String,
    /// Message content
    pub message: String,
    /// Available options
    pub options: Vec<String>,
    /// Creation time
    pub created_at: DateTime<Utc>,
}

impl ApprovalRequest {
    pub fn new(task_iri: String, stage_id: String, message: String, options: Vec<String>) -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            task_iri,
            stage_id,
            message,
            options,
            created_at: Utc::now(),
        }
    }
}

/// Approval response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalResponse {
    /// Corresponding request ID
    pub request_id: String,
    /// Stage ID
    pub stage_id: String,
    /// Whether approved
    pub approved: bool,
    /// Comments
    pub comments: Option<String>,
    /// Response time
    pub responded_at: DateTime<Utc>,
}

impl ApprovalResponse {
    pub fn approved(request_id: String, stage_id: String, comments: Option<String>) -> Self {
        Self {
            request_id,
            stage_id,
            approved: true,
            comments,
            responded_at: Utc::now(),
        }
    }

    pub fn rejected(request_id: String, stage_id: String, comments: Option<String>) -> Self {
        Self {
            request_id,
            stage_id,
            approved: false,
            comments,
            responded_at: Utc::now(),
        }
    }

    pub fn timeout(request_id: String, stage_id: String) -> Self {
        Self {
            request_id,
            stage_id,
            approved: false,
            comments: Some("Approval timeout".to_string()),
            responded_at: Utc::now(),
        }
    }
}

/// Approval state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalState {
    /// The request
    pub request: ApprovalRequest,
    /// Response (if any)
    pub response: Option<ApprovalResponse>,
    /// Whether processed
    pub processed: bool,
}

/// Human Approval Hook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanApprovalConfig {
    /// Whether enabled
    pub enabled: bool,
    /// List of approval points
    pub approval_points: Vec<ApprovalPoint>,
    /// Default timeout in seconds
    pub default_timeout_seconds: u64,
    /// Default action on timeout
    pub default_action: DefaultAction,
}

impl Default for HumanApprovalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            approval_points: Vec::new(),
            default_timeout_seconds: 3600,
            default_action: DefaultAction::Approve,
        }
    }
}

/// Approval notifier trait
#[async_trait]
pub trait ApprovalNotifier: Send + Sync {
    /// Send an approval request
    async fn notify(
        &self,
        request: &ApprovalRequest,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Wait for an approval response
    async fn wait_for_response(
        &self,
        request_id: &str,
        timeout: std::time::Duration,
    ) -> Option<ApprovalResponse>;
}

/// Channel-based approval notifier (for testing and in-process communication)
pub struct ChannelApprovalNotifier {
    pending: Arc<RwLock<HashMap<String, ApprovalState>>>,
    waiters: parking_lot::Mutex<HashMap<String, Arc<tokio::sync::Notify>>>,
}

impl ChannelApprovalNotifier {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(RwLock::new(HashMap::new())),
            waiters: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    pub fn get_pending(&self) -> Vec<ApprovalRequest> {
        let pending = self.pending.read();
        pending
            .values()
            .filter(|s| !s.processed && s.response.is_none())
            .map(|s| s.request.clone())
            .collect()
    }

    pub async fn submit_response(&self, response: ApprovalResponse) {
        {
            let mut pending = self.pending.write();
            if let Some(state) = pending.get_mut(&response.request_id) {
                state.response = Some(response.clone());
            }
        }
        if let Some(waiter) = self.waiters.lock().get(&response.request_id).cloned() {
            waiter.notify_one();
        }
    }
}

impl Default for ChannelApprovalNotifier {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ApprovalNotifier for ChannelApprovalNotifier {
    async fn notify(
        &self,
        request: &ApprovalRequest,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut pending = self.pending.write();
        pending.insert(
            request.request_id.clone(),
            ApprovalState {
                request: request.clone(),
                response: None,
                processed: false,
            },
        );
        self.waiters
            .lock()
            .entry(request.request_id.clone())
            .or_insert_with(|| Arc::new(tokio::sync::Notify::new()));
        Ok(())
    }

    async fn wait_for_response(
        &self,
        request_id: &str,
        timeout: std::time::Duration,
    ) -> Option<ApprovalResponse> {
        let waiter = self
            .waiters
            .lock()
            .entry(request_id.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Notify::new()))
            .clone();
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = waiter.notified();
            if let Some(response) = {
                let mut pending = self.pending.write();
                pending.get_mut(request_id).and_then(|state| {
                    let response = state.response.clone();
                    if response.is_some() {
                        state.processed = true;
                    }
                    response
                })
            } {
                self.waiters.lock().remove(request_id);
                return Some(response);
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                self.waiters.lock().remove(request_id);
                return None;
            }
            if tokio::time::Instant::now() >= deadline {
                self.waiters.lock().remove(request_id);
                return None;
            }
        }
    }
}

/// Human Approval Hook
pub struct HumanApprovalHook {
    config: HumanApprovalConfig,
    notifier: Arc<dyn ApprovalNotifier>,
}

impl HumanApprovalHook {
    pub fn new(config: HumanApprovalConfig, notifier: Arc<dyn ApprovalNotifier>) -> Box<Self> {
        Box::new(Self { config, notifier })
    }

    pub fn with_channel_notifier(
        config: HumanApprovalConfig,
    ) -> (Box<Self>, Arc<ChannelApprovalNotifier>) {
        let notifier = Arc::new(ChannelApprovalNotifier::new());
        let hook = Box::new(Self {
            config,
            notifier: notifier.clone(),
        });
        (hook, notifier)
    }

    fn needs_approval(&self, ctx: &HookContext) -> bool {
        if !self.config.enabled {
            return false;
        }
        self.config
            .approval_points
            .iter()
            .any(|point| Self::point_matches(point, ctx))
    }

    fn create_request(&self, ctx: &HookContext) -> ApprovalRequest {
        let stage_id = ctx
            .data
            .get("stage_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let task_iri = ctx.task_iri.clone().unwrap_or_default();

        let message = ctx
            .error
            .as_ref()
            .map(|e| format!("Execution error: {}, please confirm whether to continue", e))
            .unwrap_or_else(|| {
                format!(
                    "Stage {} completed, please confirm whether to continue",
                    stage_id
                )
            });

        ApprovalRequest::new(
            task_iri,
            stage_id,
            message,
            vec![
                "Approve".to_string(),
                "Reject".to_string(),
                "Rollback".to_string(),
            ],
        )
    }

    fn find_matching_point(&self, ctx: &HookContext) -> Option<&ApprovalPoint> {
        self.config
            .approval_points
            .iter()
            .find(|point| Self::point_matches(point, ctx))
    }

    fn point_matches(point: &ApprovalPoint, ctx: &HookContext) -> bool {
        if point.hook_point != ctx.hook_point {
            return false;
        }
        match &point.condition {
            ApprovalCondition::Always => true,
            ApprovalCondition::OnFailure => ctx.error.is_some(),
            ApprovalCondition::OnStageComplete => ctx
                .data
                .get("stage_id")
                .and_then(Value::as_str)
                .is_some_and(|stage| {
                    point.stages.is_empty() || point.stages.iter().any(|item| item == stage)
                }),
            ApprovalCondition::Custom(expression) => custom_condition_matches(expression, ctx),
        }
    }
}

/// Evaluate the deliberately small, deterministic custom approval DSL.
/// Supported forms are `error`, `no_error`, `data.KEY`, `metadata.KEY`, and
/// `data.KEY == JSON_VALUE` (likewise for metadata). Unknown expressions fail
/// closed by not requesting approval and emit a warning.
fn custom_condition_matches(expression: &str, ctx: &HookContext) -> bool {
    let expression = expression.trim();
    match expression {
        "error" => return ctx.error.is_some(),
        "no_error" => return ctx.error.is_none(),
        _ => {}
    }

    let (path, expected) = expression
        .split_once("==")
        .map_or((expression, None), |(path, value)| {
            (path.trim(), Some(value.trim()))
        });
    let value = path
        .strip_prefix("data.")
        .and_then(|key| ctx.data.get(key))
        .or_else(|| {
            path.strip_prefix("metadata.")
                .and_then(|key| ctx.metadata.get(key))
        });
    let Some(value) = value else {
        tracing::warn!(condition = %expression, "Unknown or missing custom approval condition path");
        return false;
    };
    let Some(expected) = expected else {
        return match value {
            Value::Null => false,
            Value::Bool(value) => *value,
            Value::String(value) => !value.is_empty(),
            Value::Array(value) => !value.is_empty(),
            Value::Object(value) => !value.is_empty(),
            Value::Number(_) => true,
        };
    };
    let expected_value = serde_json::from_str(expected)
        .unwrap_or_else(|_| Value::String(expected.trim_matches(['\'', '"']).to_string()));
    value == &expected_value
}

#[async_trait]
impl Hook for HumanApprovalHook {
    fn name(&self) -> &str {
        "human_approval"
    }

    fn hook_points(&self) -> Vec<HookPoint> {
        self.config
            .approval_points
            .iter()
            .map(|p| p.hook_point)
            .collect()
    }

    fn priority(&self) -> i32 {
        0 // high priority
    }

    async fn execute(&self, ctx: &mut HookContext) -> HookResult {
        if !self.needs_approval(ctx) {
            return HookResult::Continue;
        }

        let request = self.create_request(ctx);
        let request_id = request.request_id.clone();
        let point = self.find_matching_point(ctx);
        let timeout = point
            .map(|p| std::time::Duration::from_secs(p.timeout_seconds))
            .unwrap_or_else(|| std::time::Duration::from_secs(self.config.default_timeout_seconds));
        let default_action = point
            .map(|p| p.default_action.clone())
            .unwrap_or_else(|| self.config.default_action.clone());

        tracing::info!(
            request_id = %request_id,
            stage_id = %request.stage_id,
            "sending approval request"
        );

        if let Err(e) = self.notifier.notify(&request).await {
            tracing::error!("Failed to send approval request: {}", e);
            return HookResult::Continue;
        }

        match self.notifier.wait_for_response(&request_id, timeout).await {
            Some(response) if response.approved => {
                tracing::info!(request_id = %request_id, "user approved");
                HookResult::Continue
            }
            Some(response) => {
                tracing::warn!(request_id = %request_id, comments = ?response.comments, "user rejected");
                ctx.error = Some(format!("User rejected: {:?}", response.comments));
                HookResult::Abort
            }
            None => {
                tracing::warn!(request_id = %request_id, "approval timeout");
                match default_action {
                    DefaultAction::Approve => HookResult::Continue,
                    DefaultAction::Reject => {
                        ctx.error = Some("Approval timeout, auto-rejected".to_string());
                        HookResult::Abort
                    }
                    DefaultAction::Retry => HookResult::Retry,
                    DefaultAction::Abort => HookResult::Abort,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_approval_conditions_use_deterministic_context_expressions() {
        let mut context = HookContext::new(HookPoint::PhaseEnd, "agent", "DA")
            .with_data("risk", serde_json::json!("high"))
            .with_data("requires_review", serde_json::json!(true));
        context
            .metadata
            .insert("attempt".to_string(), serde_json::json!(3));

        assert!(custom_condition_matches("data.requires_review", &context));
        assert!(custom_condition_matches("data.risk == \"high\"", &context));
        assert!(custom_condition_matches("metadata.attempt == 3", &context));
        assert!(custom_condition_matches("no_error", &context));
        assert!(!custom_condition_matches("data.risk == \"low\"", &context));
        assert!(!custom_condition_matches("data.missing", &context));
    }

    #[test]
    fn stage_filter_is_consistent_for_selection_and_triggering() {
        let point = ApprovalPoint {
            hook_point: HookPoint::PhaseEnd,
            condition: ApprovalCondition::OnStageComplete,
            stages: vec!["review".to_string()],
            ..ApprovalPoint::default()
        };
        let review = HookContext::new(HookPoint::PhaseEnd, "agent", "DA")
            .with_data("stage_id", serde_json::json!("review"));
        let build = HookContext::new(HookPoint::PhaseEnd, "agent", "DA")
            .with_data("stage_id", serde_json::json!("build"));
        assert!(HumanApprovalHook::point_matches(&point, &review));
        assert!(!HumanApprovalHook::point_matches(&point, &build));
    }

    #[tokio::test]
    async fn approval_responses_are_routed_to_concurrent_waiters() {
        let notifier = Arc::new(ChannelApprovalNotifier::new());
        let first = ApprovalRequest::new(
            "iri://task/one".into(),
            "stage-one".into(),
            "approve one".into(),
            vec!["approve".into(), "reject".into()],
        );
        let second = ApprovalRequest::new(
            "iri://task/two".into(),
            "stage-two".into(),
            "approve two".into(),
            vec!["approve".into(), "reject".into()],
        );
        notifier.notify(&first).await.unwrap();
        notifier.notify(&second).await.unwrap();

        let first_waiter = {
            let notifier = notifier.clone();
            let request_id = first.request_id.clone();
            tokio::spawn(async move {
                notifier
                    .wait_for_response(&request_id, std::time::Duration::from_secs(1))
                    .await
            })
        };
        let second_waiter = {
            let notifier = notifier.clone();
            let request_id = second.request_id.clone();
            tokio::spawn(async move {
                notifier
                    .wait_for_response(&request_id, std::time::Duration::from_secs(1))
                    .await
            })
        };
        notifier
            .submit_response(ApprovalResponse::rejected(
                second.request_id.clone(),
                second.stage_id.clone(),
                None,
            ))
            .await;
        notifier
            .submit_response(ApprovalResponse::approved(
                first.request_id.clone(),
                first.stage_id.clone(),
                None,
            ))
            .await;

        let first_response = first_waiter.await.unwrap().unwrap();
        let second_response = second_waiter.await.unwrap().unwrap();
        assert_eq!(first_response.request_id, first.request_id);
        assert!(first_response.approved);
        assert_eq!(second_response.request_id, second.request_id);
        assert!(!second_response.approved);
    }

    #[tokio::test]
    async fn test_hook_manager() {
        let manager = HookManager::new();

        let hook = FunctionHook::new("test_hook", vec![HookPoint::TaskStart], 100, |ctx| {
            ctx.data.insert("hooked".to_string(), Value::Bool(true));
            HookResult::Continue
        });

        manager.register(Box::new(hook));

        let mut context = HookContext::new(HookPoint::TaskStart, "agent_1", "DA");

        let result = manager.execute(HookPoint::TaskStart, &mut context).await;

        assert_eq!(result, HookResult::Continue);
        assert_eq!(context.data.get("hooked"), Some(&Value::Bool(true)));
    }

    #[tokio::test]
    async fn replace_arc_removes_prior_registration_with_the_same_name() {
        let manager = HookManager::new();
        let old_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let new_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        manager.register_arc(Arc::new(FunctionHook::new(
            "upgradeable_hook",
            vec![HookPoint::TaskError],
            10,
            {
                let old_calls = old_calls.clone();
                move |_| {
                    old_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    HookResult::Continue
                }
            },
        )));

        let mut before = HookContext::new(HookPoint::TaskError, "agent", "DA");
        manager.execute(HookPoint::TaskError, &mut before).await;
        assert_eq!(old_calls.load(std::sync::atomic::Ordering::Relaxed), 1);

        manager.replace_arc(Arc::new(FunctionHook::new(
            "upgradeable_hook",
            vec![HookPoint::TaskError],
            10,
            {
                let new_calls = new_calls.clone();
                move |_| {
                    new_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    HookResult::Continue
                }
            },
        )));

        let mut after = HookContext::new(HookPoint::TaskError, "agent", "DA");
        manager.execute(HookPoint::TaskError, &mut after).await;
        assert_eq!(old_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(new_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_rate_limit_hook() {
        let manager = HookManager::new();
        manager.register(RateLimitHook::new(2, 60));

        let mut ctx1 = HookContext::new(HookPoint::LlmRequest, "agent_1", "DA");
        let result1 = manager.execute(HookPoint::LlmRequest, &mut ctx1).await;
        assert_eq!(result1, HookResult::Continue);

        let mut ctx2 = HookContext::new(HookPoint::LlmRequest, "agent_1", "DA");
        let result2 = manager.execute(HookPoint::LlmRequest, &mut ctx2).await;
        assert_eq!(result2, HookResult::Continue);

        let mut ctx3 = HookContext::new(HookPoint::LlmRequest, "agent_1", "DA");
        let result3 = manager.execute(HookPoint::LlmRequest, &mut ctx3).await;
        assert_eq!(result3, HookResult::Abort);
    }

    #[test]
    fn test_hook_context() {
        let ctx = HookContext::new(HookPoint::TaskStart, "agent_1", "DA")
            .with_task("task_123", "iri://task/123")
            .with_data("key", Value::String("value".to_string()));

        assert_eq!(ctx.agent_id, "agent_1");
        assert_eq!(ctx.agent_role, "DA");
        assert_eq!(ctx.task_id, Some("task_123".to_string()));
        assert_eq!(
            ctx.data.get("key"),
            Some(&Value::String("value".to_string()))
        );
    }
}
