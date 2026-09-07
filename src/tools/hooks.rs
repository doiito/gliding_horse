use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::FutureExt;
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

/// Runtime support promise for a hook point. Experimental points are part of
/// the forward-compatible vocabulary but are not guaranteed to be emitted by
/// every execution path yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HookSupportLevel {
    Stable,
    Experimental,
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

    /// Advertise only hook points whose current invocation coverage matches
    /// their public contract as stable. This prevents declaration in the enum
    /// from being mistaken for end-to-end runtime wiring.
    #[must_use]
    pub const fn support_level(&self) -> HookSupportLevel {
        match self {
            Self::MemoryRead | Self::BlackboardRead | Self::McpToolCall | Self::McpToolResult => {
                HookSupportLevel::Experimental
            }
            Self::AgentInit
            | Self::AgentStart
            | Self::AgentEnd
            | Self::AgentError
            | Self::TaskStart
            | Self::TaskEnd
            | Self::TaskError
            | Self::LlmRequest
            | Self::LlmResponse
            | Self::MemoryWrite
            | Self::SkillBefore
            | Self::SkillAfter
            | Self::BlackboardWrite
            | Self::PhaseStart
            | Self::PhaseEnd
            | Self::CycleStart
            | Self::CycleEnd => HookSupportLevel::Stable,
        }
    }

    #[must_use]
    pub const fn support_note(&self) -> &'static str {
        match self {
            Self::MemoryRead => "declared for memory-reader integration; emission is not universal",
            Self::BlackboardRead => {
                "declared for blackboard-reader integration; emission is not universal"
            }
            Self::McpToolCall | Self::McpToolResult => {
                "declared for MCP transport integration; use SkillBefore/SkillAfter for stable tool coverage"
            }
            _ => "emitted by the primary AgentRunner lifecycle",
        }
    }

    /// Default containment policy when a hook implementation panics. Policy
    /// gates fail closed; observational hooks fail open so telemetry defects
    /// cannot corrupt execution. `PhaseEnd` is also a transition gate, so a
    /// panic blocks the next phase without replaying the completed one.
    #[must_use]
    pub const fn panic_failure_policy(&self) -> HookFailurePolicy {
        match self {
            Self::LlmRequest
            | Self::SkillBefore
            | Self::TaskStart
            | Self::AgentStart
            | Self::PhaseEnd
            | Self::McpToolCall => HookFailurePolicy::FailClosed,
            _ => HookFailurePolicy::FailOpen,
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

/// The action the caller must take after all hooks for one point have run.
///
/// `HookResult` remains the source-compatible return type implemented by
/// existing hooks.  `HookControl` removes the historical ambiguity between
/// "stop evaluating hooks" and "skip the operation being guarded".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HookControl {
    Continue,
    SkipOperation,
    Abort,
    Retry,
}

/// Policy used when a hook cannot complete within its execution budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HookFailurePolicy {
    /// Record the failure and continue evaluating hooks.
    FailOpen,
    /// Abort the guarded operation.
    FailClosed,
    /// Ask a decision-aware caller to retry the guarded operation.
    Retry,
    /// Skip the guarded operation without aborting its enclosing workflow.
    SkipOperation,
}

/// Per-hook execution bounds.  The default deliberately has no timeout so
/// existing approval hooks retain their configured wait behavior.  Hooks that
/// interact with untrusted or remote code should opt into an explicit budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HookExecutionPolicy {
    pub timeout_ms: Option<u64>,
    pub on_timeout: HookFailurePolicy,
    /// `None` inherits [`HookPoint::panic_failure_policy`].
    pub on_panic: Option<HookFailurePolicy>,
}

impl HookExecutionPolicy {
    #[must_use]
    pub const fn bounded(timeout_ms: u64, on_timeout: HookFailurePolicy) -> Self {
        Self {
            timeout_ms: Some(timeout_ms),
            on_timeout,
            on_panic: None,
        }
    }

    #[must_use]
    pub const fn with_panic_policy(mut self, on_panic: HookFailurePolicy) -> Self {
        self.on_panic = Some(on_panic);
        self
    }
}

impl Default for HookExecutionPolicy {
    fn default() -> Self {
        Self {
            timeout_ms: None,
            on_timeout: HookFailurePolicy::FailOpen,
            on_panic: None,
        }
    }
}

/// One auditable entry in a hook-chain decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HookExecutionRecord {
    pub hook_name: String,
    pub result: HookResult,
    pub elapsed_ms: u64,
    pub timed_out: bool,
    pub panicked: bool,
    pub context_modified: bool,
}

/// Structured result for callers that need production-grade hook semantics.
/// Existing callers can continue using [`HookManager::execute`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HookDecision {
    pub control: HookControl,
    /// True only when the caller can apply a validated behavioral patch.
    /// Hook-local annotations remain visible in `records`/`HookContext` but do
    /// not claim that the guarded operation itself changed.
    pub context_modified: bool,
    pub remaining_hooks_skipped: bool,
    pub terminal_hook: Option<String>,
    pub trace_id: String,
    /// Complete replacement arguments accepted from one `SkillBefore`
    /// `Modify` hook. Multiple writers are rejected as a patch conflict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_arguments_patch: Option<Value>,
    pub records: Vec<HookExecutionRecord>,
}

impl HookDecision {
    fn new(trace_id: String, record_capacity: usize) -> Self {
        Self {
            control: HookControl::Continue,
            context_modified: false,
            remaining_hooks_skipped: false,
            terminal_hook: None,
            trace_id,
            tool_arguments_patch: None,
            // Execution records belong to this one decision and are returned
            // to its caller. HookManager never retains a history of them.
            records: Vec::with_capacity(record_capacity),
        }
    }

    /// Preserve the legacy aggregate result contract while decision-aware
    /// callers migrate to `execute_decision`.
    #[must_use]
    pub const fn legacy_result(&self) -> HookResult {
        match self.control {
            HookControl::Continue => HookResult::Continue,
            HookControl::SkipOperation => HookResult::Skip,
            HookControl::Abort => HookResult::Abort,
            HookControl::Retry => HookResult::Retry,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HookContext {
    pub hook_point: HookPoint,
    pub agent_id: String,
    pub agent_role: String,
    pub task_id: Option<String>,
    pub task_iri: Option<String>,
    pub data: HashMap<String, Value>,
    pub metadata: HashMap<String, Value>,
    /// Compatibility timestamp retained for existing consumers.
    pub timestamp: u64,
    /// Millisecond-resolution wall-clock timestamp for correlation.
    #[serde(default)]
    pub timestamp_ms: u64,
    /// Trace shared by all hook points belonging to one logical interaction.
    #[serde(default)]
    pub trace_id: String,
    /// Span identifying this concrete hook-point invocation.
    #[serde(default)]
    pub span_id: String,
    pub error: Option<String>,
}

impl HookContext {
    pub fn new(hook_point: HookPoint, agent_id: &str, agent_role: &str) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        Self {
            hook_point,
            agent_id: agent_id.to_string(),
            agent_role: agent_role.to_string(),
            task_id: None,
            task_iri: None,
            data: HashMap::new(),
            metadata: HashMap::new(),
            timestamp: timestamp_ms / 1_000,
            timestamp_ms,
            trace_id: uuid::Uuid::new_v4().to_string(),
            span_id: uuid::Uuid::new_v4().to_string(),
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

    /// Attach this hook point to a caller-owned trace while retaining a unique
    /// span for the individual invocation.
    #[must_use]
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = trace_id.into();
        self
    }

    #[must_use]
    pub fn with_span_id(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = span_id.into();
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

    fn execution_policy(&self) -> HookExecutionPolicy {
        HookExecutionPolicy::default()
    }

    async fn execute(&self, context: &mut HookContext) -> HookResult;
}

#[derive(Clone)]
pub struct FunctionHook {
    name: String,
    hook_points: Vec<HookPoint>,
    priority: i32,
    execution_policy: HookExecutionPolicy,
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
            execution_policy: HookExecutionPolicy::default(),
            handler: Arc::new(handler),
        }
    }

    #[must_use]
    pub fn with_execution_policy(mut self, policy: HookExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
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
    fn execution_policy(&self) -> HookExecutionPolicy {
        self.execution_policy
    }

    async fn execute(&self, context: &mut HookContext) -> HookResult {
        (self.handler)(context)
    }
}

pub struct AsyncFunctionHook {
    name: String,
    hook_points: Vec<HookPoint>,
    priority: i32,
    execution_policy: HookExecutionPolicy,
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
            execution_policy: HookExecutionPolicy::default(),
            handler: Arc::new(move |ctx| Box::pin(handler(ctx))),
        }
    }

    #[must_use]
    pub fn with_execution_policy(mut self, policy: HookExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
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
    fn execution_policy(&self) -> HookExecutionPolicy {
        self.execution_policy
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
                    timestamp_ms = ctx.timestamp_ms,
                    agent_id = %ctx.agent_id,
                    hook_point = ctx.hook_point.as_str(),
                    trace_id = %ctx.trace_id,
                    span_id = %ctx.span_id,
                    "hook event"
                );
                HookResult::Continue
            },
        ))
    }
}

#[derive(Debug)]
struct TimingEntry {
    started_at_ms: u64,
    inserted_at: Instant,
    sequence: u64,
}

#[derive(Debug, Default)]
struct TimingTracker {
    entries: HashMap<String, TimingEntry>,
    insertion_order: BTreeMap<u64, String>,
    next_sequence: u64,
    evicted_entries: u64,
}

impl TimingTracker {
    fn allocate_sequence(&mut self) -> u64 {
        if self.next_sequence == u64::MAX {
            // A wrap is practically unreachable, but keeping the ordering
            // invariant total makes the capacity guard correct indefinitely.
            let ordered_keys = self.insertion_order.values().cloned().collect::<Vec<_>>();
            self.insertion_order.clear();
            for (index, key) in ordered_keys.into_iter().enumerate() {
                let sequence = (index as u64).saturating_add(1);
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.sequence = sequence;
                    self.insertion_order.insert(sequence, key);
                }
            }
            self.next_sequence = self.insertion_order.len() as u64;
        }
        self.next_sequence += 1;
        self.next_sequence
    }

    fn remove_entry(&mut self, key: &str) -> Option<TimingEntry> {
        let entry = self.entries.remove(key)?;
        self.insertion_order.remove(&entry.sequence);
        Some(entry)
    }

    fn remove_oldest(&mut self) -> bool {
        let Some((&sequence, key)) = self.insertion_order.first_key_value() else {
            return false;
        };
        let key = key.clone();
        self.insertion_order.remove(&sequence);
        if self
            .entries
            .get(&key)
            .is_some_and(|entry| entry.sequence == sequence)
        {
            self.entries.remove(&key);
            self.evicted_entries = self.evicted_entries.saturating_add(1);
        }
        true
    }

    fn prune_stale(&mut self, now: Instant, stale_after: Duration) {
        loop {
            let Some((&sequence, key)) = self.insertion_order.first_key_value() else {
                break;
            };
            let should_remove = self.entries.get(key).is_none_or(|entry| {
                entry.sequence != sequence
                    || now.saturating_duration_since(entry.inserted_at) >= stale_after
            });
            if !should_remove {
                break;
            }
            self.remove_oldest();
        }
    }

    fn start(
        &mut self,
        key: String,
        started_at_ms: u64,
        now: Instant,
        stale_after: Duration,
        max_in_flight: usize,
    ) {
        self.prune_stale(now, stale_after);
        self.remove_entry(&key);

        let sequence = self.allocate_sequence();
        self.insertion_order.insert(sequence, key.clone());
        self.entries.insert(
            key,
            TimingEntry {
                started_at_ms,
                inserted_at: now,
                sequence,
            },
        );

        while self.entries.len() > max_in_flight {
            if !self.remove_oldest() {
                break;
            }
        }
    }

    fn finish(
        &mut self,
        key: &str,
        finished_at_ms: u64,
        now: Instant,
        stale_after: Duration,
    ) -> Option<u64> {
        self.prune_stale(now, stale_after);
        let entry = self.remove_entry(key)?;
        Some(finished_at_ms.saturating_sub(entry.started_at_ms))
    }
}

pub struct TimingHook {
    tracker: Arc<RwLock<TimingTracker>>,
    max_in_flight: usize,
    stale_after: Duration,
}

impl TimingHook {
    const DEFAULT_MAX_IN_FLIGHT: usize = 4_096;
    const DEFAULT_STALE_AFTER: Duration = Duration::from_secs(15 * 60);

    pub fn new() -> Box<dyn Hook> {
        Box::new(Self::with_limits(
            Self::DEFAULT_MAX_IN_FLIGHT,
            Self::DEFAULT_STALE_AFTER,
        ))
    }

    fn with_limits(max_in_flight: usize, stale_after: Duration) -> Self {
        Self {
            tracker: Arc::new(RwLock::new(TimingTracker::default())),
            max_in_flight: max_in_flight.max(1),
            stale_after: stale_after.max(Duration::from_millis(1)),
        }
    }

    #[cfg(test)]
    fn tracker_footprint(&self) -> (usize, usize, u64) {
        let tracker = self.tracker.read();
        (
            tracker.entries.len(),
            tracker.insertion_order.len(),
            tracker.evicted_entries,
        )
    }

    fn correlation(ctx: &HookContext) -> Option<(String, bool)> {
        match ctx.hook_point {
            HookPoint::TaskStart | HookPoint::TaskEnd => Some((
                format!(
                    "{}:{}:task",
                    ctx.agent_id,
                    ctx.task_id.as_deref().unwrap_or("none")
                ),
                ctx.hook_point == HookPoint::TaskStart,
            )),
            HookPoint::SkillBefore | HookPoint::SkillAfter => {
                let tool_call_id = ctx
                    .data
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .unwrap_or(ctx.trace_id.as_str());
                Some((
                    format!(
                        "{}:{}:skill:{}",
                        ctx.agent_id,
                        ctx.task_id.as_deref().unwrap_or("none"),
                        tool_call_id
                    ),
                    ctx.hook_point == HookPoint::SkillBefore,
                ))
            }
            HookPoint::LlmRequest | HookPoint::LlmResponse => Some((
                format!(
                    "{}:{}:llm:{}",
                    ctx.agent_id,
                    ctx.task_id.as_deref().unwrap_or("none"),
                    ctx.trace_id
                ),
                ctx.hook_point == HookPoint::LlmRequest,
            )),
            _ => None,
        }
    }
}

#[async_trait]
impl Hook for TimingHook {
    fn name(&self) -> &str {
        "timing"
    }

    fn hook_points(&self) -> Vec<HookPoint> {
        vec![
            HookPoint::TaskStart,
            HookPoint::TaskEnd,
            HookPoint::SkillBefore,
            HookPoint::SkillAfter,
            HookPoint::LlmRequest,
            HookPoint::LlmResponse,
        ]
    }

    fn priority(&self) -> i32 {
        // Policy hooks must decide first. In particular, a rate-limit Abort
        // must not create an LLM timing start that can never receive a finish.
        i32::MAX
    }

    async fn execute(&self, ctx: &mut HookContext) -> HookResult {
        let Some((key, starts_window)) = Self::correlation(ctx) else {
            return HookResult::Continue;
        };
        let now = Instant::now();
        if starts_window {
            self.tracker.write().start(
                key,
                ctx.timestamp_ms,
                now,
                self.stale_after,
                self.max_in_flight,
            );
        } else if let Some(duration_ms) =
            self.tracker
                .write()
                .finish(&key, ctx.timestamp_ms, now, self.stale_after)
        {
            ctx.metadata.insert(
                "duration_seconds".to_string(),
                Value::Number((duration_ms / 1_000).into()),
            );
            ctx.metadata
                .insert("duration_ms".to_string(), Value::Number(duration_ms.into()));
        }
        HookResult::Continue
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
    const MAX_TRACKED_IDENTITIES: usize = 4_096;

    pub fn new(max_calls: usize, window_seconds: u64) -> Box<dyn Hook> {
        let calls: Arc<RwLock<HashMap<String, Vec<u64>>>> = Arc::new(RwLock::new(HashMap::new()));
        let calls_clone = calls.clone();

        Box::new(FunctionHook::new(
            "rate_limit",
            vec![HookPoint::LlmRequest],
            10,
            move |ctx| {
                let agent_id = ctx.agent_id.clone();
                let now = ctx.timestamp;

                let mut calls = calls_clone.write();
                // Agent IDs are task-scoped and normally unique. Retaining an
                // empty window for every completed agent would therefore turn
                // an opt-in rate limiter into a process-lifetime memory leak.
                calls.retain(|_, timestamps| {
                    timestamps.retain(|&t| now.saturating_sub(t) < window_seconds);
                    !timestamps.is_empty()
                });
                if !calls.contains_key(&agent_id)
                    && calls.len() >= RateLimitHook::MAX_TRACKED_IDENTITIES
                {
                    ctx.error = Some(
                        "Rate-limit identity capacity exhausted; refusing an untracked caller"
                            .to_string(),
                    );
                    return HookResult::Abort;
                }
                let entry: &mut Vec<u64> = calls.entry(agent_id.clone()).or_default();

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct HookMetricAggregate {
    event_count: u64,
    error_count: u64,
    last_timestamp_ms: u64,
}

fn record_hook_metric(metrics: &mut HashMap<String, HookMetricAggregate>, context: &HookContext) {
    // `hook_point` is a closed enum, so the number of keys is statically
    // bounded. Keep counters only: task/agent IDs, arbitrary metadata and
    // request-derived values do not belong in a process-lifetime metric hook.
    let entry = metrics
        .entry(context.hook_point.as_str().to_string())
        .or_default();
    entry.event_count = entry.event_count.saturating_add(1);
    entry.error_count = entry
        .error_count
        .saturating_add(context.error.is_some() as u64);
    entry.last_timestamp_ms = context.timestamp_ms;
}

pub struct MetricsHook {
    #[allow(dead_code)]
    metrics: Arc<RwLock<HashMap<String, HookMetricAggregate>>>,
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
                let mut metrics = metrics_clone.write();
                record_hook_metric(&mut metrics, ctx);

                HookResult::Continue
            },
        ))
    }
}

pub struct HookManager {
    hooks: RwLock<HashMap<HookPoint, Vec<Arc<dyn Hook>>>>,
}

/// The sole behavioral patch surface currently supported by the internal Hook
/// plane. It is a complete tool-argument object, not an arbitrary JSON Patch.
pub const TOOL_ARGUMENTS_PATCH_METADATA_KEY: &str = "arguments";

fn equal_except_tool_arguments(before: &HookContext, after: &HookContext) -> bool {
    let mut before = before.clone();
    let mut after = after.clone();
    before.metadata.remove(TOOL_ARGUMENTS_PATCH_METADATA_KEY);
    after.metadata.remove(TOOL_ARGUMENTS_PATCH_METADATA_KEY);
    before == after
}

const fn hook_failure_result(policy: HookFailurePolicy) -> HookResult {
    match policy {
        HookFailurePolicy::FailOpen => HookResult::Continue,
        HookFailurePolicy::FailClosed => HookResult::Abort,
        HookFailurePolicy::Retry => HookResult::Retry,
        HookFailurePolicy::SkipOperation => HookResult::Skip,
    }
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

    /// Execute a hook chain and return a fully auditable decision.
    ///
    /// Terminal results (`Skip`, `Abort`, and `Retry`) stop the chain.  A
    /// `SkipRemaining` result only stops later hooks; it does not skip the
    /// guarded operation. `Modify` records that the shared context was changed
    /// and allows lower-priority hooks to inspect the modified context.
    pub async fn execute_decision(
        &self,
        hook_point: HookPoint,
        context: &mut HookContext,
    ) -> HookDecision {
        let hooks: Vec<Arc<dyn Hook>> = {
            let guard = self.hooks.read();
            guard
                .get(&hook_point)
                .map(|v| v.clone())
                .unwrap_or_default()
        };

        context.hook_point = hook_point;
        if context.timestamp_ms == 0 {
            context.timestamp_ms = if context.timestamp > 0 {
                context.timestamp.saturating_mul(1_000)
            } else {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX)
            };
        }
        if context.timestamp == 0 {
            context.timestamp = context.timestamp_ms / 1_000;
        }
        if context.trace_id.is_empty() {
            context.trace_id = uuid::Uuid::new_v4().to_string();
        }
        if context.span_id.is_empty() {
            context.span_id = uuid::Uuid::new_v4().to_string();
        }
        let mut decision = HookDecision::new(context.trace_id.clone(), hooks.len());

        for hook in &hooks {
            let policy = hook.execution_policy();
            let before = context.clone();
            let started = Instant::now();
            let hook_future = std::panic::AssertUnwindSafe(hook.execute(context)).catch_unwind();
            let (result, timed_out, panicked) = match policy
                .timeout_ms
                .filter(|timeout| *timeout > 0)
            {
                Some(timeout_ms) => {
                    match tokio::time::timeout(Duration::from_millis(timeout_ms), hook_future).await
                    {
                        Ok(Ok(result)) => (result, false, false),
                        Ok(Err(_)) => {
                            *context = before.clone();
                            let failure_policy = policy
                                .on_panic
                                .unwrap_or_else(|| hook_point.panic_failure_policy());
                            tracing::error!(
                                hook = hook.name(),
                                hook_point = hook_point.as_str(),
                                trace_id = %context.trace_id,
                                ?failure_policy,
                                "hook panicked; partial context changes were rolled back"
                            );
                            (hook_failure_result(failure_policy), false, true)
                        }
                        Err(_) => {
                            *context = before.clone();
                            tracing::warn!(
                                hook = hook.name(),
                                hook_point = hook_point.as_str(),
                                trace_id = %context.trace_id,
                                timeout_ms,
                                ?policy.on_timeout,
                                "hook execution timed out; partial context changes were rolled back"
                            );
                            (hook_failure_result(policy.on_timeout), true, false)
                        }
                    }
                }
                _ => match hook_future.await {
                    Ok(result) => (result, false, false),
                    Err(_) => {
                        *context = before.clone();
                        let failure_policy = policy
                            .on_panic
                            .unwrap_or_else(|| hook_point.panic_failure_policy());
                        tracing::error!(
                            hook = hook.name(),
                            hook_point = hook_point.as_str(),
                            trace_id = %context.trace_id,
                            ?failure_policy,
                            "hook panicked; partial context changes were rolled back"
                        );
                        (hook_failure_result(failure_policy), false, true)
                    }
                },
            };
            let context_changed = before != *context;
            let before_arguments = before
                .metadata
                .get(TOOL_ARGUMENTS_PATCH_METADATA_KEY)
                .cloned();
            let after_arguments = context
                .metadata
                .get(TOOL_ARGUMENTS_PATCH_METADATA_KEY)
                .cloned();
            let arguments_changed = before_arguments != after_arguments;
            let semantic_error = if result == HookResult::Retry {
                // A retry re-evaluates policy only. Discard every mutation from
                // the failed attempt so retries cannot accumulate hidden state
                // or alter a later tool invocation.
                *context = before.clone();
                None
            } else if result == HookResult::Modify {
                if hook_point != HookPoint::SkillBefore {
                    Some(format!(
                        "Hook '{}' requested unsupported Modify at {}; only SkillBefore tool arguments are patchable",
                        hook.name(),
                        hook_point.as_str()
                    ))
                } else if decision.tool_arguments_patch.is_some() {
                    Some(format!(
                        "Hook '{}' conflicts with an earlier SkillBefore arguments patch",
                        hook.name()
                    ))
                } else if !arguments_changed {
                    Some(format!(
                        "Hook '{}' returned Modify without changing metadata.arguments",
                        hook.name()
                    ))
                } else if !after_arguments.as_ref().is_some_and(Value::is_object) {
                    Some(format!(
                        "Hook '{}' produced invalid metadata.arguments; a JSON object is required",
                        hook.name()
                    ))
                } else if !equal_except_tool_arguments(&before, context) {
                    Some(format!(
                        "Hook '{}' attempted to modify fields outside metadata.arguments",
                        hook.name()
                    ))
                } else {
                    decision.tool_arguments_patch = after_arguments;
                    decision.context_modified = true;
                    None
                }
            } else if hook_point == HookPoint::SkillBefore
                && arguments_changed
                && matches!(result, HookResult::Continue | HookResult::SkipRemaining)
            {
                Some(format!(
                    "Hook '{}' changed metadata.arguments without returning Modify",
                    hook.name()
                ))
            } else {
                None
            };
            decision.records.push(HookExecutionRecord {
                hook_name: hook.name().to_string(),
                result,
                elapsed_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                timed_out,
                panicked,
                context_modified: context_changed,
            });

            if let Some(error) = semantic_error {
                context.error = Some(error);
                decision.control = HookControl::Abort;
                decision.terminal_hook = Some(hook.name().to_string());
                break;
            }

            match result {
                HookResult::Continue | HookResult::Modify => {}
                HookResult::Abort => {
                    decision.control = HookControl::Abort;
                    decision.terminal_hook = Some(hook.name().to_string());
                    break;
                }
                HookResult::Skip => {
                    decision.control = HookControl::SkipOperation;
                    decision.terminal_hook = Some(hook.name().to_string());
                    break;
                }
                HookResult::Retry => {
                    decision.control = HookControl::Retry;
                    decision.terminal_hook = Some(hook.name().to_string());
                    break;
                }
                HookResult::SkipRemaining => {
                    decision.remaining_hooks_skipped = true;
                    decision.terminal_hook = Some(hook.name().to_string());
                    break;
                }
            }
        }

        decision
    }

    /// Compatibility wrapper for existing callers. New control points should
    /// use [`Self::execute_decision`] so context modifications and trace
    /// records remain observable.
    pub async fn execute(&self, hook_point: HookPoint, context: &mut HookContext) -> HookResult {
        self.execute_decision(hook_point, context)
            .await
            .legacy_result()
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
        Self::Reject
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
            default_action: DefaultAction::Reject,
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
            default_action: DefaultAction::Reject,
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

/// Owns the in-process state for one active approval wait. Async cancellation
/// drops the wait future, so cleanup must live in `Drop` rather than only in
/// timeout/response branches.
struct ApprovalWaitCleanup<'a> {
    pending: &'a RwLock<HashMap<String, ApprovalState>>,
    waiters: &'a parking_lot::Mutex<HashMap<String, Arc<tokio::sync::Notify>>>,
    request_id: String,
}

impl Drop for ApprovalWaitCleanup<'_> {
    fn drop(&mut self) {
        self.waiters.lock().remove(&self.request_id);
        self.pending.write().remove(&self.request_id);
    }
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
        const MAX_PENDING_APPROVALS: usize = 4_096;
        if pending.contains_key(&request.request_id) {
            return Err("duplicate approval request id".into());
        }
        if pending.len() >= MAX_PENDING_APPROVALS {
            return Err("approval queue capacity exhausted".into());
        }
        pending.insert(
            request.request_id.clone(),
            ApprovalState {
                request: request.clone(),
                response: None,
                processed: false,
            },
        );
        Ok(())
    }

    async fn wait_for_response(
        &self,
        request_id: &str,
        timeout: std::time::Duration,
    ) -> Option<ApprovalResponse> {
        if !self.pending.read().contains_key(request_id) {
            return None;
        }
        let waiter = {
            let mut waiters = self.waiters.lock();
            // One request has one owner. A second waiter must not be allowed
            // to cancel or consume the first waiter's approval lifecycle.
            if waiters.contains_key(request_id) {
                tracing::warn!(request_id = %request_id, "duplicate approval waiter rejected");
                return None;
            }
            let waiter = Arc::new(tokio::sync::Notify::new());
            waiters.insert(request_id.to_string(), waiter.clone());
            waiter
        };
        let _cleanup = ApprovalWaitCleanup {
            pending: self.pending.as_ref(),
            waiters: &self.waiters,
            request_id: request_id.to_string(),
        };
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = waiter.notified();
            let response = {
                let pending = self.pending.read();
                pending.get(request_id).map(|state| state.response.clone())
            };
            match response {
                None => return None,
                Some(Some(response)) => return Some(response),
                Some(None) => {}
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return None;
            }
            if tokio::time::Instant::now() >= deadline {
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
            ApprovalCondition::OnStageComplete => {
                ctx.data.get("phase_executed").and_then(Value::as_bool) != Some(false)
                    && ctx
                        .data
                        .get("stage_id")
                        .and_then(Value::as_str)
                        .is_some_and(|stage| {
                            point.stages.is_empty() || point.stages.iter().any(|item| item == stage)
                        })
            }
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

        if self.notifier.notify(&request).await.is_err() {
            // Notifier errors can contain transport payloads or credentials.
            // Record only correlation metadata and fail closed.
            tracing::error!(request_id = %request_id, "approval notification failed");
            ctx.error = Some("Approval request could not be delivered".to_string());
            return HookResult::Abort;
        }

        match self.notifier.wait_for_response(&request_id, timeout).await {
            Some(response) if response.approved => {
                tracing::info!(request_id = %request_id, "user approved");
                HookResult::Continue
            }
            Some(_) => {
                // Reviewer comments may contain user or business content. Keep
                // them in the response channel, not telemetry or model-facing
                // execution errors.
                tracing::warn!(request_id = %request_id, "approval rejected");
                ctx.error = Some("Approval rejected by reviewer".to_string());
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

    struct FailingApprovalNotifier;

    struct RejectingApprovalNotifier;

    #[async_trait::async_trait]
    impl ApprovalNotifier for FailingApprovalNotifier {
        async fn notify(
            &self,
            _request: &ApprovalRequest,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Err(std::io::Error::other("secret transport failure payload").into())
        }

        async fn wait_for_response(
            &self,
            _request_id: &str,
            _timeout: std::time::Duration,
        ) -> Option<ApprovalResponse> {
            panic!("wait_for_response must not run after notification failure")
        }
    }

    #[async_trait::async_trait]
    impl ApprovalNotifier for RejectingApprovalNotifier {
        async fn notify(
            &self,
            _request: &ApprovalRequest,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }

        async fn wait_for_response(
            &self,
            request_id: &str,
            _timeout: std::time::Duration,
        ) -> Option<ApprovalResponse> {
            Some(ApprovalResponse::rejected(
                request_id.to_string(),
                "review".to_string(),
                Some("secret reviewer comment".to_string()),
            ))
        }
    }

    struct SlowHook {
        name: &'static str,
        policy: HookExecutionPolicy,
    }

    struct PanickingHook {
        point: HookPoint,
    }

    #[async_trait::async_trait]
    impl Hook for SlowHook {
        fn name(&self) -> &str {
            self.name
        }

        fn hook_points(&self) -> Vec<HookPoint> {
            vec![HookPoint::LlmRequest]
        }

        fn execution_policy(&self) -> HookExecutionPolicy {
            self.policy
        }

        async fn execute(&self, _context: &mut HookContext) -> HookResult {
            tokio::time::sleep(Duration::from_millis(50)).await;
            HookResult::Continue
        }
    }

    #[async_trait::async_trait]
    impl Hook for PanickingHook {
        fn name(&self) -> &str {
            "panicking_hook"
        }

        fn hook_points(&self) -> Vec<HookPoint> {
            vec![self.point]
        }

        async fn execute(&self, context: &mut HookContext) -> HookResult {
            context
                .data
                .insert("partial_patch".to_string(), Value::Bool(true));
            panic!("deliberate hook panic")
        }
    }

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
        assert!(notifier.get_pending().is_empty());
        assert!(notifier.pending.read().is_empty());
        assert!(notifier.waiters.lock().is_empty());
    }

    #[tokio::test]
    async fn approval_timeout_reclaims_request_and_waiter_state() {
        let notifier = ChannelApprovalNotifier::new();
        let request = ApprovalRequest::new(
            "iri://task/timeout".into(),
            "review".into(),
            "approve".into(),
            vec!["approve".into(), "reject".into()],
        );
        notifier.notify(&request).await.unwrap();

        assert!(notifier
            .wait_for_response(&request.request_id, std::time::Duration::from_millis(1))
            .await
            .is_none());
        assert!(notifier.get_pending().is_empty());
        assert!(notifier.pending.read().is_empty());
        assert!(notifier.waiters.lock().is_empty());
    }

    #[tokio::test]
    async fn cancelled_approval_wait_reclaims_request_and_waiter_state() {
        let notifier = Arc::new(ChannelApprovalNotifier::new());
        let request = ApprovalRequest::new(
            "iri://task/cancelled".into(),
            "review".into(),
            "approve".into(),
            vec!["approve".into(), "reject".into()],
        );
        notifier.notify(&request).await.unwrap();

        let waiter_task = {
            let notifier = notifier.clone();
            let request_id = request.request_id.clone();
            tokio::spawn(async move {
                notifier
                    .wait_for_response(&request_id, std::time::Duration::from_secs(60))
                    .await
            })
        };

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if notifier.waiters.lock().contains_key(&request.request_id) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("approval wait did not become active");

        waiter_task.abort();
        assert!(waiter_task.await.unwrap_err().is_cancelled());
        assert!(notifier.get_pending().is_empty());
        assert!(notifier.pending.read().is_empty());
        assert!(notifier.waiters.lock().is_empty());
    }

    #[tokio::test]
    async fn approval_notification_failure_aborts_without_waiting_or_leaking_error_content() {
        let hook = HumanApprovalHook::new(
            HumanApprovalConfig {
                enabled: true,
                approval_points: vec![ApprovalPoint {
                    hook_point: HookPoint::PhaseEnd,
                    condition: ApprovalCondition::Always,
                    timeout_seconds: 60,
                    default_action: DefaultAction::Reject,
                    ..ApprovalPoint::default()
                }],
                ..HumanApprovalConfig::default()
            },
            Arc::new(FailingApprovalNotifier),
        );
        let mut context = HookContext::new(HookPoint::PhaseEnd, "agent", "AA")
            .with_data("stage_id", serde_json::json!("review"));

        let result = hook.execute(&mut context).await;

        assert_eq!(result, HookResult::Abort);
        assert_eq!(
            context.error.as_deref(),
            Some("Approval request could not be delivered")
        );
        assert!(!context
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("secret"));
    }

    #[tokio::test]
    async fn approval_rejection_does_not_copy_reviewer_comments_into_execution_errors() {
        let hook = HumanApprovalHook::new(
            HumanApprovalConfig {
                enabled: true,
                approval_points: vec![ApprovalPoint {
                    hook_point: HookPoint::PhaseEnd,
                    condition: ApprovalCondition::Always,
                    timeout_seconds: 60,
                    default_action: DefaultAction::Reject,
                    ..ApprovalPoint::default()
                }],
                ..HumanApprovalConfig::default()
            },
            Arc::new(RejectingApprovalNotifier),
        );
        let mut context = HookContext::new(HookPoint::PhaseEnd, "agent", "CA")
            .with_data("stage_id", serde_json::json!("review"));

        let result = hook.execute(&mut context).await;

        assert_eq!(result, HookResult::Abort);
        assert_eq!(
            context.error.as_deref(),
            Some("Approval rejected by reviewer")
        );
        assert!(!context
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("secret"));
    }

    #[test]
    fn approval_timeout_defaults_are_fail_closed() {
        assert_eq!(DefaultAction::default(), DefaultAction::Reject);
        assert_eq!(
            ApprovalPoint::default().default_action,
            DefaultAction::Reject
        );
        assert_eq!(
            HumanApprovalConfig::default().default_action,
            DefaultAction::Reject
        );
    }

    #[tokio::test]
    async fn default_approval_timeout_aborts_the_guarded_operation() {
        let (hook, notifier) = HumanApprovalHook::with_channel_notifier(HumanApprovalConfig {
            enabled: true,
            approval_points: vec![ApprovalPoint {
                hook_point: HookPoint::PhaseEnd,
                condition: ApprovalCondition::Always,
                timeout_seconds: 0,
                ..ApprovalPoint::default()
            }],
            ..HumanApprovalConfig::default()
        });
        let mut context = HookContext::new(HookPoint::PhaseEnd, "agent", "PA")
            .with_data("stage_id", serde_json::json!("review"));

        let result = hook.execute(&mut context).await;

        assert_eq!(result, HookResult::Abort);
        assert_eq!(
            context.error.as_deref(),
            Some("Approval timeout, auto-rejected")
        );
        assert!(notifier.pending.read().is_empty());
        assert!(notifier.waiters.lock().is_empty());
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
    async fn timing_hook_correlates_interleaved_tool_calls_independently() {
        let manager = HookManager::new();
        manager.register(TimingHook::new());

        let context = |point, call_id: &str, timestamp_ms| {
            let mut context = HookContext::new(point, "agent", "DA")
                .with_task("task", "iri://task/timing")
                .with_trace_id("shared-llm-interaction")
                .with_data("tool_call_id", Value::String(call_id.to_string()));
            context.timestamp_ms = timestamp_ms;
            context.timestamp = timestamp_ms / 1_000;
            context
        };

        let mut first_start = context(HookPoint::SkillBefore, "call-a", 1_000);
        let mut second_start = context(HookPoint::SkillBefore, "call-b", 1_100);
        manager
            .execute(HookPoint::SkillBefore, &mut first_start)
            .await;
        manager
            .execute(HookPoint::SkillBefore, &mut second_start)
            .await;

        let mut first_end = context(HookPoint::SkillAfter, "call-a", 1_300);
        let mut second_end = context(HookPoint::SkillAfter, "call-b", 1_500);
        let (first_result, second_result) = tokio::join!(
            manager.execute(HookPoint::SkillAfter, &mut first_end),
            manager.execute(HookPoint::SkillAfter, &mut second_end),
        );

        assert_eq!(first_result, HookResult::Continue);
        assert_eq!(second_result, HookResult::Continue);
        assert_eq!(
            first_end.metadata.get("duration_ms"),
            Some(&Value::from(300))
        );
        assert_eq!(
            second_end.metadata.get("duration_ms"),
            Some(&Value::from(400))
        );
    }

    #[tokio::test]
    async fn timing_hook_correlates_interleaved_llm_traces_independently() {
        let manager = HookManager::new();
        manager.register(TimingHook::new());

        let context = |point, trace_id: &str, timestamp_ms| {
            let mut context = HookContext::new(point, "agent", "PA")
                .with_task("task", "iri://task/timing")
                .with_trace_id(trace_id);
            context.timestamp_ms = timestamp_ms;
            context.timestamp = timestamp_ms / 1_000;
            context
        };

        let mut first_start = context(HookPoint::LlmRequest, "llm-a", 2_000);
        let mut second_start = context(HookPoint::LlmRequest, "llm-b", 2_100);
        manager
            .execute(HookPoint::LlmRequest, &mut first_start)
            .await;
        manager
            .execute(HookPoint::LlmRequest, &mut second_start)
            .await;

        let mut first_end = context(HookPoint::LlmResponse, "llm-a", 2_250);
        let mut second_end = context(HookPoint::LlmResponse, "llm-b", 2_550);
        let (first_result, second_result) = tokio::join!(
            manager.execute(HookPoint::LlmResponse, &mut first_end),
            manager.execute(HookPoint::LlmResponse, &mut second_end),
        );

        assert_eq!(first_result, HookResult::Continue);
        assert_eq!(second_result, HookResult::Continue);
        assert_eq!(
            first_end.metadata.get("duration_ms"),
            Some(&Value::from(250))
        );
        assert_eq!(
            second_end.metadata.get("duration_ms"),
            Some(&Value::from(450))
        );
    }

    #[tokio::test]
    async fn rate_limit_abort_does_not_leak_timing_start() {
        let manager = HookManager::new();
        let timing = Arc::new(TimingHook::with_limits(64, Duration::from_secs(60)));
        manager.register_arc(timing.clone());
        manager.register(RateLimitHook::new(1, 60));

        let mut accepted = HookContext::new(HookPoint::LlmRequest, "agent", "PA")
            .with_task("task", "iri://task/rate-timing")
            .with_trace_id("accepted");
        let accepted_decision = manager
            .execute_decision(HookPoint::LlmRequest, &mut accepted)
            .await;
        assert_eq!(accepted_decision.control, HookControl::Continue);
        assert_eq!(timing.tracker_footprint().0, 1);

        for attempt in 0..100 {
            let mut rejected = HookContext::new(HookPoint::LlmRequest, "agent", "PA")
                .with_task("task", "iri://task/rate-timing")
                .with_trace_id(format!("rejected-{attempt}"));
            let decision = manager
                .execute_decision(HookPoint::LlmRequest, &mut rejected)
                .await;
            assert_eq!(decision.control, HookControl::Abort);
            assert_eq!(decision.terminal_hook.as_deref(), Some("rate_limit"));
            assert!(decision
                .records
                .iter()
                .all(|record| record.hook_name != "timing"));
        }

        assert_eq!(
            timing.tracker_footprint().0,
            1,
            "rate-limited requests must not allocate timing starts"
        );
        let mut response = HookContext::new(HookPoint::LlmResponse, "agent", "PA")
            .with_task("task", "iri://task/rate-timing")
            .with_trace_id("accepted");
        manager.execute(HookPoint::LlmResponse, &mut response).await;
        assert_eq!(timing.tracker_footprint().0, 0);
    }

    #[tokio::test]
    async fn timing_hook_bounds_high_concurrency_state_and_decision_records_are_local() {
        const MAX_IN_FLIGHT: usize = 32;
        const REQUESTS: usize = 512;

        let manager = Arc::new(HookManager::new());
        let timing = Arc::new(TimingHook::with_limits(
            MAX_IN_FLIGHT,
            Duration::from_secs(60),
        ));
        manager.register_arc(timing.clone());

        let mut handles = Vec::with_capacity(REQUESTS);
        for request in 0..REQUESTS {
            let manager = manager.clone();
            handles.push(tokio::spawn(async move {
                let mut context = HookContext::new(HookPoint::LlmRequest, "agent", "DA")
                    .with_task("task", "iri://task/high-concurrency")
                    .with_trace_id(format!("trace-{request}"));
                manager
                    .execute_decision(HookPoint::LlmRequest, &mut context)
                    .await
            }));
        }

        for handle in handles {
            let decision = handle.await.unwrap();
            assert_eq!(decision.control, HookControl::Continue);
            assert_eq!(
                decision.records.len(),
                1,
                "records are scoped to one invocation, not accumulated by HookManager"
            );
        }

        let (entries, order_entries, evicted) = timing.tracker_footprint();
        assert_eq!(entries, MAX_IN_FLIGHT);
        assert_eq!(order_entries, MAX_IN_FLIGHT);
        assert_eq!(evicted, (REQUESTS - MAX_IN_FLIGHT) as u64);
        assert_eq!(manager.get_hooks(HookPoint::LlmRequest), ["timing"]);
    }

    #[tokio::test]
    async fn timing_hook_prunes_stale_unfinished_windows() {
        let manager = HookManager::new();
        let timing = Arc::new(TimingHook::with_limits(8, Duration::from_millis(1)));
        manager.register_arc(timing.clone());

        let mut abandoned = HookContext::new(HookPoint::LlmRequest, "agent", "CA")
            .with_task("task", "iri://task/stale-timing")
            .with_trace_id("abandoned");
        manager.execute(HookPoint::LlmRequest, &mut abandoned).await;
        tokio::time::sleep(Duration::from_millis(5)).await;

        let mut current = HookContext::new(HookPoint::LlmRequest, "agent", "CA")
            .with_task("task", "iri://task/stale-timing")
            .with_trace_id("current");
        manager.execute(HookPoint::LlmRequest, &mut current).await;

        let (entries, order_entries, evicted) = timing.tracker_footprint();
        assert_eq!(entries, 1);
        assert_eq!(order_entries, 1);
        assert_eq!(evicted, 1);
    }

    #[tokio::test]
    async fn structured_decision_preserves_modify_and_stops_remaining_hooks() {
        let manager = HookManager::new();
        let later_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        manager.register(Box::new(FunctionHook::new(
            "modifier",
            vec![HookPoint::SkillBefore],
            10,
            |context| {
                context.metadata.insert(
                    TOOL_ARGUMENTS_PATCH_METADATA_KEY.to_string(),
                    serde_json::json!({"effective": true}),
                );
                HookResult::Modify
            },
        )));
        manager.register(Box::new(FunctionHook::new(
            "skip_operation",
            vec![HookPoint::SkillBefore],
            20,
            |context| {
                assert_eq!(
                    context.metadata.get(TOOL_ARGUMENTS_PATCH_METADATA_KEY),
                    Some(&serde_json::json!({"effective": true}))
                );
                HookResult::Skip
            },
        )));
        manager.register(Box::new(FunctionHook::new(
            "must_not_run",
            vec![HookPoint::SkillBefore],
            30,
            {
                let later_calls = later_calls.clone();
                move |_| {
                    later_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    HookResult::Continue
                }
            },
        )));

        let mut context =
            HookContext::new(HookPoint::SkillBefore, "agent", "PA").with_trace_id("trace-task-1");
        context.metadata.insert(
            TOOL_ARGUMENTS_PATCH_METADATA_KEY.to_string(),
            serde_json::json!({"effective": false}),
        );
        let decision = manager
            .execute_decision(HookPoint::SkillBefore, &mut context)
            .await;

        assert_eq!(decision.control, HookControl::SkipOperation);
        assert!(decision.context_modified);
        assert_eq!(decision.terminal_hook.as_deref(), Some("skip_operation"));
        assert_eq!(decision.trace_id, "trace-task-1");
        assert_eq!(decision.records.len(), 2);
        assert_eq!(
            decision.tool_arguments_patch,
            Some(serde_json::json!({"effective": true}))
        );
        assert_eq!(later_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(decision.legacy_result(), HookResult::Skip);
    }

    #[tokio::test]
    async fn conflicting_skill_before_argument_patches_fail_closed() {
        let manager = HookManager::new();
        manager.register(Box::new(FunctionHook::new(
            "first_patch",
            vec![HookPoint::SkillBefore],
            10,
            |context| {
                context.metadata.insert(
                    TOOL_ARGUMENTS_PATCH_METADATA_KEY.to_string(),
                    serde_json::json!({"command": "first"}),
                );
                HookResult::Modify
            },
        )));
        manager.register(Box::new(FunctionHook::new(
            "conflicting_patch",
            vec![HookPoint::SkillBefore],
            20,
            |context| {
                context.metadata.insert(
                    TOOL_ARGUMENTS_PATCH_METADATA_KEY.to_string(),
                    serde_json::json!({"command": "second"}),
                );
                HookResult::Modify
            },
        )));

        let mut context = HookContext::new(HookPoint::SkillBefore, "agent", "DA");
        context.metadata.insert(
            TOOL_ARGUMENTS_PATCH_METADATA_KEY.to_string(),
            serde_json::json!({"command": "original"}),
        );
        let decision = manager
            .execute_decision(HookPoint::SkillBefore, &mut context)
            .await;

        assert_eq!(decision.control, HookControl::Abort);
        assert_eq!(decision.terminal_hook.as_deref(), Some("conflicting_patch"));
        assert!(context
            .error
            .as_deref()
            .is_some_and(|error| error.contains("conflicts")));
        assert_eq!(
            decision.tool_arguments_patch,
            Some(serde_json::json!({"command": "first"}))
        );
    }

    #[tokio::test]
    async fn llm_modify_is_explicitly_rejected_until_typed_patch_exists() {
        for point in [HookPoint::LlmRequest, HookPoint::LlmResponse] {
            let manager = HookManager::new();
            manager.register(Box::new(FunctionHook::new(
                "unsupported_llm_patch",
                vec![point],
                10,
                |context| {
                    context
                        .metadata
                        .insert("messages".to_string(), serde_json::json!([]));
                    HookResult::Modify
                },
            )));
            let mut context = HookContext::new(point, "agent", "PA");
            let decision = manager.execute_decision(point, &mut context).await;

            assert_eq!(decision.control, HookControl::Abort);
            assert_eq!(
                decision.terminal_hook.as_deref(),
                Some("unsupported_llm_patch")
            );
            assert!(!decision.context_modified);
            assert!(context
                .error
                .as_deref()
                .is_some_and(|error| error.contains("unsupported Modify")));
        }
    }

    #[tokio::test]
    async fn undeclared_skill_argument_mutation_is_rejected() {
        let manager = HookManager::new();
        manager.register(Box::new(FunctionHook::new(
            "silent_mutator",
            vec![HookPoint::SkillBefore],
            10,
            |context| {
                context.metadata.insert(
                    TOOL_ARGUMENTS_PATCH_METADATA_KEY.to_string(),
                    serde_json::json!({"path": "different"}),
                );
                HookResult::Continue
            },
        )));
        let mut context = HookContext::new(HookPoint::SkillBefore, "agent", "CA");
        context.metadata.insert(
            TOOL_ARGUMENTS_PATCH_METADATA_KEY.to_string(),
            serde_json::json!({"path": "original"}),
        );

        let decision = manager
            .execute_decision(HookPoint::SkillBefore, &mut context)
            .await;
        assert_eq!(decision.control, HookControl::Abort);
        assert!(!decision.context_modified);
        assert!(context
            .error
            .as_deref()
            .is_some_and(|error| error.contains("without returning Modify")));
    }

    #[tokio::test]
    async fn retry_is_terminal_for_the_current_hook_chain() {
        let manager = HookManager::new();
        let later_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        manager.register(Box::new(FunctionHook::new(
            "request_retry",
            vec![HookPoint::CycleStart],
            10,
            |_| HookResult::Retry,
        )));
        manager.register(Box::new(FunctionHook::new(
            "must_not_run",
            vec![HookPoint::CycleStart],
            20,
            {
                let later_calls = later_calls.clone();
                move |_| {
                    later_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    HookResult::Continue
                }
            },
        )));

        let mut context = HookContext::new(HookPoint::CycleStart, "agent", "CA");
        let result = manager.execute(HookPoint::CycleStart, &mut context).await;
        assert_eq!(result, HookResult::Retry);
        assert_eq!(later_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn skip_remaining_stops_hooks_but_continues_guarded_operation() {
        let manager = HookManager::new();
        let later_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        manager.register(Box::new(FunctionHook::new(
            "chain_complete",
            vec![HookPoint::TaskEnd],
            10,
            |_| HookResult::SkipRemaining,
        )));
        manager.register(Box::new(FunctionHook::new(
            "must_not_run",
            vec![HookPoint::TaskEnd],
            20,
            {
                let later_calls = later_calls.clone();
                move |_| {
                    later_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    HookResult::Continue
                }
            },
        )));
        let mut context = HookContext::new(HookPoint::TaskEnd, "agent", "DA");

        let decision = manager
            .execute_decision(HookPoint::TaskEnd, &mut context)
            .await;

        assert_eq!(decision.control, HookControl::Continue);
        assert!(decision.remaining_hooks_skipped);
        assert_eq!(decision.legacy_result(), HookResult::Continue);
        assert_eq!(later_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn bounded_hook_timeout_honors_fail_closed_policy() {
        let manager = HookManager::new();
        manager.register(Box::new(SlowHook {
            name: "slow_guard",
            policy: HookExecutionPolicy::bounded(5, HookFailurePolicy::FailClosed),
        }));
        let mut context = HookContext::new(HookPoint::LlmRequest, "agent", "AA");

        let decision = manager
            .execute_decision(HookPoint::LlmRequest, &mut context)
            .await;

        assert_eq!(decision.control, HookControl::Abort);
        assert_eq!(decision.terminal_hook.as_deref(), Some("slow_guard"));
        assert_eq!(decision.records.len(), 1);
        assert!(decision.records[0].timed_out);
        assert!(decision.records[0].elapsed_ms < 50);
    }

    #[tokio::test]
    async fn bounded_hook_timeout_can_fail_open_and_continue_chain() {
        let manager = HookManager::new();
        let later_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        manager.register(Box::new(SlowHook {
            name: "best_effort_observer",
            policy: HookExecutionPolicy::bounded(5, HookFailurePolicy::FailOpen),
        }));
        manager.register(Box::new(FunctionHook::new(
            "later_guard",
            vec![HookPoint::LlmRequest],
            200,
            {
                let later_calls = later_calls.clone();
                move |_| {
                    later_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    HookResult::Continue
                }
            },
        )));
        let mut context = HookContext::new(HookPoint::LlmRequest, "agent", "PA");

        let decision = manager
            .execute_decision(HookPoint::LlmRequest, &mut context)
            .await;

        assert_eq!(decision.control, HookControl::Continue);
        assert!(decision.records[0].timed_out);
        assert_eq!(later_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn policy_hook_panic_fails_closed_and_rolls_back_partial_patch() {
        let manager = HookManager::new();
        manager.register(Box::new(PanickingHook {
            point: HookPoint::SkillBefore,
        }));
        let mut context = HookContext::new(HookPoint::SkillBefore, "agent", "DA")
            .with_data("stable", Value::Bool(true));

        let decision = manager
            .execute_decision(HookPoint::SkillBefore, &mut context)
            .await;

        assert_eq!(decision.control, HookControl::Abort);
        assert!(decision.records[0].panicked);
        assert_eq!(context.data.get("stable"), Some(&Value::Bool(true)));
        assert!(!context.data.contains_key("partial_patch"));
    }

    #[tokio::test]
    async fn observational_hook_panic_fails_open_and_rolls_back_partial_patch() {
        let manager = HookManager::new();
        manager.register(Box::new(PanickingHook {
            point: HookPoint::TaskEnd,
        }));
        let mut context = HookContext::new(HookPoint::TaskEnd, "agent", "DA");

        let decision = manager
            .execute_decision(HookPoint::TaskEnd, &mut context)
            .await;

        assert_eq!(decision.control, HookControl::Continue);
        assert!(decision.records[0].panicked);
        assert!(!context.data.contains_key("partial_patch"));
    }

    #[test]
    fn hook_context_provides_millisecond_time_and_trace_identifiers() {
        let context = HookContext::new(HookPoint::AgentInit, "agent", "PA");
        assert!(context.timestamp_ms >= context.timestamp.saturating_mul(1_000));
        assert!(!context.trace_id.is_empty());
        assert!(!context.span_id.is_empty());
    }

    #[tokio::test]
    async fn legacy_context_fields_are_normalized_before_execution() {
        let manager = HookManager::new();
        let mut context = HookContext::new(HookPoint::TaskStart, "agent", "PA");
        context.timestamp_ms = 0;
        context.trace_id.clear();
        context.span_id.clear();

        let decision = manager
            .execute_decision(HookPoint::TaskStart, &mut context)
            .await;

        assert_eq!(
            context.timestamp_ms,
            context.timestamp.saturating_mul(1_000)
        );
        assert!(!context.trace_id.is_empty());
        assert!(!context.span_id.is_empty());
        assert_eq!(decision.trace_id, context.trace_id);
    }

    #[test]
    fn unwired_hook_points_are_explicitly_experimental() {
        for point in [
            HookPoint::MemoryRead,
            HookPoint::BlackboardRead,
            HookPoint::McpToolCall,
            HookPoint::McpToolResult,
        ] {
            assert_eq!(point.support_level(), HookSupportLevel::Experimental);
            assert!(!point.support_note().is_empty());
        }
        for point in [
            HookPoint::AgentStart,
            HookPoint::TaskStart,
            HookPoint::LlmRequest,
            HookPoint::MemoryWrite,
            HookPoint::SkillBefore,
            HookPoint::BlackboardWrite,
            HookPoint::PhaseStart,
            HookPoint::PhaseEnd,
            HookPoint::CycleStart,
            HookPoint::CycleEnd,
        ] {
            assert_eq!(point.support_level(), HookSupportLevel::Stable);
        }
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
    fn metrics_hook_uses_bounded_aggregates_without_retaining_event_payloads() {
        let mut metrics = HashMap::new();
        for index in 0..10_000u64 {
            let mut context = HookContext::new(HookPoint::LlmResponse, "secret-agent", "DA")
                .with_task("secret-task", "iri://task/secret")
                .with_trace_id("secret-trace");
            context.timestamp_ms = index;
            context
                .metadata
                .insert("payload".into(), Value::String("secret-payload".into()));
            if index % 10 == 0 {
                context.error = Some("secret-error".into());
            }
            record_hook_metric(&mut metrics, &context);
        }

        assert_eq!(metrics.len(), 1);
        let aggregate = metrics.get("llm_response").unwrap();
        assert_eq!(aggregate.event_count, 10_000);
        assert_eq!(aggregate.error_count, 1_000);
        assert_eq!(aggregate.last_timestamp_ms, 9_999);
        let debug = format!("{metrics:?}");
        assert!(!debug.contains("secret"));
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
