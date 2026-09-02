//! Event Bus - Async event notification system with O(1) bitmap routing
//!
//! The event bus enables loose coupling between agents through
//! asynchronous event notifications.
//!
//! Key features:
//! - O(1) bitmap routing for type-based event matching
//! - Zero-copy event broadcasting using Arc
//! - SSE streaming support for real-time clients

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::ops::BitOr;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::debug;

/// Event types in the PDCA system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EventType {
    // Task lifecycle
    TaskCreated,
    TaskStarted,
    TaskCompleted,
    TaskFailed,
    TaskArchived,

    // PDCA phase events
    PlanStarted,
    PlanCompleted,
    DoStarted,
    DoCompleted,
    CheckStarted,
    CheckCompleted,
    ActStarted,
    ActCompleted,

    // Node events
    NodeCreated,
    NodeUpdated,
    NodeDeleted,

    // Agent events
    AgentStarted,
    AgentCompleted,
    AgentError,

    // System events
    CycleIteration,
    ThresholdExceeded,
    InterventionRequired,

    // Memory events
    MemoryInvalidate,
    MemoryWriteBack,
    MemoryPrefetch,
    MemoryLoad,

    // 5W2H constraint events
    DeadlineApproaching,
    BudgetExceeded,

    // Human approval events
    HumanApprovalRequired,
    HumanApprovalResult,

    // User supplementary input event
    UserSupplementaryInput,

    // ===== Batch Agent events =====
    BatchAgentRegistered,
    BatchAgentStarted,
    BatchAgentStopped,
    BatchAgentError,
    BatchExtractionStarted,
    BatchExtractionCompleted,
    BatchExtractionFailed,
    BatchEntityDetected,
    BatchRelationDetected,
    BatchIntentDetected,
    BatchDecisionDetected,
    BatchContextInjected,

    // ===== Batch Agent Maintenance events =====
    BatchSkillMergeSuggested,
    BatchSkillMergeApplied,
    BatchFragmentRefined,
    BatchEntityResolved,
    BatchEntityMergeConflict,
    BatchFailurePatternDetected,
    BatchHealthReportGenerated,
    BatchMemoryCompacted,
    BatchLinkRecommended,
    BatchLinkApplied,
    BatchTemplateAnalysisReady,

    // ===== Workspace file events =====
    WorkspaceFileCreated,
    WorkspaceFileModified,
    WorkspaceFileRemoved,
    WorkspaceFileStale,

    // Custom
    Custom(String),
}

impl EventType {
    pub fn as_str(&self) -> &str {
        match self {
            EventType::TaskCreated => "TASK_CREATED",
            EventType::TaskStarted => "TASK_STARTED",
            EventType::TaskCompleted => "TASK_COMPLETED",
            EventType::TaskFailed => "TASK_FAILED",
            EventType::TaskArchived => "TASK_ARCHIVED",
            EventType::PlanStarted => "PLAN_STARTED",
            EventType::PlanCompleted => "PLAN_COMPLETED",
            EventType::DoStarted => "DO_STARTED",
            EventType::DoCompleted => "DO_COMPLETED",
            EventType::CheckStarted => "CHECK_STARTED",
            EventType::CheckCompleted => "CHECK_COMPLETED",
            EventType::ActStarted => "ACT_STARTED",
            EventType::ActCompleted => "ACT_COMPLETED",
            EventType::NodeCreated => "NODE_CREATED",
            EventType::NodeUpdated => "NODE_UPDATED",
            EventType::NodeDeleted => "NODE_DELETED",
            EventType::AgentStarted => "AGENT_STARTED",
            EventType::AgentCompleted => "AGENT_COMPLETED",
            EventType::AgentError => "AGENT_ERROR",
            EventType::CycleIteration => "CYCLE_ITERATION",
            EventType::ThresholdExceeded => "THRESHOLD_EXCEEDED",
            EventType::InterventionRequired => "INTERVENTION_REQUIRED",
            EventType::MemoryInvalidate => "MEMORY_INVALIDATE",
            EventType::MemoryWriteBack => "MEMORY_WRITE_BACK",
            EventType::MemoryPrefetch => "MEMORY_PREFETCH",
            EventType::MemoryLoad => "MEMORY_LOAD",
            EventType::DeadlineApproaching => "DEADLINE_APPROACHING",
            EventType::BudgetExceeded => "BUDGET_EXCEEDED",
            EventType::HumanApprovalRequired => "HUMAN_APPROVAL_REQUIRED",
            EventType::HumanApprovalResult => "HUMAN_APPROVAL_RESULT",
            EventType::UserSupplementaryInput => "USER_SUPPLEMENTARY_INPUT",
            EventType::BatchAgentRegistered => "BATCH_AGENT_REGISTERED",
            EventType::BatchAgentStarted => "BATCH_AGENT_STARTED",
            EventType::BatchAgentStopped => "BATCH_AGENT_STOPPED",
            EventType::BatchAgentError => "BATCH_AGENT_ERROR",
            EventType::BatchExtractionStarted => "BATCH_EXTRACTION_STARTED",
            EventType::BatchExtractionCompleted => "BATCH_EXTRACTION_COMPLETED",
            EventType::BatchExtractionFailed => "BATCH_EXTRACTION_FAILED",
            EventType::BatchEntityDetected => "BATCH_ENTITY_DETECTED",
            EventType::BatchRelationDetected => "BATCH_RELATION_DETECTED",
            EventType::BatchIntentDetected => "BATCH_INTENT_DETECTED",
            EventType::BatchDecisionDetected => "BATCH_DECISION_DETECTED",
            EventType::BatchContextInjected => "BATCH_CONTEXT_INJECTED",
            EventType::BatchSkillMergeSuggested => "BATCH_SKILL_MERGE_SUGGESTED",
            EventType::BatchSkillMergeApplied => "BATCH_SKILL_MERGE_APPLIED",
            EventType::BatchFragmentRefined => "BATCH_FRAGMENT_REFINED",
            EventType::BatchEntityResolved => "BATCH_ENTITY_RESOLVED",
            EventType::BatchEntityMergeConflict => "BATCH_ENTITY_MERGE_CONFLICT",
            EventType::BatchFailurePatternDetected => "BATCH_FAILURE_PATTERN_DETECTED",
            EventType::BatchHealthReportGenerated => "BATCH_HEALTH_REPORT_GENERATED",
            EventType::BatchMemoryCompacted => "BATCH_MEMORY_COMPACTED",
            EventType::BatchLinkRecommended => "BATCH_LINK_RECOMMENDED",
            EventType::BatchLinkApplied => "BATCH_LINK_APPLIED",
            EventType::BatchTemplateAnalysisReady => "BATCH_TEMPLATE_ANALYSIS_READY",
            EventType::WorkspaceFileCreated => "WORKSPACE_FILE_CREATED",
            EventType::WorkspaceFileModified => "WORKSPACE_FILE_MODIFIED",
            EventType::WorkspaceFileRemoved => "WORKSPACE_FILE_REMOVED",
            EventType::WorkspaceFileStale => "WORKSPACE_FILE_STALE",
            EventType::Custom(s) => s.as_str(),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "TASK_CREATED" => EventType::TaskCreated,
            "TASK_STARTED" => EventType::TaskStarted,
            "TASK_COMPLETED" => EventType::TaskCompleted,
            "TASK_FAILED" => EventType::TaskFailed,
            "TASK_ARCHIVED" => EventType::TaskArchived,
            "PLAN_STARTED" => EventType::PlanStarted,
            "PLAN_COMPLETED" => EventType::PlanCompleted,
            "DO_STARTED" => EventType::DoStarted,
            "DO_COMPLETED" => EventType::DoCompleted,
            "CHECK_STARTED" => EventType::CheckStarted,
            "CHECK_COMPLETED" => EventType::CheckCompleted,
            "ACT_STARTED" => EventType::ActStarted,
            "ACT_COMPLETED" => EventType::ActCompleted,
            "NODE_CREATED" => EventType::NodeCreated,
            "NODE_UPDATED" => EventType::NodeUpdated,
            "NODE_DELETED" => EventType::NodeDeleted,
            "AGENT_STARTED" => EventType::AgentStarted,
            "AGENT_COMPLETED" => EventType::AgentCompleted,
            "AGENT_ERROR" => EventType::AgentError,
            "CYCLE_ITERATION" => EventType::CycleIteration,
            "THRESHOLD_EXCEEDED" => EventType::ThresholdExceeded,
            "INTERVENTION_REQUIRED" => EventType::InterventionRequired,
            "MEMORY_INVALIDATE" => EventType::MemoryInvalidate,
            "MEMORY_WRITE_BACK" => EventType::MemoryWriteBack,
            "MEMORY_PREFETCH" => EventType::MemoryPrefetch,
            "MEMORY_LOAD" => EventType::MemoryLoad,
            "DEADLINE_APPROACHING" => EventType::DeadlineApproaching,
            "BUDGET_EXCEEDED" => EventType::BudgetExceeded,
            "HUMAN_APPROVAL_REQUIRED" => EventType::HumanApprovalRequired,
            "HUMAN_APPROVAL_RESULT" => EventType::HumanApprovalResult,
            "USER_SUPPLEMENTARY_INPUT" => EventType::UserSupplementaryInput,
            "BATCH_AGENT_REGISTERED" => EventType::BatchAgentRegistered,
            "BATCH_AGENT_STARTED" => EventType::BatchAgentStarted,
            "BATCH_AGENT_STOPPED" => EventType::BatchAgentStopped,
            "BATCH_AGENT_ERROR" => EventType::BatchAgentError,
            "BATCH_EXTRACTION_STARTED" => EventType::BatchExtractionStarted,
            "BATCH_EXTRACTION_COMPLETED" => EventType::BatchExtractionCompleted,
            "BATCH_EXTRACTION_FAILED" => EventType::BatchExtractionFailed,
            "BATCH_ENTITY_DETECTED" => EventType::BatchEntityDetected,
            "BATCH_RELATION_DETECTED" => EventType::BatchRelationDetected,
            "BATCH_INTENT_DETECTED" => EventType::BatchIntentDetected,
            "BATCH_DECISION_DETECTED" => EventType::BatchDecisionDetected,
            "BATCH_CONTEXT_INJECTED" => EventType::BatchContextInjected,
            "BATCH_SKILL_MERGE_SUGGESTED" => EventType::BatchSkillMergeSuggested,
            "BATCH_SKILL_MERGE_APPLIED" => EventType::BatchSkillMergeApplied,
            "BATCH_FRAGMENT_REFINED" => EventType::BatchFragmentRefined,
            "BATCH_ENTITY_RESOLVED" => EventType::BatchEntityResolved,
            "BATCH_ENTITY_MERGE_CONFLICT" => EventType::BatchEntityMergeConflict,
            "BATCH_FAILURE_PATTERN_DETECTED" => EventType::BatchFailurePatternDetected,
            "BATCH_HEALTH_REPORT_GENERATED" => EventType::BatchHealthReportGenerated,
            "BATCH_MEMORY_COMPACTED" => EventType::BatchMemoryCompacted,
            "BATCH_LINK_RECOMMENDED" => EventType::BatchLinkRecommended,
            "BATCH_LINK_APPLIED" => EventType::BatchLinkApplied,
            "BATCH_TEMPLATE_ANALYSIS_READY" => EventType::BatchTemplateAnalysisReady,
            "WORKSPACE_FILE_CREATED" => EventType::WorkspaceFileCreated,
            "WORKSPACE_FILE_MODIFIED" => EventType::WorkspaceFileModified,
            "WORKSPACE_FILE_REMOVED" => EventType::WorkspaceFileRemoved,
            "WORKSPACE_FILE_STALE" => EventType::WorkspaceFileStale,
            other => EventType::Custom(other.to_string()),
        }
    }
}

impl FromStr for EventType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(EventType::from_str(s))
    }
}

/// Dynamically sized event-type bitset.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventTypeMask(Vec<u64>);

impl EventTypeMask {
    fn singleton(index: usize) -> Self {
        let mut words = vec![0; index / 64 + 1];
        words[index / 64] = 1u64 << (index % 64);
        Self(words)
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    fn intersects(&self, other: &Self) -> bool {
        self.0
            .iter()
            .zip(&other.0)
            .any(|(left, right)| left & right != 0)
    }

    fn union_assign(&mut self, other: &Self) {
        if self.0.len() < other.0.len() {
            self.0.resize(other.0.len(), 0);
        }
        for (index, word) in other.0.iter().enumerate() {
            self.0[index] |= word;
        }
    }
}

impl BitOr for EventTypeMask {
    type Output = Self;

    fn bitor(mut self, rhs: Self) -> Self::Output {
        self.union_assign(&rhs);
        self
    }
}

/// Registry assigning each event type a position in a dynamic bitset.
#[derive(Debug, Clone)]
pub struct TypeMask {
    masks: HashMap<String, usize>,
    next_bit: usize,
}

impl TypeMask {
    pub fn new() -> Self {
        Self {
            masks: HashMap::new(),
            next_bit: 0,
        }
    }

    pub fn get_or_create_mask(&mut self, type_name: &str) -> EventTypeMask {
        if let Some(&index) = self.masks.get(type_name) {
            return EventTypeMask::singleton(index);
        }

        let index = self.next_bit;
        self.next_bit += 1;
        self.masks.insert(type_name.to_string(), index);
        EventTypeMask::singleton(index)
    }

    pub fn combine_masks(&self, types: &[String]) -> EventTypeMask {
        let mut combined = EventTypeMask::default();
        for index in types
            .iter()
            .filter_map(|event_type| self.masks.get(event_type))
        {
            combined.union_assign(&EventTypeMask::singleton(*index));
        }
        combined
    }

    pub fn get_mask(&self, type_name: &str) -> Option<EventTypeMask> {
        self.masks
            .get(type_name)
            .map(|index| EventTypeMask::singleton(*index))
    }

    pub fn type_count(&self) -> usize {
        self.masks.len()
    }
}

impl Default for TypeMask {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal event with bitmap for O(1) routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalEvent {
    pub event_id: u64,
    pub change_type: String,
    pub affected_iri: Arc<str>,
    pub affected_types_mask: EventTypeMask,
    pub scope_iri: Arc<str>,
    pub timestamp: i64,
    pub payload: String,
}

/// An event in the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for EventPriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug, Clone)]
pub struct Event {
    pub event_id: String,
    pub task_iri: String,
    pub event_type: String,
    pub source_agent_iri: String,
    pub payload: String,
    pub payload_json_ld: String,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
    pub type_mask: EventTypeMask,
    pub priority: EventPriority,
}

/// Subscription with bitmap routing
#[derive(Debug, Clone)]
pub struct Subscription {
    pub subscriber_id: String,
    pub type_mask: EventTypeMask,
    pub scope_iri: Option<String>,
    pub event_types: Vec<String>,
}

impl Subscription {
    pub fn new(subscriber_id: String) -> Self {
        Self {
            subscriber_id,
            type_mask: EventTypeMask::default(),
            scope_iri: None,
            event_types: Vec::new(),
        }
    }

    pub fn with_type_mask(mut self, mask: EventTypeMask) -> Self {
        self.type_mask = mask;
        self
    }

    pub fn with_scope(mut self, scope_iri: String) -> Self {
        self.scope_iri = Some(scope_iri);
        self
    }

    pub fn with_event_types(mut self, types: Vec<String>) -> Self {
        self.event_types = types;
        self
    }

    /// O(1) check if event matches subscription
    pub fn matches(&self, event: &Event) -> bool {
        if !self.type_mask.is_empty() && !event.type_mask.intersects(&self.type_mask) {
            return false;
        }

        if let Some(ref scope) = self.scope_iri {
            if &event.task_iri != scope {
                return false;
            }
        }

        if !self.event_types.is_empty() && !self.event_types.contains(&event.event_type) {
            return false;
        }

        true
    }
}

/// Event bus configuration
#[derive(Debug, Clone)]
pub struct EventBusConfig {
    /// Buffer size for broadcast channel
    pub buffer_size: usize,

    /// Maximum events to keep in history
    pub max_history: usize,
}

impl Default for EventBusConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            max_history: 10000,
        }
    }
}

/// The event bus with O(1) bitmap routing
pub struct EventBus {
    /// Broadcast sender
    sender: broadcast::Sender<Event>,

    /// Event counter
    event_count: AtomicU64,

    /// Type mask registry
    type_mask: std::sync::Mutex<TypeMask>,

    /// Process-local bounded audit history. This is intentionally separate
    /// from broadcast delivery and is not a durable event log.
    history: std::sync::Mutex<VecDeque<Event>>,
    max_history: usize,
}

/// A broadcast receiver that applies a `Subscription` before exposing an
/// event. Lag/closed errors retain Tokio broadcast semantics so callers can
/// decide how to recover rather than silently losing them.
pub struct FilteredEventReceiver {
    receiver: broadcast::Receiver<Event>,
    subscription: Subscription,
}

impl FilteredEventReceiver {
    pub async fn recv(&mut self) -> Result<Event, broadcast::error::RecvError> {
        loop {
            let event = self.receiver.recv().await?;
            if self.subscription.matches(&event) {
                return Ok(event);
            }
        }
    }
}

impl EventBus {
    /// Create a new event bus
    pub fn new(buffer_size: usize) -> Self {
        Self::with_config(EventBusConfig {
            buffer_size,
            ..EventBusConfig::default()
        })
    }

    /// Create an event bus with independently configured broadcast and
    /// process-local history capacities.
    pub fn with_config(config: EventBusConfig) -> Self {
        let (sender, _) = broadcast::channel(config.buffer_size);

        Self {
            sender,
            event_count: AtomicU64::new(0),
            type_mask: std::sync::Mutex::new(TypeMask::new()),
            history: std::sync::Mutex::new(VecDeque::with_capacity(config.max_history)),
            max_history: config.max_history,
        }
    }

    /// Emit an event
    pub async fn emit(
        &self,
        task_iri: &str,
        event_type: &str,
        source_agent_iri: &str,
        payload: &str,
    ) -> String {
        self.emit_with_priority(
            task_iri,
            event_type,
            source_agent_iri,
            payload,
            EventPriority::Normal,
        )
        .await
    }

    pub async fn emit_with_priority(
        &self,
        task_iri: &str,
        event_type: &str,
        source_agent_iri: &str,
        payload: &str,
        priority: EventPriority,
    ) -> String {
        let sequence = self.event_count.fetch_add(1, Ordering::Relaxed);
        let event_id = format!("evt_{}", uuid::Uuid::new_v4().hyphenated());

        let type_mask = {
            let mut mask = self.type_mask.lock().expect("type_mask Mutex poisoned");
            mask.get_or_create_mask(event_type)
        };

        let event = Event {
            event_id: event_id.clone(),
            task_iri: task_iri.to_string(),
            event_type: event_type.to_string(),
            source_agent_iri: source_agent_iri.to_string(),
            payload: payload.to_string(),
            payload_json_ld: payload.to_string(),
            timestamp: Utc::now(),
            sequence,
            type_mask,
            priority,
        };

        debug!(
            event_id = %event_id,
            event_type = %event_type,
            task_iri = %task_iri,
            type_mask = ?event.type_mask,
            "Event emitted"
        );

        if self.max_history > 0 {
            let mut history = self.history.lock().expect("event history Mutex poisoned");
            if history.len() >= self.max_history {
                history.pop_front();
            }
            history.push_back(event.clone());
        }

        let _ = self.sender.send(event);

        event_id
    }

    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.sender.subscribe()
    }

    /// Subscribe with a specific subscription filter. Events that do not
    /// match are consumed from this receiver and never exposed to its caller.
    pub fn subscribe_with_filter(&self, subscription: Subscription) -> FilteredEventReceiver {
        FilteredEventReceiver {
            receiver: self.sender.subscribe(),
            subscription,
        }
    }

    /// Register a type for bitmap routing
    pub fn register_type(&self, type_name: &str) -> EventTypeMask {
        let mut mask = self.type_mask.lock().expect("type_mask Mutex poisoned");
        mask.get_or_create_mask(type_name)
    }

    /// Get combined mask for multiple types
    pub fn get_combined_mask(&self, types: &[String]) -> EventTypeMask {
        let mask = self.type_mask.lock().expect("type_mask Mutex poisoned");
        mask.combine_masks(types)
    }

    /// Get event count
    pub fn event_count(&self) -> u64 {
        self.event_count.load(Ordering::Relaxed)
    }

    /// Get subscriber count
    pub fn subscriber_count(&self) -> u64 {
        self.sender.receiver_count() as u64
    }

    /// Get registered type count
    pub fn type_count(&self) -> usize {
        self.type_mask
            .lock()
            .expect("type_mask Mutex poisoned")
            .type_count()
    }

    /// Return up to `limit` matching events in chronological order from the
    /// bounded in-process history. It is useful for diagnostics and task
    /// audits, not for durable replay.
    pub fn recent_events(&self, filter: &EventFilter, limit: usize) -> Vec<Event> {
        if limit == 0 {
            return Vec::new();
        }
        let history = self.history.lock().expect("event history Mutex poisoned");
        let mut result: Vec<Event> = history
            .iter()
            .rev()
            .filter(|event| filter.matches(event))
            .take(limit)
            .cloned()
            .collect();
        result.reverse();
        result
    }

    pub fn history_len(&self) -> usize {
        self.history
            .lock()
            .expect("event history Mutex poisoned")
            .len()
    }

    pub fn try_recv(&self) -> Result<Event, broadcast::error::TryRecvError> {
        let mut receiver = self.sender.subscribe();
        receiver.try_recv()
    }

    pub fn spawn_consumer<F, Fut>(&self, event_types: Vec<String>, handler: F)
    where
        F: Fn(Event) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let mut receiver = self.sender.subscribe();
        let handler = Arc::new(handler);
        tokio::spawn(async move {
            loop {
                match receiver.recv().await {
                    Ok(event) => {
                        if event_types.is_empty() || event_types.contains(&event.event_type) {
                            handler(event).await;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("EventBus consumer lagged by {} events", n);
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }
}

/// Event filter for subscription
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    /// Filter by task IRI
    pub task_iri: Option<String>,

    /// Filter by event types
    pub event_types: Vec<String>,

    /// Filter by source agent
    pub source_agent: Option<String>,

    /// Type mask for O(1) routing
    pub type_mask: EventTypeMask,
}

impl EventFilter {
    /// Check if an event matches the filter
    pub fn matches(&self, event: &Event) -> bool {
        if !self.type_mask.is_empty() && !event.type_mask.intersects(&self.type_mask) {
            return false;
        }

        if let Some(ref task_iri) = self.task_iri {
            if &event.task_iri != task_iri {
                return false;
            }
        }

        if !self.event_types.is_empty() && !self.event_types.contains(&event.event_type) {
            return false;
        }

        if let Some(ref source) = self.source_agent {
            if &event.source_agent_iri != source {
                return false;
            }
        }

        true
    }

    /// Create filter with type mask for O(1) matching
    pub fn with_type_mask(mut self, mask: EventTypeMask) -> Self {
        self.type_mask = mask;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_event_bus() {
        let bus = EventBus::new(100);
        let mut receiver = bus.subscribe();

        let event_id = bus
            .emit(
                "iri://task_1",
                "TEST_EVENT",
                "iri://agent_1",
                r#"{"test": true}"#,
            )
            .await;

        assert!(!event_id.is_empty());

        let event = receiver.recv().await.unwrap();
        assert_eq!(event.event_type, "TEST_EVENT");
    }

    #[tokio::test]
    async fn bounded_history_keeps_newest_matching_events_in_chronological_order() {
        let bus = EventBus::with_config(EventBusConfig {
            buffer_size: 8,
            max_history: 2,
        });
        bus.emit("task:a", "FIRST", "agent:a", "{}").await;
        bus.emit("task:b", "SECOND", "agent:b", "{}").await;
        bus.emit("task:a", "THIRD", "agent:a", "{}").await;

        assert_eq!(bus.history_len(), 2);
        let task_a = bus.recent_events(
            &EventFilter {
                task_iri: Some("task:a".to_string()),
                ..EventFilter::default()
            },
            10,
        );
        assert_eq!(
            task_a
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec!["THIRD"]
        );

        let all = bus.recent_events(&EventFilter::default(), 10);
        assert_eq!(
            all.iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec!["SECOND", "THIRD"]
        );
    }

    #[tokio::test]
    async fn filtered_subscription_skips_non_matching_broadcast_events() {
        let bus = EventBus::new(8);
        let task_mask = bus.register_type("TASK_MATCH");
        let mut receiver = bus.subscribe_with_filter(
            Subscription::new("task-only".to_string())
                .with_scope("task:target".to_string())
                .with_type_mask(task_mask),
        );

        bus.emit("task:other", "TASK_MATCH", "agent", "{}").await;
        bus.emit("task:target", "UNRELATED", "agent", "{}").await;
        bus.emit("task:target", "TASK_MATCH", "agent", "{}").await;

        let event = receiver.recv().await.unwrap();
        assert_eq!(event.task_iri, "task:target");
        assert_eq!(event.event_type, "TASK_MATCH");
    }

    #[test]
    fn test_type_mask() {
        let mut mask = TypeMask::new();

        let mask1 = mask.get_or_create_mask("PLAN_NODE");
        let mask2 = mask.get_or_create_mask("CODE_ARTIFACT");
        let mask3 = mask.get_or_create_mask("REVIEW_RESULT");

        assert_ne!(mask1, mask2);
        assert_ne!(mask2, mask3);

        let combined = mask.combine_masks(&["PLAN_NODE".to_string(), "CODE_ARTIFACT".to_string()]);
        assert_eq!(combined, mask1.clone() | mask2.clone());

        assert!(combined.intersects(&mask1));
        assert!(combined.intersects(&mask2));
        assert!(!combined.intersects(&mask3));
    }

    #[tokio::test]
    async fn dynamic_type_mask_supports_more_than_sixty_four_event_types() {
        let bus = EventBus::new(8);
        let mut last = EventTypeMask::default();
        for index in 0..130 {
            last = bus.register_type(&format!("CUSTOM_{index}"));
        }
        assert_eq!(bus.type_count(), 130);
        assert!(!last.is_empty());

        let mut receiver = bus.subscribe_with_filter(
            Subscription::new("late-type".to_string())
                .with_type_mask(last)
                .with_event_types(vec!["CUSTOM_129".to_string()]),
        );
        bus.emit("task", "CUSTOM_129", "agent", "{}").await;
        assert_eq!(receiver.recv().await.unwrap().event_type, "CUSTOM_129");
    }

    #[test]
    fn subscriber_count_tracks_receiver_lifetime() {
        let bus = EventBus::new(8);
        assert_eq!(bus.subscriber_count(), 0);
        let receiver = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 1);
        drop(receiver);
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn test_subscription_matching() {
        let mut type_mask = TypeMask::new();
        let plan_mask = type_mask.get_or_create_mask("PLAN_NODE");
        let code_mask = type_mask.get_or_create_mask("CODE_ARTIFACT");

        let subscription = Subscription::new("sub_1".to_string())
            .with_type_mask(plan_mask.clone() | code_mask.clone());

        let plan_event = Event {
            event_id: "evt_1".to_string(),
            task_iri: "iri://task_1".to_string(),
            event_type: "PLAN_NODE".to_string(),
            source_agent_iri: "iri://agent_1".to_string(),
            payload: "{}".to_string(),
            payload_json_ld: "{}".to_string(),
            timestamp: Utc::now(),
            sequence: 1,
            type_mask: plan_mask,
            priority: EventPriority::Normal,
        };

        let review_event = Event {
            event_id: "evt_2".to_string(),
            task_iri: "iri://task_1".to_string(),
            event_type: "REVIEW_RESULT".to_string(),
            source_agent_iri: "iri://agent_1".to_string(),
            payload: "{}".to_string(),
            payload_json_ld: "{}".to_string(),
            timestamp: Utc::now(),
            sequence: 2,
            type_mask: type_mask.get_or_create_mask("REVIEW_RESULT"),
            priority: EventPriority::Normal,
        };

        assert!(subscription.matches(&plan_event));
        assert!(!subscription.matches(&review_event));
    }

    #[test]
    fn test_event_type_5w2h_variants() {
        assert_eq!(
            EventType::DeadlineApproaching.as_str(),
            "DEADLINE_APPROACHING"
        );
        assert_eq!(EventType::BudgetExceeded.as_str(), "BUDGET_EXCEEDED");
        assert_eq!(
            "DEADLINE_APPROACHING".parse::<EventType>(),
            Ok(EventType::DeadlineApproaching)
        );
        assert_eq!(
            "BUDGET_EXCEEDED".parse::<EventType>(),
            Ok(EventType::BudgetExceeded)
        );
    }
}
