/// Shared types for the RootCauseEngine module.
///
/// Architecture Layer: L1 — Enforcement
/// See design: PR-res/superpowers-skills-full-integration-design.md §2
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ════════════════════════════════════════════════════════════════════════
// Core Types
// ════════════════════════════════════════════════════════════════════════

/// A single level in a traceback chain.
/// Each level represents one step in the 5-level backward trace:
/// L1: Symptom → L2: Direct Caller → L3: Intermediate → L4: Context → L5: Root Cause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceLevel {
    /// Level number (1-based, 5 = root cause)
    pub level: u8,
    /// Label: "symptom" | "intermediate" | "root_cause"
    pub label: String,
    /// Human-readable description of what happened at this level
    pub description: String,
    /// Source file:line where this level was captured
    pub source_location: String,
    /// Whether this level is identified as the root cause
    pub is_root_cause: bool,
    /// Evidence collected at this level
    pub evidence: Evidence,
}

/// A complete trace chain from symptom to root cause.
/// Produced by BackwardTracer::trace_backward().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceChain {
    /// Unique trace identifier
    pub trace_id: String,
    /// Levels in order from symptom (L1) to root cause (L5)
    pub levels: Vec<TraceLevel>,
    /// Agent ID that triggered the trace
    pub agent_id: String,
    /// Task ID being traced
    pub task_id: Option<String>,
    /// Timestamp when trace was initiated
    pub timestamp: u64,
    /// Whether the trace reached a definitive root cause
    pub resolved: bool,
}

impl TraceChain {
    pub fn new(trace_id: &str, agent_id: &str) -> Self {
        Self {
            trace_id: trace_id.to_string(),
            levels: Vec::new(),
            agent_id: agent_id.to_string(),
            task_id: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            resolved: false,
        }
    }

    pub fn add_level(&mut self, level: TraceLevel) {
        if level.is_root_cause {
            self.resolved = true;
        }
        self.levels.push(level);
    }

    pub fn root_level(&self) -> Option<&TraceLevel> {
        self.levels.iter().find(|l| l.is_root_cause)
    }

    pub fn has_root_cause(&self) -> bool {
        self.levels.iter().any(|l| l.is_root_cause)
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    pub fn summary(&self) -> String {
        let root = self
            .root_level()
            .map(|l| l.description.as_str())
            .unwrap_or("Unidentified root cause");
        format!(
            "Trace[{}]: {} levels, root={}, agent={}",
            self.trace_id,
            self.depth(),
            root,
            self.agent_id
        )
    }
}

/// Evidence collected at a single trace level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Source of the evidence (file, log, error message, etc.)
    pub source: String,
    /// The actual extracted value or observation
    pub value: Value,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Optional supporting evidence references
    pub references: Vec<String>,
}

impl Evidence {
    pub fn new(source: &str, value: Value, confidence: f64) -> Self {
        Self {
            source: source.to_string(),
            value,
            confidence,
            references: Vec::new(),
        }
    }

    pub fn with_references(mut self, refs: Vec<String>) -> Self {
        self.references = refs;
        self
    }
}

/// Errors that can occur during root cause tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RootCauseError {
    TraceDeadEnd {
        message: String,
        last_known: Box<TraceLevel>,
    },
    InvalidEvidenceChain {
        message: String,
        errors: Vec<String>,
    },
    InsufficientEvidence {
        message: String,
        min_confidence: f64,
        actual_confidence: f64,
    },
    Internal {
        message: String,
    },
}

impl std::fmt::Display for RootCauseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TraceDeadEnd { message, .. } => write!(f, "Trace dead end: {}", message),
            Self::InvalidEvidenceChain { message, .. } => {
                write!(f, "Invalid evidence chain: {}", message)
            }
            Self::InsufficientEvidence { message, .. } => {
                write!(f, "Insufficient evidence: {}", message)
            }
            Self::Internal { message } => write!(f, "Internal error: {}", message),
        }
    }
}

impl std::error::Error for RootCauseError {}

/// Validation error from EvidenceChainManager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainValidationError {
    pub errors: Vec<String>,
}

impl std::fmt::Display for ChainValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Chain validation errors: {}", self.errors.join("; "))
    }
}

/// A defense recommendation generated by DefenseInDepthManager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefenseRecommendation {
    pub layer: DefenseLayer,
    pub title: String,
    pub description: String,
    pub priority: u8,
}

/// The 4 defense layers from defense-in-depth methodology
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DefenseLayer {
    /// Prevent errors at entry point
    EntryValidation,
    /// Validate business logic assumptions
    BusinessLogic,
    /// Guard against environmental issues
    EnvironmentGuard,
    /// Instrument for observability
    Instrumentation,
}

impl DefenseLayer {
    pub fn name(&self) -> &'static str {
        match self {
            Self::EntryValidation => "entry validation",
            Self::BusinessLogic => "business logic",
            Self::EnvironmentGuard => "environment guard",
            Self::Instrumentation => "observability",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::EntryValidation => "Validate input parameters and preconditions at entry",
            Self::BusinessLogic => "Add defensive checks in business logic layer",
            Self::EnvironmentGuard => "Add guard measures in environment layer",
            Self::Instrumentation => "Add logging and monitoring on critical paths",
        }
    }
}

/// Context passed to the tracer during a trace session
#[derive(Debug, Clone, Default)]
pub struct TraceContext {
    pub task_type: String,
    pub failure_description: String,
    pub available_logs: Vec<String>,
    pub available_metrics: HashMap<String, Value>,
    pub max_depth: u8,
}

/// Configuration for the RootCauseEngine
#[derive(Debug, Clone, PartialEq)]
pub struct RootCauseConfig {
    pub max_trace_depth: u8,
    pub min_confidence: f64,
    pub enable_auto_trace: bool,
    pub enable_defense_recommendations: bool,
    pub trace_timeout_ms: u64,
}

impl Default for RootCauseConfig {
    fn default() -> Self {
        Self {
            max_trace_depth: 5,
            min_confidence: 0.7,
            enable_auto_trace: true,
            enable_defense_recommendations: true,
            trace_timeout_ms: 30_000,
        }
    }
}
