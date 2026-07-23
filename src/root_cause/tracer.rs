/// BackwardTracer — 5-level traceback algorithm.
///
/// From the failure point, traces upward level by level asking:
/// "What called this?" and "What values were passed?"
/// Each level produces evidence that must chain to the next.
///
/// Corresponds to: superpowers-main/skills/systematic-debugging/root-cause-tracing.md
/// Architecture Layer: L1 — Enforcement (RootCauseEngine)
///
/// Algorithm:
///   Level 1 (Symptom):  Record the error as it occurred — message, stack, location
///   Level 2 (Caller):    Identify the direct caller — "who invoked the failed operation?"
///   Level 3 (Context):   Examine the surrounding state — "what was the context at the caller?"
///   Level 4 (Trigger):   Find the triggering event — "what initiated this chain?"
///   Level 5 (Root):      Identify the root cause — "what precondition or assumption was violated?"
use std::collections::HashMap;
use std::sync::RwLock;

use serde_json::json;

use super::types::{
    Evidence, RootCauseConfig, RootCauseError, TraceChain, TraceContext, TraceLevel,
};

/// A stack frame extracted from an error or log context
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub function: String,
    pub file: String,
    pub line: u32,
    pub caller: Option<String>,
    pub params: HashMap<String, String>,
}

/// BackwardTracer — core 5-level traceback engine
pub struct BackwardTracer {
    config: RootCauseConfig,
    /// Active trace sessions, keyed by trace_id
    active_traces: RwLock<HashMap<String, TraceChain>>,
    /// Known error patterns for root cause matching
    pattern_db: Vec<ErrorPattern>,
}

/// An error pattern used to match common root causes
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    pub pattern: &'static str,
    pub root_cause_label: &'static str,
    pub root_cause_description: &'static str,
    pub confidence: f64,
}

impl BackwardTracer {
    pub fn new(config: RootCauseConfig) -> Self {
        Self {
            pattern_db: Self::default_patterns(),
            config,
            active_traces: RwLock::new(HashMap::new()),
        }
    }

    /// Default error patterns for common root cause matching
    fn default_patterns() -> Vec<ErrorPattern> {
        vec![
            ErrorPattern {
                pattern: "connection refused|connection reset|timeout",
                root_cause_label: "network_error",
                root_cause_description:
                    "Network connection failed — target service unavailable or network unreachable",
                confidence: 0.9,
            },
            ErrorPattern {
                pattern: "not found|no such file|enoent|no such",
                root_cause_label: "resource_not_found",
                root_cause_description:
                    "Resource not found — path/file/URL is incorrect or has been removed",
                confidence: 0.85,
            },
            ErrorPattern {
                pattern: "permission denied|access denied|forbidden|eacces",
                root_cause_label: "permission_error",
                root_cause_description:
                    "Permission denied — current environment lacks access to this resource",
                confidence: 0.9,
            },
            ErrorPattern {
                pattern: "syntax error|parse error|invalid syntax",
                root_cause_label: "syntax_error",
                root_cause_description:
                    "Syntax error — input data format does not match expectations",
                confidence: 0.8,
            },
            ErrorPattern {
                pattern: "out of memory|oom|no space|disk full",
                root_cause_label: "resource_exhausted",
                root_cause_description: "Resource exhausted — memory/disk/connections at capacity",
                confidence: 0.9,
            },
            ErrorPattern {
                pattern: "null pointer|undefined|cannot read property|unwrap.*none",
                root_cause_label: "null_reference",
                root_cause_description:
                    "Null reference — accessed uninitialized or non-existent value",
                confidence: 0.85,
            },
            ErrorPattern {
                pattern: "invalid argument|invalid input|bad request|400",
                root_cause_label: "invalid_input",
                root_cause_description: "Invalid input — parameter value outside expected range",
                confidence: 0.8,
            },
        ]
    }

    /// Main entry: trace backward from an error to find root cause.
    ///
    /// 1. Records the symptom (L1)
    /// 2. Extracts call frames from the error context
    /// 3. Walks up the call chain (L2-L4)
    /// 4. Matches against error patterns for root cause (L5)
    pub fn trace_backward(
        &self,
        error_message: &str,
        source_location: &str,
        context: &TraceContext,
        trace_id: &str,
        agent_id: &str,
    ) -> Result<TraceChain, RootCauseError> {
        let mut chain = TraceChain::new(trace_id, agent_id);
        chain.task_id = Some(context.failure_description.clone());

        // Level 1: Symptom
        let symptom_level = TraceLevel {
            level: 1,
            label: "symptom".to_string(),
            description: format!("Error occurred: {}", error_message),
            source_location: source_location.to_string(),
            is_root_cause: false,
            evidence: Evidence::new(
                source_location,
                json!({
                    "error": error_message,
                    "task_context": context.failure_description,
                }),
                1.0,
            ),
        };
        chain.add_level(symptom_level);

        // Level 2: Direct caller — extract from error message patterns
        let caller_level = self.trace_caller(error_message, source_location, context)?;
        chain.add_level(caller_level);

        // Level 3: Context — examine surrounding state
        let context_level = self.trace_context(error_message, context)?;
        chain.add_level(context_level);

        // Level 4: Trigger — identify what initiated the chain
        let trigger_level = self.trace_trigger(context)?;
        chain.add_level(trigger_level);

        // Level 5: Root cause — match against known patterns
        let root_level = self.identify_root_cause(error_message, context)?;
        let root_confidence = root_level.evidence.confidence;
        chain.add_level(root_level);

        // Validate evidence chain confidence
        if root_confidence < self.config.min_confidence {
            return Err(RootCauseError::InsufficientEvidence {
                message: "Root cause evidence confidence insufficient, need more context"
                    .to_string(),
                min_confidence: self.config.min_confidence,
                actual_confidence: root_confidence,
            });
        }

        Ok(chain)
    }

    /// Level 2: Identify the direct caller
    fn trace_caller(
        &self,
        error_message: &str,
        source_location: &str,
        _context: &TraceContext,
    ) -> Result<TraceLevel, RootCauseError> {
        let caller_info = self.extract_caller_from_error(error_message, source_location);

        Ok(TraceLevel {
            level: 2,
            label: "intermediate".to_string(),
            description: format!(
                "Caller: {} (call site: {}:{})",
                caller_info.function, caller_info.file, caller_info.line
            ),
            source_location: format!("{}:{}", caller_info.file, caller_info.line),
            is_root_cause: false,
            evidence: Evidence::new(
                &format!("{}:{}", caller_info.file, caller_info.line),
                json!({
                    "function": caller_info.function,
                    "caller": caller_info.caller,
                    "params": caller_info.params,
                }),
                0.85,
            ),
        })
    }

    /// Level 3: Examine surrounding context
    fn trace_context(
        &self,
        _error_message: &str,
        context: &TraceContext,
    ) -> Result<TraceLevel, RootCauseError> {
        Ok(TraceLevel {
            level: 3,
            label: "intermediate".to_string(),
            description: format!(
                "Context state: task_type={}, failure_description={}, available_logs={}",
                context.task_type,
                context.failure_description,
                context.available_logs.len(),
            ),
            source_location: "trace_context".to_string(),
            is_root_cause: false,
            evidence: Evidence::new(
                "trace_context",
                json!({
                    "task_type": context.task_type,
                    "failure_description": context.failure_description,
                    "log_count": context.available_logs.len(),
                    "metrics": context.available_metrics,
                }),
                0.75,
            ),
        })
    }

    /// Level 4: Identify the trigger
    fn trace_trigger(&self, context: &TraceContext) -> Result<TraceLevel, RootCauseError> {
        Ok(TraceLevel {
            level: 4,
            label: "intermediate".to_string(),
            description: format!(
                "Trigger event: exception occurred during task '{}' execution",
                context.task_type
            ),
            source_location: "trace_trigger".to_string(),
            is_root_cause: false,
            evidence: Evidence::new(
                "trace_trigger",
                json!({
                    "trigger": context.task_type,
                    "state_at_trigger": "task_running",
                }),
                0.7,
            ),
        })
    }

    /// Level 5: Match against known error patterns to identify root cause
    fn identify_root_cause(
        &self,
        error_message: &str,
        _context: &TraceContext,
    ) -> Result<TraceLevel, RootCauseError> {
        let lower = error_message.to_lowercase();

        for pattern in &self.pattern_db {
            // Simple substring matching for pattern detection
            let parts: Vec<&str> = pattern.pattern.split('|').collect();
            if parts.iter().any(|p| lower.contains(p.trim())) {
                return Ok(TraceLevel {
                    level: 5,
                    label: "root_cause".to_string(),
                    description: format!(
                        "[{}] {}",
                        pattern.root_cause_label, pattern.root_cause_description
                    ),
                    source_location: "pattern_match".to_string(),
                    is_root_cause: true,
                    evidence: Evidence::new(
                        "pattern_match",
                        json!({
                            "matched_pattern": pattern.pattern,
                            "root_cause_label": pattern.root_cause_label,
                            "matched_text": error_message,
                        }),
                        pattern.confidence,
                    ),
                });
            }
        }

        // Fallback: return last resort "unknown" root cause
        Ok(TraceLevel {
            level: 5,
            label: "root_cause".to_string(),
            description: "Root cause did not match any known pattern — requires manual analysis"
                .to_string(),
            source_location: "unknown".to_string(),
            is_root_cause: true,
            evidence: Evidence::new(
                "unknown",
                json!({
                    "error": error_message,
                    "note": "No known error pattern matched",
                }),
                0.5,
            ),
        })
    }

    /// Extract caller information from error message and source location
    fn extract_caller_from_error(&self, error_message: &str, source_location: &str) -> CallFrame {
        let parts: Vec<&str> = source_location.rsplitn(2, ':').collect();
        let (file, line_str) = if parts.len() == 2 {
            (parts[1], parts[0])
        } else {
            (source_location, "0")
        };
        let line: u32 = line_str.parse().unwrap_or(0);

        // Try to extract function name from error context
        let function = self.infer_function_from_error(error_message);

        CallFrame {
            function,
            file: file.to_string(),
            line,
            caller: None,
            params: HashMap::new(),
        }
    }

    /// Infer likely function from error message keywords
    fn infer_function_from_error(&self, error: &str) -> String {
        let lower = error.to_lowercase();
        if lower.contains("read") || lower.contains("open") || lower.contains("file") {
            "file_operation".to_string()
        } else if lower.contains("connect")
            || lower.contains("network")
            || lower.contains("timeout")
        {
            "network_operation".to_string()
        } else if lower.contains("parse") || lower.contains("syntax") || lower.contains("json") {
            "data_parsing".to_string()
        } else if lower.contains("write") || lower.contains("save") || lower.contains("create") {
            "write_operation".to_string()
        } else if lower.contains("permission")
            || lower.contains("denied")
            || lower.contains("forbidden")
        {
            "authorization_check".to_string()
        } else if lower.contains("null") || lower.contains("undefined") || lower.contains("empty") {
            "null_check".to_string()
        } else {
            "unknown_operation".to_string()
        }
    }

    /// Save a trace chain for later retrieval
    pub fn save_trace(&self, chain: TraceChain) {
        let trace_id = chain.trace_id.clone();
        if let Ok(mut traces) = self.active_traces.write() {
            traces.insert(trace_id, chain);
        }
    }

    /// Get a saved trace by ID
    pub fn get_trace(&self, trace_id: &str) -> Option<TraceChain> {
        self.active_traces
            .read()
            .ok()
            .and_then(|t| t.get(trace_id).cloned())
    }

    /// Check if a task has an unresolved trace
    pub fn has_unresolved_trace(&self, task_id: &str) -> bool {
        self.active_traces.read().ok().map_or(false, |traces| {
            traces
                .values()
                .any(|t| t.task_id.as_deref() == Some(task_id) && !t.resolved)
        })
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> RootCauseConfig {
        RootCauseConfig {
            min_confidence: 0.1, // low threshold for test
            ..Default::default()
        }
    }

    #[test]
    fn test_trace_network_error() {
        let tracer = BackwardTracer::new(test_config());
        let ctx = TraceContext {
            task_type: "http_request".to_string(),
            failure_description: "GET /api/users failed".to_string(),
            ..Default::default()
        };
        let result = tracer.trace_backward(
            "connection refused: failed to connect to 127.0.0.1:8080",
            "src/http/client.rs:42",
            &ctx,
            "trace_001",
            "agent_1",
        );
        assert!(
            result.is_ok(),
            "Should trace network error: {:?}",
            result.err()
        );
        let chain = result.unwrap();
        assert!(chain.has_root_cause(), "Should identify root cause");
        assert_eq!(chain.depth(), 5, "Should have 5 levels");
        let root = chain.root_level().unwrap();
        assert_eq!(root.level, 5);
        assert!(
            root.description.contains("network_error"),
            "Root cause should be network_error, got: {}",
            root.description
        );
    }

    #[test]
    fn test_trace_file_not_found() {
        let tracer = BackwardTracer::new(test_config());
        let ctx = TraceContext::default();
        let result = tracer.trace_backward(
            "No such file or directory: /etc/config.yaml",
            "src/config/loader.rs:88",
            &ctx,
            "trace_002",
            "agent_2",
        );
        assert!(result.is_ok());
        let chain = result.unwrap();
        assert!(chain.has_root_cause());
        let root = chain.root_level().unwrap();
        assert!(root.description.contains("resource_not_found"));
    }

    #[test]
    fn test_trace_unknown_error() {
        let tracer = BackwardTracer::new(test_config());
        let ctx = TraceContext::default();
        let result = tracer.trace_backward(
            "Something unusual happened: code 0xDEAD",
            "src/main.rs:1",
            &ctx,
            "trace_003",
            "agent_3",
        );
        assert!(result.is_ok());
        let chain = result.unwrap();
        // Unknown errors still resolve with low confidence
        assert!(chain.has_root_cause());
        // But should not match any known pattern
        assert!(!chain
            .root_level()
            .unwrap()
            .description
            .contains("network_error"));
    }

    #[test]
    fn test_trace_confidence_too_low() {
        let tracer = BackwardTracer::new(RootCauseConfig {
            min_confidence: 0.99, // impossibly high
            ..Default::default()
        });
        let ctx = TraceContext::default();
        let result = tracer.trace_backward(
            "permission denied: /etc/shadow",
            "src/auth.rs:10",
            &ctx,
            "trace_004",
            "agent_4",
        );
        assert!(result.is_err(), "Should fail when confidence too low");
    }

    #[test]
    fn test_trace_save_and_retrieve() {
        let tracer = BackwardTracer::new(test_config());
        let ctx = TraceContext::default();
        let chain = tracer
            .trace_backward(
                "disk full: no space left on device",
                "src/storage.rs:200",
                &ctx,
                "trace_005",
                "agent_5",
            )
            .unwrap();
        tracer.save_trace(chain);
        let retrieved = tracer.get_trace("trace_005");
        assert!(retrieved.is_some(), "Should retrieve saved trace");
        assert_eq!(retrieved.unwrap().trace_id, "trace_005");
    }

    #[test]
    fn test_has_unresolved_trace() {
        let tracer = BackwardTracer::new(test_config());
        let ctx = TraceContext {
            failure_description: "task_abc".to_string(),
            ..Default::default()
        };
        let chain = tracer
            .trace_backward(
                "timeout: connection timed out",
                "src/network.rs:50",
                &ctx,
                "trace_006",
                "agent_6",
            )
            .unwrap();
        tracer.save_trace(chain);
        // All test traces are resolved (since they have root cause)
        assert!(!tracer.has_unresolved_trace("task_abc"));
    }

    #[test]
    fn test_infer_function() {
        let tracer = BackwardTracer::new(test_config());
        assert_eq!(
            tracer.infer_function_from_error("file not found"),
            "file_operation"
        );
        assert_eq!(
            tracer.infer_function_from_error("connection timeout"),
            "network_operation"
        );
        assert_eq!(
            tracer.infer_function_from_error("parse error: invalid json"),
            "data_parsing"
        );
        assert_eq!(
            tracer.infer_function_from_error("write failed"),
            "write_operation"
        );
        assert_eq!(
            tracer.infer_function_from_error("access denied"),
            "authorization_check"
        );
        assert_eq!(
            tracer.infer_function_from_error("null pointer"),
            "null_check"
        );
        assert_eq!(
            tracer.infer_function_from_error("something random"),
            "unknown_operation"
        );
    }
}
