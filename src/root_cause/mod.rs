/// RootCauseEngine — 5-level traceback + evidence chain + defense depth engine.
///
/// Integrates with HookManager to automatically trace errors and validate
/// that root cause analysis is complete before allowing remediation.
///
/// Architecture Layer: L1 — Enforcement
/// See design: PR-res/superpowers-skills-full-integration-design.md §2
pub mod config;
pub mod defense;
pub mod evidence;
pub mod tracer;
pub mod types;

pub use types::*;

use std::sync::Arc;

use crate::causal::fused::{FusedRootCause, FusedRootCauseEngine};
use crate::tools::hooks::{FunctionHook, HookManager, HookPoint, HookResult};

/// RootCauseEngine — main entry point for root cause tracing and defense generation.
///
/// Combines:
/// - BackwardTracer (5-level traceback)
/// - EvidenceChainManager (chain validation)
/// - DefenseInDepthManager (defense recommendations)
///
/// Usage:
/// ```rust,ignore
/// let engine = RootCauseEngine::new(config);
/// engine.register_hooks(&hook_manager);
/// let chain = engine.trace("error msg", "file.rs:42", &context)?;
/// let report = engine.evidence_report(&chain);
/// let defenses = engine.defense_recommendations(&chain);
/// ```
pub struct RootCauseEngine {
    tracer: tracer::BackwardTracer,
    evidence: evidence::EvidenceChainManager,
    defense: defense::DefenseInDepthManager,
    config: types::RootCauseConfig,
    /// Optional three-dimensional fusion engine for enriched root cause analysis.
    fused_engine: Option<FusedRootCauseEngine>,
}

impl RootCauseEngine {
    pub fn new(config: types::RootCauseConfig) -> Self {
        Self {
            tracer: tracer::BackwardTracer::new(config.clone()),
            evidence: evidence::EvidenceChainManager::new(config.clone()),
            defense: defense::DefenseInDepthManager::new(config.clone()),
            config,
            fused_engine: None,
        }
    }

    pub fn with_fused_engine(mut self, engine: FusedRootCauseEngine) -> Self {
        self.fused_engine = Some(engine);
        self
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(types::RootCauseConfig::default())
    }

    // ── Core Operations ──

    /// Run a full trace: backward trace → evidence validation → defense recommendations.
    pub fn trace(
        &self,
        error_message: &str,
        source_location: &str,
        context: &types::TraceContext,
    ) -> Result<TracedResult, types::RootCauseError> {
        let trace_id = format!("trace_{}", uuid_or_fallback());
        let agent_id = "agent";

        // Step 1: Backward trace (5 levels)
        let chain = self.tracer.trace_backward(
            error_message,
            source_location,
            context,
            &trace_id,
            agent_id,
        )?;

        // Step 2: Validate evidence chain
        self.evidence.validate_chain(&chain).map_err(|e| {
            types::RootCauseError::InvalidEvidenceChain {
                message: e.to_string(),
                errors: e.errors,
            }
        })?;

        // Step 3: Compute derived values before consuming chain
        let evidence_report = self.evidence.evidence_report(&chain);
        let confidence = self.evidence.chain_confidence(&chain);

        let defenses = if self.config.enable_defense_recommendations {
            self.defense.targeted_recommendations(&chain)
        } else {
            Vec::new()
        };

        // Step 4 (optional): Three-dimensional fusion enrichment
        let fused = self.fused_engine.as_ref().map(|fe| {
            let error_iri = extract_iri_from_error(error_message);
            fe.fuse(&chain, &error_iri)
        });

        // Store for later retrieval
        self.tracer.save_trace(chain.clone());

        Ok(TracedResult {
            chain,
            evidence_report,
            defenses,
            confidence,
            fused_root_cause: fused,
        })
    }

    // ── Sub-Operations ──

    pub fn validate_evidence_chain(
        &self,
        chain: &types::TraceChain,
    ) -> Result<(), types::ChainValidationError> {
        self.evidence.validate_chain(chain)
    }

    pub fn defense_recommendations(
        &self,
        chain: &types::TraceChain,
    ) -> Vec<types::DefenseRecommendation> {
        self.defense.targeted_recommendations(chain)
    }

    pub fn evidence_report(&self, chain: &types::TraceChain) -> String {
        self.evidence.evidence_report(chain)
    }

    pub fn has_unresolved_trace(&self, task_id: &str) -> bool {
        self.tracer.has_unresolved_trace(task_id)
    }

    pub fn get_trace(&self, trace_id: &str) -> Option<types::TraceChain> {
        self.tracer.get_trace(trace_id)
    }

    // ── Hook Integration ──

    /// Register hooks on a HookManager for automatic trace on TaskError.
    ///
    /// Registers:
    /// 1. `TaskError` → auto-trace (priority 90, run before other error handlers)
    /// 2. `PhaseEnd` → check unresolved traces before allowing DO→next phase (priority 50)
    pub fn register_hooks(&self, hook_manager: &HookManager, _agent_id: &str) {
        let engine = Arc::new(self.clone_inner());

        // Hook 1: TaskError → auto traceback (synchronous)
        hook_manager.replace_arc(Arc::new(FunctionHook::new(
            "root_cause_trace",
            vec![HookPoint::TaskError],
            90,
            {
                let engine = engine.clone();
                move |ctx: &mut crate::tools::hooks::HookContext| {
                    if let Some(error) = &ctx.error {
                        let source = ctx
                            .metadata
                            .get("source_location")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let trace_ctx = types::TraceContext {
                            task_type: ctx.agent_role.clone(),
                            failure_description: error.clone(),
                            ..Default::default()
                        };
                        match engine.trace(error, source, &trace_ctx) {
                            Ok(result) => {
                                ctx.data.insert(
                                    "root_cause_trace".to_string(),
                                    serde_json::json!({
                                        "trace_id": result.chain.trace_id,
                                        "resolved": result.chain.resolved,
                                        "confidence": result.confidence,
                                    }),
                                );
                                tracing::info!(
                                    "RootCause trace complete: {} (resolved={}, confidence={:.2})",
                                    result.chain.trace_id,
                                    result.chain.resolved,
                                    result.confidence,
                                );
                            }
                            Err(e) => {
                                tracing::warn!("RootCause trace failed: {}", e);
                                ctx.data.insert(
                                    "root_cause_error".to_string(),
                                    serde_json::json!(e.to_string()),
                                );
                            }
                        }
                    }
                    HookResult::Continue
                }
            },
        )));

        // Hook 2: PhaseEnd → check unresolved traces before allowing phase transition
        hook_manager.replace_arc(Arc::new(FunctionHook::new(
            "root_cause_principle_check",
            vec![HookPoint::PhaseEnd],
            50,
            {
                let engine = engine.clone();
                move |ctx: &mut crate::tools::hooks::HookContext| {
                    let phase = ctx.data.get("phase")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if phase != "DO" {
                        return HookResult::Continue;
                    }
                    if let Some(task_id) = &ctx.task_id {
                        if engine.has_unresolved_trace(task_id) {
                            ctx.error = Some(
                                "Conduct violation: Root cause analysis incomplete before fix. Complete root cause tracing before fixing.".to_string()
                            );
                            tracing::warn!(
                                "RootCause principle violation: unresolved trace for task {}",
                                task_id
                            );
                            return HookResult::Abort;
                        }
                    }
                    HookResult::Continue
                }
            },
        )));
    }

    fn clone_inner(&self) -> Self {
        Self {
            tracer: tracer::BackwardTracer::new(self.config.clone()),
            evidence: evidence::EvidenceChainManager::new(self.config.clone()),
            defense: defense::DefenseInDepthManager::new(self.config.clone()),
            config: self.config.clone(),
            fused_engine: self.fused_engine.clone(),
        }
    }
}

/// Complete output of a trace operation
#[derive(Debug, Clone)]
pub struct TracedResult {
    pub chain: types::TraceChain,
    pub evidence_report: String,
    pub defenses: Vec<types::DefenseRecommendation>,
    pub confidence: f64,
    /// Optional three-dimensional fused root cause analysis.
    pub fused_root_cause: Option<FusedRootCause>,
}

/// Extract an IRI from an error message for three-dimensional fusion.
/// Falls back to a placeholder IRI if no recognizable IRI is found.
fn extract_iri_from_error(error: &str) -> String {
    // Simple heuristic: find the first `iri:` or `http` substring
    for token in error.split_whitespace() {
        let cleaned = token.trim_matches(|c: char| c.is_ascii_punctuation());
        if cleaned.starts_with("iri:")
            || cleaned.starts_with("http://")
            || cleaned.starts_with("https://")
        {
            return cleaned.to_string();
        }
    }
    format!("error:{}", uuid_or_fallback())
}

fn uuid_or_fallback() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("rc{:x}", nanos)
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_full_trace() {
        let engine = RootCauseEngine::default();
        let context = types::TraceContext {
            task_type: "http_request".to_string(),
            failure_description: "Failed to fetch data".to_string(),
            ..Default::default()
        };
        let result = engine.trace(
            "connection refused: error connecting to server",
            "src/http/client.rs:42",
            &context,
        );
        assert!(
            result.is_ok(),
            "Full trace should succeed: {:?}",
            result.err()
        );
        let traced = result.unwrap();
        assert!(!traced.chain.levels.is_empty(), "Should have trace levels");
        assert!(
            !traced.defenses.is_empty(),
            "Should have defense recommendations"
        );
        assert!(traced.confidence > 0.0, "Should have confidence");
    }

    #[test]
    fn test_engine_no_root_cause() {
        let engine = RootCauseEngine::new(types::RootCauseConfig {
            min_confidence: 0.99, // impossible
            ..Default::default()
        });
        let context = types::TraceContext::default();
        let result = engine.trace("something unknown happened", "src/main.rs:1", &context);
        assert!(result.is_err(), "Should fail when confidence too low");
    }

    #[test]
    fn test_engine_has_unresolved() {
        let engine = RootCauseEngine::default();
        // Before any trace, no task should be unresolved
        assert!(!engine.has_unresolved_trace("nonexistent_task"));
    }

    #[test]
    fn test_traced_result_structure() {
        let engine = RootCauseEngine::default();
        let context = types::TraceContext::default();
        let result = engine
            .trace("permission denied", "src/auth.rs:10", &context)
            .unwrap();
        assert!(result.evidence_report.contains("Evidence Chain Report"));
        assert!(
            result.chain.trace_id.starts_with("trace_") || result.chain.trace_id.starts_with("rc")
        );
    }
}
