/// DefenseInDepthManager — generates defense recommendations from trace analysis.
///
/// Maps the defense-in-depth methodology to concrete recommendations:
/// Layer 1: EntryValidation — Prevent errors at entry point
/// Layer 2: BusinessLogic — Validate business logic assumptions
/// Layer 3: EnvironmentGuard — Guard against environmental issues
/// Layer 4: Instrumentation — Instrument for observability
///
/// Corresponds to: superpowers-main/skills/systematic-debugging/defense-in-depth.md
/// Architecture Layer: L1 — Enforcement (RootCauseEngine)
use super::types::{DefenseLayer, DefenseRecommendation, RootCauseConfig, TraceChain};

/// DefenseInDepthManager converts root cause analysis into actionable defense recommendations.
pub struct DefenseInDepthManager {
    #[allow(dead_code)]
    config: RootCauseConfig,
}

impl DefenseInDepthManager {
    pub fn new(config: RootCauseConfig) -> Self {
        Self { config }
    }

    /// Generate defense recommendations from a trace chain's root cause.
    pub fn generate_recommendations(&self, chain: &TraceChain) -> Vec<DefenseRecommendation> {
        let root = match chain.root_level() {
            Some(r) => r,
            None => return vec![],
        };

        let mut recommendations = Vec::new();

        // Layer 1: Entry validation — validate inputs/preconditions at entry point
        recommendations.push(DefenseRecommendation {
            layer: DefenseLayer::EntryValidation,
            title: "Entry Parameter Validation".to_string(),
            description: format!(
                "Add input parameter validation and precondition checks at root cause location '{}', ensuring parameters are within expected range",
                root.source_location
            ),
            priority: 1,
        });

        // Layer 2: Business logic — add defensive checks in business logic
        recommendations.push(DefenseRecommendation {
            layer: DefenseLayer::BusinessLogic,
            title: "Business Logic Defense".to_string(),
            description: format!(
                "Add business logic layer defensive checks around '{}', including null protection, boundary checks, and state validation",
                root.description.chars().take(50).collect::<String>()
            ),
            priority: 2,
        });

        // Layer 3: Environment guard — add guard clauses for environmental issues
        recommendations.push(DefenseRecommendation {
            layer: DefenseLayer::EnvironmentGuard,
            title: "Environment Anomaly Protection".to_string(),
            description: "Add environment check guards (disk space, network connectivity, service health checks, etc.), verify environment readiness before operations".to_string(),
            priority: 3,
        });

        // Layer 4: Instrumentation — add logging and monitoring
        recommendations.push(DefenseRecommendation {
            layer: DefenseLayer::Instrumentation,
            title: "Critical Path Monitoring".to_string(),
            description: format!(
                "Add tracing logs and performance monitoring on root cause path '{}', enabling fast identification of similar issues in the future",
                root.source_location
            ),
            priority: 4,
        });

        recommendations
    }

    /// Generate targeted recommendations based on the specific root cause label
    pub fn targeted_recommendations(&self, chain: &TraceChain) -> Vec<DefenseRecommendation> {
        let root = match chain.root_level() {
            Some(r) => r,
            None => return vec![],
        };

        let label = root
            .evidence
            .value
            .get("root_cause_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match label {
            "network_error" => vec![
                DefenseRecommendation {
                    layer: DefenseLayer::EntryValidation,
                    title: "Connection Timeout Retry".into(),
                    description: "Add exponential backoff retry mechanism".into(),
                    priority: 1,
                },
                DefenseRecommendation {
                    layer: DefenseLayer::Instrumentation,
                    title: "Network Probing".into(),
                    description: "Add pre-connection network health check".into(),
                    priority: 2,
                },
            ],
            "resource_not_found" => vec![
                DefenseRecommendation {
                    layer: DefenseLayer::EntryValidation,
                    title: "Path Validation".into(),
                    description: "Verify path/URL exists before accessing resource".into(),
                    priority: 1,
                },
                DefenseRecommendation {
                    layer: DefenseLayer::BusinessLogic,
                    title: "Degradation Handling".into(),
                    description: "Provide degradation alternatives when resource does not exist"
                        .into(),
                    priority: 2,
                },
            ],
            "permission_error" => vec![
                DefenseRecommendation {
                    layer: DefenseLayer::BusinessLogic,
                    title: "Permission Pre-check".into(),
                    description: "Check current permissions meet requirements before operation"
                        .into(),
                    priority: 1,
                },
                DefenseRecommendation {
                    layer: DefenseLayer::EnvironmentGuard,
                    title: "Escalation Advice".into(),
                    description:
                        "Provide clear escalation guidance when permissions are insufficient".into(),
                    priority: 2,
                },
            ],
            "null_reference" => vec![
                DefenseRecommendation {
                    layer: DefenseLayer::EntryValidation,
                    title: "Null Protection".into(),
                    description: "Check for null/None before dereferencing".into(),
                    priority: 1,
                },
                DefenseRecommendation {
                    layer: DefenseLayer::BusinessLogic,
                    title: "Default Values".into(),
                    description: "Provide safe defaults for potentially null fields".into(),
                    priority: 2,
                },
            ],
            "resource_exhausted" => vec![
                DefenseRecommendation {
                    layer: DefenseLayer::EnvironmentGuard,
                    title: "Resource Monitoring".into(),
                    description:
                        "Check resource usage before operations, alert when exceeding thresholds"
                            .into(),
                    priority: 1,
                },
                DefenseRecommendation {
                    layer: DefenseLayer::BusinessLogic,
                    title: "Rate Limiting & Circuit Breaker".into(),
                    description:
                        "Add rate limiting and circuit breaker to prevent cascading failures".into(),
                    priority: 2,
                },
            ],
            "invalid_input" => vec![
                DefenseRecommendation {
                    layer: DefenseLayer::EntryValidation,
                    title: "Parameter Validation".into(),
                    description: "Perform complete parameter validation at entry point".into(),
                    priority: 1,
                },
                DefenseRecommendation {
                    layer: DefenseLayer::Instrumentation,
                    title: "Parameter Logging".into(),
                    description: "Log key parameter values for debugging".into(),
                    priority: 2,
                },
            ],
            "syntax_error" => vec![
                DefenseRecommendation {
                    layer: DefenseLayer::EntryValidation,
                    title: "Format Validation".into(),
                    description: "Validate input format before parsing".into(),
                    priority: 1,
                },
                DefenseRecommendation {
                    layer: DefenseLayer::BusinessLogic,
                    title: "Format Specification".into(),
                    description: "Provide clear error when input format does not meet expectations"
                        .into(),
                    priority: 2,
                },
            ],
            _ => self.generate_recommendations(chain),
        }
    }

    /// Prioritize recommendations based on chain confidence
    pub fn prioritize(&self, recommendations: &mut [DefenseRecommendation], chain: &TraceChain) {
        let confidence = chain
            .root_level()
            .map(|r| r.evidence.confidence)
            .unwrap_or(0.5);
        for rec in recommendations.iter_mut() {
            // Lower priority number = more urgent
            if confidence < 0.6 {
                rec.priority = rec.priority.saturating_sub(1);
            }
        }
        recommendations.sort_by_key(|r| r.priority);
    }

    /// Summary of all recommendations
    pub fn summary(&self, recommendations: &[DefenseRecommendation]) -> String {
        if recommendations.is_empty() {
            return "No defense recommendations".to_string();
        }
        let mut s = String::from("Defense-in-Depth Recommendations:\n");
        for rec in recommendations {
            s.push_str(&format!(
                "  [{}.{}] {}: {}\n",
                rec.layer.name(),
                rec.priority,
                rec.title,
                rec.description
            ));
        }
        s
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_chain_with_root(root_label: &str) -> TraceChain {
        let mut chain = TraceChain::new("defense_test", "agent_1");
        chain.add_level(super::super::types::TraceLevel {
            level: 1,
            label: "symptom".into(),
            description: "error".into(),
            source_location: "src/main.rs:1".into(),
            is_root_cause: false,
            evidence: super::super::types::Evidence::new("src/main.rs:1", json!("error"), 0.9),
        });
        chain.add_level(super::super::types::TraceLevel {
            level: 5,
            label: "root_cause".into(),
            description: "network error".into(),
            source_location: "src/net.rs:50".into(),
            is_root_cause: true,
            evidence: super::super::types::Evidence::new(
                "src/net.rs:50",
                json!({"root_cause_label": root_label, "error": "test"}),
                0.85,
            ),
        });
        chain
    }

    #[test]
    fn test_generates_4_layers() {
        let manager = DefenseInDepthManager::new(RootCauseConfig::default());
        let chain = make_chain_with_root("unknown");
        let recs = manager.generate_recommendations(&chain);
        assert_eq!(recs.len(), 4, "Should generate 4 defense layers");
        // Verify all 4 layers present
        let layers: Vec<_> = recs.iter().map(|r| r.layer).collect();
        assert!(layers.contains(&DefenseLayer::EntryValidation));
        assert!(layers.contains(&DefenseLayer::BusinessLogic));
        assert!(layers.contains(&DefenseLayer::EnvironmentGuard));
        assert!(layers.contains(&DefenseLayer::Instrumentation));
    }

    #[test]
    fn test_targeted_network_error() {
        let manager = DefenseInDepthManager::new(RootCauseConfig::default());
        let chain = make_chain_with_root("network_error");
        let recs = manager.targeted_recommendations(&chain);
        assert_eq!(recs.len(), 2);
        assert!(recs[0].title.contains("Retry"));
    }

    #[test]
    fn test_targeted_null_reference() {
        let manager = DefenseInDepthManager::new(RootCauseConfig::default());
        let chain = make_chain_with_root("null_reference");
        let recs = manager.targeted_recommendations(&chain);
        assert_eq!(recs.len(), 2);
        assert!(recs[0].title.contains("Null Protection"));
    }

    #[test]
    fn test_empty_chain_returns_empty() {
        let manager = DefenseInDepthManager::new(RootCauseConfig::default());
        let chain = TraceChain::new("empty", "agent_1");
        let recs = manager.generate_recommendations(&chain);
        assert!(recs.is_empty());
    }

    #[test]
    fn test_prioritize_sorts_by_priority() {
        let manager = DefenseInDepthManager::new(RootCauseConfig::default());
        let chain = make_chain_with_root("unknown");
        let mut recs = manager.generate_recommendations(&chain);
        manager.prioritize(&mut recs, &chain);
        for i in 1..recs.len() {
            assert!(
                recs[i - 1].priority <= recs[i].priority,
                "Priorities should be sorted ascending"
            );
        }
    }

    #[test]
    fn test_summary_format() {
        let manager = DefenseInDepthManager::new(RootCauseConfig::default());
        let chain = make_chain_with_root("network_error");
        let recs = manager.generate_recommendations(&chain);
        let summary = manager.summary(&recs);
        assert!(summary.contains("Defense-in-Depth"));
        assert!(summary.contains("Entry"));
    }
}
