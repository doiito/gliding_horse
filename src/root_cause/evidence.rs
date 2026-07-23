use super::types::{ChainValidationError, RootCauseConfig, TraceChain, TraceLevel};

/// EvidenceChainManager validates the continuity and integrity of trace evidence chains.
pub struct EvidenceChainManager {
    config: RootCauseConfig,
}

impl EvidenceChainManager {
    pub fn new(config: RootCauseConfig) -> Self {
        Self { config }
    }

    /// Validate an entire trace chain's evidence integrity.
    /// Returns Ok(()) if all checks pass, or Err with detailed failures.
    pub fn validate_chain(&self, chain: &TraceChain) -> Result<(), ChainValidationError> {
        let mut errors = Vec::new();

        // 1. Each level must have evidence with a source
        for level in &chain.levels {
            self.validate_level(level, &mut errors);
        }

        // 2. Evidence must chain continuously between adjacent levels
        for window in chain.levels.windows(2) {
            self.validate_continuity(&window[0], &window[1], &mut errors);
        }

        // 3. Chain must have a root cause
        if !chain.has_root_cause() {
            errors.push("Root cause not reached: traceback terminated prematurely".to_string());
        }

        // 4. Minimum trace depth
        if chain.depth() < 2 {
            errors.push(format!(
                "Evidence chain depth insufficient: {} levels (minimum 2)",
                chain.depth()
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ChainValidationError { errors })
        }
    }

    /// Validate a single level's evidence
    fn validate_level(&self, level: &TraceLevel, errors: &mut Vec<String>) {
        if level.evidence.source.is_empty() {
            errors.push(format!("Level {}: missing evidence source", level.level));
        }
        if level.evidence.confidence < 0.0 || level.evidence.confidence > 1.0 {
            errors.push(format!(
                "Level {}: confidence out of range ({})",
                level.level, level.evidence.confidence
            ));
        }
        if level.evidence.confidence < self.config.min_confidence {
            errors.push(format!(
                "Level {}: confidence too low ({:.2} < {:.2})",
                level.level, level.evidence.confidence, self.config.min_confidence
            ));
        }
        if level.description.is_empty() {
            errors.push(format!("Level {}: level description is empty", level.level));
        }
    }

    /// Validate that two adjacent levels have continuous evidence
    fn validate_continuity(&self, a: &TraceLevel, b: &TraceLevel, errors: &mut Vec<String>) {
        if b.level != a.level + 1 {
            errors.push(format!(
                "Level {} → {}: level numbers are not sequential",
                a.level, b.level
            ));
        }
        // Evidence source duplication check: only flag if source AND description are identical
        if a.evidence.source == b.evidence.source
            && a.description == b.description
            && !a.evidence.source.is_empty()
        {
            errors.push(format!(
                "Level {} → {}: evidence source and description are fully duplicated ({})",
                a.level, b.level, a.evidence.source
            ));
        }
    }

    /// Compute aggregate confidence across the entire chain
    pub fn chain_confidence(&self, chain: &TraceChain) -> f64 {
        if chain.levels.is_empty() {
            return 0.0;
        }
        // Geometric mean of all confidence values
        let product: f64 = chain.levels.iter().map(|l| l.evidence.confidence).product();
        product.powf(1.0 / chain.levels.len() as f64)
    }

    /// Find the weakest link (lowest confidence level)
    pub fn weakest_evidence<'a>(&self, chain: &'a TraceChain) -> Option<&'a TraceLevel> {
        chain.levels.iter().min_by(|a, b| {
            a.evidence
                .confidence
                .partial_cmp(&b.evidence.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Generate a human-readable evidence report
    pub fn evidence_report(&self, chain: &TraceChain) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "===== Evidence Chain Report [{}] =====\n",
            chain.trace_id
        ));
        report.push_str(&format!(
            "Agent: {} | Task: {}\n\n",
            chain.agent_id,
            chain.task_id.as_deref().unwrap_or("N/A"),
        ));

        for level in &chain.levels {
            let flag = if level.is_root_cause {
                " [Root Cause]"
            } else {
                ""
            };
            report.push_str(&format!("  L{} {}{}\n", level.level, level.label, flag));
            report.push_str(&format!("    Description: {}\n", level.description));
            report.push_str(&format!("    Source: {}\n", level.evidence.source));
            report.push_str(&format!(
                "    Confidence: {:.2}\n",
                level.evidence.confidence
            ));
            report.push('\n');
        }

        report.push_str(&format!(
            "Overall Confidence: {:.2}\n",
            self.chain_confidence(chain)
        ));
        report.push_str(&format!(
            "Status: {}",
            if chain.resolved {
                "Root Cause Identified"
            } else {
                "Root Cause Not Identified"
            }
        ));
        report
    }

    /// Check that each level has at least the minimum confidence
    pub fn all_levels_confident(&self, chain: &TraceChain) -> bool {
        chain
            .levels
            .iter()
            .all(|l| l.evidence.confidence >= self.config.min_confidence)
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::root_cause::Evidence;
    use serde_json::json;

    fn make_chain() -> TraceChain {
        let mut chain = TraceChain::new("test_chain", "agent_1");
        chain.add_level(TraceLevel {
            level: 1,
            label: "symptom".into(),
            description: "error occurred".into(),
            source_location: "file.rs:10".into(),
            is_root_cause: false,
            evidence: Evidence::new("file.rs:10", json!("error"), 0.9),
        });
        chain.add_level(TraceLevel {
            level: 2,
            label: "intermediate".into(),
            description: "caller".into(),
            source_location: "caller.rs:20".into(),
            is_root_cause: false,
            evidence: Evidence::new("caller.rs:20", json!("caller"), 0.8),
        });
        chain.add_level(TraceLevel {
            level: 3,
            label: "root_cause".into(),
            description: "root cause".into(),
            source_location: "root.rs:1".into(),
            is_root_cause: true,
            evidence: Evidence::new("root.rs:1", json!("root"), 0.85),
        });
        chain
    }

    #[test]
    fn test_valid_chain_passes() {
        let manager = EvidenceChainManager::new(RootCauseConfig::default());
        let chain = make_chain();
        assert!(manager.validate_chain(&chain).is_ok());
    }

    #[test]
    fn test_missing_root_cause_fails() {
        let manager = EvidenceChainManager::new(RootCauseConfig::default());
        let mut chain = make_chain();
        chain.levels.last_mut().unwrap().is_root_cause = false;
        assert!(manager.validate_chain(&chain).is_err());
    }

    #[test]
    fn test_empty_source_fails() {
        let manager = EvidenceChainManager::new(RootCauseConfig::default());
        let mut chain = make_chain();
        chain.levels[0].evidence.source.clear();
        assert!(manager.validate_chain(&chain).is_err());
    }

    #[test]
    fn test_discontinuous_levels_fails() {
        let manager = EvidenceChainManager::new(RootCauseConfig::default());
        let mut chain = make_chain();
        chain.levels[1].level = 5; // skip
        assert!(manager.validate_chain(&chain).is_err());
    }

    #[test]
    fn test_chain_confidence() {
        let manager = EvidenceChainManager::new(RootCauseConfig::default());
        let chain = make_chain();
        let conf = manager.chain_confidence(&chain);
        assert!((conf - 0.85).abs() < 0.1, "Expected ~0.85, got {}", conf);
    }

    #[test]
    fn test_chain_too_shallow_fails() {
        let manager = EvidenceChainManager::new(RootCauseConfig::default());
        let mut chain = TraceChain::new("shallow", "agent_1");
        chain.add_level(TraceLevel {
            level: 1,
            label: "symptom".into(),
            description: "only level".into(),
            source_location: "x.rs:1".into(),
            is_root_cause: true,
            evidence: Evidence::new("x.rs:1", json!("x"), 0.9),
        });
        assert!(manager.validate_chain(&chain).is_err());
    }

    #[test]
    fn test_evidence_report() {
        let manager = EvidenceChainManager::new(RootCauseConfig::default());
        let chain = make_chain();
        let report = manager.evidence_report(&chain);
        assert!(report.contains("Evidence Chain Report"));
        assert!(report.contains("test_chain"));
        assert!(report.contains("Root Cause"));
    }

    #[test]
    fn test_weakest_evidence() {
        let manager = EvidenceChainManager::new(RootCauseConfig::default());
        let chain = make_chain();
        let weakest = manager.weakest_evidence(&chain).unwrap();
        assert_eq!(weakest.level, 2); // level 2 has confidence 0.8, which is lowest
    }

    #[test]
    fn test_duplicate_source_and_description_fails() {
        let manager = EvidenceChainManager::new(RootCauseConfig::default());
        let mut chain = make_chain();
        chain.levels[1].evidence.source = "file.rs:10".to_string(); // same as L1
        chain.levels[1].description = chain.levels[0].description.clone(); // same as L1
        let result = manager.validate_chain(&chain);
        assert!(result.is_err());
        let error_msg = format!("{:?}", result);
        assert!(
            error_msg.contains("fully duplicated"),
            "Should detect fully duplicate level"
        );
    }

    #[test]
    fn test_same_source_different_desc_ok() {
        let manager = EvidenceChainManager::new(RootCauseConfig::default());
        let mut chain = make_chain();
        chain.levels[1].evidence.source = "file.rs:10".to_string(); // same as L1 but diff desc
                                                                    // descriptions differ (L1="error occurred", L2="caller")
        assert!(
            manager.validate_chain(&chain).is_ok(),
            "Same source with different descriptions is valid"
        );
    }
}
