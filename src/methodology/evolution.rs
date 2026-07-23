/// EvolutionEngine (Phase 4) — Self-evolution through violation learning and effectiveness metrics.
///
/// Records methodology violations as they occur, identifies frequent patterns,
/// and tracks per-methodology effectiveness. This creates a feedback loop that
/// enables the system to improve its methodology definitions over time.
///
/// ## Data Flow
///
/// ```text
/// MethodologyGate ──► EvolutionEngine ──► LearnedPatterns
///      │                    │                    │
///      │  anti-pattern      │  aggregate by      │  feed back to
///      │  blocks/warnings   │  pattern + role     │  constitution
///      ▼                    ▼                    ▼
///   ViolationRecord    PatternLearner       MethodologyMetrics
/// ```
///
/// Architecture Layer: L4 — Self-Evolution (Iterative)
/// See design: PR-res/superpowers-skills-full-integration-design.md §4
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::methodology::RedFlagSeverity;

// ════════════════════════════════════════════════════════════════════════
// Core Types
// ════════════════════════════════════════════════════════════════════════

/// What kind of methodology pattern was violated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternType {
    RedFlag,
    AntiPattern,
}

impl PatternType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RedFlag => "red_flag",
            Self::AntiPattern => "anti_pattern",
        }
    }
}

/// A single violation event recorded during execution
#[derive(Debug, Clone)]
pub struct ViolationRecord {
    /// Which methodology was violated
    pub methodology_id: String,
    /// Type of pattern
    pub pattern_type: PatternType,
    /// Pattern name (from RedFlagEntry.pattern or AntiPatternEntry.name)
    pub pattern_name: String,
    /// Severity of the violation
    pub severity: RedFlagSeverity,
    /// Which agent role triggered it
    pub agent_role: String,
    /// Tool being used when violation occurred
    pub tool_name: Option<String>,
    /// Whether execution was blocked
    pub blocked: bool,
    /// When it happened (unix seconds)
    pub timestamp: u64,
    /// Associated task
    pub task_id: Option<String>,
    /// Any additional context
    pub description: String,
}

/// A pattern learned from aggregating similar violations
#[derive(Debug, Clone)]
pub struct LearnedPattern {
    /// Human-readable description of the pattern
    pub pattern_description: String,
    /// Which methodology it relates to
    pub methodology_id: String,
    /// Which pattern type
    pub pattern_type: PatternType,
    /// How many times this has been observed
    pub frequency: u64,
    /// First observation (unix seconds)
    pub first_seen: u64,
    /// Most recent observation (unix seconds)
    pub last_seen: u64,
    /// Typical severity
    pub severity: RedFlagSeverity,
    /// Which roles most commonly trigger this
    pub frequent_roles: Vec<String>,
    /// Whether this pattern is actively being used for enforcement
    pub active: bool,
}

/// Per-methodology effectiveness metrics
#[derive(Debug, Clone)]
pub struct MethodologyMetrics {
    /// Methodology ID
    pub methodology_id: String,
    /// How many times this methodology was activated
    pub activation_count: u64,
    /// How many violations were detected
    pub total_violations: u64,
    /// How many blocks occurred
    pub block_count: u64,
    /// How many non-blocking warnings
    pub warning_count: u64,
    /// How many tool calls passed without any methodology violation
    pub pass_count: u64,
    /// When it was last activated
    pub last_activated: Option<u64>,
    /// Derived effectiveness score (0.0 = ineffective, 1.0 = perfect)
    pub effectiveness_score: f64,
}

/// Summary report for AA/SA consumption
#[derive(Debug, Clone)]
pub struct EvolutionReport {
    /// Total violations recorded
    pub total_violations: u64,
    /// Top violation patterns
    pub top_patterns: Vec<LearnedPattern>,
    /// Methodology metrics
    pub methodology_metrics: Vec<MethodologyMetrics>,
    /// Overall system health score
    pub health_score: f64,
    /// Number of active methodologies
    pub active_methodologies: usize,
}

// ════════════════════════════════════════════════════════════════════════
// EvolutionEngine
// ════════════════════════════════════════════════════════════════════════

/// Records methodology violations, learns patterns, and tracks effectiveness.
pub struct EvolutionEngine {
    /// All recorded violations (ring buffer via VecDeque)
    violations: VecDeque<ViolationRecord>,
    /// Per-methodology metrics
    metrics: HashMap<String, MethodologyMetrics>,
    /// Maximum number of violations to retain (ring buffer)
    max_records: usize,
}

impl EvolutionEngine {
    /// Create a new engine with default capacity (10,000 records).
    pub fn new() -> Self {
        Self {
            violations: VecDeque::new(),
            metrics: HashMap::new(),
            max_records: 10_000,
        }
    }

    /// Create with a specific maximum record capacity.
    pub fn with_max_records(max: usize) -> Self {
        Self {
            violations: VecDeque::new(),
            metrics: HashMap::new(),
            max_records: max,
        }
    }

    // ─── Recording ───

    /// Record a methodology violation.
    ///
    /// Updates both the violation log and the per-methodology metrics.
    pub fn record_violation(&mut self, record: ViolationRecord) {
        let is_block = record.blocked;
        let sev = record.severity;
        let mid = record.methodology_id.clone();
        let mid_for_metrics = mid.clone();

        if self.violations.len() >= self.max_records {
            self.violations.pop_front();
        }
        self.violations.push_back(record);

        let metrics = self
            .metrics
            .entry(mid)
            .or_insert_with(|| MethodologyMetrics {
                methodology_id: mid_for_metrics,
                activation_count: 0,
                total_violations: 0,
                block_count: 0,
                warning_count: 0,
                pass_count: 0,
                last_activated: None,
                effectiveness_score: 1.0,
            });
        metrics.total_violations += 1;
        if is_block {
            metrics.block_count += 1;
        } else {
            metrics.warning_count += 1;
        }
        metrics.effectiveness_score = compute_effectiveness(
            metrics.activation_count,
            metrics.total_violations,
            metrics.pass_count,
        );

        match sev {
            RedFlagSeverity::Critical => {
                metrics.effectiveness_score *= 0.8;
            }
            RedFlagSeverity::Warning => {
                metrics.effectiveness_score *= 0.9;
            }
            RedFlagSeverity::Info => {
                metrics.effectiveness_score *= 0.95;
            }
        }
    }

    /// Record a successful tool call (no violation).
    pub fn record_pass(&mut self, methodology_id: &str) {
        let metrics = self
            .metrics
            .entry(methodology_id.to_string())
            .or_insert_with(|| MethodologyMetrics {
                methodology_id: methodology_id.to_string(),
                activation_count: 0,
                total_violations: 0,
                block_count: 0,
                warning_count: 0,
                pass_count: 0,
                last_activated: None,
                effectiveness_score: 1.0,
            });
        metrics.pass_count += 1;
        metrics.effectiveness_score = compute_effectiveness(
            metrics.activation_count,
            metrics.total_violations,
            metrics.pass_count,
        );
    }

    /// Record a methodology activation.
    pub fn record_activation(&mut self, methodology_id: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let metrics = self
            .metrics
            .entry(methodology_id.to_string())
            .or_insert_with(|| MethodologyMetrics {
                methodology_id: methodology_id.to_string(),
                activation_count: 0,
                total_violations: 0,
                block_count: 0,
                warning_count: 0,
                pass_count: 0,
                last_activated: None,
                effectiveness_score: 1.0,
            });
        metrics.activation_count += 1;
        metrics.last_activated = Some(now);
        metrics.effectiveness_score = compute_effectiveness(
            metrics.activation_count,
            metrics.total_violations,
            metrics.pass_count,
        );
    }

    // ─── Pattern Learning ───

    /// Identify the most frequent violation patterns, grouped by methodology + pattern name.
    pub fn learn_patterns(&self) -> Vec<LearnedPattern> {
        let mut groups: HashMap<(String, String, PatternType), Vec<&ViolationRecord>> =
            HashMap::new();

        for v in &self.violations {
            let key = (
                v.methodology_id.clone(),
                v.pattern_name.clone(),
                v.pattern_type,
            );
            groups.entry(key).or_default().push(v);
        }

        let mut patterns: Vec<LearnedPattern> = groups
            .into_iter()
            .map(|((mid, pname, ptype), records)| {
                let freq = records.len() as u64;
                let first = records.iter().map(|r| r.timestamp).min().unwrap_or(0);
                let last = records.iter().map(|r| r.timestamp).max().unwrap_or(0);
                let severity = records
                    .iter()
                    .map(|r| r.severity)
                    .max_by_key(|s| match s {
                        RedFlagSeverity::Critical => 3,
                        RedFlagSeverity::Warning => 2,
                        RedFlagSeverity::Info => 1,
                    })
                    .unwrap_or(RedFlagSeverity::Info);

                let mut role_counts: HashMap<&str, usize> = HashMap::new();
                for r in &records {
                    *role_counts.entry(&r.agent_role).or_insert(0) += 1;
                }
                let mut frequent_role_pairs: Vec<(usize, String)> = role_counts
                    .into_iter()
                    .map(|(r, c)| (c, r.to_string()))
                    .collect::<Vec<_>>();
                frequent_role_pairs.sort_by(|a, b| b.0.cmp(&a.0));

                LearnedPattern {
                    pattern_description: format!("{} — {} ({} occurrences)", mid, pname, freq),
                    methodology_id: mid,
                    pattern_type: ptype,
                    frequency: freq,
                    first_seen: first,
                    last_seen: last,
                    severity,
                    frequent_roles: frequent_role_pairs.into_iter().map(|(_, r)| r).collect(),
                    active: true,
                }
            })
            .collect();

        patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        patterns
    }

    // ─── Metrics Queries ───

    /// Get metrics for a specific methodology.
    pub fn get_metrics(&self, methodology_id: &str) -> Option<&MethodologyMetrics> {
        self.metrics.get(methodology_id)
    }

    /// Get all methodology metrics.
    pub fn all_metrics(&self) -> Vec<&MethodologyMetrics> {
        self.metrics.values().collect()
    }

    /// Get the top N violation patterns by frequency.
    pub fn top_patterns(&self, n: usize) -> Vec<LearnedPattern> {
        let mut patterns = self.learn_patterns();
        patterns.truncate(n);
        patterns
    }

    /// Get all violations recorded.
    pub fn all_violations(&self) -> impl Iterator<Item = &ViolationRecord> {
        self.violations.iter()
    }

    /// Count of recorded violations.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Generate a summary report for AA/SA consumption.
    pub fn generate_report(&self) -> EvolutionReport {
        let patterns = self.learn_patterns();
        let top: Vec<LearnedPattern> = patterns.into_iter().take(10).collect();
        let metrics: Vec<MethodologyMetrics> = self.metrics.values().cloned().collect();
        let total_violations = self.violations.len() as u64;

        let health_score = if metrics.is_empty() {
            1.0
        } else {
            let sum: f64 = metrics.iter().map(|m| m.effectiveness_score).sum();
            sum / metrics.len() as f64
        };

        EvolutionReport {
            total_violations,
            top_patterns: top,
            methodology_metrics: metrics,
            health_score,
            active_methodologies: self.metrics.len(),
        }
    }

    /// Generate a prompt section for AA, summarizing current methodology evolution state.
    pub fn aa_evolution_briefing(&self) -> String {
        let report = self.generate_report();

        let mut sections = vec![
            "\n## 📊 Methodology Evolution Report".to_string(),
            format!("System Health Score: {:.1}%", report.health_score * 100.0),
            format!("Total Violations Recorded: {}", report.total_violations),
            format!("Active Methodologies: {}", report.active_methodologies),
        ];

        if !report.top_patterns.is_empty() {
            sections.push("\n### High-Frequency Violation Patterns".to_string());
            for (i, p) in report.top_patterns.iter().enumerate().take(5) {
                let sev = match p.severity {
                    RedFlagSeverity::Critical => "🔴",
                    RedFlagSeverity::Warning => "🟡",
                    RedFlagSeverity::Info => "🔵",
                };
                sections.push(format!(
                    "{}. {} {} (frequency: {}, roles: {})",
                    i + 1,
                    sev,
                    p.pattern_description,
                    p.frequency,
                    p.frequent_roles.join(", "),
                ));
            }
        }

        if !report.methodology_metrics.is_empty() {
            sections.push("\n### Methodology Effectiveness".to_string());
            for m in report.methodology_metrics.iter().take(5) {
                let status = if m.effectiveness_score > 0.8 {
                    "✅"
                } else if m.effectiveness_score > 0.5 {
                    "⚠️"
                } else {
                    "🔴"
                };
                sections.push(format!(
                    "{} {} — effectiveness: {:.1}% (activated {} times / violations {} / passes {})",
                    status,
                    m.methodology_id,
                    m.effectiveness_score * 100.0,
                    m.activation_count,
                    m.total_violations,
                    m.pass_count,
                ));
            }
        }

        sections.join("\n")
    }
}

impl Default for EvolutionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe handle for EvolutionEngine
pub struct EvolutionEngineHandle {
    inner: std::sync::Arc<parking_lot::RwLock<EvolutionEngine>>,
}

impl EvolutionEngineHandle {
    pub fn new(engine: EvolutionEngine) -> Self {
        Self {
            inner: std::sync::Arc::new(parking_lot::RwLock::new(engine)),
        }
    }

    pub fn inner(&self) -> std::sync::Arc<parking_lot::RwLock<EvolutionEngine>> {
        self.inner.clone()
    }
}

impl Clone for EvolutionEngineHandle {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════

fn compute_effectiveness(activations: u64, violations: u64, passes: u64) -> f64 {
    let total = activations + violations + passes;
    if total == 0 {
        return 1.0;
    }
    let score = (passes as f64 + 1.0) / (total as f64 + 1.0);
    score.clamp(0.0, 1.0)
}

// ════════════════════════════════════════════════════════════════════════
// MethodologyGate integration: ViolationReporter
// ════════════════════════════════════════════════════════════════════════

use crate::methodology::gate::AntiPatternGateResult;
use crate::methodology::RedFlagEntry;

/// Helper to convert MethodologyGate results into ViolationRecords.
pub struct ViolationReporter;

impl ViolationReporter {
    /// Convert an anti-pattern gate result to a violation record.
    pub fn from_anti_pattern(
        result: &AntiPatternGateResult,
        agent_role: &str,
        task_id: Option<String>,
        timestamp: u64,
    ) -> ViolationRecord {
        ViolationRecord {
            methodology_id: result.methodology_id.clone(),
            pattern_type: PatternType::AntiPattern,
            pattern_name: result.anti_pattern_name.clone(),
            severity: if result.should_block {
                RedFlagSeverity::Critical
            } else {
                RedFlagSeverity::Warning
            },
            agent_role: agent_role.to_string(),
            tool_name: None,
            blocked: result.should_block,
            timestamp,
            task_id,
            description: result.description.clone(),
        }
    }

    /// Convert a red flag entry to a violation record.
    pub fn from_red_flag(
        methodology_id: &str,
        flag: &RedFlagEntry,
        agent_role: &str,
        tool_name: Option<String>,
        task_id: Option<String>,
        timestamp: u64,
    ) -> ViolationRecord {
        ViolationRecord {
            methodology_id: methodology_id.to_string(),
            pattern_type: PatternType::RedFlag,
            pattern_name: flag.pattern.to_string(),
            severity: flag.severity,
            agent_role: agent_role.to_string(),
            tool_name,
            blocked: matches!(flag.severity, RedFlagSeverity::Critical),
            timestamp,
            task_id,
            description: flag.pattern.to_string(),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_violation(
        mid: &str,
        pattern: &str,
        sev: RedFlagSeverity,
        role: &str,
        blocked: bool,
    ) -> ViolationRecord {
        ViolationRecord {
            methodology_id: mid.to_string(),
            pattern_type: PatternType::AntiPattern,
            pattern_name: pattern.to_string(),
            severity: sev,
            agent_role: role.to_string(),
            tool_name: Some("bash".to_string()),
            blocked,
            timestamp: 1000,
            task_id: Some("task_1".to_string()),
            description: format!("Test violation: {}", pattern),
        }
    }

    // ─── Recording ───

    #[test]
    fn test_record_violation() {
        let mut engine = EvolutionEngine::new();
        assert_eq!(engine.violation_count(), 0);

        engine.record_violation(sample_violation(
            "methodology:index-priority",
            "full traversal",
            RedFlagSeverity::Critical,
            "DA",
            true,
        ));

        assert_eq!(engine.violation_count(), 1);
    }

    #[test]
    fn test_record_pass() {
        let mut engine = EvolutionEngine::new();
        engine.record_pass("methodology:index-priority");

        let metrics = engine.get_metrics("methodology:index-priority").unwrap();
        assert_eq!(metrics.pass_count, 1);
    }

    #[test]
    fn test_record_activation() {
        let mut engine = EvolutionEngine::new();
        engine.record_activation("methodology:systematic-debugging");

        let metrics = engine
            .get_metrics("methodology:systematic-debugging")
            .unwrap();
        assert_eq!(metrics.activation_count, 1);
        assert!(metrics.last_activated.is_some());
    }

    #[test]
    fn test_ring_buffer_capacity() {
        let mut engine = EvolutionEngine::with_max_records(5);
        for i in 0..10 {
            engine.record_violation(sample_violation(
                "methodology:test",
                &format!("pattern_{}", i),
                RedFlagSeverity::Info,
                "DA",
                false,
            ));
        }
        assert_eq!(engine.violation_count(), 5);
    }

    // ─── Pattern Learning ───

    #[test]
    fn test_learn_patterns_groups_by_pattern() {
        let mut engine = EvolutionEngine::new();
        engine.record_violation(sample_violation(
            "methodology:index-priority",
            "full traversal",
            RedFlagSeverity::Critical,
            "DA",
            true,
        ));
        engine.record_violation(sample_violation(
            "methodology:index-priority",
            "full traversal",
            RedFlagSeverity::Critical,
            "DA",
            true,
        ));
        engine.record_violation(sample_violation(
            "methodology:cost-awareness",
            "no alternative comparison",
            RedFlagSeverity::Warning,
            "PA",
            false,
        ));

        let patterns = engine.learn_patterns();
        assert_eq!(patterns.len(), 2, "Should have 2 distinct patterns");

        let top = patterns.first().unwrap();
        assert_eq!(top.frequency, 2);
        assert_eq!(top.methodology_id, "methodology:index-priority");
    }

    #[test]
    fn test_top_patterns_limit() {
        let mut engine = EvolutionEngine::new();
        for i in 0..10 {
            engine.record_violation(sample_violation(
                "methodology:test",
                &format!("pattern_{}", i),
                RedFlagSeverity::Info,
                "DA",
                false,
            ));
        }
        let top = engine.top_patterns(3);
        assert_eq!(top.len(), 3);
    }

    // ─── Effectiveness ───

    #[test]
    fn test_effectiveness_starts_perfect() {
        let mut engine = EvolutionEngine::new();
        engine.record_pass("methodology:test");
        let metrics = engine.get_metrics("methodology:test").unwrap();
        assert!((metrics.effectiveness_score - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_effectiveness_decreases_with_violations() {
        let mut engine = EvolutionEngine::new();
        engine.record_pass("methodology:test");
        engine.record_pass("methodology:test");
        engine.record_violation(sample_violation(
            "methodology:test",
            "bad",
            RedFlagSeverity::Critical,
            "DA",
            true,
        ));

        let metrics = engine.get_metrics("methodology:test").unwrap();
        assert!(metrics.effectiveness_score < 1.0);
        assert!(metrics.total_violations == 1);
    }

    #[test]
    fn test_no_metrics_for_unrecorded() {
        let engine = EvolutionEngine::new();
        assert!(engine.get_metrics("methodology:nonexistent").is_none());
    }

    // ─── Report ───

    #[test]
    fn test_empty_report() {
        let engine = EvolutionEngine::new();
        let report = engine.generate_report();
        assert_eq!(report.total_violations, 0);
        assert!((report.health_score - 1.0).abs() < 0.01);
        assert_eq!(report.active_methodologies, 0);
    }

    #[test]
    fn test_report_with_data() {
        let mut engine = EvolutionEngine::new();
        engine.record_violation(sample_violation(
            "methodology:index-priority",
            "full traversal",
            RedFlagSeverity::Critical,
            "DA",
            true,
        ));
        engine.record_pass("methodology:index-priority");
        engine.record_activation("methodology:index-priority");

        let report = engine.generate_report();
        assert_eq!(report.total_violations, 1);
        assert!(report.active_methodologies >= 1);
    }

    #[test]
    fn test_aa_briefing_not_empty() {
        let mut engine = EvolutionEngine::new();
        engine.record_violation(sample_violation(
            "methodology:index-priority",
            "full traversal",
            RedFlagSeverity::Critical,
            "DA",
            true,
        ));
        let briefing = engine.aa_evolution_briefing();
        assert!(!briefing.is_empty());
        assert!(briefing.contains("Evolution Report"));
        assert!(briefing.contains("System Health"));
    }

    #[test]
    fn test_aa_briefing_empty() {
        let engine = EvolutionEngine::new();
        let briefing = engine.aa_evolution_briefing();
        assert!(!briefing.is_empty());
        assert!(briefing.contains("System Health Score: 100.0%"));
    }

    // ─── ViolationReporter ───

    #[test]
    fn test_violation_from_anti_pattern() {
        let result = AntiPatternGateResult {
            methodology_id: "methodology:test".to_string(),
            anti_pattern_name: "Test Pattern".to_string(),
            description: "Test description".to_string(),
            gate_ask: "Should you do this?".to_string(),
            gate_action: "STOP".to_string(),
            should_block: true,
            message: "⚠️ Blocked".to_string(),
        };

        let record =
            ViolationReporter::from_anti_pattern(&result, "DA", Some("task_1".to_string()), 1000);
        assert_eq!(record.methodology_id, "methodology:test");
        assert_eq!(record.pattern_name, "Test Pattern");
        assert!(record.blocked);
        assert_eq!(record.agent_role, "DA");
    }

    // ─── Edge Cases ───

    #[test]
    fn test_thousands_of_violations() {
        let mut engine = EvolutionEngine::new();
        for i in 0..1000 {
            engine.record_violation(sample_violation(
                "methodology:test",
                &format!("pattern_{}", i % 5),
                RedFlagSeverity::Info,
                "DA",
                false,
            ));
        }
        assert_eq!(engine.violation_count(), 1000);
        let patterns = engine.learn_patterns();
        assert_eq!(
            patterns.len(),
            5,
            "Should have 5 distinct patterns from 1000 violations"
        );
    }

    #[test]
    fn test_all_metrics_query() {
        let mut engine = EvolutionEngine::new();
        engine.record_activation("methodology:a");
        engine.record_activation("methodology:b");
        engine.record_activation("methodology:c");

        let all = engine.all_metrics();
        assert_eq!(all.len(), 3);
    }
}
