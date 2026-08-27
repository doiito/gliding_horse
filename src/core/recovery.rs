//! Structured recovery protocol shared by the SA execution layers.
//!
//! CA reports evidence and scope; SA converts that report into a recovery
//! directive.  The roles never call one another directly.

use serde::{Deserialize, Serialize};

use super::five_w2h::{AuditStatus, DimensionAuditResult};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditVerdict {
    Pass,
    Conditional,
    Fail,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RepairScope {
    Step,
    Phase,
    Task,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecoveryReason {
    Accepted,
    LocalExecutionGap,
    PlanInvalid,
    NonConvergent,
    DependencyBlocked,
    EvidenceMissing,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecoveryDirective {
    Accept,
    RetryDa,
    ReplanPa,
    Blocked,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrchestrationMode {
    /// LLM-created PA→DA→CA→AA plan, with SA-level PDCA re-entry.
    Pdca,
    /// External JSON-LD workflow whose graph topology is preserved.
    Dag,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditFinding {
    pub dimension: String,
    pub message: String,
    pub evidence: String,
    pub scope: RepairScope,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditReport {
    pub verdict: AuditVerdict,
    pub failed_dimensions: Vec<String>,
    pub findings: Vec<AuditFinding>,
    pub scope: RepairScope,
    pub reason: Option<RecoveryReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionReport {
    pub mode: OrchestrationMode,
    pub directive: RecoveryDirective,
    pub reason: RecoveryReason,
    pub scope: RepairScope,
    pub plan_revision: u32,
}

impl AuditReport {
    pub fn from_results(results: &[DimensionAuditResult]) -> Self {
        let failures: Vec<&DimensionAuditResult> = results
            .iter()
            .filter(|result| matches!(result.status, AuditStatus::Fail(_)))
            .collect();
        let warnings = results
            .iter()
            .filter(|result| matches!(result.status, AuditStatus::Warning(_)))
            .count();

        let findings: Vec<AuditFinding> = failures
            .iter()
            .map(|result| {
                let message = match &result.status {
                    AuditStatus::Fail(message) => message.clone(),
                    _ => "audit failed".to_string(),
                };
                AuditFinding {
                    dimension: result.dimension.clone(),
                    message,
                    evidence: result.evidence.clone(),
                    scope: scope_for_dimension(&result.dimension),
                }
            })
            .collect();

        let scope = findings
            .iter()
            .map(|finding| finding.scope)
            .max_by_key(|scope| match scope {
                RepairScope::Step => 0,
                RepairScope::Phase => 1,
                RepairScope::Task => 2,
            })
            .unwrap_or(RepairScope::Step);

        let verdict = if !failures.is_empty() {
            AuditVerdict::Fail
        } else if warnings > 0 {
            AuditVerdict::Conditional
        } else {
            AuditVerdict::Pass
        };
        let reason = if failures.is_empty() {
            None
        } else if scope == RepairScope::Task {
            Some(RecoveryReason::PlanInvalid)
        } else {
            Some(RecoveryReason::LocalExecutionGap)
        };

        Self {
            verdict,
            failed_dimensions: failures
                .iter()
                .map(|result| result.dimension.clone())
                .collect(),
            findings,
            scope,
            reason,
        }
    }

    pub fn failed(&self) -> bool {
        self.verdict == AuditVerdict::Fail
    }
}

pub fn scope_for_dimension(_dimension: &str) -> RepairScope {
    // A 5W2H dimension identifies which acceptance boundary failed, not which
    // business role owns the repair. For example, a `why` failure can simply
    // mean that DA omitted a required test; sending that directly to PA would
    // discard a valid plan and waste a complete PDCA cycle. Start with a local
    // executable repair. `track_non_convergence` promotes repeated identical
    // failures to task scope, where SA re-enters PA with the accumulated CA
    // evidence.
    RepairScope::Step
}

pub fn select_directive(
    report: &AuditReport,
    local_repairs_used: u32,
    local_repair_limit: u32,
) -> RecoveryDirective {
    if !report.failed() {
        return RecoveryDirective::Accept;
    }
    if report.scope == RepairScope::Task || local_repairs_used >= local_repair_limit {
        RecoveryDirective::ReplanPa
    } else {
        RecoveryDirective::RetryDa
    }
}

/// Mark repeated identical CA failures as a task-level non-convergence.
/// Two consecutive identical failed-dimension sets are enough to stop local
/// DA retries; the next decision must be made by PA with a changed plan.
pub fn track_non_convergence(
    report: &mut AuditReport,
    previous_signature: &mut Option<Vec<String>>,
    repeated_failures: &mut u32,
) {
    if !report.failed() {
        *previous_signature = None;
        *repeated_failures = 0;
        return;
    }

    let mut signature = report.failed_dimensions.clone();
    signature.sort();
    if previous_signature.as_ref() == Some(&signature) {
        *repeated_failures = repeated_failures.saturating_add(1);
    } else {
        *repeated_failures = 1;
        *previous_signature = Some(signature);
    }

    if *repeated_failures >= 2 {
        report.scope = RepairScope::Task;
        report.reason = Some(RecoveryReason::NonConvergent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(dimension: &str) -> DimensionAuditResult {
        DimensionAuditResult {
            dimension: dimension.to_string(),
            status: AuditStatus::Fail("gap".to_string()),
            evidence: "evidence".to_string(),
            details: vec![],
        }
    }

    #[test]
    fn acceptance_dimension_does_not_prejudge_repair_owner() {
        let report = AuditReport::from_results(&[finding("why")]);
        assert_eq!(report.scope, RepairScope::Step);
        assert_eq!(select_directive(&report, 0, 3), RecoveryDirective::RetryDa);
    }

    #[test]
    fn local_scope_uses_da_until_limit() {
        let report = AuditReport::from_results(&[finding("how")]);
        assert_eq!(report.scope, RepairScope::Step);
        assert_eq!(select_directive(&report, 0, 3), RecoveryDirective::RetryDa);
        assert_eq!(select_directive(&report, 3, 3), RecoveryDirective::ReplanPa);
    }

    #[test]
    fn repeated_ca_failures_become_non_convergent() {
        let mut previous = None;
        let mut repeats = 0;
        let mut first = AuditReport::from_results(&[finding("how")]);
        track_non_convergence(&mut first, &mut previous, &mut repeats);
        assert_eq!(first.scope, RepairScope::Step);

        let mut second = AuditReport::from_results(&[finding("how")]);
        track_non_convergence(&mut second, &mut previous, &mut repeats);
        assert_eq!(second.scope, RepairScope::Task);
        assert_eq!(second.reason, Some(RecoveryReason::NonConvergent));
        assert_eq!(select_directive(&second, 1, 3), RecoveryDirective::ReplanPa);
    }

    #[test]
    fn orchestration_modes_are_distinguishable_in_reports() {
        let pdca = DecisionReport {
            mode: OrchestrationMode::Pdca,
            directive: RecoveryDirective::Accept,
            reason: RecoveryReason::Accepted,
            scope: RepairScope::Task,
            plan_revision: 1,
        };
        let dag = DecisionReport {
            mode: OrchestrationMode::Dag,
            ..pdca.clone()
        };
        assert_ne!(pdca.mode, dag.mode);
        assert!(serde_json::to_string(&dag).unwrap().contains("Dag"));
    }
}
