//! Structured recovery protocol shared by the SA execution layers.
//!
//! CA reports evidence and scope; SA converts that report into a recovery
//! directive.  The roles never call one another directly.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    /// CA obtained runtime evidence but failed to serialize its kernel-owned
    /// terminal contract. Recovery may re-dispatch one fresh, isolated CA to
    /// encode the retained evidence; it must not repeat broad verification or
    /// grant DA mutation authority.
    TerminalContractInvalid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecoveryDirective {
    Accept,
    /// Re-run an isolated Check Agent to obtain missing acceptance evidence.
    /// This never grants workspace-mutation authority and must not be
    /// collapsed into `RetryDa` by the outer PDCA controller.
    RetryCa,
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
    /// Stable, non-prose identifiers used only for convergence identity.
    /// Examples are a referenced artifact path or test identifier. Full CA
    /// evidence remains in `evidence` for repair, but timestamps/turn counts
    /// and wording must not make the same defect appear new forever.
    #[serde(default)]
    pub identity_keys: Vec<String>,
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
                    identity_keys: Vec::new(),
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

/// Attach concrete CA evidence while deriving stable, mechanically
/// identifiable repair targets. This intentionally does not split or infer
/// free-form natural-language requirements: only explicitly labelled failed
/// criteria, path-like tokens, test identifiers and failure/exit codes
/// participate in convergence identity.
pub fn enrich_findings_with_evidence(report: &mut AuditReport, evidence: &str) {
    let identity_keys = actionable_evidence_keys(evidence);
    for finding in &mut report.findings {
        finding.evidence = evidence.to_string();
        finding.identity_keys = identity_keys.clone();
    }
}

fn actionable_evidence_keys(evidence: &str) -> Vec<String> {
    let mut keys = Vec::new();
    let mut actionable_lines = Vec::new();
    // A CA may state a natural-language criterion explicitly. Only labelled
    // *failed* criteria are identity-bearing; generic criterion inventories
    // include already-satisfied requirements and would hide a changing gap.
    for line in evidence.lines() {
        let normalized_line = line.trim().trim_start_matches(|character: char| {
            character.is_whitespace()
                || matches!(character, '-' | '*' | '•' | '[' | ']' | '`')
                || character.is_ascii_digit()
                || character == '.'
        });
        let lower_line = normalized_line.to_lowercase();
        let criterion = [
            "failed criterion:",
            "failed criterion：",
            "unmet criterion:",
            "unmet criterion：",
            "unsatisfied criterion:",
            "unsatisfied criterion：",
            "failed_criterion:",
            "failed_criterion：",
            "未满足验收标准:",
            "未满足验收标准：",
            "失败验收标准:",
            "失败验收标准：",
        ]
        .iter()
        .find_map(|prefix| lower_line.strip_prefix(prefix));
        if let Some(criterion) = criterion {
            let suffix_start = [
                " | ",
                " evidence:",
                " evidence：",
                " observed:",
                " observed：",
            ]
            .iter()
            .filter_map(|delimiter| criterion.find(delimiter))
            .min()
            .unwrap_or(criterion.len());
            let stable_value = criterion[..suffix_start]
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            if !stable_value.is_empty() {
                keys.push(format!(
                    "criterion:{}",
                    stable_value.chars().take(240).collect::<String>()
                ));
            }
        }
        if [
            "fail",
            "error",
            "missing",
            "absent",
            "unmet",
            "unsatisfied",
            "not found",
            "does not exist",
            "must ",
            "required",
            "create ",
            "fix ",
            "exit_code=",
            "error_code=",
            "failure_code=",
            "失败",
            "缺失",
            "不存在",
            "未满足",
            "未生成",
            "必须",
            "需要创建",
        ]
        .iter()
        .any(|marker| lower_line.contains(marker))
        {
            actionable_lines.push(line);
        }
    }

    // Restrict identifiers to actionable failure lines. Successful-test
    // inventories and unrelated tool output may vary between CA turns, but
    // that incidental evidence must not reset an otherwise identical defect.
    for raw in actionable_lines.iter().flat_map(|line| {
        line.split(|character: char| {
            character.is_whitespace() || matches!(character, ',' | '，' | '、' | ';' | '；')
        })
    }) {
        let token = raw.trim_matches(|character: char| {
            matches!(
                character,
                '`' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | '：' | ':'
            )
        });
        let lower = token
            .trim_end_matches(|character: char| matches!(character, '.' | '。' | '!' | '！'))
            .to_lowercase();
        if !lower.starts_with("http://")
            && !lower.starts_with("https://")
            && !lower.starts_with("iri://")
            && lower.contains('/')
            && lower
                .chars()
                .all(|character| character.is_alphanumeric() || "._-/\\".contains(character))
        {
            keys.push(format!("path:{lower}"));
        }
        let identifier =
            lower.trim_matches(|character: char| !character.is_alphanumeric() && character != '_');
        if identifier.starts_with("test_") || identifier.contains("::test_") {
            keys.push(format!("test:{identifier}"));
        }
        if lower.starts_with("exit_code=")
            || lower.starts_with("exit-code=")
            || lower.starts_with("error_code=")
            || lower.starts_with("failure_code=")
        {
            keys.push(format!("code:{lower}"));
        }
    }
    keys.sort();
    keys.dedup();
    keys.truncate(32);
    keys
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
    if matches!(
        report.reason,
        Some(RecoveryReason::EvidenceMissing | RecoveryReason::TerminalContractInvalid)
    ) {
        return if report.scope == RepairScope::Task || local_repairs_used >= local_repair_limit {
            // Re-planning cannot manufacture an omitted verifier receipt and
            // would replay already-completed DA side effects. Stop the
            // automatic loop after the bounded CA-only recheck instead.
            RecoveryDirective::Blocked
        } else {
            RecoveryDirective::RetryCa
        };
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

    // A 5W2H dimension is only a routing category. Two different concrete CA
    // findings commonly share `why`; treating them as identical prematurely
    // escalates useful local repairs, while resetting this signature between
    // outer cycles creates the opposite retry explosion. Sign the normalized
    // typed findings, falling back to dimensions only for legacy callers.
    let mut signature = if report.findings.is_empty() {
        report.failed_dimensions.clone()
    } else {
        report
            .findings
            .iter()
            .map(|finding| {
                let normalized = format!(
                    "{:?}\n{}\n{}\n{}",
                    report.reason,
                    finding.dimension.trim().to_lowercase(),
                    finding
                        .message
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" "),
                    finding.identity_keys.join("\n"),
                )
                .to_lowercase();
                let digest = Sha256::digest(normalized.as_bytes());
                format!(
                    "{}:sha256:{}",
                    finding.dimension,
                    hex::encode(&digest[..12])
                )
            })
            .collect::<Vec<_>>()
    };
    signature.sort();
    if previous_signature.as_ref() == Some(&signature) {
        *repeated_failures = repeated_failures.saturating_add(1);
    } else {
        *repeated_failures = 1;
        *previous_signature = Some(signature);
    }

    // Repeating an implementation defect means the current DA repair is not
    // converging and legitimately requires PA. Repeating an evidence gap has
    // different ownership: PA/DA cannot manufacture a verifier receipt. Keep
    // it CA-owned until the independent CA recheck budget is exhausted, at
    // which point `select_directive` returns `Blocked` without replaying work.
    if *repeated_failures >= 2
        && !matches!(
            report.reason,
            Some(RecoveryReason::EvidenceMissing | RecoveryReason::TerminalContractInvalid)
        )
    {
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

    fn report_with_concrete_evidence(evidence: &str) -> AuditReport {
        let mut report = AuditReport {
            verdict: AuditVerdict::Fail,
            failed_dimensions: vec!["why".to_string()],
            findings: vec![AuditFinding {
                dimension: "why".to_string(),
                message: "CA overall verdict is FAIL".to_string(),
                evidence: evidence.to_string(),
                scope: RepairScope::Step,
                identity_keys: Vec::new(),
            }],
            scope: RepairScope::Step,
            reason: Some(RecoveryReason::LocalExecutionGap),
        };
        enrich_findings_with_evidence(&mut report, evidence);
        report
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
    fn missing_evidence_retries_ca_without_granting_mutation() {
        let mut report = AuditReport::from_results(&[finding("how")]);
        report.reason = Some(RecoveryReason::EvidenceMissing);

        assert_eq!(select_directive(&report, 0, 1), RecoveryDirective::RetryCa);
        assert_eq!(select_directive(&report, 1, 1), RecoveryDirective::Blocked);

        report.scope = RepairScope::Task;
        assert_eq!(select_directive(&report, 0, 1), RecoveryDirective::Blocked);
    }

    #[test]
    fn repeated_missing_evidence_stays_ca_owned_until_its_budget_is_exhausted() {
        let mut previous = None;
        let mut repeats = 0;
        let mut first = AuditReport::from_results(&[finding("how")]);
        first.reason = Some(RecoveryReason::EvidenceMissing);
        first.scope = RepairScope::Phase;
        track_non_convergence(&mut first, &mut previous, &mut repeats);

        let mut second = first.clone();
        track_non_convergence(&mut second, &mut previous, &mut repeats);

        assert_eq!(repeats, 2);
        assert_eq!(second.scope, RepairScope::Phase);
        assert_eq!(second.reason, Some(RecoveryReason::EvidenceMissing));
        assert_eq!(select_directive(&second, 1, 2), RecoveryDirective::RetryCa);
        assert_eq!(select_directive(&second, 2, 2), RecoveryDirective::Blocked);
    }

    #[test]
    fn repeated_terminal_contract_failure_stays_ca_owned_and_stops_at_budget() {
        let mut previous = None;
        let mut repeats = 0;
        let mut first = AuditReport::from_results(&[finding("why")]);
        first.reason = Some(RecoveryReason::TerminalContractInvalid);
        first.scope = RepairScope::Phase;
        track_non_convergence(&mut first, &mut previous, &mut repeats);

        let mut second = first.clone();
        track_non_convergence(&mut second, &mut previous, &mut repeats);

        assert_eq!(repeats, 2);
        assert_eq!(second.scope, RepairScope::Phase);
        assert_eq!(second.reason, Some(RecoveryReason::TerminalContractInvalid));
        assert_eq!(select_directive(&second, 0, 1), RecoveryDirective::RetryCa);
        assert_eq!(select_directive(&second, 1, 1), RecoveryDirective::Blocked);
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
    fn same_dimension_with_a_new_concrete_defect_is_not_false_non_convergence() {
        let mut previous = None;
        let mut repeats = 0;
        let mut missing_docs =
            report_with_concrete_evidence("docs/README.md and docs/report.md are missing");
        track_non_convergence(&mut missing_docs, &mut previous, &mut repeats);
        assert_eq!(repeats, 1);

        let mut later_test_failure =
            report_with_concrete_evidence("test_cli_divide_by_zero now fails");
        track_non_convergence(&mut later_test_failure, &mut previous, &mut repeats);
        assert_eq!(repeats, 1);
        assert_eq!(later_test_failure.scope, RepairScope::Step);

        let mut same_test_failure =
            report_with_concrete_evidence("test_cli_divide_by_zero now fails");
        track_non_convergence(&mut same_test_failure, &mut previous, &mut repeats);
        assert_eq!(same_test_failure.scope, RepairScope::Task);
        assert_eq!(
            same_test_failure.reason,
            Some(RecoveryReason::NonConvergent)
        );
    }

    #[test]
    fn same_concrete_defect_with_different_incidental_evidence_is_non_convergent() {
        let mut previous = None;
        let mut repeats = 0;
        let mut first = report_with_concrete_evidence(
            "turn 4 at 12:01:03: docs/README.md is still missing; archive iri://task/a/turn_4",
        );
        track_non_convergence(&mut first, &mut previous, &mut repeats);
        assert_eq!(first.findings[0].identity_keys, vec!["path:docs/readme.md"]);
        assert_eq!(first.scope, RepairScope::Step);

        let mut second = report_with_concrete_evidence(
            "turn 11 at 12:09:44: rerun confirms docs/README.md remains missing; archive iri://task/a/turn_11",
        );
        track_non_convergence(&mut second, &mut previous, &mut repeats);
        assert_eq!(repeats, 2);
        assert_eq!(second.scope, RepairScope::Task);
        assert_eq!(second.reason, Some(RecoveryReason::NonConvergent));
    }

    #[test]
    fn passing_artifact_inventory_does_not_change_failure_identity() {
        let mut previous = None;
        let mut repeats = 0;
        let mut first = report_with_concrete_evidence(
            "docs/README.md is missing\nsrc/calculator.py verified PASS",
        );
        track_non_convergence(&mut first, &mut previous, &mut repeats);
        assert_eq!(first.findings[0].identity_keys, vec!["path:docs/readme.md"]);

        let mut second = report_with_concrete_evidence(
            "docs/README.md remains missing\nsrc/cli.py verified PASS",
        );
        track_non_convergence(&mut second, &mut previous, &mut repeats);
        assert_eq!(
            second.findings[0].identity_keys,
            vec!["path:docs/readme.md"]
        );
        assert_eq!(second.scope, RepairScope::Task);
    }

    #[test]
    fn explicitly_labelled_failed_criterion_distinguishes_non_path_defects() {
        let mut previous = None;
        let mut repeats = 0;
        let mut arithmetic = report_with_concrete_evidence(
            "Failed criterion: decimal addition preserves precision | observed: turn 4",
        );
        track_non_convergence(&mut arithmetic, &mut previous, &mut repeats);
        assert_eq!(repeats, 1);
        assert_eq!(
            arithmetic.findings[0].identity_keys,
            vec!["criterion:decimal addition preserves precision"]
        );

        let mut cli = report_with_concrete_evidence(
            "Failed criterion: invalid input returns a readable error | observed: turn 9",
        );
        track_non_convergence(&mut cli, &mut previous, &mut repeats);
        assert_eq!(repeats, 1);
        assert_eq!(cli.scope, RepairScope::Step);
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
