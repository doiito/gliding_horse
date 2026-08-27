/// Methodology Integration Layer (Phase 3) — Role-specific prompt addendums.
///
/// Generates methodology-aware prompt sections for each agent role:
/// - PA: Plan-review gate + Granularity check
/// - CA: Dual-stage audit + Anti-pattern detection  
/// - AA: Pressure testing + Meta-test protocol
/// - SA: Methodology auto-trigger awareness
///
/// These addendums are injected into agent system prompts alongside the
/// constitution behavioral policy, instructing agents to apply specific
/// methodology disciplines during execution.
///
/// Architecture Layer: L2/L1 boundary — Methodology → Agent Prompt
/// See design: PR-res/superpowers-skills-full-integration-design.md §3
use crate::core::agent_instance::AgentRole;
use crate::methodology::MethodologyRegistry;

/// Generates role-specific methodology prompt addendums.
///
/// Stateless — creates a fresh registry each time, following the same
/// pattern as `build_constitution_prompt` in `system_prompt.rs`.
pub struct MethodologyPromptInjector;

impl MethodologyPromptInjector {
    /// Build the full methodology addendum for an agent role.
    ///
    /// Returns `None` for roles that only need the constitution baseline (DA).
    pub fn build_for_role(role: AgentRole) -> Option<String> {
        let registry = crate::methodology::global_registry();
        match role {
            AgentRole::Plan => Some(Self::pa_plan_review_gate(registry)),
            AgentRole::Check => Some(Self::ca_dual_stage_audit(registry)),
            AgentRole::Act => Some(Self::aa_pressure_test(registry)),
            AgentRole::Do => Some(Self::da_execution_discipline(registry)),
        }
    }

    /// Build methodology addendum for SA (used in sa.rs prompt assembly).
    pub fn build_for_sa() -> String {
        let registry = crate::methodology::global_registry();
        Self::sa_methodology_awareness(registry)
    }

    // ─── PA: Plan-Review Gate + Granularity Check ───

    fn pa_plan_review_gate(registry: &MethodologyRegistry) -> String {
        let complexity = registry.get("methodology:complexity-assessment");
        let boundary = registry.get("methodology:boundary-enforcement");

        let mut sections = vec![
            "\n## 📋 Methodology Discipline — Plan Review Gate".to_string(),
            "As the Planning Agent, you must follow these methodology disciplines:".to_string(),
        ];

        sections.push("\n### 1. Step Granularity Check".to_string());
        sections.push(
            "For each plan step, evaluate whether its granularity is appropriate:\n\
            - ✅ Too coarse: a step contains multiple unrelated operations → split\n\
            - ✅ Too fine: a step contains only atomic operations → merge\n\
            - ✅ Standard: a step contains a group of related operations with clear input and output\n\
            Example: 'create file' and 'edit file' should not be two steps; merge into 'implement feature X'"
                .to_string(),
        );

        if let Some(c) = complexity {
            let reds: Vec<&str> = c.red_flags.iter().map(|r| r.pattern).collect();
            sections.push("\n### 2. Complexity Matching".to_string());
            sections.push(format!(
                "Selected complexity level must match actual task requirements. Prohibited:\n\
                - {}: choosing a lower level for convenience\n\
                - {}: choosing a higher level for show",
                reds.first().unwrap_or(&"convenience downgrade"),
                reds.get(1).unwrap_or(&"show-off upgrade"),
            ));
        }

        if let Some(b) = boundary {
            let aps: Vec<&str> = b.anti_patterns.iter().map(|ap| ap.name).collect();
            sections.push("\n### 3. Boundary Check".to_string());
            sections.push(format!(
                "Plans must not contain boundary violations:\n\
                - Check whether each step's responsibility is within the Agent's authority\n\
                - Do not include {} operations in the plan\n\
                - If violations are found, flag and suggest corrections",
                aps.first().unwrap_or(&"boundary overstep"),
            ));
        }

        sections.join("\n")
    }

    // ─── CA: Dual-Stage Audit + Anti-Pattern Detection ───

    fn ca_dual_stage_audit(registry: &MethodologyRegistry) -> String {
        let debugging = registry.get("methodology:systematic-debugging");
        let verification = registry.get("methodology:verification-before-completion");

        let mut sections = vec![
            "\n## 📋 Methodology Discipline — Dual-Stage Audit + Anti-Pattern Detection"
                .to_string(),
            "As the Checking Agent, you must perform a dual-stage audit:".to_string(),
        ];

        sections.push("\n### Stage 1: Output Verification".to_string());
        sections.push(
            "Verify that execution results meet task requirements:\n\
            - Check whether artifacts are complete and correct\n\
            - Verify whether success criteria are met\n\
            - Confirm whether all steps have been completed\n\
            - Check whether files actually exist (don't rely solely on summary)"
                .to_string(),
        );

        sections.push("\n### Stage 2: Methodology Compliance Check".to_string());
        sections.push(
            "Check whether the execution process complied with methodology requirements:\n\
            - Did you verify before claiming completion? (Verification-Before-Completion)\n\
            - Are there known anti-patterns? (e.g., blind retry, random modification, full traversal)\n\
            - Was cost awareness followed? (avoid unnecessary large output, no-alternative comparisons)\n\
            - Were operations within authorized scope? (Least-Privilege / Boundary-Enforcement)"
                .to_string(),
        );

        if let Some(d) = debugging {
            let aps: Vec<&str> = d.anti_patterns.iter().map(|ap| ap.name).collect();
            sections.push("\n### Anti-Pattern Detection".to_string());
            sections.push(format!(
                "During the audit, focus on detecting these anti-patterns:\n\
                - {}: modifying multiple places without verifying each?\n\
                - no-evidence claim: claiming completion without verification evidence?\n\
                - implementation first: writing code before tests?",
                aps.first().unwrap_or(&"Shotgun Debugging"),
            ));
        }

        if let Some(v) = verification {
            let reds: Vec<&str> = v.red_flags.iter().map(|r| r.pattern).collect();
            sections.push("\n### Evidence Requirements".to_string());
            sections.push(format!(
                "CA's own audit conclusions also need verification evidence:\n\
                - Avoid the '{}' situation\n\
                - Attach actual inspection results to every PASS/FAIL conclusion\n\
                - Use tools to verify before drawing conclusions",
                reds.first().unwrap_or(&"claiming pass without running"),
            ));
        }

        sections.join("\n")
    }

    // ─── AA: Pressure Testing + Meta-Test Protocol ───

    fn aa_pressure_test(registry: &MethodologyRegistry) -> String {
        let _debugging = registry.get("methodology:systematic-debugging");
        let _boundary = registry.get("methodology:boundary-enforcement");

        vec![
            "\n## 📋 Methodology Discipline — Pressure Testing + Meta-Test Protocol".to_string(),
            "As the Decision Agent, you must perform pressure testing:".to_string(),
            "".to_string(),
            "### 1. Pressure Testing".to_string(),
            "Before approving CA's audit conclusions, test the supplied evidence for concrete vulnerabilities:".to_string(),
            "- Examine from the opposite perspective: 'If this conclusion is wrong, which supplied fact would expose the error?'".to_string(),
            "- Check whether CA only verified surface-level results without inspecting required dimensions".to_string(),
            "- Raise a challenge only when the supplied evidence contains a specific contradiction, unsupported criterion, or material gap".to_string(),
            "- Do not invent a challenge merely to demonstrate independence".to_string(),
            "".to_string(),
            "### 2. Meta-Test Protocol".to_string(),
            "Assess whether CA's verification process is reliable from the evidence directly supplied to you:".to_string(),
            "- Did CA report actual verification actions/results rather than an unsupported assertion?".to_string(),
            "- Did CA check all acceptance criteria and relevant dimensions?".to_string(),
            "- Are CA's conclusions supported by clear, internally consistent evidence?".to_string(),
            "- If a material criterion lacks evidence, identify that exact criterion and request a supplementary check".to_string(),
            "- Do not reject solely because an archive or execution tool is unavailable to AA; AA is a non-mutating decision role".to_string(),
            "".to_string(),
            "### 3. Decision Responsibility".to_string(),
            "The final decision rests with you; you are accountable for its consequences:".to_string(),
            "- Don't be CA's 'rubber stamp' — make independent judgments".to_string(),
            "- Consider task goals, actual results, and risks comprehensively".to_string(),
            "- If rollback is needed, provide specific rollback recommendations".to_string(),
        ]
        .join("\n")
    }

    // ─── SA: Methodology Auto-Trigger Awareness ───

    fn sa_methodology_awareness(registry: &MethodologyRegistry) -> String {
        let always_on: Vec<&str> = registry
            .all()
            .iter()
            .filter(|m| {
                matches!(
                    m.activation,
                    crate::methodology::ActivationCondition::Always
                )
            })
            .map(|m| m.name)
            .collect();

        let role_triggers: Vec<&str> = registry
            .all()
            .iter()
            .filter(|m| {
                matches!(
                    m.activation,
                    crate::methodology::ActivationCondition::OnAgentRole(_)
                )
            })
            .map(|m| m.name)
            .collect();

        let mut sections = vec![
            "\n## 📋 Methodology Discipline — Auto-Trigger Protocol".to_string(),
            "As the Supervisor Agent, consider the following methodologies when creating plans:"
                .to_string(),
        ];

        if !always_on.is_empty() {
            sections.push("\n### Always-Active Methodologies".to_string());
            sections.push(format!(
                "The following methodologies are always active and must be reflected in plans:\n- {}",
                always_on.join("\n- ")
            ));
        }

        if !role_triggers.is_empty() {
            sections.push("\n### Role-Triggered Methodologies".to_string());
            sections.push(format!(
                "The following methodologies auto-trigger when specific Agent roles execute:\n- {}",
                role_triggers.join("\n- ")
            ));
        }

        sections.push("\n### Plan Generation Requirements".to_string());
        sections.push(
            "When generating plans, ensure:\n\
            - Each step's responsibilities match the corresponding methodology\n\
            - Reserve sufficient verification space for DA steps\n\
            - CA steps include dual-stage audit (output verification + methodology compliance)\n\
            - AA steps include pressure testing requirements\n\
            - Step granularity is reasonable: group related operations into one step"
                .to_string(),
        );

        sections.join("\n")
    }

    // ─── DA: Execution Discipline ───

    fn da_execution_discipline(registry: &MethodologyRegistry) -> String {
        let debugging = registry.get("methodology:systematic-debugging");
        let verification = registry.get("methodology:verification-before-completion");

        let mut sections = vec![
            "\n## 📋 Methodology Discipline — Execution Standards".to_string(),
            "As the Execution Agent, you must follow these execution standards:".to_string(),
        ];

        sections.push("\n### 1. Least Privilege Execution".to_string());
        sections.push(
            "Strictly limit operation scope in each step:\n\
            - Only execute assigned tasks; do not exceed authority\n\
            - Do not modify system configuration outside responsibilities\n\
            - If scope expansion is needed, flag and request guidance"
                .to_string(),
        );

        if let Some(v) = verification {
            let reds: Vec<&str> = v.red_flags.iter().map(|r| r.pattern).collect();
            sections.push("\n### 2. Verification Before Completion".to_string());
            sections.push(format!(
                "Before claiming completion, verification evidence must be provided:\n\
                - Avoid '{}' — verify at runtime\n\
                - Check actual effects after tool operations\n\
                - Confirm content correctness after file writes",
                reds.first().unwrap_or(&"assumed success"),
            ));
        }

        if let Some(d) = debugging {
            let aps: Vec<&str> = d.anti_patterns.iter().map(|ap| ap.name).collect();
            sections.push("\n### 3. Exception Handling Standards".to_string());
            sections.push(format!(
                "When encountering execution failures:\n\
                - Analyze root cause before retrying; no blind retries\n\
                - Avoid {} — change one thing at a time and verify\n\
                - Stop and report after 3 consecutive failures",
                aps.first().unwrap_or(&"random modification"),
            ));
        }

        sections.push("\n### 4. Cost Awareness".to_string());
        sections.push(
            "Maintain sensitivity to costs (tokens/time) during execution:\n\
            - Prioritize precise tools over large-output traversal\n\
            - Read only the needed portions of files, not the entire file\n\
            - Do not execute unnecessary verification steps beyond task requirements"
                .to_string(),
        );

        sections.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pa_addendum_exists() {
        let addendum = MethodologyPromptInjector::build_for_role(AgentRole::Plan);
        assert!(addendum.is_some(), "PA should have a methodology addendum");
        let text = addendum.unwrap();
        assert!(
            text.contains("Granularity Check"),
            "PA addendum should mention granularity"
        );
        assert!(
            text.contains("Complexity Matching"),
            "PA addendum should mention complexity"
        );
        assert!(
            text.contains("Boundary Check"),
            "PA addendum should mention boundary"
        );
    }

    #[test]
    fn test_ca_addendum_exists() {
        let addendum = MethodologyPromptInjector::build_for_role(AgentRole::Check);
        assert!(addendum.is_some(), "CA should have a methodology addendum");
        let text = addendum.unwrap();
        assert!(
            text.contains("Dual-Stage Audit"),
            "CA addendum should mention dual-stage audit"
        );
        assert!(text.contains("Stage 1"), "CA addendum should have Stage 1");
        assert!(text.contains("Stage 2"), "CA addendum should have Stage 2");
        assert!(
            text.contains("Anti-Pattern Detection"),
            "CA addendum should mention anti-pattern detection"
        );
    }

    #[test]
    fn test_aa_addendum_exists() {
        let addendum = MethodologyPromptInjector::build_for_role(AgentRole::Act);
        assert!(addendum.is_some(), "AA should have a methodology addendum");
        let text = addendum.unwrap();
        assert!(
            text.contains("Pressure Testing"),
            "AA addendum should mention pressure testing"
        );
        assert!(
            text.contains("Meta-Test"),
            "AA addendum should mention meta-test"
        );
        assert!(
            text.contains("rubber stamp"),
            "AA addendum should warn against rubber-stamping"
        );
        assert!(
            text.contains("Do not invent a challenge"),
            "AA must not manufacture a failure when CA evidence is complete"
        );
        assert!(
            text.contains("non-mutating decision role"),
            "AA must respect its no-tool decision boundary"
        );
    }

    #[test]
    fn test_da_addendum_exists() {
        let addendum = MethodologyPromptInjector::build_for_role(AgentRole::Do);
        assert!(
            addendum.is_some(),
            "DA should now have a methodology addendum"
        );
        let text = addendum.unwrap();
        assert!(
            text.contains("Least Privilege Execution"),
            "DA addendum should mention least privilege"
        );
        assert!(
            text.contains("Cost Awareness"),
            "DA addendum should mention cost awareness"
        );
    }

    #[test]
    fn test_sa_addendum_exists() {
        let text = MethodologyPromptInjector::build_for_sa();
        assert!(!text.is_empty(), "SA addendum should not be empty");
        assert!(
            text.contains("Always-Active Methodologies"),
            "SA addendum should mention always-on"
        );
        assert!(
            text.contains("Plan Generation Requirements"),
            "SA addendum should have plan generation requirements"
        );
    }

    #[test]
    fn test_addendum_includes_methodology_names() {
        let registry = MethodologyRegistry::new();
        let always_on_count = registry
            .all()
            .iter()
            .filter(|m| {
                matches!(
                    m.activation,
                    crate::methodology::ActivationCondition::Always
                )
            })
            .count();

        let sa_text = MethodologyPromptInjector::build_for_sa();
        assert!(
            always_on_count == 0 || sa_text.contains("Methodology"),
            "SA addendum should reference methodologies"
        );
    }

    #[test]
    fn test_pa_addendum_references_granularity_rules() {
        let text = MethodologyPromptInjector::build_for_role(AgentRole::Plan).unwrap();
        assert!(
            text.contains("Too coarse") || text.contains("Too fine"),
            "PA granularity check should define granularity levels"
        );
    }

    #[test]
    fn test_ca_addendum_has_evidence_requirement() {
        let text = MethodologyPromptInjector::build_for_role(AgentRole::Check).unwrap();
        assert!(
            text.contains("Evidence"),
            "CA dual-stage audit should require evidence"
        );
    }

    #[test]
    fn test_aa_addendum_mentions_decision_responsibility() {
        let text = MethodologyPromptInjector::build_for_role(AgentRole::Act).unwrap();
        assert!(
            text.contains("Decision Responsibility"),
            "AA pressure testing should mention decision responsibility"
        );
    }

    #[test]
    fn test_all_roles_have_consistent_structure() {
        for role in &[
            AgentRole::Plan,
            AgentRole::Check,
            AgentRole::Act,
            AgentRole::Do,
        ] {
            if let Some(text) = MethodologyPromptInjector::build_for_role(*role) {
                assert!(
                    text.starts_with("\n##"),
                    "{} addendum should start with a header",
                    role
                );
                assert!(text.len() > 100, "{} addendum should be substantial", role);
            }
        }
    }
}
