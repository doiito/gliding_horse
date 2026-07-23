use std::collections::HashSet;
use std::time::Instant;

use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::types::{
    GraphInvariant, SkillGraphNode, SkillLinkType, TrustLevel, VerificationResult, Violation,
    ViolationSeverity,
};

pub struct GraphVerifier;

impl GraphVerifier {
    pub fn new() -> Self {
        Self
    }

    pub fn verify_all(&self, store: &SkillGraphStore) -> Vec<VerificationResult> {
        vec![
            self.verify(store, GraphInvariant::Acyclicity),
            self.verify(store, GraphInvariant::LinkTargetExists),
            self.verify(store, GraphInvariant::CompositeReachability),
            self.verify(store, GraphInvariant::NoDeprecatedPrerequisites),
            self.verify(store, GraphInvariant::Valid5W2H),
            self.verify(store, GraphInvariant::ValidSecurityLevels),
        ]
    }

    pub fn verify(&self, store: &SkillGraphStore, invariant: GraphInvariant) -> VerificationResult {
        let start = Instant::now();
        let (passed, violations) = match invariant {
            GraphInvariant::Acyclicity => self.verify_acyclicity(store),
            GraphInvariant::LinkTargetExists => self.verify_link_targets(store),
            GraphInvariant::CompositeReachability => self.verify_composite_reachability(store),
            GraphInvariant::NoDeprecatedPrerequisites => {
                self.verify_no_deprecated_prerequisites(store)
            }
            GraphInvariant::Valid5W2H => self.verify_5w2h(store),
            GraphInvariant::ValidSecurityLevels => self.verify_security_levels(store),
        };
        VerificationResult {
            invariant,
            passed,
            violations,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    fn verify_acyclicity(&self, store: &SkillGraphStore) -> (bool, Vec<Violation>) {
        let skills = store.list_all_skills();
        let mut violations = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();

        for skill in &skills {
            if visited.contains(&skill.skill_iri) {
                continue;
            }
            let mut stack = HashSet::new();
            if has_prerequisite_cycle(&skill.skill_iri, &skills, &mut visited, &mut stack) {
                let affected: Vec<String> = stack.into_iter().collect();
                violations.push(Violation {
                    severity: ViolationSeverity::Error,
                    description: "Cycle detected in prerequisite chain".to_string(),
                    affected_iris: affected,
                    suggestion: "Remove or update a prerequisite link to break the cycle"
                        .to_string(),
                });
            }
        }

        (violations.is_empty(), violations)
    }

    fn verify_link_targets(&self, store: &SkillGraphStore) -> (bool, Vec<Violation>) {
        let skills = store.list_all_skills();
        let registered: HashSet<String> = skills.iter().map(|s| s.skill_iri.clone()).collect();
        let mut violations = Vec::new();

        for skill in &skills {
            for link in &skill.links {
                if !registered.contains(&link.target_iri) {
                    violations.push(Violation {
                        severity: ViolationSeverity::Error,
                        description: format!(
                            "Skill '{}' has link to non-existent target '{}'",
                            skill.name, link.target_iri
                        ),
                        affected_iris: vec![skill.skill_iri.clone(), link.target_iri.clone()],
                        suggestion: "Register the target skill or remove the dangling link"
                            .to_string(),
                    });
                }
            }
        }

        (violations.is_empty(), violations)
    }

    fn verify_composite_reachability(&self, store: &SkillGraphStore) -> (bool, Vec<Violation>) {
        let hyperedges = store.list_hyperedges();
        let registered: HashSet<String> = store
            .list_all_skills()
            .into_iter()
            .map(|s| s.skill_iri)
            .collect();
        let mut violations = Vec::new();

        for hyperedge in &hyperedges {
            for component_iri in &hyperedge.components {
                if !registered.contains(component_iri) {
                    violations.push(Violation {
                        severity: ViolationSeverity::Error,
                        description: format!(
                            "Hyperedge '{}' references non-existent component '{}'",
                            hyperedge.name, component_iri
                        ),
                        affected_iris: vec![hyperedge.hyperedge_id.clone(), component_iri.clone()],
                        suggestion: "Register the component skill or remove it from the hyperedge"
                            .to_string(),
                    });
                }
            }
        }

        (violations.is_empty(), violations)
    }

    fn verify_no_deprecated_prerequisites(
        &self,
        store: &SkillGraphStore,
    ) -> (bool, Vec<Violation>) {
        let skills = store.list_all_skills();
        let mut violations = Vec::new();

        for skill in &skills {
            if skill.maturity == "deprecated" {
                continue;
            }
            for link in &skill.links {
                if link.link_type != SkillLinkType::Prerequisite {
                    continue;
                }
                if let Some(target) = store.get_skill(&link.target_iri) {
                    if target.maturity == "deprecated" {
                        violations.push(Violation {
                            severity: ViolationSeverity::Warning,
                            description: format!(
                                "Skill '{}' depends on deprecated prerequisite '{}'",
                                skill.name, target.name
                            ),
                            affected_iris: vec![skill.skill_iri.clone(), link.target_iri.clone()],
                            suggestion: format!(
                                "Update '{}' to use a non-deprecated alternative",
                                skill.name
                            ),
                        });
                    }
                }
            }
        }

        (violations.is_empty(), violations)
    }

    fn verify_5w2h(&self, store: &SkillGraphStore) -> (bool, Vec<Violation>) {
        let skills = store.list_all_skills();
        let mut violations = Vec::new();

        for skill in &skills {
            let w2h = &skill.w2h;
            if w2h.what.trim().is_empty() || w2h.why.trim().is_empty() {
                violations.push(Violation {
                    severity: ViolationSeverity::Warning,
                    description: format!(
                        "Skill '{}' has incomplete 5W2H metadata (what/why)",
                        skill.name
                    ),
                    affected_iris: vec![skill.skill_iri.clone()],
                    suggestion: "Fill in the 'what' and 'why' fields in the skill's 5W2H metadata"
                        .to_string(),
                });
            }
        }

        (violations.is_empty(), violations)
    }

    fn verify_security_levels(&self, store: &SkillGraphStore) -> (bool, Vec<Violation>) {
        let skills = store.list_all_skills();
        let mut violations = Vec::new();

        for skill in &skills {
            match &skill.security_info {
                Some(info) => {
                    if info.trust_level == TrustLevel::Untrusted {
                        violations.push(Violation {
                            severity: ViolationSeverity::Warning,
                            description: format!(
                                "Skill '{}' has Untrusted security level",
                                skill.name
                            ),
                            affected_iris: vec![skill.skill_iri.clone()],
                            suggestion: "Review and increase the trust level if appropriate"
                                .to_string(),
                        });
                    }
                }
                None => {
                    violations.push(Violation {
                        severity: ViolationSeverity::Warning,
                        description: format!(
                            "Skill '{}' has no security information configured",
                            skill.name
                        ),
                        affected_iris: vec![skill.skill_iri.clone()],
                        suggestion: "Add security information with an appropriate trust level"
                            .to_string(),
                    });
                }
            }
        }

        (violations.is_empty(), violations)
    }
}

fn has_prerequisite_cycle(
    iri: &str,
    skills: &[SkillGraphNode],
    visited: &mut HashSet<String>,
    stack: &mut HashSet<String>,
) -> bool {
    if stack.contains(iri) {
        return true;
    }
    if visited.contains(iri) {
        return false;
    }
    visited.insert(iri.to_string());
    stack.insert(iri.to_string());

    if let Some(skill) = skills.iter().find(|s| s.skill_iri == iri) {
        for link in &skill.links {
            if link.link_type == SkillLinkType::Prerequisite {
                if has_prerequisite_cycle(&link.target_iri, skills, visited, stack) {
                    return true;
                }
            }
        }
    }

    stack.remove(iri);
    false
}
