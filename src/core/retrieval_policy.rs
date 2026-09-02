//! Whitelisted, side-effect-free retrieval policy arms.
//!
//! These arms may alter only the order of already retrieved, bounded context
//! hints. They cannot change a model, an index, a skill graph or any tool/effect
//! permission, so they form the safe candidate set for `ConstrainedPolicy`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalPolicyArm {
    Baseline,
    ExperienceFirst,
    KnowledgeFirst,
    SkillFirst,
}

impl RetrievalPolicyArm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::ExperienceFirst => "experience_first",
            Self::KnowledgeFirst => "knowledge_first",
            Self::SkillFirst => "skill_first",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "baseline" => Some(Self::Baseline),
            "experience_first" => Some(Self::ExperienceFirst),
            "knowledge_first" => Some(Self::KnowledgeFirst),
            "skill_first" => Some(Self::SkillFirst),
            _ => None,
        }
    }

    /// Build the complete safe candidate set available for the current task.
    pub fn eligible(
        skill_count: usize,
        knowledge_count: usize,
        experience_count: usize,
    ) -> Vec<Self> {
        let mut arms = vec![Self::Baseline];
        if knowledge_count > 0 {
            arms.push(Self::KnowledgeFirst);
        }
        if experience_count > 0 {
            arms.push(Self::ExperienceFirst);
        }
        if skill_count > 0 {
            arms.push(Self::SkillFirst);
        }
        arms
    }

    pub fn candidate_names(
        skill_count: usize,
        knowledge_count: usize,
        experience_count: usize,
    ) -> Vec<String> {
        Self::eligible(skill_count, knowledge_count, experience_count)
            .into_iter()
            .map(|arm| arm.as_str().to_string())
            .collect()
    }

    /// Materialize the arm's source ordering. Baseline is an actual ablation
    /// and returns no durable history at all.
    pub fn order_hints(
        self,
        experience: &[String],
        skills: &[String],
        knowledge: &[String],
    ) -> Vec<String> {
        let mut hints = Vec::with_capacity(experience.len() + skills.len() + knowledge.len());
        match self {
            Self::Baseline => {}
            Self::ExperienceFirst => {
                hints.extend_from_slice(experience);
                hints.extend_from_slice(knowledge);
                hints.extend_from_slice(skills);
            }
            Self::KnowledgeFirst => {
                hints.extend_from_slice(knowledge);
                hints.extend_from_slice(experience);
                hints.extend_from_slice(skills);
            }
            Self::SkillFirst => {
                hints.extend_from_slice(skills);
                hints.extend_from_slice(experience);
                hints.extend_from_slice(knowledge);
            }
        }
        hints
    }
}

impl std::fmt::Display for RetrievalPolicyArm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_catalog_never_exposes_an_unbounded_arm() {
        assert_eq!(
            RetrievalPolicyArm::candidate_names(0, 0, 0),
            vec!["baseline"]
        );
        assert_eq!(
            RetrievalPolicyArm::candidate_names(1, 1, 1),
            vec![
                "baseline",
                "knowledge_first",
                "experience_first",
                "skill_first"
            ]
        );
        assert!(RetrievalPolicyArm::parse("change_hnsw").is_none());
    }

    #[test]
    fn baseline_is_a_true_history_ablation() {
        let experience = vec!["experience".into()];
        let skills = vec!["skill".into()];
        let knowledge = vec!["knowledge".into()];
        assert!(RetrievalPolicyArm::Baseline
            .order_hints(&experience, &skills, &knowledge)
            .is_empty());
        assert_eq!(
            RetrievalPolicyArm::KnowledgeFirst.order_hints(&experience, &skills, &knowledge),
            vec!["knowledge", "experience", "skill"]
        );
    }
}
