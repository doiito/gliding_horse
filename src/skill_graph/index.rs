use std::collections::{HashMap, HashSet};

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::skill_graph::types::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub skill_iri: String,
    pub name: String,
    pub node_type: SkillNodeType,
    pub maturity: String,
    pub tags: Vec<String>,
    pub what: String,
    pub why: String,
    pub stack: Vec<String>,
    pub role: Option<String>,
    pub success_rate: f32,
    pub avg_token_cost: u32,
}

impl IndexEntry {
    pub fn from_skill(skill: &SkillGraphNode) -> Self {
        Self {
            skill_iri: skill.skill_iri.clone(),
            name: skill.name.clone(),
            node_type: skill.node_type,
            maturity: skill.maturity.clone(),
            tags: skill.tags.clone(),
            what: skill.w2h.what.clone(),
            why: skill.w2h.why.clone(),
            stack: skill.w2h.where_.target_stack.clone(),
            role: skill.w2h.who.required_agent_role.clone(),
            success_rate: skill.graph_meta.success_rate,
            avg_token_cost: skill.w2h.how_much.avg_token_cost,
        }
    }

    pub fn matches_tag(&self, tag: &str) -> bool {
        self.tags
            .iter()
            .any(|t| t.to_lowercase() == tag.to_lowercase())
    }

    pub fn matches_stack(&self, stack: &str) -> bool {
        self.stack
            .iter()
            .any(|s| s.to_lowercase() == stack.to_lowercase())
    }

    pub fn matches_role(&self, role: &str) -> bool {
        self.role
            .as_ref()
            .map_or(false, |r| r.to_lowercase() == role.to_lowercase())
    }

    pub fn matches_maturity(&self, allowed: &[&str]) -> bool {
        allowed.iter().any(|m| *m == self.maturity)
    }
}

pub struct PreAggregatedIndex {
    tag_index: DashMap<String, Vec<String>>,
    stack_index: DashMap<String, Vec<String>>,
    role_index: DashMap<String, Vec<String>>,
    transitive_deps: RwLock<HashMap<String, HashSet<String>>>,
    skill_summaries: DashMap<String, IndexEntry>,
}

impl PreAggregatedIndex {
    pub fn new() -> Self {
        Self {
            tag_index: DashMap::new(),
            stack_index: DashMap::new(),
            role_index: DashMap::new(),
            transitive_deps: RwLock::new(HashMap::new()),
            skill_summaries: DashMap::new(),
        }
    }

    pub fn index_skill(&self, skill: &SkillGraphNode) {
        let entry = IndexEntry::from_skill(skill);
        let iri = skill.skill_iri.clone();

        self.skill_summaries.insert(iri.clone(), entry);

        for tag in &skill.tags {
            let tag_key = tag.to_lowercase();
            self.tag_index
                .entry(tag_key)
                .or_insert_with(Vec::new)
                .push(iri.clone());
        }

        for stack in &skill.w2h.where_.target_stack {
            let stack_key = stack.to_lowercase();
            self.stack_index
                .entry(stack_key)
                .or_insert_with(Vec::new)
                .push(iri.clone());
        }

        if let Some(ref role) = skill.w2h.who.required_agent_role {
            let role_key = role.to_lowercase();
            self.role_index
                .entry(role_key)
                .or_insert_with(Vec::new)
                .push(iri.clone());
        }

        self.compute_transitive_deps(&skill.skill_iri, &skill.links);
        debug!(
            "Indexing skill: {} (tags={}, stack={}, role={:?})",
            iri,
            skill.tags.len(),
            skill.w2h.where_.target_stack.len(),
            skill.w2h.who.required_agent_role
        );
    }

    pub fn remove_skill(&self, skill_iri: &str) {
        if let Some((_, entry)) = self.skill_summaries.remove(skill_iri) {
            for tag in &entry.tags {
                if let Some(mut list) = self.tag_index.get_mut(&tag.to_lowercase()) {
                    list.retain(|iri| iri != skill_iri);
                }
            }
            for stack in &entry.stack {
                if let Some(mut list) = self.stack_index.get_mut(&stack.to_lowercase()) {
                    list.retain(|iri| iri != skill_iri);
                }
            }
            if let Some(ref role) = entry.role {
                if let Some(mut list) = self.role_index.get_mut(&role.to_lowercase()) {
                    list.retain(|iri| iri != skill_iri);
                }
            }
        }

        self.transitive_deps.write().remove(skill_iri);
    }

    pub fn update_skill(&self, skill: &SkillGraphNode) {
        self.remove_skill(&skill.skill_iri);
        self.index_skill(skill);
    }

    pub fn find_by_tag(&self, tag: &str) -> Vec<String> {
        self.tag_index
            .get(&tag.to_lowercase())
            .map(|v| v.value().clone())
            .unwrap_or_default()
    }

    pub fn find_by_stack(&self, stack: &str) -> Vec<String> {
        self.stack_index
            .get(&stack.to_lowercase())
            .map(|v| v.value().clone())
            .unwrap_or_default()
    }

    pub fn find_by_role(&self, role: &str) -> Vec<String> {
        self.role_index
            .get(&role.to_lowercase())
            .map(|v| v.value().clone())
            .unwrap_or_default()
    }

    pub fn get_transitive_deps(&self, skill_iri: &str) -> HashSet<String> {
        self.transitive_deps
            .read()
            .get(skill_iri)
            .cloned()
            .unwrap_or_default()
    }

    pub fn get_summary(&self, skill_iri: &str) -> Option<IndexEntry> {
        self.skill_summaries
            .get(skill_iri)
            .map(|e| e.value().clone())
    }

    pub fn find_by_tags_intersection(&self, tags: &[&str]) -> Vec<String> {
        if tags.is_empty() {
            return Vec::new();
        }

        let mut result: Option<HashSet<String>> = None;
        for tag in tags {
            let skill_iris: HashSet<String> = self.find_by_tag(tag).into_iter().collect();
            result = Some(match result {
                Some(prev) => prev.intersection(&skill_iris).cloned().collect(),
                None => skill_iris,
            });
        }
        result.map(|s| s.into_iter().collect()).unwrap_or_default()
    }

    pub fn find_by_criteria(
        &self,
        tags: &[&str],
        stack: Option<&str>,
        role: Option<&str>,
        min_success_rate: Option<f32>,
        allowed_maturities: &[&str],
    ) -> Vec<String> {
        let mut candidates: Option<HashSet<String>> = None;

        if !tags.is_empty() {
            let tag_results: HashSet<String> =
                self.find_by_tags_intersection(tags).into_iter().collect();
            candidates = Some(tag_results);
        }

        if let Some(s) = stack {
            let stack_results: HashSet<String> = self.find_by_stack(s).into_iter().collect();
            candidates = Some(match candidates {
                Some(prev) => prev.intersection(&stack_results).cloned().collect(),
                None => stack_results,
            });
        }

        if let Some(r) = role {
            let role_results: HashSet<String> = self.find_by_role(r).into_iter().collect();
            candidates = Some(match candidates {
                Some(prev) => prev.intersection(&role_results).cloned().collect(),
                None => role_results,
            });
        }

        let candidates = candidates.unwrap_or_else(|| {
            self.skill_summaries
                .iter()
                .map(|e| e.key().clone())
                .collect()
        });

        candidates
            .into_iter()
            .filter(|iri| {
                if let Some(entry) = self.skill_summaries.get(iri) {
                    if let Some(min_rate) = min_success_rate {
                        if entry.success_rate < min_rate {
                            return false;
                        }
                    }
                    if !allowed_maturities.is_empty() && !entry.matches_maturity(allowed_maturities)
                    {
                        return false;
                    }
                    true
                } else {
                    false
                }
            })
            .collect()
    }

    fn compute_transitive_deps(&self, skill_iri: &str, links: &[SkillLink]) {
        let mut deps = HashSet::new();
        for link in links {
            if link.link_type == SkillLinkType::Prerequisite {
                deps.insert(link.target_iri.clone());

                if let Some(transitive) = self.transitive_deps.read().get(&link.target_iri) {
                    for dep in transitive {
                        deps.insert(dep.clone());
                    }
                }
            }
        }

        if !deps.is_empty() {
            self.transitive_deps
                .write()
                .insert(skill_iri.to_string(), deps);
        }

        let affected: Vec<String> = self
            .transitive_deps
            .read()
            .iter()
            .filter(|(_, deps)| deps.contains(skill_iri))
            .map(|(iri, _)| iri.clone())
            .collect();

        if !affected.is_empty() {
            let new_deps: HashSet<String> = self
                .transitive_deps
                .read()
                .get(skill_iri)
                .cloned()
                .unwrap_or_default();

            let mut transitive = self.transitive_deps.write();
            for affected_iri in &affected {
                if let Some(existing) = transitive.get_mut(affected_iri) {
                    for dep in &new_deps {
                        existing.insert(dep.clone());
                    }
                }
            }
        }
    }

    pub fn stats(&self) -> IndexStats {
        IndexStats {
            total_skills: self.skill_summaries.len(),
            tag_count: self.tag_index.len(),
            stack_count: self.stack_index.len(),
            role_count: self.role_index.len(),
            transitive_dep_entries: self.transitive_deps.read().len(),
        }
    }
}

impl Default for PreAggregatedIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_skills: usize,
    pub tag_count: usize,
    pub stack_count: usize,
    pub role_count: usize,
    pub transitive_dep_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_skill(
        iri: &str,
        name: &str,
        tags: Vec<&str>,
        stack: Vec<&str>,
        role: Option<&str>,
    ) -> SkillGraphNode {
        let mut skill =
            SkillGraphNode::new(iri, name, &format!("{} description", name)).with_5w2h(Skill5W2H {
                what: name.to_string(),
                why: format!("Why {}", name),
                who: SkillRole {
                    role_name: "Test".to_string(),
                    required_agent_role: role.map(|r| r.to_string()),
                },
                when: SkillTrigger {
                    applicable_phases: vec!["Do".to_string()],
                    trigger_condition: None,
                    deadline_constraint: None,
                },
                where_: SkillContext {
                    target_stack: stack.iter().map(|s| s.to_string()).collect(),
                    repo_pattern: None,
                },
                how: SkillApproach {
                    approach: "Test".to_string(),
                    plan_iri: None,
                },
                how_much: SkillCost {
                    avg_token_cost: 500,
                    avg_duration_seconds: 5,
                    max_sub_agents: 1,
                },
            });
        for tag in tags {
            skill = skill.with_tag(tag);
        }
        skill
    }

    #[test]
    fn test_tag_index() {
        let index = PreAggregatedIndex::new();
        let skill = create_test_skill(
            "iri://skills/jwt",
            "JWT Auth",
            vec!["auth", "jwt", "rust"],
            vec!["rust"],
            Some("DA"),
        );
        index.index_skill(&skill);

        let result = index.find_by_tag("auth");
        assert_eq!(result.len(), 1);

        let result = index.find_by_tag("unknown");
        assert!(result.is_empty());
    }

    #[test]
    fn test_stack_index() {
        let index = PreAggregatedIndex::new();
        let skill = create_test_skill(
            "iri://skills/jwt",
            "JWT Auth",
            vec!["auth"],
            vec!["rust", "axum"],
            Some("DA"),
        );
        index.index_skill(&skill);

        let result = index.find_by_stack("rust");
        assert_eq!(result.len(), 1);

        let result = index.find_by_stack("axum");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_role_index() {
        let index = PreAggregatedIndex::new();
        let skill = create_test_skill(
            "iri://skills/jwt",
            "JWT Auth",
            vec!["auth"],
            vec!["rust"],
            Some("DA"),
        );
        index.index_skill(&skill);

        let result = index.find_by_role("DA");
        assert_eq!(result.len(), 1);

        let result = index.find_by_role("PA");
        assert!(result.is_empty());
    }

    #[test]
    fn test_tags_intersection() {
        let index = PreAggregatedIndex::new();
        let s1 = create_test_skill("iri://skills/s1", "S1", vec!["auth", "jwt"], vec![], None);
        let s2 = create_test_skill("iri://skills/s2", "S2", vec!["auth", "oauth"], vec![], None);
        let s3 = create_test_skill("iri://skills/s3", "S3", vec!["jwt"], vec![], None);
        index.index_skill(&s1);
        index.index_skill(&s2);
        index.index_skill(&s3);

        let result = index.find_by_tags_intersection(&["auth"]);
        assert_eq!(result.len(), 2);

        let result = index.find_by_tags_intersection(&["auth", "jwt"]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_find_by_criteria() {
        let index = PreAggregatedIndex::new();
        let s1 = create_test_skill(
            "iri://skills/s1",
            "S1",
            vec!["auth"],
            vec!["rust"],
            Some("DA"),
        );
        let s2 = create_test_skill(
            "iri://skills/s2",
            "S2",
            vec!["auth"],
            vec!["python"],
            Some("PA"),
        );
        index.index_skill(&s1);
        index.index_skill(&s2);

        let result = index.find_by_criteria(&["auth"], Some("rust"), None, None, &[]);
        assert_eq!(result.len(), 1);

        let result = index.find_by_criteria(&["auth"], None, Some("PA"), None, &[]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_remove_skill() {
        let index = PreAggregatedIndex::new();
        let skill = create_test_skill(
            "iri://skills/jwt",
            "JWT Auth",
            vec!["auth"],
            vec!["rust"],
            Some("DA"),
        );
        index.index_skill(&skill);

        assert_eq!(index.find_by_tag("auth").len(), 1);
        index.remove_skill("iri://skills/jwt");
        assert!(index.find_by_tag("auth").is_empty());
    }

    #[test]
    fn test_transitive_deps() {
        let index = PreAggregatedIndex::new();
        let s1 = create_test_skill("iri://skills/s1", "S1", vec![], vec![], None);
        let mut s2 = create_test_skill("iri://skills/s2", "S2", vec![], vec![], None);
        s2.add_link(SkillLink {
            link_type: SkillLinkType::Prerequisite,
            target_iri: "iri://skills/s1".to_string(),
            strength: LinkStrength::Required,
            description: "Requires S1".to_string(),
        });
        let mut s3 = create_test_skill("iri://skills/s3", "S3", vec![], vec![], None);
        s3.add_link(SkillLink {
            link_type: SkillLinkType::Prerequisite,
            target_iri: "iri://skills/s2".to_string(),
            strength: LinkStrength::Required,
            description: "Requires S2".to_string(),
        });

        index.index_skill(&s1);
        index.index_skill(&s2);
        index.index_skill(&s3);

        let deps = index.get_transitive_deps("iri://skills/s3");
        assert!(deps.contains("iri://skills/s1"));
        assert!(deps.contains("iri://skills/s2"));

        let deps = index.get_transitive_deps("iri://skills/s2");
        assert!(deps.contains("iri://skills/s1"));
    }

    #[test]
    fn test_index_stats() {
        let index = PreAggregatedIndex::new();
        let skill = create_test_skill(
            "iri://skills/jwt",
            "JWT Auth",
            vec!["auth", "jwt"],
            vec!["rust"],
            Some("DA"),
        );
        index.index_skill(&skill);

        let stats = index.stats();
        assert_eq!(stats.total_skills, 1);
        assert_eq!(stats.tag_count, 2);
        assert_eq!(stats.stack_count, 1);
        assert_eq!(stats.role_count, 1);
    }
}
