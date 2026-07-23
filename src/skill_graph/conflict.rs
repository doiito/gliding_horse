use crate::core::CoreError;
use crate::skill_graph::types::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRule {
    pub rule_id: String,
    pub rule_type: ConflictRuleType,
    pub description: String,
    pub severity: ConflictSeverity,
    pub auto_resolvable: bool,
    pub resolution_strategy: ResolutionStrategy,
    pub conditions: Vec<ConflictCondition>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictRuleType {
    ResourceAccess,
    DependencyCycle,
    PermissionOverlap,
    SemanticDuplicate,
    VersionMismatch,
    TemporalOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictCondition {
    pub field: String,
    pub operator: ConditionOperator,
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
    Matches,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictReport {
    pub report_id: String,
    pub generated_at: DateTime<Utc>,
    pub total_conflicts: usize,
    pub by_severity: HashMap<String, usize>,
    pub by_type: HashMap<String, usize>,
    pub conflicts: Vec<SkillConflict>,
    pub auto_resolvable_count: usize,
    pub recommendations: Vec<String>,
}

impl ConflictReport {
    pub fn new() -> Self {
        Self {
            report_id: format!("report:{}", uuid::Uuid::new_v4()),
            generated_at: Utc::now(),
            total_conflicts: 0,
            by_severity: HashMap::new(),
            by_type: HashMap::new(),
            conflicts: Vec::new(),
            auto_resolvable_count: 0,
            recommendations: Vec::new(),
        }
    }

    pub fn add_conflict(&mut self, conflict: SkillConflict) {
        let severity_key = format!("{:?}", conflict.severity);
        let type_key = format!("{:?}", conflict.conflict_type);

        *self.by_severity.entry(severity_key).or_insert(0) += 1;
        *self.by_type.entry(type_key).or_insert(0) += 1;

        if conflict.auto_resolvable {
            self.auto_resolvable_count += 1;
        }

        self.conflicts.push(conflict);
        self.total_conflicts += 1;
    }

    pub fn generate_recommendations(&mut self) {
        if self.total_conflicts == 0 {
            self.recommendations
                .push("No conflicts detected. System is healthy.".to_string());
            return;
        }

        if let Some(&critical_count) = self.by_severity.get("Critical") {
            if critical_count > 0 {
                self.recommendations.push(format!(
                    "{} critical conflicts require immediate attention",
                    critical_count
                ));
            }
        }

        if self.auto_resolvable_count > 0 {
            self.recommendations.push(format!(
                "{} conflicts can be auto-resolved",
                self.auto_resolvable_count
            ));
        }

        if let Some(&resource_count) = self.by_type.get("Resource") {
            if resource_count > 0 {
                self.recommendations
                    .push("Consider resource partitioning to reduce conflicts".to_string());
            }
        }

        if let Some(&semantic_count) = self.by_type.get("Semantic") {
            if semantic_count > 2 {
                self.recommendations.push(
                    "Multiple semantic duplicates detected. Consider skill consolidation."
                        .to_string(),
                );
            }
        }
    }
}

pub struct ConflictDetectionEngine {
    skills: Arc<RwLock<HashMap<String, SkillGraphNode>>>,
    rules: RwLock<Vec<ConflictRule>>,
    conflict_history: RwLock<HashMap<String, Vec<SkillConflict>>>,
}

impl ConflictDetectionEngine {
    pub fn new(skills: Arc<RwLock<HashMap<String, SkillGraphNode>>>) -> Self {
        let mut rules = Vec::new();
        Self::init_default_rules(&mut rules);

        Self {
            skills,
            rules: RwLock::new(rules),
            conflict_history: RwLock::new(HashMap::new()),
        }
    }

    fn init_default_rules(rules: &mut Vec<ConflictRule>) {
        rules.push(ConflictRule {
            rule_id: "rule-resource-001".to_string(),
            rule_type: ConflictRuleType::ResourceAccess,
            description: "Skills accessing the same resource concurrently".to_string(),
            severity: ConflictSeverity::High,
            auto_resolvable: true,
            resolution_strategy: ResolutionStrategy::PreferHigherTrust,
            conditions: vec![],
        });

        rules.push(ConflictRule {
            rule_id: "rule-dependency-001".to_string(),
            rule_type: ConflictRuleType::DependencyCycle,
            description: "Circular dependency detected between skills".to_string(),
            severity: ConflictSeverity::Critical,
            auto_resolvable: false,
            resolution_strategy: ResolutionStrategy::RequireManual,
            conditions: vec![],
        });

        rules.push(ConflictRule {
            rule_id: "rule-permission-001".to_string(),
            rule_type: ConflictRuleType::PermissionOverlap,
            description: "Overlapping permissions between skills".to_string(),
            severity: ConflictSeverity::Medium,
            auto_resolvable: true,
            resolution_strategy: ResolutionStrategy::KeepBoth,
            conditions: vec![],
        });

        rules.push(ConflictRule {
            rule_id: "rule-semantic-001".to_string(),
            rule_type: ConflictRuleType::SemanticDuplicate,
            description: "Skills with similar functionality".to_string(),
            severity: ConflictSeverity::Low,
            auto_resolvable: true,
            resolution_strategy: ResolutionStrategy::PreferNewer,
            conditions: vec![],
        });
    }

    pub async fn detect_all_conflicts(&self) -> Result<ConflictReport, CoreError> {
        let mut report = ConflictReport::new();
        let skills = self.skills.read().await;

        let skill_list: Vec<&SkillGraphNode> = skills.values().collect();

        for i in 0..skill_list.len() {
            for j in (i + 1)..skill_list.len() {
                let skill_a = skill_list[i];
                let skill_b = skill_list[j];

                if let Some(conflict) = self.detect_resource_conflict(skill_a, skill_b) {
                    report.add_conflict(conflict);
                }

                if let Some(conflict) = self.detect_permission_conflict(skill_a, skill_b) {
                    report.add_conflict(conflict);
                }

                if let Some(conflict) = self.detect_semantic_conflict(skill_a, skill_b) {
                    report.add_conflict(conflict);
                }

                if let Some(conflict) = self.detect_version_conflict(skill_a, skill_b) {
                    report.add_conflict(conflict);
                }
            }
        }

        for skill in &skill_list {
            if let Some(conflict) = self.detect_dependency_cycle(skill, &skills).await {
                report.add_conflict(conflict);
            }
        }

        report.generate_recommendations();
        Ok(report)
    }

    fn detect_resource_conflict(
        &self,
        skill_a: &SkillGraphNode,
        skill_b: &SkillGraphNode,
    ) -> Option<SkillConflict> {
        let resources_a: HashSet<String> = self.extract_resources(skill_a);
        let resources_b: HashSet<String> = self.extract_resources(skill_b);

        let overlap: Vec<&String> = resources_a.intersection(&resources_b).collect();

        if overlap.is_empty() {
            return None;
        }

        let description = format!(
            "Skills '{}' and '{}' share resources: {:?}",
            skill_a.name, skill_b.name, overlap
        );

        Some(
            SkillConflict::new(
                ConflictType::Resource,
                vec![skill_a.skill_iri.clone(), skill_b.skill_iri.clone()],
                &description,
            )
            .with_severity(ConflictSeverity::High)
            .mark_auto_resolvable(),
        )
    }

    fn detect_permission_conflict(
        &self,
        skill_a: &SkillGraphNode,
        skill_b: &SkillGraphNode,
    ) -> Option<SkillConflict> {
        let perms_a = skill_a.security_info.as_ref()?;
        let perms_b = skill_b.security_info.as_ref()?;

        let mut overlapping_perms = Vec::new();

        for perm_a in &perms_a.permissions {
            for perm_b in &perms_b.permissions {
                if perm_a.action == perm_b.action
                    && perm_a.resource_pattern == perm_b.resource_pattern
                    && perm_a.resource_pattern != "*"
                {
                    overlapping_perms.push(perm_a.permission_id.clone());
                }
            }
        }

        if overlapping_perms.is_empty() {
            return None;
        }

        let description = format!(
            "Skills '{}' and '{}' have overlapping permissions: {:?}",
            skill_a.name, skill_b.name, overlapping_perms
        );

        Some(
            SkillConflict::new(
                ConflictType::Permission,
                vec![skill_a.skill_iri.clone(), skill_b.skill_iri.clone()],
                &description,
            )
            .with_severity(ConflictSeverity::Medium)
            .mark_auto_resolvable(),
        )
    }

    fn detect_semantic_conflict(
        &self,
        skill_a: &SkillGraphNode,
        skill_b: &SkillGraphNode,
    ) -> Option<SkillConflict> {
        let similarity = self.calculate_semantic_similarity(skill_a, skill_b);

        if similarity < 0.7 {
            return None;
        }

        let description = format!(
            "Skills '{}' and '{}' have high semantic similarity ({:.2}%)",
            skill_a.name,
            skill_b.name,
            similarity * 100.0
        );

        Some(
            SkillConflict::new(
                ConflictType::Semantic,
                vec![skill_a.skill_iri.clone(), skill_b.skill_iri.clone()],
                &description,
            )
            .with_severity(ConflictSeverity::Low)
            .mark_auto_resolvable(),
        )
    }

    fn detect_version_conflict(
        &self,
        skill_a: &SkillGraphNode,
        skill_b: &SkillGraphNode,
    ) -> Option<SkillConflict> {
        if skill_a.name != skill_b.name {
            return None;
        }

        if skill_a.version == skill_b.version {
            return None;
        }

        let description = format!(
            "Skills '{}' have different versions: {} vs {}",
            skill_a.name, skill_a.version, skill_b.version
        );

        Some(
            SkillConflict::new(
                ConflictType::Version,
                vec![skill_a.skill_iri.clone(), skill_b.skill_iri.clone()],
                &description,
            )
            .with_severity(ConflictSeverity::Medium)
            .mark_auto_resolvable(),
        )
    }

    async fn detect_dependency_cycle(
        &self,
        start_skill: &SkillGraphNode,
        all_skills: &HashMap<String, SkillGraphNode>,
    ) -> Option<SkillConflict> {
        let mut visited = HashSet::new();
        let mut path = Vec::new();
        let mut cycle_skills = Vec::new();

        if self.has_cycle(
            start_skill,
            all_skills,
            &mut visited,
            &mut path,
            &mut cycle_skills,
        ) {
            let description = format!("Dependency cycle detected: {}", cycle_skills.join(" -> "));

            return Some(
                SkillConflict::new(ConflictType::Dependency, cycle_skills, &description)
                    .with_severity(ConflictSeverity::Critical),
            );
        }

        None
    }

    fn has_cycle(
        &self,
        skill: &SkillGraphNode,
        all_skills: &HashMap<String, SkillGraphNode>,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
        cycle_skills: &mut Vec<String>,
    ) -> bool {
        if path.contains(&skill.skill_iri) {
            let cycle_start = path
                .iter()
                .position(|s| s == &skill.skill_iri)
                .expect("skill_iri confirmed in path by contains() check above");
            cycle_skills.extend(path[cycle_start..].iter().cloned());
            cycle_skills.push(skill.skill_iri.clone());
            return true;
        }

        if visited.contains(&skill.skill_iri) {
            return false;
        }

        visited.insert(skill.skill_iri.clone());
        path.push(skill.skill_iri.clone());

        for link in &skill.links {
            if link.link_type == SkillLinkType::Prerequisite {
                if let Some(dep_skill) = all_skills.get(&link.target_iri) {
                    if self.has_cycle(dep_skill, all_skills, visited, path, cycle_skills) {
                        return true;
                    }
                }
            }
        }

        path.pop();
        false
    }

    fn extract_resources(&self, skill: &SkillGraphNode) -> HashSet<String> {
        let mut resources = HashSet::new();

        if let Some(ref content) = skill.content {
            for step in &content.steps {
                for ref_iri in &step.references {
                    if ref_iri.starts_with("resource:") {
                        resources.insert(ref_iri.clone());
                    }
                }
            }
        }

        for link in &skill.links {
            if link.link_type == SkillLinkType::Prerequisite {
                resources.insert(format!("skill:{}", link.target_iri));
            }
        }

        resources
    }

    fn calculate_semantic_similarity(
        &self,
        skill_a: &SkillGraphNode,
        skill_b: &SkillGraphNode,
    ) -> f32 {
        let mut score = 0.0f32;
        let mut weight = 0.0f32;

        let name_sim = self.string_similarity(&skill_a.name, &skill_b.name);
        score += name_sim * 0.3;
        weight += 0.3;

        let desc_sim = self.string_similarity(&skill_a.description, &skill_b.description);
        score += desc_sim * 0.2;
        weight += 0.2;

        let tags_a: HashSet<&str> = skill_a.tags.iter().map(|s| s.as_str()).collect();
        let tags_b: HashSet<&str> = skill_b.tags.iter().map(|s| s.as_str()).collect();
        let tag_intersection = tags_a.intersection(&tags_b).count();
        let tag_union = tags_a.union(&tags_b).count();
        let tag_sim = if tag_union > 0 {
            tag_intersection as f32 / tag_union as f32
        } else {
            0.0
        };
        score += tag_sim * 0.3;
        weight += 0.3;

        let what_sim = self.string_similarity(&skill_a.w2h.what, &skill_b.w2h.what);
        score += what_sim * 0.2;
        weight += 0.2;

        score / weight
    }

    fn string_similarity(&self, a: &str, b: &str) -> f32 {
        if a == b {
            return 1.0;
        }

        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        let words_a: HashSet<&str> = a_lower.split_whitespace().collect();
        let words_b: HashSet<&str> = b_lower.split_whitespace().collect();

        if words_a.is_empty() && words_b.is_empty() {
            return 1.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }

    pub async fn resolve_conflict(
        &self,
        conflict: &SkillConflict,
        strategy: ResolutionStrategy,
    ) -> Result<ConflictResolution, CoreError> {
        let skills = self.skills.read().await;

        let resolution = match strategy {
            ResolutionStrategy::PreferHigherTrust => {
                self.resolve_by_trust(&conflict.skill_iris, &skills)?
            }
            ResolutionStrategy::PreferNewer => {
                self.resolve_by_version(&conflict.skill_iris, &skills)?
            }
            ResolutionStrategy::PreferSystem => {
                self.resolve_by_source(&conflict.skill_iris, &skills)?
            }
            ResolutionStrategy::KeepBoth => ConflictResolution::new(
                ResolutionStrategy::KeepBoth,
                "Both skills retained with namespace separation",
            ),
            ResolutionStrategy::Merge => ConflictResolution::new(
                ResolutionStrategy::Merge,
                "Skills will be merged into a composite skill",
            ),
            ResolutionStrategy::RemoveConflict => ConflictResolution::new(
                ResolutionStrategy::RemoveConflict,
                "Conflict source will be removed",
            ),
            ResolutionStrategy::RequireManual => {
                return Err(CoreError::ValidationFailed {
                    message: "This conflict requires manual resolution".to_string(),
                });
            }
        };

        {
            let mut history = self.conflict_history.write().await;
            for skill_iri in &conflict.skill_iris {
                history
                    .entry(skill_iri.clone())
                    .or_insert_with(Vec::new)
                    .push(conflict.clone());
            }
        }

        Ok(resolution)
    }

    fn resolve_by_trust(
        &self,
        skill_iris: &[String],
        skills: &HashMap<String, SkillGraphNode>,
    ) -> Result<ConflictResolution, CoreError> {
        let mut best_skill: Option<&SkillGraphNode> = None;
        let mut best_trust = TrustLevel::Untrusted;

        for iri in skill_iris {
            if let Some(skill) = skills.get(iri) {
                let trust = skill.get_trust_level();
                if trust as u8 > best_trust as u8 {
                    best_trust = trust;
                    best_skill = Some(skill);
                }
            }
        }

        match best_skill {
            Some(skill) => Ok(ConflictResolution::new(
                ResolutionStrategy::PreferHigherTrust,
                &format!(
                    "Selected '{}' with trust level {:?}",
                    skill.name, best_trust
                ),
            )),
            None => Err(CoreError::ValidationFailed {
                message: "No skills found for resolution".to_string(),
            }),
        }
    }

    fn resolve_by_version(
        &self,
        skill_iris: &[String],
        skills: &HashMap<String, SkillGraphNode>,
    ) -> Result<ConflictResolution, CoreError> {
        let mut best_skill: Option<&SkillGraphNode> = None;
        let mut best_version = "0.0.0".to_string();

        for iri in skill_iris {
            if let Some(skill) = skills.get(iri) {
                if self.compare_versions(&skill.version, &best_version) {
                    best_version = skill.version.clone();
                    best_skill = Some(skill);
                }
            }
        }

        match best_skill {
            Some(skill) => Ok(ConflictResolution::new(
                ResolutionStrategy::PreferNewer,
                &format!("Selected '{}' version {}", skill.name, skill.version),
            )),
            None => Err(CoreError::ValidationFailed {
                message: "No skills found for resolution".to_string(),
            }),
        }
    }

    fn resolve_by_source(
        &self,
        skill_iris: &[String],
        skills: &HashMap<String, SkillGraphNode>,
    ) -> Result<ConflictResolution, CoreError> {
        let mut best_skill: Option<&SkillGraphNode> = None;

        for iri in skill_iris {
            if let Some(skill) = skills.get(iri) {
                if let Some(ref security) = skill.security_info {
                    if matches!(security.source, SkillSource::SystemBuiltin) {
                        best_skill = Some(skill);
                        break;
                    }
                }
            }
        }

        match best_skill {
            Some(skill) => Ok(ConflictResolution::new(
                ResolutionStrategy::PreferSystem,
                &format!("Selected system skill '{}'", skill.name),
            )),
            None => Ok(ConflictResolution::new(
                ResolutionStrategy::PreferSystem,
                "No system skill found, using first available",
            )),
        }
    }

    fn compare_versions(&self, a: &str, b: &str) -> bool {
        let parse_version =
            |v: &str| -> Vec<u32> { v.split('.').filter_map(|s| s.parse().ok()).collect() };

        let va = parse_version(a);
        let vb = parse_version(b);

        for i in 0..va.len().max(vb.len()) {
            let na = va.get(i).unwrap_or(&0);
            let nb = vb.get(i).unwrap_or(&0);
            if na > nb {
                return true;
            }
            if na < nb {
                return false;
            }
        }
        false
    }

    pub async fn add_rule(&self, rule: ConflictRule) {
        let mut rules = self.rules.write().await;
        rules.push(rule);
    }

    pub async fn get_rules(&self) -> Vec<ConflictRule> {
        let rules = self.rules.read().await;
        rules.clone()
    }

    pub async fn get_conflict_history(&self, skill_iri: &str) -> Option<Vec<SkillConflict>> {
        let history = self.conflict_history.read().await;
        history.get(skill_iri).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_skill(iri: &str, name: &str) -> SkillGraphNode {
        SkillGraphNode::new(iri, name, &format!("Description for {}", name))
    }

    #[test]
    fn test_conflict_report() {
        let mut report = ConflictReport::new();

        let conflict = SkillConflict::new(
            ConflictType::Resource,
            vec!["iri://skills/a".to_string()],
            "Test conflict",
        );

        report.add_conflict(conflict);
        report.generate_recommendations();

        assert_eq!(report.total_conflicts, 1);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_conflict_rule() {
        let rule = ConflictRule {
            rule_id: "test-rule".to_string(),
            rule_type: ConflictRuleType::ResourceAccess,
            description: "Test rule".to_string(),
            severity: ConflictSeverity::High,
            auto_resolvable: true,
            resolution_strategy: ResolutionStrategy::PreferHigherTrust,
            conditions: vec![],
        };

        assert!(rule.auto_resolvable);
        assert_eq!(rule.severity, ConflictSeverity::High);
    }

    #[test]
    fn test_string_similarity() {
        let engine = ConflictDetectionEngine::new(Arc::new(RwLock::new(HashMap::new())));

        let sim = engine.string_similarity("hello world", "hello world");
        assert_eq!(sim, 1.0);

        let sim = engine.string_similarity("hello world", "hello");
        assert!(sim >= 0.5);

        let sim = engine.string_similarity("hello", "goodbye");
        assert!(sim < 0.5);
    }

    #[test]
    fn test_compare_versions() {
        let engine = ConflictDetectionEngine::new(Arc::new(RwLock::new(HashMap::new())));

        assert!(engine.compare_versions("2.0.0", "1.0.0"));
        assert!(!engine.compare_versions("1.0.0", "2.0.0"));
        assert!(engine.compare_versions("1.1.0", "1.0.0"));
        assert!(engine.compare_versions("1.0.1", "1.0.0"));
    }

    #[tokio::test]
    async fn test_detect_semantic_conflict() {
        let mut skills = HashMap::new();

        let skill_a = SkillGraphNode::new(
            "iri://skills/jwt-auth",
            "JWT Authentication",
            "Implement JWT authentication in Rust",
        )
        .with_tag("authentication");

        let skill_b = SkillGraphNode::new(
            "iri://skills/jwt-auth-v2",
            "JWT Authentication",
            "Implement JWT authentication in Rust",
        )
        .with_tag("authentication");

        skills.insert(skill_a.skill_iri.clone(), skill_a);
        skills.insert(skill_b.skill_iri.clone(), skill_b);

        let engine = ConflictDetectionEngine::new(Arc::new(RwLock::new(skills)));
        let report = engine.detect_all_conflicts().await.unwrap();

        assert!(report.total_conflicts > 0);
    }

    #[tokio::test]
    async fn test_no_conflict() {
        let mut skills = HashMap::new();

        let skill_a =
            SkillGraphNode::new("iri://skills/auth", "Authentication", "User authentication");

        let skill_b = SkillGraphNode::new("iri://skills/logging", "Logging", "Application logging");

        skills.insert(skill_a.skill_iri.clone(), skill_a);
        skills.insert(skill_b.skill_iri.clone(), skill_b);

        let engine = ConflictDetectionEngine::new(Arc::new(RwLock::new(skills)));
        let report = engine.detect_all_conflicts().await.unwrap();

        assert_eq!(report.total_conflicts, 0);
    }
}
