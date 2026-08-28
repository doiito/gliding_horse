use std::collections::HashSet;
use std::sync::Arc;

use serde_json::Value;
use tracing::{debug, info};

use crate::memory::hyperspace_store::HyperspaceStore;
use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::types::*;
use crate::CoreError;

#[derive(Debug, Clone, Default)]
pub struct Task5W2H {
    pub what: String,
    pub why: String,
    pub who: Option<String>,
    pub when_phase: Option<String>,
    pub where_context: Option<String>,
    pub how_approach: Option<String>,
    pub constraints: Vec<String>,
}

impl Task5W2H {
    pub fn new(what: &str, why: &str) -> Self {
        Self {
            what: what.to_string(),
            why: why.to_string(),
            ..Default::default()
        }
    }

    pub fn with_phase(mut self, phase: &str) -> Self {
        self.when_phase = Some(phase.to_string());
        self
    }

    pub fn with_agent_role(mut self, role: &str) -> Self {
        self.who = Some(role.to_string());
        self
    }

    pub fn with_constraint(mut self, constraint: &str) -> Self {
        self.constraints.push(constraint.to_string());
        self
    }
}

#[derive(Debug, Clone)]
pub struct SkillMatch {
    pub skill: SkillGraphNode,
    pub relevance_score: f32,
    pub match_reasons: Vec<String>,
    pub required_dependencies: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SkillConflict {
    pub skill_a: String,
    pub skill_b: String,
    pub conflict_type: String,
    pub description: String,
}

pub struct SkillDiscoveryEngine {
    graph_store: Arc<SkillGraphStore>,
    vector_store: Option<Arc<HyperspaceStore>>,
}

impl SkillDiscoveryEngine {
    fn lexical_terms(text: &str) -> HashSet<String> {
        const STOP_WORDS: &[&str] = &[
            "and", "the", "for", "with", "from", "into", "that", "this", "task", "current",
            "using", "only", "without", "return", "complete",
        ];
        text.to_lowercase()
            .split(|character: char| !character.is_alphanumeric())
            .filter(|term| term.chars().count() >= 3)
            .filter(|term| !STOP_WORDS.contains(term))
            .map(str::to_string)
            .collect()
    }

    /// Deterministic retrieval fallback for installations without a semantic
    /// embedding provider. It requires multiple shared distinctive terms and
    /// never turns the hash-vector availability fallback into relevance.
    fn lexical_relevance(task: &Task5W2H, skill: &SkillGraphNode) -> Option<f32> {
        if let (Some(task_role), Some(skill_role)) = (
            task.who.as_deref(),
            skill.w2h.who.required_agent_role.as_deref(),
        ) {
            let explicitly_allowed = skill.tags.iter().any(|tag| {
                tag.strip_prefix("allowed-role:")
                    .is_some_and(|role| role.eq_ignore_ascii_case(task_role))
            });
            if !task_role.eq_ignore_ascii_case(skill_role) && !explicitly_allowed {
                return None;
            }
        }
        if let Some(phase) = task.when_phase.as_deref() {
            // Task5W2H.when is often a temporal value (Immediate/Scheduled),
            // whereas skill phases are BizAgent phases. Enforce only when the
            // caller supplied a real business phase; comparing different
            // ontologies silently hid otherwise relevant skills.
            let business_phase = matches!(
                phase.to_ascii_lowercase().as_str(),
                "plan" | "pa" | "do" | "da" | "check" | "ca" | "act" | "aa"
            );
            if business_phase
                && !skill.w2h.when.applicable_phases.is_empty()
                && !skill
                    .w2h
                    .when
                    .applicable_phases
                    .iter()
                    .any(|candidate| candidate.eq_ignore_ascii_case(phase))
            {
                return None;
            }
        }
        let task_text = [
            task.what.as_str(),
            task.why.as_str(),
            task.how_approach.as_deref().unwrap_or_default(),
        ]
        .join(" ");
        let mut skill_parts = vec![
            skill.name.as_str(),
            skill.description.as_str(),
            skill.w2h.what.as_str(),
            skill.w2h.why.as_str(),
            skill.w2h.how.approach.as_str(),
        ];
        skill_parts.extend(skill.tags.iter().map(String::as_str));
        let task_terms = Self::lexical_terms(&task_text);
        let skill_terms = Self::lexical_terms(&skill_parts.join(" "));
        let shared = task_terms.intersection(&skill_terms).count();
        if shared < 2 {
            return None;
        }
        let smaller = task_terms.len().min(skill_terms.len()).max(1);
        let coverage = shared as f32 / smaller as f32;
        (coverage >= 0.15).then(|| (0.55 + coverage.min(1.0) * 0.25).min(0.8))
    }

    pub fn new(graph_store: Arc<SkillGraphStore>) -> Self {
        Self {
            graph_store,
            vector_store: None,
        }
    }

    pub fn with_vector_store(mut self, vector_store: Arc<HyperspaceStore>) -> Self {
        self.vector_store = Some(vector_store);
        self
    }

    /// Index one skill for semantic discovery.
    ///
    /// This is deliberately explicit and async: embedding can fail and callers
    /// must decide whether a graph write should proceed without semantic
    /// retrieval, rather than hiding a detached background task in
    /// `SkillGraphStore::register_skill`.
    pub async fn index_skill(&self, skill: &SkillGraphNode) -> Result<(), CoreError> {
        let Some(vector_store) = &self.vector_store else {
            return Ok(());
        };

        let mut parts = vec![
            skill.name.clone(),
            skill.description.clone(),
            skill.w2h.what.clone(),
            skill.w2h.why.clone(),
            skill.w2h.how.approach.clone(),
        ];
        if let Some(content) = &skill.content {
            parts.push(content.summary.clone());
            parts.extend(content.steps.iter().map(|step| step.action.clone()));
        }
        let text = parts
            .into_iter()
            .filter(|part| !part.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        vector_store
            .upsert_with_metadata(
                &skill.skill_iri,
                &text,
                &skill.tags,
                Some(skill.graph_meta.success_rate),
                Some(&vec!["skill:Skill".to_string()]),
                Some("graph:skills"),
            )
            .await
            .map(|_| ())
    }

    /// Rebuild the semantic skill index from the authoritative graph state.
    /// Existing IRIs are upserted, so this is safe to run during startup and
    /// also repairs indexes created before skill-vector synchronization.
    pub async fn index_all_skills(&self) -> Result<usize, CoreError> {
        let skills = self.graph_store.list_all_skills();
        for skill in &skills {
            self.index_skill(skill).await?;
        }
        Ok(skills.len())
    }

    pub async fn discover_for_task(&self, task: &Task5W2H) -> Vec<SkillMatch> {
        info!("Discovering skills for task: what={}", task.what);

        let mut seen_iris = HashSet::new();
        let mut matches: Vec<SkillMatch> = Vec::new();

        // Phase 1: keyword-based 5W2H matching
        let keyword_matches = self.graph_store.find_skills_by_5w2h(
            Some(&task.what),
            Some(&task.why),
            task.when_phase.as_deref(),
            task.who.as_deref(),
            None,
        );

        for skill in keyword_matches {
            let mut match_reasons = Vec::new();
            let mut score = 0.5f32;

            if skill
                .w2h
                .what
                .to_lowercase()
                .contains(&task.what.to_lowercase())
            {
                match_reasons.push("what matched".to_string());
                score += 0.2;
            }

            if skill
                .w2h
                .why
                .to_lowercase()
                .contains(&task.why.to_lowercase())
            {
                match_reasons.push("why matched".to_string());
                score += 0.15;
            }

            if let Some(ref phase) = task.when_phase {
                if skill
                    .w2h
                    .when
                    .applicable_phases
                    .iter()
                    .any(|p| p.to_lowercase() == phase.to_lowercase())
                {
                    match_reasons.push(format!("phase matched: {}", phase));
                    score += 0.1;
                }
            }

            if let Some(ref role) = task.who {
                if let Some(ref required_role) = skill.w2h.who.required_agent_role {
                    if required_role.to_lowercase() == role.to_lowercase() {
                        match_reasons.push(format!("role matched: {}", role));
                        score += 0.1;
                    }
                }
            }

            score = score.min(1.0);

            let deps = self.graph_store.resolve_dependencies(&skill.skill_iri);
            let required_deps: Vec<String> = deps
                .iter()
                .filter(|d| *d != &skill.skill_iri)
                .cloned()
                .collect();

            seen_iris.insert(skill.skill_iri.clone());
            matches.push(SkillMatch {
                skill,
                relevance_score: score,
                match_reasons,
                required_dependencies: required_deps,
            });
        }

        // Phase 1b: safe lexical overlap. The historical 5W2H query requires
        // the skill text to contain the complete task sentence, which is too
        // strict for accumulated/generalized skills. Multiple-term overlap
        // gives deterministic retrieval when only fallback embeddings exist,
        // while the threshold keeps unrelated skills out.
        for skill in self.graph_store.list_all_skills() {
            if seen_iris.contains(&skill.skill_iri) {
                continue;
            }
            let Some(score) = Self::lexical_relevance(task, &skill) else {
                continue;
            };
            seen_iris.insert(skill.skill_iri.clone());
            let required_dependencies = self
                .graph_store
                .resolve_dependencies(&skill.skill_iri)
                .into_iter()
                .filter(|dependency| dependency != &skill.skill_iri)
                .collect();
            matches.push(SkillMatch {
                skill,
                relevance_score: score,
                match_reasons: vec!["deterministic lexical overlap".to_string()],
                required_dependencies,
            });
        }

        // Structured task effects are stronger evidence than fallback vector
        // similarity. Applications may declare a generic workspace mutation
        // contract without teaching the kernel their business domain; map it
        // only to explicit file-write capabilities already present in the
        // authoritative skill graph.
        let requires_workspace_mutation = task.constraints.iter().any(|constraint| {
            let normalized = constraint.to_ascii_lowercase().replace(' ', "");
            normalized == "required_effect=workspace_mutation"
                || normalized == "required_effect:workspace_mutation"
        });
        if requires_workspace_mutation {
            for skill in self.graph_store.list_all_skills() {
                let has_file_capability = skill.tags.iter().any(|tag| {
                    let tag = tag.to_ascii_lowercase();
                    tag.ends_with("fileoperation") || tag.ends_with("file-operation")
                });
                let has_write_capability = skill.tags.iter().any(|tag| {
                    let tag = tag.to_ascii_lowercase();
                    tag.ends_with("writeoperation") || tag.ends_with("write-operation")
                });
                if has_file_capability
                    && has_write_capability
                    && seen_iris.insert(skill.skill_iri.clone())
                {
                    let required_dependencies = self
                        .graph_store
                        .resolve_dependencies(&skill.skill_iri)
                        .into_iter()
                        .filter(|dependency| dependency != &skill.skill_iri)
                        .collect();
                    matches.push(SkillMatch {
                        skill,
                        relevance_score: 0.9,
                        match_reasons: vec![
                            "required workspace mutation capability matched".to_string()
                        ],
                        required_dependencies,
                    });
                }
            }
        }

        // Applications may nominate one non-executable workflow skill as the
        // home for their accumulated, CA-validated knowledge. The kernel only
        // resolves an existing graph IRI; it neither invents the application
        // skill nor grants it tool permissions.
        let preferred_learning_skill = task.constraints.iter().find_map(|constraint| {
            constraint
                .strip_prefix("learning_skill_iri=")
                .or_else(|| constraint.strip_prefix("learning_skill_iri:"))
                .map(str::trim)
        });
        if let Some(skill_iri) = preferred_learning_skill {
            if let Some(skill) = self.graph_store.get_skill(skill_iri) {
                if seen_iris.insert(skill.skill_iri.clone()) {
                    let required_dependencies = self
                        .graph_store
                        .resolve_dependencies(&skill.skill_iri)
                        .into_iter()
                        .filter(|dependency| dependency != &skill.skill_iri)
                        .collect();
                    matches.push(SkillMatch {
                        skill,
                        relevance_score: 1.0,
                        match_reasons: vec![
                            "application learning skill explicitly selected".to_string()
                        ],
                        required_dependencies,
                    });
                }
            }
        }

        // Phase 2: semantic vector search for complementary results
        // Hash fallback vectors are an availability mechanism, not semantic
        // evidence. Their top-k always returns something and previously
        // injected unrelated/destructive skills into otherwise simple tasks.
        // Keep deterministic 5W2H matching, but only trust semantic ranking
        // from a real embedding provider.
        if self
            .vector_store
            .as_ref()
            .is_some_and(|store| store.embedding_provider() != "fallback")
        {
            let query = [
                task.what.as_str(),
                task.why.as_str(),
                task.how_approach.as_deref().unwrap_or(""),
            ]
            .join(" ");
            if let Ok(semantic_matches) = self.semantic_search(&query, 5).await {
                for sm in semantic_matches {
                    if seen_iris.insert(sm.skill.skill_iri.clone()) {
                        matches.push(sm);
                    }
                }
            }
        }

        matches.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!(
            "Found {} matching skills ({} keyword, {} semantic)",
            matches.len(),
            matches
                .iter()
                .filter(|m| m.match_reasons.iter().any(|r| r != "semantic match"))
                .count(),
            matches
                .iter()
                .filter(|m| m.match_reasons.contains(&"semantic match".to_string()))
                .count()
        );
        matches
    }

    pub fn match_by_5w2h(
        &self,
        what: &str,
        why: Option<&str>,
        phase: Option<&str>,
        role: Option<&str>,
    ) -> Vec<SkillGraphNode> {
        self.graph_store
            .find_skills_by_5w2h(Some(what), why, phase, role, None)
    }

    pub fn expand_dependencies(&self, skill_iri: &str) -> Vec<String> {
        self.graph_store.resolve_dependencies(skill_iri)
    }

    pub fn check_conflicts(&self, skill_iris: &[&str]) -> Vec<SkillConflict> {
        let mut conflicts = Vec::new();

        for i in 0..skill_iris.len() {
            for j in (i + 1)..skill_iris.len() {
                let skill_a = self.graph_store.get_skill(skill_iris[i]);
                let skill_b = self.graph_store.get_skill(skill_iris[j]);

                if let (Some(a), Some(b)) = (skill_a, skill_b) {
                    let mut found_alternative = false;
                    for link in &a.links {
                        if link.target_iri == skill_iris[j] {
                            if link.link_type == SkillLinkType::Alternative {
                                found_alternative = true;
                            }
                        }
                    }
                    for link in &b.links {
                        if link.target_iri == skill_iris[i] {
                            if link.link_type == SkillLinkType::Alternative {
                                found_alternative = true;
                            }
                        }
                    }
                    if found_alternative {
                        conflicts.push(SkillConflict {
                            skill_a: skill_iris[i].to_string(),
                            skill_b: skill_iris[j].to_string(),
                            conflict_type: "alternative".to_string(),
                            description: format!(
                                "{} is an alternative to {}",
                                skill_iris[j], skill_iris[i]
                            ),
                        });
                    }

                    let tags_a: HashSet<&String> = a.tags.iter().collect();
                    let tags_b: HashSet<&String> = b.tags.iter().collect();

                    if tags_a.contains(&"exclusive".to_string())
                        && tags_b.contains(&"exclusive".to_string())
                    {
                        conflicts.push(SkillConflict {
                            skill_a: skill_iris[i].to_string(),
                            skill_b: skill_iris[j].to_string(),
                            conflict_type: "exclusive".to_string(),
                            description: "Both skills are marked as exclusive".to_string(),
                        });
                    }
                }
            }
        }

        conflicts
    }

    pub fn find_skill_chain(&self, start_iri: &str, end_iri: &str) -> Option<Vec<String>> {
        let mut visited = HashSet::new();
        let mut path = Vec::new();

        if self.find_path_recursive(start_iri, end_iri, &mut visited, &mut path) {
            Some(path)
        } else {
            None
        }
    }

    fn find_path_recursive(
        &self,
        current: &str,
        target: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) -> bool {
        if current == target {
            path.push(current.to_string());
            return true;
        }

        if visited.contains(current) {
            return false;
        }
        visited.insert(current.to_string());

        if let Some(skill) = self.graph_store.get_skill(current) {
            for link in &skill.links {
                if link.link_type == SkillLinkType::Composition
                    || link.link_type == SkillLinkType::Related
                {
                    if self.find_path_recursive(&link.target_iri, target, visited, path) {
                        path.insert(0, current.to_string());
                        return true;
                    }
                }
            }
        }

        false
    }

    pub async fn semantic_search(
        &self,
        query: &str,
        limit: u64,
    ) -> Result<Vec<SkillMatch>, CoreError> {
        if let Some(ref vector_store) = self.vector_store {
            // Scope retrieval to skill entries only (indexed with skill:Skill type).
            let filter = crate::memory::hyperspace_store::HybridSearchFilter::new()
                .with_jsonld_types(vec!["skill:Skill".to_string()]);
            let results = vector_store
                .search_with_filter(query, &filter, limit)
                .await
                .map_err(|e| CoreError::Internal {
                    message: format!("Vector search failed: {}", e),
                })?;

            let mut matches = Vec::new();
            for result in results {
                if let Some(skill) = self.graph_store.get_skill(&result.iri) {
                    matches.push(SkillMatch {
                        skill,
                        relevance_score: result.score,
                        match_reasons: vec!["semantic match".to_string()],
                        required_dependencies: self.graph_store.resolve_dependencies(&result.iri),
                    });
                }
            }

            Ok(matches)
        } else {
            Ok(Vec::new())
        }
    }

    /// Hybrid retrieval: free-text search plus tag-structure boost.
    ///
    /// `should_tags` rank entries carrying those tags higher without excluding
    /// untagged entries — combining semantic similarity with tag overlap.
    pub async fn hybrid_search(
        &self,
        query: &str,
        should_tags: &[String],
        limit: u64,
    ) -> Result<Vec<SkillMatch>, CoreError> {
        let Some(ref vector_store) = self.vector_store else {
            return Ok(Vec::new());
        };
        let results = vector_store
            .hybrid_search(query, &[], should_tags, None, limit)
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("Vector search failed: {}", e),
            })?;

        let mut matches = Vec::new();
        for result in results {
            if let Some(skill) = self.graph_store.get_skill(&result.iri) {
                matches.push(SkillMatch {
                    skill,
                    relevance_score: result.score,
                    match_reasons: vec!["hybrid match".to_string()],
                    required_dependencies: self.graph_store.resolve_dependencies(&result.iri),
                });
            }
        }
        Ok(matches)
    }

    pub fn get_recommended_skills(&self, skill_iri: &str) -> Vec<(String, String)> {
        let mut recommended = Vec::new();

        if let Some(skill) = self.graph_store.get_skill(skill_iri) {
            for link in &skill.links {
                if link.link_type == SkillLinkType::Related
                    && link.strength == LinkStrength::Recommended
                {
                    recommended.push((link.target_iri.clone(), link.description.clone()));
                }
            }
        }

        recommended
    }

    pub fn get_skill_tree(&self, root_iri: &str, max_depth: u32) -> Value {
        let mut tree = serde_json::json!({
            "root": root_iri,
            "nodes": []
        });

        let mut nodes = Vec::new();
        self.build_skill_tree_recursive(root_iri, 0, max_depth, &mut nodes);

        if let Some(obj) = tree.as_object_mut() {
            obj.insert("nodes".to_string(), serde_json::json!(nodes));
        }

        tree
    }

    fn build_skill_tree_recursive(
        &self,
        skill_iri: &str,
        depth: u32,
        max_depth: u32,
        nodes: &mut Vec<Value>,
    ) {
        if depth > max_depth {
            return;
        }

        if let Some(skill) = self.graph_store.get_skill(skill_iri) {
            let mut children = Vec::new();

            for link in &skill.links {
                if link.link_type == SkillLinkType::Composition {
                    children.push(link.target_iri.clone());
                    self.build_skill_tree_recursive(&link.target_iri, depth + 1, max_depth, nodes);
                }
            }

            nodes.push(serde_json::json!({
                "iri": skill.skill_iri,
                "name": skill.name,
                "depth": depth,
                "children": children
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embedding_service::FallbackEmbeddingService;

    fn setup_test_store() -> Arc<SkillGraphStore> {
        let store = Arc::new(SkillGraphStore::new());

        let jwt_skill = SkillGraphNode::new(
            "iri://skills/jwt-auth",
            "JWT Authentication",
            "Implement JWT authentication",
        )
        .with_5w2h(
            Skill5W2H::new("JWT authentication", "Secure API access")
                .with_phase("Do")
                .with_agent_role("DA"),
        )
        .with_tag("authentication")
        .with_tag("security");

        let mut oauth_skill = SkillGraphNode::new(
            "iri://skills/oauth-auth",
            "OAuth Authentication",
            "Implement OAuth authentication",
        )
        .with_5w2h(
            Skill5W2H::new("OAuth authentication", "Third-party auth")
                .with_phase("Do")
                .with_agent_role("DA"),
        );
        oauth_skill.add_alternative("iri://skills/jwt-auth", "JWT is simpler for internal APIs");

        let mut middleware_skill = SkillGraphNode::new(
            "iri://skills/rust-middleware",
            "Rust Middleware",
            "Implement middleware in Rust",
        )
        .with_5w2h(
            Skill5W2H::new("Middleware", "Request processing pipeline")
                .with_phase("Do")
                .with_agent_role("DA"),
        );
        middleware_skill.add_related("iri://skills/jwt-auth", "JWT often used with middleware");

        store.register_skill(jwt_skill).unwrap();
        store.register_skill(oauth_skill).unwrap();
        store.register_skill(middleware_skill).unwrap();

        store
    }

    #[tokio::test]
    async fn test_discover_for_task() {
        let store = setup_test_store();
        let engine = SkillDiscoveryEngine::new(store);

        let task = Task5W2H::new("JWT authentication", "Secure API")
            .with_phase("Do")
            .with_agent_role("DA");

        let matches = engine.discover_for_task(&task).await;

        assert!(!matches.is_empty());
        assert!(matches
            .iter()
            .any(|m| m.skill.skill_iri == "iri://skills/jwt-auth"));
    }

    #[tokio::test]
    async fn test_index_all_skills_populates_semantic_discovery() {
        let store = setup_test_store();
        let dir = tempfile::tempdir().unwrap();
        let vector_store = Arc::new(
            HyperspaceStore::open(
                dir.path(),
                Arc::new(FallbackEmbeddingService::with_dimension(32)),
            )
            .unwrap(),
        );
        let engine = SkillDiscoveryEngine::new(store).with_vector_store(vector_store);

        assert_eq!(engine.index_all_skills().await.unwrap(), 3);
        let matches = engine
            .semantic_search("JWT Authentication", 5)
            .await
            .unwrap();
        assert!(matches
            .iter()
            .any(|m| m.skill.skill_iri == "iri://skills/jwt-auth"));
    }

    #[tokio::test]
    async fn fallback_embeddings_do_not_fabricate_skill_relevance() {
        let store = setup_test_store();
        let dir = tempfile::tempdir().unwrap();
        let vector_store = Arc::new(
            HyperspaceStore::open(
                dir.path(),
                Arc::new(FallbackEmbeddingService::with_dimension(32)),
            )
            .unwrap(),
        );
        let engine = SkillDiscoveryEngine::new(store).with_vector_store(vector_store);
        engine.index_all_skills().await.unwrap();

        let matches = engine
            .discover_for_task(&Task5W2H::new(
                "unrelated banana astronomy",
                "observe distant galaxies",
            ))
            .await;

        assert!(matches.is_empty(), "got fabricated matches: {matches:?}");
    }

    #[tokio::test]
    async fn lexical_fallback_retrieves_a_relevant_accumulated_skill() {
        let store = Arc::new(SkillGraphStore::new());
        store
            .register_skill(
                SkillGraphNode::new(
                    "iri://skills/evidence-locator",
                    "single-file evidence locator",
                    "Read a named file once and cite the exact matching evidence line",
                )
                .with_5w2h(
                    Skill5W2H::new(
                        "Extract one exact field from one named text file",
                        "Minimize redundant file reads while retaining evidence",
                    )
                    .with_agent_role("DA")
                    .with_phase("Do"),
                )
                .with_tag("allowed-role:CA"),
            )
            .unwrap();
        store
            .register_skill(SkillGraphNode::new(
                "iri://skills/unrelated",
                "astronomy image classifier",
                "Classify telescope pictures of distant galaxies",
            ))
            .unwrap();
        let engine = SkillDiscoveryEngine::new(store);
        let matches = engine
            .discover_for_task(&Task5W2H {
                what: "Read fixture.txt and extract the exact ANSWER value".to_string(),
                why: "Cite the exact evidence line and avoid redundant file reads".to_string(),
                who: Some("CA".to_string()),
                when_phase: Some("Immediate".to_string()),
                ..Task5W2H::default()
            })
            .await;

        assert!(matches
            .iter()
            .any(|matched| matched.skill.skill_iri == "iri://skills/evidence-locator"));
        assert!(!matches
            .iter()
            .any(|matched| matched.skill.skill_iri == "iri://skills/unrelated"));
    }

    #[tokio::test]
    async fn structured_workspace_effect_discovers_only_explicit_write_capabilities() {
        let store = Arc::new(SkillGraphStore::new());
        store
            .register_skill(
                SkillGraphNode::new("iri://skills/file-write", "file_write", "Write a file")
                    .with_tag("iri://skill-types/FileOperation")
                    .with_tag("iri://skill-types/WriteOperation"),
            )
            .unwrap();
        store
            .register_skill(
                SkillGraphNode::new("iri://skills/file-read", "file_read", "Read a file")
                    .with_tag("iri://skill-types/FileOperation")
                    .with_tag("iri://skill-types/ReadOperation"),
            )
            .unwrap();
        let engine = SkillDiscoveryEngine::new(store);
        let task = Task5W2H::new("opaque multilingual objective", "complete task")
            .with_constraint("required_effect=workspace_mutation");

        let matches = engine.discover_for_task(&task).await;

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].skill.skill_iri, "iri://skills/file-write");
        assert_eq!(matches[0].relevance_score, 0.9);
    }

    #[tokio::test]
    async fn application_learning_skill_is_resolved_without_execution_authority() {
        let store = Arc::new(SkillGraphStore::new());
        store
            .register_skill(
                SkillGraphNode::new(
                    "iri://skills/application-workflow",
                    "application workflow",
                    "Non-executable learning home",
                )
                .with_tag("non-executable-learning-skill"),
            )
            .unwrap();
        let engine = SkillDiscoveryEngine::new(store);
        let task = Task5W2H::new("unrelated wording", "current task")
            .with_constraint("learning_skill_iri=iri://skills/application-workflow");

        let matches = engine.discover_for_task(&task).await;

        assert_eq!(matches.len(), 1);
        assert_eq!(
            matches[0].skill.skill_iri,
            "iri://skills/application-workflow"
        );
        assert_eq!(matches[0].relevance_score, 1.0);
        assert!(matches[0]
            .match_reasons
            .iter()
            .any(|reason| reason.contains("explicitly selected")));
    }

    #[test]
    fn test_expand_dependencies() {
        let store = Arc::new(SkillGraphStore::new());

        let skill_a = SkillGraphNode::new("iri://skills/a", "A", "Skill A");
        let mut skill_b = SkillGraphNode::new("iri://skills/b", "B", "Skill B");
        skill_b.add_prerequisite("iri://skills/a", "A is required");

        store.register_skill(skill_a).unwrap();
        store.register_skill(skill_b).unwrap();

        let engine = SkillDiscoveryEngine::new(store);
        let deps = engine.expand_dependencies("iri://skills/b");

        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&"iri://skills/a".to_string()));
    }

    #[test]
    fn test_check_conflicts() {
        let store = setup_test_store();
        let engine = SkillDiscoveryEngine::new(store);

        let conflicts =
            engine.check_conflicts(&["iri://skills/jwt-auth", "iri://skills/oauth-auth"]);

        assert!(!conflicts.is_empty());
        assert!(conflicts.iter().any(|c| c.conflict_type == "alternative"));
    }

    #[test]
    fn test_get_recommended_skills() {
        let store = setup_test_store();
        let engine = SkillDiscoveryEngine::new(store);

        let recommended = engine.get_recommended_skills("iri://skills/rust-middleware");

        assert!(!recommended.is_empty());
        assert!(recommended
            .iter()
            .any(|(iri, _)| iri == "iri://skills/jwt-auth"));
    }

    #[test]
    fn test_find_skill_chain() {
        let store = Arc::new(SkillGraphStore::new());

        let skill_a = SkillGraphNode::new("iri://skills/a", "A", "Skill A");
        let mut skill_b = SkillGraphNode::new("iri://skills/b", "B", "Skill B");
        skill_b.add_related("iri://skills/a", "Related to A");
        let mut skill_c = SkillGraphNode::new("iri://skills/c", "C", "Skill C");
        skill_c.add_related("iri://skills/b", "Related to B");

        store.register_skill(skill_a).unwrap();
        store.register_skill(skill_b).unwrap();
        store.register_skill(skill_c).unwrap();

        let engine = SkillDiscoveryEngine::new(store);
        let chain = engine.find_skill_chain("iri://skills/c", "iri://skills/a");

        assert!(chain.is_some());
        let chain = chain.unwrap();
        assert!(chain.contains(&"iri://skills/c".to_string()));
        assert!(chain.contains(&"iri://skills/a".to_string()));
    }

    #[test]
    fn test_get_skill_tree() {
        let store = Arc::new(SkillGraphStore::new());

        let mut parent = SkillGraphNode::new("iri://skills/parent", "Parent", "Parent skill");
        parent.node_type = SkillNodeType::Composite;
        parent.add_link(SkillLink::new(
            SkillLinkType::Composition,
            "iri://skills/child1".to_string(),
        ));
        parent.add_link(SkillLink::new(
            SkillLinkType::Composition,
            "iri://skills/child2".to_string(),
        ));

        let child1 = SkillGraphNode::new("iri://skills/child1", "Child 1", "First child");
        let child2 = SkillGraphNode::new("iri://skills/child2", "Child 2", "Second child");

        store.register_skill(parent).unwrap();
        store.register_skill(child1).unwrap();
        store.register_skill(child2).unwrap();

        let engine = SkillDiscoveryEngine::new(store);
        let tree = engine.get_skill_tree("iri://skills/parent", 2);

        assert!(tree.get("nodes").and_then(|n| n.as_array()).unwrap().len() >= 1);
    }
}
