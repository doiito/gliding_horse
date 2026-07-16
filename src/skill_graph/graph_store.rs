use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::Utc;
use oxigraph::store::Store;
use parking_lot::RwLock;
use serde_json::Value;
use tracing::{debug, error, info, warn};

use crate::memory::l0_store::L0Store;
use crate::memory::l2_blackboard::Blackboard;
use crate::skill_graph::discovery::SkillDiscoveryEngine;
use crate::skill_graph::embedding::SkillGraphEmbedder;
use crate::skill_graph::index::PreAggregatedIndex;
use crate::skill_graph::types::*;
use crate::snapshots::{GraphMutation, TimelineStore};
use crate::CoreError;

const SKILL_GRAPH_NAMED_GRAPH: &str = "system:skill_graph";

pub struct SkillGraphStore {
    skills: RwLock<HashMap<String, SkillGraphNode>>,
    mocs: RwLock<HashMap<String, MOCNode>>,
    fragments: RwLock<HashMap<String, KnowledgeFragment>>,
    hyperedges: RwLock<HashMap<String, Hyperedge>>,
    snapshot_history: RwLock<Vec<SnapshotRecord>>,
    index: PreAggregatedIndex,
    blackboard: Option<Arc<Blackboard>>,
    l0_store: Option<Arc<L0Store>>,
    oxi_store: Option<Arc<Store>>,
    /// Optional TimelineStore for recording graph mutations.
    /// When set, every structural change to the skill graph is recorded so that
    /// `TimelineStore::pending_mutations()` reflects real activity.
    timeline: Option<Arc<TimelineStore>>,
}

impl SkillGraphStore {
    pub fn new() -> Self {
        Self {
            skills: RwLock::new(HashMap::new()),
            mocs: RwLock::new(HashMap::new()),
            fragments: RwLock::new(HashMap::new()),
            hyperedges: RwLock::new(HashMap::new()),
            snapshot_history: RwLock::new(Vec::new()),
            index: PreAggregatedIndex::new(),
            blackboard: None,
            l0_store: None,
            oxi_store: None,
            timeline: None,
        }
    }

    /// Attach a TimelineStore that receives a `GraphMutation` for every
    /// structural change applied to this skill graph. This is what makes
    /// `TimelineStore::pending_mutations()` / `snapshot_count()` reflect real
    /// graph activity instead of staying permanently at 0.
    pub fn with_timeline(mut self, timeline: Arc<TimelineStore>) -> Self {
        self.timeline = Some(timeline);
        self
    }

    /// Record a structural mutation against the attached TimelineStore (if any).
    /// No SnapshotBackend is supplied, so this only appends to the incremental
    /// mutation log — the engine still takes explicit full snapshots per task.
    fn emit_mutation(&self, mutation: GraphMutation) {
        if let Some(ref tl) = self.timeline {
            tl.record_mutation(mutation, None);
        }
    }

    pub fn with_blackboard(mut self, blackboard: Arc<Blackboard>) -> Self {
        self.blackboard = Some(blackboard);
        self
    }

    pub fn with_l0_store(mut self, l0_store: Arc<L0Store>) -> Self {
        self.l0_store = Some(l0_store);
        self
    }

    pub fn with_oxi_store(mut self, store: Arc<Store>) -> Self {
        self.oxi_store = Some(store);
        self
    }

    pub fn get_index(&self) -> &PreAggregatedIndex {
        &self.index
    }

    // ── P0-1: Oxigraph sync helpers ─────────────────────────────────────

    fn sync_sparql(&self, sparql: &str) {
        if let Some(ref store) = self.oxi_store {
            if let Err(e) = store.update(sparql) {
                error!("Oxigraph sync failed: {}. SPARQL was: {}", e, sparql);
            }
        }
    }

    fn sync_skill_insert(&self, skill: &SkillGraphNode) {
        if skill.storage_tier == StorageTier::L1Session {
            return;
        }
        let sparql = skill.to_sparql_insert(SKILL_GRAPH_NAMED_GRAPH);
        self.sync_sparql(&sparql);
    }

    fn sync_skill_delete(&self, iri: &str) {
        let sparql = format!(
            "DELETE WHERE {{ GRAPH <{}> {{ <{}> ?p ?o }} }}",
            SKILL_GRAPH_NAMED_GRAPH, iri
        );
        self.sync_sparql(&sparql);
    }

    pub fn register_skill(&self, skill: SkillGraphNode) -> Result<(), CoreError> {
        let iri = skill.skill_iri.clone();
        info!("Registering skill to graph: {} ({})", skill.name, iri);

        if let Some(_blackboard) = &self.blackboard {
            let json_ld = skill.to_json_ld();
            let json_str = serde_json::to_string(&json_ld).unwrap_or_default();
            debug!("Skill JSON-LD generated: {} bytes", json_str.len());
        }

        self.index.index_skill(&skill);
        self.skills.write().insert(iri.clone(), skill.clone());
        self.emit_mutation(GraphMutation::SkillRegistered(skill));

        // P0-1: sync to Oxigraph
        if let Some(skill) = self.skills.read().get(&iri) {
            self.sync_skill_insert(skill);
        }

        if let Some(ref l0_store) = self.l0_store {
            let now = Utc::now();
            let entry = crate::memory::l0_store::L0Entry {
                iri: iri.clone(),
                content: format!("skill:{}", iri),
                importance: 0.5,
                access_count: 0,
                created_at: now,
                last_accessed: now,
                tags: vec!["skill".to_string(), "skill_graph".to_string()],
                metadata: serde_json::Map::new(),
                mesi_state: crate::memory::l0_store::MesiState::Shared,
                content_hash: String::new(),
                named_graph: Some(SKILL_GRAPH_NAMED_GRAPH.to_string()),

                jsonld_context: None,
                jsonld_types: vec!["skill:Skill".to_string()],
                hyperspace_point_id: None,
            };
            l0_store.store(&iri, &serde_json::to_string(&entry).unwrap_or_default())?;
            debug!("Skill written to L0 store: {}", iri);
        }

        Ok(())
    }

    pub fn get_skill(&self, skill_iri: &str) -> Option<SkillGraphNode> {
        self.skills.read().get(skill_iri).cloned()
    }

    pub fn update_skill(&self, skill: SkillGraphNode) -> Result<(), CoreError> {
        let iri = skill.skill_iri.clone();
        let old = self.skills.read().get(&iri).cloned();
        if let Some(old) = old {
            self.index.update_skill(&skill);
            // P0-1: sync update to Oxigraph (delete old + insert new)
            self.sync_skill_delete(&iri);
            self.sync_skill_insert(&skill);
            self.skills.write().insert(iri, skill.clone());
            self.emit_mutation(GraphMutation::SkillUpdated { old, new: skill });
            Ok(())
        } else {
            Err(CoreError::SkillNotFound {
                iri: format!("Skill not found: {}", iri),
            })
        }
    }

    pub fn remove_skill(&self, skill_iri: &str) -> Result<(), CoreError> {
        if let Some(removed) = self.skills.write().remove(skill_iri) {
            self.index.remove_skill(skill_iri);
            // P0-1: sync delete from Oxigraph
            self.sync_skill_delete(skill_iri);
            info!("Skill removed from graph: {}", skill_iri);
            self.emit_mutation(GraphMutation::SkillRemoved(removed));
            Ok(())
        } else {
            Err(CoreError::SkillNotFound {
                iri: format!("Skill not found: {}", skill_iri),
            })
        }
    }

    pub fn add_link(
        &self,
        source_iri: &str,
        target_iri: &str,
        link_type: SkillLinkType,
        strength: LinkStrength,
        description: &str,
    ) -> Result<(), CoreError> {
        let mut skills = self.skills.write();
        
        if let Some(source) = skills.get_mut(source_iri) {
            source.links.push(SkillLink {
                link_type,
                target_iri: target_iri.to_string(),
                strength,
                description: description.to_string(),
            });
            debug!("Adding link: {} -> {} ({:?})", source_iri, target_iri, link_type);

            let updated = skills.get(source_iri).cloned();
            drop(skills);
            if let Some(skill) = updated {
                self.index.update_skill(&skill);
                // P0-1: sync link triple to Oxigraph
                self.sync_skill_insert(&skill);
            }
            self.emit_mutation(GraphMutation::LinkAdded {
                source: source_iri.to_string(),
                target: target_iri.to_string(),
                link_type,
            });
            Ok(())
        } else {
            Err(CoreError::SkillNotFound {
                iri: format!("Source skill not found: {}", source_iri),
            })
        }
    }

    pub fn traverse_links(
        &self,
        start_iri: &str,
        link_types: Option<&[SkillLinkType]>,
        max_depth: u32,
    ) -> Vec<(String, SkillLinkType, u32)> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue: Vec<(String, u32)> = vec![(start_iri.to_string(), 0)];

        while let Some((current_iri, depth)) = queue.pop() {
            if depth >= max_depth {
                continue;
            }

            if visited.contains(&current_iri) {
                continue;
            }
            visited.insert(current_iri.clone());

            let skills = self.skills.read();
            if let Some(skill) = skills.get(&current_iri) {
                for link in &skill.links {
                    if let Some(types) = link_types {
                        if !types.contains(&link.link_type) {
                            continue;
                        }
                    }

                    result.push((link.target_iri.clone(), link.link_type, depth + 1));

                    if !visited.contains(&link.target_iri) {
                        queue.push((link.target_iri.clone(), depth + 1));
                    }
                }
            }
        }

        result
    }

    pub fn resolve_dependencies(&self, skill_iri: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        self.resolve_dependencies_recursive(skill_iri, &mut result, &mut visited);
        result
    }

    fn resolve_dependencies_recursive(
        &self,
        skill_iri: &str,
        result: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) {
        if visited.contains(skill_iri) {
            return;
        }
        visited.insert(skill_iri.to_string());

        let skills = self.skills.read();
        if let Some(skill) = skills.get(skill_iri) {
            for link in &skill.links {
                if link.link_type == SkillLinkType::Prerequisite {
                    self.resolve_dependencies_recursive(&link.target_iri, result, visited);
                }
            }
        }

        result.push(skill_iri.to_string());
    }

    pub fn find_alternatives(&self, skill_iri: &str) -> Vec<(String, String)> {
        let skills = self.skills.read();
        
        if let Some(skill) = skills.get(skill_iri) {
            skill
                .get_alternatives()
                .iter()
                .map(|link| (link.target_iri.clone(), link.description.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn find_skills_by_5w2h(
        &self,
        what_keyword: Option<&str>,
        why_keyword: Option<&str>,
        phase: Option<&str>,
        agent_role: Option<&str>,
        min_success_rate: Option<f32>,
    ) -> Vec<SkillGraphNode> {
        let role_candidates: Option<Vec<String>> = agent_role.map(|role| {
            self.index.find_by_role(role)
        });

        let skills = self.skills.read();
        
        skills
            .values()
            .filter(|skill| {
                if let Some(ref candidates) = role_candidates {
                    if !candidates.contains(&skill.skill_iri) {
                        return false;
                    }
                }

                if let Some(kw) = what_keyword {
                    if !skill.w2h.what.to_lowercase().contains(&kw.to_lowercase()) {
                        return false;
                    }
                }

                if let Some(kw) = why_keyword {
                    if !skill.w2h.why.to_lowercase().contains(&kw.to_lowercase()) {
                        return false;
                    }
                }

                if let Some(p) = phase {
                    if !skill.w2h.when.applicable_phases.iter().any(|ph| ph.to_lowercase() == p.to_lowercase()) {
                        return false;
                    }
                }

                if let Some(min_rate) = min_success_rate {
                    if skill.graph_meta.success_rate < min_rate {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect()
    }

    pub fn find_skills_by_tags(&self, tags: &[&str]) -> Vec<SkillGraphNode> {
        let candidates = self.index.find_by_tags_intersection(tags);
        if !candidates.is_empty() {
            let skills = self.skills.read();
            return candidates
                .iter()
                .filter_map(|iri| skills.get(iri).cloned())
                .collect();
        }

        let skills = self.skills.read();
        skills
            .values()
            .filter(|skill| {
                tags.iter().all(|tag| {
                    skill.tags.iter().any(|t| t.to_lowercase() == tag.to_lowercase())
                })
            })
            .cloned()
            .collect()
    }

    pub fn register_moc(&self, moc: MOCNode) -> Result<(), CoreError> {
        let moc_iri = moc.moc_iri.clone();
        info!("Registering MOC node: {} ({})", moc.name, moc_iri);
        self.mocs.write().insert(moc_iri, moc.clone());
        self.emit_mutation(GraphMutation::MOCAdded(moc));
        Ok(())
    }

    pub fn get_moc(&self, moc_iri: &str) -> Option<MOCNode> {
        self.mocs.read().get(moc_iri).cloned()
    }

    pub fn list_mocs(&self) -> Vec<MOCNode> {
        self.mocs.read().values().cloned().collect()
    }

    pub fn create_fragment(
        &self,
        fragment_iri: &str,
        attached_to: &str,
        problem: &str,
        recommendation: &str,
        discoverer: Option<&str>,
    ) -> Result<KnowledgeFragment, CoreError> {
        let mut fragment = KnowledgeFragment::new(fragment_iri, attached_to, problem, recommendation);
        
        if let Some(d) = discoverer {
            fragment = fragment.with_discoverer(d);
        }

        info!("Creating knowledge fragment: {} -> {}", attached_to, fragment_iri);
        
        self.fragments.write().insert(fragment_iri.to_string(), fragment.clone());
        
        let skill_clone = {
            self.skills.read().get(attached_to).cloned()
        };
        
        if let Some(mut skill) = skill_clone {
            skill.graph_meta.known_failure_modes.push(FailureMode {
                mode: problem.to_string(),
                discovered_by: discoverer.map(|s| s.to_string()),
                mitigation: recommendation.to_string(),
            });
            self.index.update_skill(&skill);
            self.skills.write().insert(attached_to.to_string(), skill);
        }

        Ok(fragment)
    }

    pub fn get_fragments_for_skill(&self, skill_iri: &str) -> Vec<KnowledgeFragment> {
        self.fragments
            .read()
            .values()
            .filter(|f| f.attached_to == skill_iri)
            .cloned()
            .collect()
    }

    pub fn list_fragments(&self) -> Vec<KnowledgeFragment> {
        self.fragments.read().values().cloned().collect()
    }

    pub fn record_skill_usage(&self, skill_iri: &str, success: bool) -> Result<(), CoreError> {
        let mut skills = self.skills.write();
        
        if let Some(skill) = skills.get_mut(skill_iri) {
            skill.graph_meta.record_usage(success);
            debug!(
                "Recording skill usage: {} (success={}, rate={:.2})",
                skill_iri, success, skill.graph_meta.success_rate
            );

            let updated = skills.get(skill_iri).cloned();
            drop(skills);
            if let Some(skill) = updated {
                self.index.update_skill(&skill);
            }
            Ok(())
        } else {
            Err(CoreError::SkillNotFound {
                iri: format!("Skill not found: {}", skill_iri),
            })
        }
    }

    pub fn get_skill_at_level(&self, skill_iri: &str, level: DisclosureLevel) -> Option<Value> {
        if level == DisclosureLevel::MOCIndex || level == DisclosureLevel::Summary5W2H {
            if let Some(entry) = self.index.get_summary(skill_iri) {
                return Some(match level {
                    DisclosureLevel::MOCIndex => {
                        serde_json::json!({
                            "@id": entry.skill_iri,
                            "name": entry.name,
                            "node_type": format!("{:?}", entry.node_type),
                            "tags": entry.tags
                        })
                    }
                    DisclosureLevel::Summary5W2H => {
                        serde_json::json!({
                            "@id": entry.skill_iri,
                            "name": entry.name,
                            "what": entry.what,
                            "why": entry.why,
                            "tags": entry.tags,
                            "success_rate": entry.success_rate
                        })
                    }
                    _ => unreachable!(),
                });
            }
        }

        let skills = self.skills.read();
        let skill = skills.get(skill_iri)?;

        Some(match level {
            DisclosureLevel::MOCIndex => {
                serde_json::json!({
                    "@id": skill.skill_iri,
                    "name": skill.name,
                    "description": skill.description,
                    "node_type": format!("{:?}", skill.node_type),
                    "tags": skill.tags
                })
            }
            DisclosureLevel::Summary5W2H => {
                serde_json::json!({
                    "@id": skill.skill_iri,
                    "name": skill.name,
                    "what": skill.w2h.what,
                    "why": skill.w2h.why,
                    "when": skill.w2h.when.applicable_phases,
                    "who": skill.w2h.who.required_agent_role,
                    "tags": skill.tags,
                    "success_rate": skill.graph_meta.success_rate
                })
            }
            DisclosureLevel::LinksExpanded => {
                let links: Vec<Value> = skill.links.iter().map(|l| {
                    serde_json::json!({
                        "type": format!("{:?}", l.link_type),
                        "target": l.target_iri,
                        "strength": format!("{:?}", l.strength),
                        "description": l.description
                    })
                }).collect();
                
                serde_json::json!({
                    "@id": skill.skill_iri,
                    "name": skill.name,
                    "what": skill.w2h.what,
                    "why": skill.w2h.why,
                    "links": links,
                    "success_rate": skill.graph_meta.success_rate
                })
            }
            DisclosureLevel::SchemaSteps => {
                let steps: Vec<Value> = skill.content.as_ref().map(|c| {
                    c.steps.iter().map(|s| {
                        serde_json::json!({
                            "step_id": s.step_id,
                            "order": s.order,
                            "action": s.action
                        })
                    }).collect()
                }).unwrap_or_default();
                
                serde_json::json!({
                    "@id": skill.skill_iri,
                    "name": skill.name,
                    "what": skill.w2h.what,
                    "steps": steps,
                    "validation": skill.content.as_ref().and_then(|c| c.validation.as_ref().map(|v| {
                        serde_json::json!({
                            "method": v.method,
                            "success_condition": v.success_condition
                        })
                    }))
                })
            }
            DisclosureLevel::FullContent => {
                skill.to_json_ld()
            }
        })
    }

    pub fn list_all_skills(&self) -> Vec<SkillGraphNode> {
        self.skills.read().values().cloned().collect()
    }

    pub fn skill_count(&self) -> usize {
        self.skills.read().len()
    }

    // ── LLRU cold archive ──────────────────────────────────────────────

    /// Archive cold skills (not used since `cutoff`) to L0 storage.
    ///
    /// Each archived skill has its `storage_tier` set to `L0Permanent` and
    /// is persisted synchronously to the L0 store.  Skills remain registered
    /// in the in-memory graph — only the tier label changes.
    ///
    /// Returns the number of skills archived.
    pub fn archive_cold_skills(&self, cutoff: &chrono::DateTime<chrono::Utc>) -> Result<usize, CoreError> {
        let cold_iris: Vec<String> = self
            .list_all_skills()
            .into_iter()
            .filter(|s| {
                s.storage_tier != StorageTier::L0Permanent
                    && s.last_used_at
                        .as_ref()
                        .map(|used| used < cutoff)
                        .unwrap_or(true)
            })
            .map(|s| s.skill_iri.clone())
            .collect();

        if cold_iris.is_empty() {
            return Ok(0);
        }

        let l0 = self.l0_store.as_ref().ok_or_else(|| CoreError::Internal {
            message: "L0 store not configured — cannot archive".to_string(),
        })?;

        let mut archived = 0usize;
        for iri in &cold_iris {
            // Serialize the skill as JSON-LD for L0 storage
            let mut skills = self.skills.write();
            if let Some(skill) = skills.get_mut(iri) {
                skill.storage_tier = StorageTier::L0Permanent;

                let now = Utc::now();
                let entry = crate::memory::l0_store::L0Entry {
                    iri: iri.clone(),
                    content: serde_json::to_string(&skill.to_json_ld()).unwrap_or_default(),
                    importance: skill.graph_meta.success_rate as f32,
                    access_count: 0,
                    created_at: skill.created_at,
                    last_accessed: now,
                    tags: skill.tags.clone(),
                    metadata: serde_json::Map::new(),
                    mesi_state: crate::memory::l0_store::MesiState::Exclusive,
                    content_hash: String::new(),
                    named_graph: Some(crate::skill_graph::graph_store::SKILL_GRAPH_NAMED_GRAPH.to_string()),
                    jsonld_context: None,
                    jsonld_types: vec!["skill:Skill".to_string()],
                    hyperspace_point_id: None,
                };
                let json = serde_json::to_string(&entry).map_err(|e| CoreError::Internal {
                    message: format!("Serialization error during archive: {e}"),
                })?;
                l0.store(iri, &json)?;
                archived += 1;
            }
        }

        info!(count = archived, "LLRU archive: cold skills migrated to L0");
        Ok(archived)
    }

    /// Find skills whose `updated_at` is older than `max_age` (i.e. stale).
    ///
    /// These skills should be re-indexed to refresh the in-memory index.
    /// Skills with `storage_tier == L1Session` are excluded (ephemeral).
    pub fn find_stale_skills(&self, max_age: &chrono::Duration) -> Vec<SkillGraphNode> {
        let cutoff = Utc::now() - *max_age;
        self.list_all_skills()
            .into_iter()
            .filter(|s| {
                s.storage_tier != StorageTier::L1Session && s.updated_at < cutoff
            })
            .collect()
    }

    /// Re-index all stale skills whose `updated_at` is older than `max_age`.
    ///
    /// Returns the number of skills that were re-indexed.
    pub fn reindex_stale_skills(&self, max_age: &chrono::Duration) -> usize {
        let stale = self.find_stale_skills(max_age);
        let count = stale.len();
        for skill in &stale {
            self.index.update_skill(skill);
        }
        if count > 0 {
            info!(count = count, "Re-indexed stale skills");
        }
        count
    }

    // ── P1-1: Hyperedge CRUD ───────────────────────────────────────────

    pub fn register_hyperedge(&self, hyperedge: Hyperedge) -> Result<(), CoreError> {
        let id = hyperedge.hyperedge_id.clone();
        info!("Registering hyperedge: {} ({})", hyperedge.name, id);
        self.hyperedges.write().insert(id, hyperedge.clone());
        self.emit_mutation(GraphMutation::HyperedgeAdded(hyperedge));
        Ok(())
    }

    pub fn get_hyperedge(&self, hyperedge_id: &str) -> Option<Hyperedge> {
        self.hyperedges.read().get(hyperedge_id).cloned()
    }

    pub fn list_hyperedges(&self) -> Vec<Hyperedge> {
        self.hyperedges.read().values().cloned().collect()
    }

    pub fn remove_hyperedge(&self, hyperedge_id: &str) -> Result<(), CoreError> {
        if self.hyperedges.write().remove(hyperedge_id).is_some() {
            info!("Hyperedge removed: {}", hyperedge_id);
            self.emit_mutation(GraphMutation::HyperedgeRemoved(hyperedge_id.to_string()));
            Ok(())
        } else {
            Err(CoreError::SkillNotFound {
                iri: format!("Hyperedge not found: {}", hyperedge_id),
            })
        }
    }

    pub fn get_composites_containing(&self, skill_iri: &str) -> Vec<Hyperedge> {
        self.hyperedges
            .read()
            .values()
            .filter(|he| he.components.contains(&skill_iri.to_string()))
            .cloned()
            .collect()
    }

    pub fn resolve_hyperedge(&self, hyperedge_id: &str) -> Result<Vec<SkillGraphNode>, CoreError> {
        let he = self
            .hyperedges
            .read()
            .get(hyperedge_id)
            .cloned()
            .ok_or_else(|| CoreError::SkillNotFound {
                iri: format!("Hyperedge not found: {}", hyperedge_id),
            })?;

        let skills = self.skills.read();
        let mut resolved = Vec::new();
        for comp_iri in &he.components {
            if let Some(skill) = skills.get(comp_iri) {
                resolved.push(skill.clone());
            } else {
                warn!("Hyperedge {} references missing skill {}", hyperedge_id, comp_iri);
            }
        }
        Ok(resolved)
    }

    pub fn migrate_composition_links(&self) -> Result<usize, CoreError> {
        let skills = self.skills.read();
        let mut hyperedges_to_create = Vec::new();
        let mut processed = 0usize;

        // Find all skills that have Composition links pointing to the same composite target
        let mut target_to_components: HashMap<String, Vec<String>> = HashMap::new();
        for (iri, skill) in skills.iter() {
            for link in &skill.links {
                if link.link_type == SkillLinkType::Composition
                    && link.strength == LinkStrength::Required
                {
                    target_to_components
                        .entry(link.target_iri.clone())
                        .or_default()
                        .push(iri.clone());
                }
            }
        }

        for (target, components) in target_to_components {
            if components.len() >= 2 {
                let he = Hyperedge::new(
                    &format!("hyperedge:composite/{}", target.trim_start_matches("iri://skills/")),
                    &format!("Hyperedge for {}", target),
                    &format!("Migrated from Composition links targeting {}", target),
                    components,
                );
                hyperedges_to_create.push(he);
                processed += 1;
            }
        }

        drop(skills);
        for he in hyperedges_to_create {
            self.hyperedges.write().insert(he.hyperedge_id.clone(), he);
        }
        Ok(processed)
    }

    // ── P1-4: Temporal Versioning ───────────────────────────────────────

    pub fn snapshot(&self, label: &str) -> Result<String, CoreError> {
        let snapshot_id = format!(
            "snapshot:{}_{}",
            label,
            Utc::now().format("%Y%m%d_%H%M%S")
        );

        let skill_count = self.skills.read().len();
        let record = SnapshotRecord {
            snapshot_id: snapshot_id.clone(),
            label: label.to_string(),
            timestamp: Utc::now(),
            skill_count,
        };
        self.snapshot_history.write().push(record);

        // Persist snapshot JSON to Oxigraph if available
        if let Some(ref store) = self.oxi_store {
            let snapshot_graph = format!("system:skill_graph_snapshot/{}", snapshot_id);
            let copy = format!(
                "DROP SILENT GRAPH <{}> ; \
                 INSERT {{ GRAPH <{}> {{ ?s ?p ?o }} }} \
                 WHERE {{ GRAPH <{}> {{ ?s ?p ?o }} }}",
                snapshot_graph, snapshot_graph, SKILL_GRAPH_NAMED_GRAPH
            );
            if let Err(e) = store.update(copy.as_str()) {
                warn!("Oxigraph snapshot failed: {}", e);
            }
        }

        info!("Snapshot created: {} ({} skills)", snapshot_id, skill_count);
        Ok(snapshot_id)
    }

    pub fn list_snapshots(&self) -> Vec<SnapshotRecord> {
        self.snapshot_history.read().clone()
    }

    // ── P2-1: Hybrid Search ─────────────────────────────────────────────

    /// Fuse two ranked result sets using Reciprocal Rank Fusion.
    /// alpha=1.0 → pure text, alpha=0.0 → pure structural.
    pub fn fuse_results(
        text_hits: &[(String, f32)],
        struct_hits: &[(String, f32)],
        alpha: f32,
    ) -> Vec<FusedHit> {
        let k = 60.0f32;
        let mut score_map: HashMap<String, f32> = HashMap::new();

        for (rank, (iri, _)) in text_hits.iter().enumerate() {
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            *score_map.entry(iri.clone()).or_insert(0.0) += alpha * rrf;
        }
        for (rank, (iri, _)) in struct_hits.iter().enumerate() {
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            *score_map.entry(iri.clone()).or_insert(0.0) += (1.0 - alpha) * rrf;
        }

        let mut fused: Vec<FusedHit> = score_map
            .into_iter()
            .map(|(iri, score)| FusedHit { iri, score })
            .collect();
        fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        fused
    }

    pub async fn hybrid_skill_search(
        &self,
        skill_iri: &str,
        text_ranked: Vec<(String, f32)>,
        struct_ranked: Vec<(String, f32)>,
        alpha: f32,
        embedder: Option<&SkillGraphEmbedder>,
    ) -> Vec<(String, SkillLinkType, f32)> {
        let struct_ranked = if struct_ranked.is_empty() {
            if let Some(emb) = embedder {
                emb.rank_by_similarity(skill_iri, 20)
            } else {
                struct_ranked
            }
        } else {
            struct_ranked
        };

        let fused = Self::fuse_results(&text_ranked, &struct_ranked, alpha);
        let mut results: Vec<(String, SkillLinkType, f32)> = fused
            .into_iter()
            .filter(|h| h.iri != skill_iri)
            .take(10)
            .map(|h| (h.iri, SkillLinkType::Related, h.score))
            .collect();
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Suggest related skills for a given skill.
    ///
    /// Uses semantic search via `SkillDiscoveryEngine` when available,
    /// falling back to Jaccard tag overlap when no engine is provided
    /// or the engine has no vector store configured.
    pub async fn suggest_links(
        &self,
        skill_iri: &str,
        engine: Option<&SkillDiscoveryEngine>,
    ) -> Vec<(String, SkillLinkType, f32)> {
        // Build query from skill metadata (drop lock before potential await)
        let query = {
            let skills = self.skills.read();
            skills.get(skill_iri).map(|s| {
                let tags = s.tags.join(" ");
                format!("{} {} {}", s.name, s.description, tags)
            })
        };

        if let Some(ref q) = query {
            if let Some(engine) = engine {
                if let Ok(matches) = engine.semantic_search(q, 10).await {
                    if !matches.is_empty() {
                        return matches
                            .into_iter()
                            .map(|m| {
                                (m.skill.skill_iri.clone(), SkillLinkType::Related, m.relevance_score)
                            })
                            .collect();
                    }
                }
            }
        }

        let mut suggestions = Vec::new();
        let skills = self.skills.read();

        if let Some(source) = skills.get(skill_iri) {
            let source_tags: HashSet<&String> = source.tags.iter().collect();

            for (other_iri, other) in skills.iter() {
                if other_iri == skill_iri {
                    continue;
                }

                let other_tags: HashSet<&String> = other.tags.iter().collect();
                let common_tags = source_tags.intersection(&other_tags).count();

                if common_tags > 0 {
                    let similarity = common_tags as f32
                        / ((source_tags.len() + other_tags.len()) as f32 / 2.0);

                    if similarity > 0.3 {
                        suggestions.push((other_iri.clone(), SkillLinkType::Related, similarity));
                    }
                }
            }
        }

        suggestions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(10);
        suggestions
    }

    /// Bulk read skills by a list of IRIs.
    /// Returns only the skills that exist, in the same order as the input.
    /// Missing IRIs are silently skipped.
    pub fn bulk_read_skills(&self, iris: &[&str]) -> Vec<SkillGraphNode> {
        let skills = self.skills.read();
        iris.iter()
            .filter_map(|iri| skills.get(*iri).cloned())
            .collect()
    }

    /// Create a composite skill from a set of component skill IRIs.
    /// Composition links are created both on the composite and on each component.
    pub fn create_composite_skill(
        &self,
        target_iri: &str,
        name: &str,
        description: &str,
        component_iris: &[String],
        reason: &str,
    ) -> Result<SkillGraphNode, CoreError> {
        let links: Vec<SkillLink> = component_iris
            .iter()
            .map(|comp_iri| SkillLink {
                link_type: SkillLinkType::Composition,
                target_iri: comp_iri.to_string(),
                strength: LinkStrength::Required,
                description: format!("Composite part: {}", reason),
            })
            .collect();

        for comp_iri in component_iris {
            let existing = self.skills.read().get(comp_iri).cloned();
            if let Some(mut comp) = existing {
                let already_linked = comp.links.iter().any(|l| {
                    l.link_type == SkillLinkType::Composition && l.target_iri == target_iri
                });
                if !already_linked {
                    comp.links.push(SkillLink {
                        link_type: SkillLinkType::Composition,
                        target_iri: target_iri.to_string(),
                        strength: LinkStrength::Recommended,
                        description: "Is a composite skill".to_string(),
                    });
                    self.index.update_skill(&comp);
                    self.skills.write().insert(comp_iri.clone(), comp);
                }
            }
        }

        let comp_tags: Vec<String> = {
            let snap: Vec<_> = component_iris
                .iter()
                .filter_map(|iri| self.skills.read().get(iri).cloned())
                .collect();
            snap.iter().flat_map(|s| s.tags.clone()).collect()
        };

        let composite = SkillGraphNode {
            skill_iri: target_iri.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            version: "1.0.0".to_string(),
            node_type: SkillNodeType::Composite,
            maturity: "experimental".to_string(),
            tags: {
                let mut t: Vec<String> = comp_tags;
                t.sort();
                t.dedup();
                t
            },
            w2h: Skill5W2H::default(),
            links,
            graph_meta: SkillGraphMeta::new(),
            content: None,
            attached_to: None,
            security_info: None,
            mcp_server_id: None,
            storage_tier: StorageTier::L0Permanent,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_used_at: None,
        };

        self.register_skill(composite.clone())?;
        Ok(composite)
    }

    /// Mark a skill as deprecated. It stays in the store for reference but is
    /// removed from the search index.
    pub fn deprecate_skill(&self, skill_iri: &str) -> Result<(), CoreError> {
        let (old, new) = {
            let mut skills = self.skills.write();
            match skills.get_mut(skill_iri) {
                Some(skill) => {
                    let old = skill.clone();
                    skill.maturity = "deprecated".to_string();
                    let new = skill.clone();
                    (Some(old), Some(new))
                }
                None => (None, None),
            }
        };
        if let (Some(old), Some(new)) = (old, new) {
            self.index.remove_skill(skill_iri);
            self.skills.write().insert(skill_iri.to_string(), new.clone());
            info!("Skill marked as deprecated: {}", skill_iri);
            self.emit_mutation(GraphMutation::SkillUpdated { old, new });
            Ok(())
        } else {
            Err(CoreError::SkillNotFound {
                iri: format!("Skill not found: {}", skill_iri),
            })
        }
    }

    /// Batch add multiple links. Returns count of successfully added links.
    pub fn batch_add_links(
        &self,
        links: &[(
            String,
            String,
            SkillLinkType,
            LinkStrength,
            String,
        )],
    ) -> usize {
        let mut success_count = 0usize;
        for (source, target, link_type, strength, description) in links {
            match self.add_link(source, target, *link_type, *strength, description) {
                Ok(()) => success_count += 1,
                Err(e) => {
                    warn!("Batch add link failed ({} -> {}): {:?}", source, target, e);
                }
            }
        }
        success_count
    }

    pub fn index_stats(&self) -> crate::skill_graph::index::IndexStats {
        self.index.stats()
    }
}

impl Default for SkillGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_skill_records_timeline_mutation() {
        let timeline = Arc::new(TimelineStore::new(100, 10));
        let store = SkillGraphStore::new().with_timeline(timeline.clone());

        assert_eq!(timeline.pending_mutations(), 0);

        let skill = SkillGraphNode::new("iri://skills/tl-test", "TL Test", "verifies mutation log");
        store.register_skill(skill).unwrap();

        // The structural mutation must now be reflected in the timeline.
        assert_eq!(timeline.pending_mutations(), 1);

        // Without an attached timeline, no mutation is recorded.
        let store2 = SkillGraphStore::new();
        let skill2 = SkillGraphNode::new("iri://skills/tl-test2", "TL Test 2", "no timeline");
        store2.register_skill(skill2).unwrap();
        assert_eq!(timeline.pending_mutations(), 1);
    }

    #[test]
    fn test_register_and_get_skill() {
        let store = SkillGraphStore::new();
        let skill = SkillGraphNode::new(
            "iri://skills/test-skill",
            "Test Skill",
            "A test skill",
        );

        store.register_skill(skill.clone()).unwrap();
        
        let retrieved = store.get_skill("iri://skills/test-skill").unwrap();
        assert_eq!(retrieved.name, "Test Skill");
    }

    #[test]
    fn test_add_link() {
        let store = SkillGraphStore::new();
        
        let skill1 = SkillGraphNode::new("iri://skills/skill1", "Skill 1", "First skill");
        let skill2 = SkillGraphNode::new("iri://skills/skill2", "Skill 2", "Second skill");
        
        store.register_skill(skill1).unwrap();
        store.register_skill(skill2).unwrap();
        
        store.add_link(
            "iri://skills/skill1",
            "iri://skills/skill2",
            SkillLinkType::Prerequisite,
            LinkStrength::Required,
            "Skill 2 is required before Skill 1",
        ).unwrap();
        
        let skill = store.get_skill("iri://skills/skill1").unwrap();
        assert_eq!(skill.links.len(), 1);
        assert_eq!(skill.links[0].link_type, SkillLinkType::Prerequisite);
    }

    #[test]
    fn test_resolve_dependencies() {
        let store = SkillGraphStore::new();
        
        let skill_a = SkillGraphNode::new("iri://skills/a", "A", "Skill A");
        let mut skill_b = SkillGraphNode::new("iri://skills/b", "B", "Skill B");
        skill_b.add_prerequisite("iri://skills/a", "A is required");
        let mut skill_c = SkillGraphNode::new("iri://skills/c", "C", "Skill C");
        skill_c.add_prerequisite("iri://skills/b", "B is required");
        
        store.register_skill(skill_a).unwrap();
        store.register_skill(skill_b).unwrap();
        store.register_skill(skill_c).unwrap();
        
        let deps = store.resolve_dependencies("iri://skills/c");
        
        assert_eq!(deps, vec!["iri://skills/a", "iri://skills/b", "iri://skills/c"]);
    }

    #[test]
    fn test_find_skills_by_5w2h() {
        let store = SkillGraphStore::new();
        
        let w2h = Skill5W2H::new("JWT Authentication", "Secure API access")
            .with_phase("Do")
            .with_agent_role("DA");
        
        let skill = SkillGraphNode::new(
            "iri://skills/jwt-auth",
            "JWT Auth",
            "JWT authentication implementation",
        ).with_5w2h(w2h);
        
        store.register_skill(skill).unwrap();
        
        let results = store.find_skills_by_5w2h(
            Some("jwt"),
            None,
            Some("Do"),
            Some("DA"),
            None,
        );
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].skill_iri, "iri://skills/jwt-auth");
    }

    #[test]
    fn test_find_alternatives() {
        let store = SkillGraphStore::new();
        
        let mut skill1 = SkillGraphNode::new("iri://skills/jwt", "JWT", "JWT auth");
        skill1.add_alternative("iri://skills/oauth", "OAuth is an alternative");
        
        let skill2 = SkillGraphNode::new("iri://skills/oauth", "OAuth", "OAuth auth");
        
        store.register_skill(skill1).unwrap();
        store.register_skill(skill2).unwrap();
        
        let alts = store.find_alternatives("iri://skills/jwt");
        assert_eq!(alts.len(), 1);
        assert_eq!(alts[0].0, "iri://skills/oauth");
    }

    #[test]
    fn test_disclosure_levels() {
        let store = SkillGraphStore::new();
        
        let skill = SkillGraphNode::new(
            "iri://skills/test",
            "Test Skill",
            "A test skill",
        ).with_tag("testing");
        
        store.register_skill(skill).unwrap();
        
        let l1 = store.get_skill_at_level("iri://skills/test", DisclosureLevel::MOCIndex).unwrap();
        assert!(l1.get("name").is_some());
        assert!(l1.get("what").is_none());
        
        let l2 = store.get_skill_at_level("iri://skills/test", DisclosureLevel::Summary5W2H).unwrap();
        assert!(l2.get("what").is_some());
    }

    #[test]
    fn test_record_usage() {
        let store = SkillGraphStore::new();
        
        let skill = SkillGraphNode::new("iri://skills/test", "Test", "Test skill");
        store.register_skill(skill).unwrap();
        
        store.record_skill_usage("iri://skills/test", true).unwrap();
        store.record_skill_usage("iri://skills/test", false).unwrap();
        
        let updated = store.get_skill("iri://skills/test").unwrap();
        assert_eq!(updated.graph_meta.usage_count, 2);
        assert_eq!(updated.graph_meta.success_rate, 0.5);
    }

    #[test]
    fn test_moc_node() {
        let store = SkillGraphStore::new();
        
        let mut moc = MOCNode::new(
            "iri://moc/auth",
            "Authentication",
            "Authentication skills",
        );
        moc.add_entry_point("iri://skills/jwt");
        moc.add_entry_point("iri://skills/oauth");
        
        store.register_moc(moc).unwrap();
        
        let retrieved = store.get_moc("iri://moc/auth").unwrap();
        assert_eq!(retrieved.skill_count, 2);
    }

    #[test]
    fn test_knowledge_fragment() {
        let store = SkillGraphStore::new();
        
        let skill = SkillGraphNode::new("iri://skills/jwt", "JWT", "JWT auth");
        store.register_skill(skill).unwrap();
        
        let fragment = store.create_fragment(
            "iri://fragment/jwt-1",
            "iri://skills/jwt",
            "Token expiration issues",
            "Use refresh tokens",
            Some("agent:ca/001"),
        ).unwrap();
        
        assert_eq!(fragment.problem, "Token expiration issues");
        
        let fragments = store.get_fragments_for_skill("iri://skills/jwt");
        assert_eq!(fragments.len(), 1);
    }

    #[tokio::test]
    async fn test_suggest_links() {
        let store = SkillGraphStore::new();
        
        let skill1 = SkillGraphNode::new("iri://skills/rust-auth", "Rust Auth", "Auth in Rust")
            .with_tag("rust")
            .with_tag("authentication");
        
        let skill2 = SkillGraphNode::new("iri://skills/rust-crypto", "Rust Crypto", "Crypto in Rust")
            .with_tag("rust")
            .with_tag("cryptography");
        
        let skill3 = SkillGraphNode::new("iri://skills/python-auth", "Python Auth", "Auth in Python")
            .with_tag("python")
            .with_tag("authentication");
        
        store.register_skill(skill1).unwrap();
        store.register_skill(skill2).unwrap();
        store.register_skill(skill3).unwrap();
        
        let suggestions = store.suggest_links("iri://skills/rust-auth", None).await;
        
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|(iri, _, _)| iri == "iri://skills/rust-crypto"));
    }

    #[test]
    fn test_index_auto_updated_on_register() {
        let store = SkillGraphStore::new();

        let skill = SkillGraphNode::new("iri://skills/jwt", "JWT Auth", "JWT")
            .with_tag("auth")
            .with_tag("jwt");
        store.register_skill(skill).unwrap();

        let stats = store.index_stats();
        assert_eq!(stats.total_skills, 1);
        assert_eq!(stats.tag_count, 2);
    }

    #[test]
    fn test_find_by_tags_uses_index() {
        let store = SkillGraphStore::new();

        let s1 = SkillGraphNode::new("iri://skills/s1", "S1", "S1")
            .with_tag("auth")
            .with_tag("jwt");
        let s2 = SkillGraphNode::new("iri://skills/s2", "S2", "S2")
            .with_tag("auth")
            .with_tag("oauth");
        store.register_skill(s1).unwrap();
        store.register_skill(s2).unwrap();

        let results = store.find_skills_by_tags(&["auth"]);
        assert_eq!(results.len(), 2);

        let results = store.find_skills_by_tags(&["auth", "jwt"]);
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_hybrid_search_with_embedder() {
        let store = Arc::new(SkillGraphStore::new());
        let embedder = SkillGraphEmbedder::new(store.clone());

        let foundational = SkillGraphNode::new("iri://skills/foundation", "Foundation", "Base")
            .with_tag("core");
        store.register_skill(foundational).unwrap();

        let mut intermediate =
            SkillGraphNode::new("iri://skills/intermediate", "Intermediate", "Mid-level")
                .with_tag("core");
        intermediate.add_prerequisite("iri://skills/foundation", "Needs foundation");
        store.register_skill(intermediate).unwrap();

        let mut advanced =
            SkillGraphNode::new("iri://skills/advanced", "Advanced", "Top-level")
                .with_tag("core");
        advanced.add_prerequisite("iri://skills/intermediate", "Needs intermediate");
        store.register_skill(advanced).unwrap();

        let unrelated = SkillGraphNode::new("iri://skills/unrelated", "Unrelated", "Different")
            .with_tag("other");
        store.register_skill(unrelated).unwrap();

        let results = store
            .hybrid_skill_search("iri://skills/intermediate", vec![], vec![], 1.0, Some(&embedder))
            .await;
        assert!(!results.is_empty(), "Should find structural neighbors");
        assert!(
            results.iter().any(|(iri, _, _)| iri == "iri://skills/foundation"),
            "Foundational skill (same chain) should appear"
        );
        assert!(
            results.iter().any(|(iri, _, _)| iri == "iri://skills/advanced"),
            "Advanced skill (same chain) should appear"
        );

        let text_only: Vec<(String, f32)> = vec![
            ("iri://skills/foundation".to_string(), 0.9),
            ("iri://skills/advanced".to_string(), 0.8),
            ("iri://skills/unrelated".to_string(), 0.1),
        ];
        let results = store
            .hybrid_skill_search(
                "iri://skills/intermediate",
                text_only,
                vec![],
                1.0,
                None,
            )
            .await;
        assert_eq!(results.len(), 3);
        // With alpha=1.0 (pure text), foundation should be first
        assert_eq!(results[0].0, "iri://skills/foundation");

        let text_ranked: Vec<(String, f32)> = vec![
            ("iri://skills/unrelated".to_string(), 0.9),
            ("iri://skills/foundation".to_string(), 0.2),
        ];
        let struct_ranked: Vec<(String, f32)> = vec![
            ("iri://skills/foundation".to_string(), 0.9),
            ("iri://skills/unrelated".to_string(), 0.1),
        ];
        let results = store
            .hybrid_skill_search(
                "iri://skills/intermediate",
                text_ranked,
                struct_ranked,
                0.5,
                None,
            )
            .await;
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_search_self_excluded() {
        let store = Arc::new(SkillGraphStore::new());
        let embedder = SkillGraphEmbedder::new(store.clone());

        let skill = SkillGraphNode::new("iri://skills/self", "Self", "Self");
        store.register_skill(skill).unwrap();

        let other = SkillGraphNode::new("iri://skills/other", "Other", "Other")
            .with_tag("related");
        store.register_skill(other).unwrap();

        let results = store
            .hybrid_skill_search("iri://skills/self", vec![], vec![], 1.0, Some(&embedder))
            .await;
        assert!(
            results.iter().all(|(iri, _, _)| iri != "iri://skills/self"),
            "Self should be excluded from results"
        );
    }

    #[tokio::test]
    async fn test_hybrid_search_empty_store() {
        let store = Arc::new(SkillGraphStore::new());
        let embedder = SkillGraphEmbedder::new(store.clone());

        let results = store
            .hybrid_skill_search(
                "iri://skills/nonexistent",
                vec![],
                vec![],
                0.5,
                Some(&embedder),
            )
            .await;
        assert!(results.is_empty(), "Empty store should return no results");
    }
}
