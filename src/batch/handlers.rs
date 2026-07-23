use serde_json::{json, Value};
use tracing::warn;

use crate::batch::types::ExtractionResult;
use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::types::{LinkStrength, SkillLinkType};

/// A pending event to be emitted by the caller (who has an async runtime).
#[derive(Debug, Clone)]
pub struct PendingEvent {
    pub event_type: String,
    pub payload: Value,
}

/// Summary of what a maintenance handler did.
#[derive(Debug, Clone, Default)]
pub struct HandlerOutcome {
    pub handler_name: String,
    pub actions_taken: Vec<String>,
    pub success_count: usize,
    pub error_count: usize,
    pub pending_events: Vec<PendingEvent>,
}

/// Dispatch to the correct handler based on agent name.
pub fn run_handler(
    agent_name: &str,
    result: &ExtractionResult,
    graph: &SkillGraphStore,
) -> HandlerOutcome {
    match agent_name {
        "skill_merge" => handle_skill_merge(result, graph),
        "fragment_refine" => handle_fragment_refine(result, graph),
        "entity_resolution" => handle_entity_resolution(result, graph),
        "failure_mining" => handle_failure_mining(result, graph),
        "skill_health" => handle_skill_health(result, graph),
        "memory_compact" => handle_memory_compact(result, graph),
        "link_recommend" => handle_link_recommend(result, graph),
        "template_analyze" => handle_template_analyze(result, graph),
        other => {
            warn!("Unknown maintenance handler: {}", other);
            HandlerOutcome {
                handler_name: other.to_string(),
                ..Default::default()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Skill Merge Agent
// ---------------------------------------------------------------------------
fn handle_skill_merge(result: &ExtractionResult, graph: &SkillGraphStore) -> HandlerOutcome {
    let mut outcome = HandlerOutcome {
        handler_name: "skill_merge".to_string(),
        ..Default::default()
    };

    for entity in &result.entities {
        if entity.confidence < 0.6 {
            continue;
        }

        let merge_relations: Vec<_> = result
            .relations
            .iter()
            .filter(|r| r.relation.to_lowercase() == "merge" && r.from == entity.name)
            .collect();

        if merge_relations.is_empty() {
            continue;
        }

        let component_iris: Vec<String> = merge_relations.iter().map(|r| r.to.clone()).collect();

        let target_iri = format!("iri://skills/composite/{}", entity.name);
        match graph.create_composite_skill(
            &target_iri,
            &entity.name,
            entity.description.as_deref().unwrap_or("Merged skill"),
            &component_iris,
            &format!("auto-merge from batch: confidence={:.2}", entity.confidence),
        ) {
            Ok(skill) => {
                outcome.actions_taken.push(format!(
                    "Created composite skill {} from {} components",
                    skill.name,
                    component_iris.len()
                ));
                outcome.success_count += 1;
                outcome.pending_events.push(PendingEvent {
                    event_type: "BATCH_SKILL_MERGE_APPLIED".into(),
                    payload: json!({ "composite_iri": target_iri, "components": component_iris }),
                });
            }
            Err(e) => {
                warn!("Skill merge failed for {}: {:?}", entity.name, e);
                outcome.error_count += 1;
            }
        }
    }

    outcome
}

// ---------------------------------------------------------------------------
// 2. Fragment Refine Agent
// ---------------------------------------------------------------------------
fn handle_fragment_refine(result: &ExtractionResult, graph: &SkillGraphStore) -> HandlerOutcome {
    let mut outcome = HandlerOutcome {
        handler_name: "fragment_refine".to_string(),
        ..Default::default()
    };

    for entity in &result.entities {
        if entity.confidence < 0.6 {
            continue;
        }
        let problem = entity.description.as_deref().unwrap_or("Refinement needed");
        let fragment_iri = format!("iri://fragment/refined/{}", entity.name);
        let skill_iri = format!("iri://skills/{}", entity.name);

        match graph.create_fragment(
            &fragment_iri,
            &skill_iri,
            problem,
            problem,
            Some("batch:fragment_refine"),
        ) {
            Ok(frag) => {
                outcome.actions_taken.push(format!(
                    "Created fragment {} for skill {}",
                    frag.fragment_iri, skill_iri
                ));
                outcome.success_count += 1;
                outcome.pending_events.push(PendingEvent {
                    event_type: "BATCH_FRAGMENT_REFINED".into(),
                    payload: json!({ "fragment_iri": fragment_iri, "attached_to": skill_iri }),
                });
            }
            Err(e) => {
                warn!("Fragment refine failed for {}: {:?}", entity.name, e);
                outcome.error_count += 1;
            }
        }
    }

    outcome
}

// ---------------------------------------------------------------------------
// 3. Entity Resolution Agent
// ---------------------------------------------------------------------------
fn handle_entity_resolution(result: &ExtractionResult, graph: &SkillGraphStore) -> HandlerOutcome {
    let mut outcome = HandlerOutcome {
        handler_name: "entity_resolution".to_string(),
        ..Default::default()
    };

    let all_skills = graph.list_all_skills();
    for entity in &result.entities {
        if entity.confidence < 0.5 {
            continue;
        }

        let mut matched = false;
        for skill in &all_skills {
            if skill.name.to_lowercase() == entity.name.to_lowercase()
                || skill
                    .tags
                    .iter()
                    .any(|t| t.to_lowercase() == entity.name.to_lowercase())
            {
                matched = true;
                if let Err(e) = graph.add_link(
                    &skill.skill_iri,
                    &format!("iri://concept/{}", entity.name),
                    SkillLinkType::Related,
                    LinkStrength::Recommended,
                    &format!(
                        "Entity resolution: {}",
                        entity.description.as_deref().unwrap_or("")
                    ),
                ) {
                    warn!("Entity resolution add_link failed: {:?}", e);
                    outcome.error_count += 1;
                } else {
                    outcome.actions_taken.push(format!(
                        "Resolved entity '{}' to skill '{}'",
                        entity.name, skill.name
                    ));
                    outcome.success_count += 1;
                    outcome.pending_events.push(PendingEvent {
                        event_type: "BATCH_ENTITY_RESOLVED".into(),
                        payload: json!({ "entity": entity.name, "resolved_to": skill.skill_iri }),
                    });
                }
                break;
            }
        }

        if !matched {
            outcome.actions_taken.push(format!(
                "Entity '{}' (conf={:.2}) had no matching skill — unresolved",
                entity.name, entity.confidence
            ));
        }
    }

    outcome
}

// ---------------------------------------------------------------------------
// 4. Failure Mining Agent
// ---------------------------------------------------------------------------
fn handle_failure_mining(result: &ExtractionResult, graph: &SkillGraphStore) -> HandlerOutcome {
    let mut outcome = HandlerOutcome {
        handler_name: "failure_mining".to_string(),
        ..Default::default()
    };

    for entity in &result.entities {
        if entity.confidence < 0.65 {
            continue;
        }
        let problem = entity
            .description
            .as_deref()
            .unwrap_or("Unknown failure pattern");
        let skill_iri = format!("iri://skills/{}", entity.name);

        let frag_iri = format!("iri://fragment/failure/{}", entity.name);
        match graph.create_fragment(
            &frag_iri,
            &skill_iri,
            problem,
            &format!(
                "Auto-detected failure. {} actions suggested.",
                result.key_decisions.len()
            ),
            Some("batch:failure_mining"),
        ) {
            Ok(frag) => {
                outcome.actions_taken.push(format!(
                    "Recorded failure pattern '{}' for skill {}",
                    frag.problem, skill_iri
                ));
                outcome.success_count += 1;
                outcome.pending_events.push(PendingEvent {
                    event_type: "BATCH_FAILURE_PATTERN_DETECTED".into(),
                    payload: json!({ "pattern": problem, "skill": skill_iri }),
                });
            }
            Err(e) => {
                warn!("Failure mining fragment creation failed: {:?}", e);
                outcome.error_count += 1;
            }
        }
    }

    outcome
}

// ---------------------------------------------------------------------------
// 5. Skill Health Agent
// ---------------------------------------------------------------------------
fn handle_skill_health(result: &ExtractionResult, graph: &SkillGraphStore) -> HandlerOutcome {
    let mut outcome = HandlerOutcome {
        handler_name: "skill_health".to_string(),
        ..Default::default()
    };

    let _ = result; // extraction result context_summary is available for richer reporting
    let all = graph.list_all_skills();
    let deprecated: Vec<_> = all.iter().filter(|s| s.maturity == "deprecated").collect();
    let total = all.len();
    let stats = graph.index_stats();

    outcome.actions_taken.push(format!(
        "Health report: {} skills total, {} deprecated, {} tags, {} roles",
        total,
        deprecated.len(),
        stats.tag_count,
        stats.role_count,
    ));
    outcome.success_count = total;

    outcome.pending_events.push(PendingEvent {
        event_type: "BATCH_HEALTH_REPORT_GENERATED".into(),
        payload: json!({
            "total_skills": total,
            "deprecated": deprecated.len(),
            "tag_count": stats.tag_count,
            "role_count": stats.role_count,
        }),
    });

    outcome
}

// ---------------------------------------------------------------------------
// 6. Memory Compact Agent
// ---------------------------------------------------------------------------
fn handle_memory_compact(result: &ExtractionResult, graph: &SkillGraphStore) -> HandlerOutcome {
    let mut outcome = HandlerOutcome {
        handler_name: "memory_compact".to_string(),
        ..Default::default()
    };

    for entity in &result.entities {
        if entity.confidence < 0.7 {
            continue;
        }
        let skill_iri = format!("iri://skills/{}", entity.name);

        if graph.get_skill(&skill_iri).is_some() {
            match graph.deprecate_skill(&skill_iri) {
                Ok(()) => {
                    outcome.actions_taken.push(format!(
                        "Deprecated skill {} for memory compaction",
                        skill_iri
                    ));
                    outcome.success_count += 1;
                    outcome.pending_events.push(PendingEvent {
                        event_type: "BATCH_MEMORY_COMPACTED".into(),
                        payload: json!({ "deprecated_iri": skill_iri }),
                    });
                }
                Err(e) => {
                    warn!(
                        "Memory compact deprecation failed for {}: {:?}",
                        skill_iri, e
                    );
                    outcome.error_count += 1;
                }
            }
        }
    }

    if outcome.success_count == 0 {
        outcome
            .actions_taken
            .push("No skills identified for compaction".to_string());
    }

    outcome
}

// ---------------------------------------------------------------------------
// 7. Link Recommend Agent
// ---------------------------------------------------------------------------
fn handle_link_recommend(result: &ExtractionResult, graph: &SkillGraphStore) -> HandlerOutcome {
    let mut outcome = HandlerOutcome {
        handler_name: "link_recommend".to_string(),
        ..Default::default()
    };

    let mut batch_links: Vec<(String, String, SkillLinkType, LinkStrength, String)> = Vec::new();
    for rel in &result.relations {
        if rel.confidence < 0.5 {
            continue;
        }
        let link_type = match rel.relation.to_lowercase().as_str() {
            "prerequisite" | "depends_on" => SkillLinkType::Prerequisite,
            "alternative" | "replaces" => SkillLinkType::Alternative,
            "related" | "similar_to" => SkillLinkType::Related,
            "composition" | "part_of" => SkillLinkType::Composition,
            "extends" | "specializes" => SkillLinkType::Extends,
            _ => SkillLinkType::Related,
        };
        let strength = if rel.confidence > 0.8 {
            LinkStrength::Required
        } else if rel.confidence > 0.6 {
            LinkStrength::Recommended
        } else {
            LinkStrength::Optional
        };
        let from_iri = format!("iri://skills/{}", rel.from);
        let to_iri = format!("iri://skills/{}", rel.to);
        batch_links.push((
            from_iri,
            to_iri,
            link_type,
            strength,
            rel.properties
                .get("description")
                .cloned()
                .unwrap_or_default(),
        ));
    }

    if !batch_links.is_empty() {
        let added = graph.batch_add_links(&batch_links);
        outcome.success_count = added;
        outcome.actions_taken.push(format!(
            "Added {} batch links from {} extracted relations",
            added,
            batch_links.len(),
        ));
        outcome.pending_events.push(PendingEvent {
            event_type: "BATCH_LINK_APPLIED".into(),
            payload: json!({ "links_added": added, "total_candidates": batch_links.len() }),
        });
    } else {
        outcome
            .actions_taken
            .push("No link recommendations from extraction".to_string());
    }

    outcome
}

// ---------------------------------------------------------------------------
// 8. Template Analyze Agent
// ---------------------------------------------------------------------------
fn handle_template_analyze(result: &ExtractionResult, graph: &SkillGraphStore) -> HandlerOutcome {
    let mut outcome = HandlerOutcome {
        handler_name: "template_analyze".to_string(),
        ..Default::default()
    };

    let _ = graph; // future: could validate template coverage against existing skills
    let entity_count = result.entities.len();
    let relation_count = result.relations.len();
    let decision_count = result.key_decisions.len();

    outcome.actions_taken.push(format!(
        "Template analysis complete: {} entities, {} relations, {} decisions",
        entity_count, relation_count, decision_count,
    ));
    outcome.success_count = 1;

    outcome.pending_events.push(PendingEvent {
        event_type: "BATCH_TEMPLATE_ANALYSIS_READY".into(),
        payload: json!({
            "entities": entity_count,
            "relations": relation_count,
            "decisions": decision_count,
            "context_summary": result.context_summary,
        }),
    });

    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::types::{
        DetectedIntent, ExtractedDecision, ExtractedEntity, ExtractedRelation, ExtractionResult,
    };
    use crate::skill_graph::graph_store::SkillGraphStore;
    use crate::skill_graph::types::SkillGraphNode;
    use chrono::Utc;
    use std::collections::HashMap;

    fn make_result() -> ExtractionResult {
        ExtractionResult {
            batch_id: "test-batch".into(),
            extracted_at: Utc::now(),
            entities: vec![
                ExtractedEntity {
                    name: "auth_combo".into(),
                    entity_type: "composite".into(),
                    description: Some("Combined auth service".into()),
                    aliases: vec![],
                    confidence: 0.85,
                    source_messages: vec![],
                },
                ExtractedEntity {
                    name: "failure_pattern_x".into(),
                    entity_type: "failure".into(),
                    description: Some("Timeout in retry loop".into()),
                    aliases: vec![],
                    confidence: 0.75,
                    source_messages: vec![],
                },
                ExtractedEntity {
                    name: "unresolved_concept".into(),
                    entity_type: "concept".into(),
                    description: Some("A new concept".into()),
                    aliases: vec![],
                    confidence: 0.6,
                    source_messages: vec![],
                },
            ],
            relations: vec![
                ExtractedRelation {
                    from: "auth_combo".into(),
                    relation: "merge".into(),
                    to: "iri://skills/jwt".into(),
                    properties: HashMap::new(),
                    confidence: 0.9,
                },
                ExtractedRelation {
                    from: "auth_combo".into(),
                    relation: "merge".into(),
                    to: "iri://skills/oauth".into(),
                    properties: HashMap::new(),
                    confidence: 0.9,
                },
                ExtractedRelation {
                    from: "jwt".into(),
                    relation: "prerequisite".into(),
                    to: "oauth".into(),
                    properties: {
                        let mut m = HashMap::new();
                        m.insert("description".into(), "JWT required before OAuth".into());
                        m
                    },
                    confidence: 0.75,
                },
            ],
            intent: Some(DetectedIntent {
                intent_type: "maintenance".into(),
                confidence: 0.8,
                details: HashMap::new(),
            }),
            key_decisions: vec![ExtractedDecision {
                decision: "Merge skills".into(),
                rationale: Some("They overlap significantly".into()),
                evidence: vec![],
                confidence: "0.8".into(),
            }],
            context_summary: "Analysis of skill graph health".into(),
            llm_calls: 1,
            tokens_consumed: 500,
            confidence_scores: HashMap::new(),
            raw_response: None,
        }
    }

    #[test]
    fn test_handler_dispatch_unknown() {
        let result = make_result();
        let graph = SkillGraphStore::new();
        let outcome = run_handler("nonexistent", &result, &graph);
        assert_eq!(outcome.handler_name, "nonexistent");
        assert_eq!(outcome.success_count, 0);
    }

    #[test]
    fn test_handle_skill_merge() {
        let result = make_result();
        let graph = SkillGraphStore::new();

        let jwt = SkillGraphNode::new("iri://skills/jwt", "JWT", "JWT auth");
        let oauth = SkillGraphNode::new("iri://skills/oauth", "OAuth", "OAuth auth");
        graph.register_skill(jwt).unwrap();
        graph.register_skill(oauth).unwrap();

        let outcome = handle_skill_merge(&result, &graph);
        assert!(
            outcome.success_count > 0,
            "Expected at least one merge action"
        );
        assert!(
            outcome
                .actions_taken
                .iter()
                .any(|a| a.contains("composite")),
            "Actions should mention composite creation: {:?}",
            outcome.actions_taken,
        );

        let composite = graph.get_skill("iri://skills/composite/auth_combo");
        assert!(composite.is_some(), "Composite skill should exist");
        if let Some(skill) = composite {
            assert!(
                skill
                    .links
                    .iter()
                    .any(|l| l.target_iri == "iri://skills/jwt"),
                "Composite should link to jwt"
            );
        }
    }

    #[test]
    fn test_handle_entity_resolution() {
        let result = make_result();
        let graph = SkillGraphStore::new();

        let skill = SkillGraphNode::new(
            "iri://skills/auth_combo",
            "auth_combo",
            "Combined auth service",
        );
        graph.register_skill(skill).unwrap();

        let outcome = handle_entity_resolution(&result, &graph);
        assert!(
            outcome.success_count > 0
                || outcome
                    .actions_taken
                    .iter()
                    .any(|a| a.contains("unresolved")),
            "Expected at least resolution or unresolvable notice: {:?}",
            outcome.actions_taken,
        );
    }

    #[test]
    fn test_handle_link_recommend() {
        let result = make_result();
        let graph = SkillGraphStore::new();

        let jwt = SkillGraphNode::new("iri://skills/jwt", "jwt", "JWT");
        let oauth = SkillGraphNode::new("iri://skills/oauth", "oauth", "OAuth");
        graph.register_skill(jwt).unwrap();
        graph.register_skill(oauth).unwrap();

        let outcome = handle_link_recommend(&result, &graph);
        assert!(outcome.success_count > 0, "Expected links to be added");
    }

    #[test]
    fn test_handle_skill_health() {
        let result = make_result();
        let graph = SkillGraphStore::new();

        let s1 = SkillGraphNode::new("iri://skills/s1", "S1", "Skill 1").with_tag("auth");
        let s2 = SkillGraphNode::new("iri://skills/s2", "S2", "Skill 2").with_tag("db");
        graph.register_skill(s1).unwrap();
        graph.register_skill(s2).unwrap();

        let outcome = handle_skill_health(&result, &graph);
        assert!(
            outcome.success_count >= 2,
            "Health report should count all skills"
        );
        assert!(
            outcome
                .actions_taken
                .iter()
                .any(|a| a.contains("Health report")),
            "Should contain health report summary"
        );
    }

    #[test]
    fn test_handle_failure_mining() {
        let result = make_result();
        let graph = SkillGraphStore::new();

        let skill = SkillGraphNode::new(
            "iri://skills/failure_pattern_x",
            "failure_pattern_x",
            "Has failures",
        );
        graph.register_skill(skill).unwrap();

        let outcome = handle_failure_mining(&result, &graph);
        assert!(
            outcome.success_count > 0,
            "Expected failure patterns recorded"
        );
    }

    #[test]
    fn test_handle_memory_compact() {
        let result = make_result();
        let graph = SkillGraphStore::new();

        let skill = SkillGraphNode::new(
            "iri://skills/failure_pattern_x",
            "failure_pattern_x",
            "Deprecate me",
        );
        graph.register_skill(skill).unwrap();

        let outcome = handle_memory_compact(&result, &graph);
        assert!(outcome.success_count > 0, "Expected deprecation");
    }

    #[test]
    fn test_handle_template_analyze() {
        let result = make_result();
        let graph = SkillGraphStore::new();

        let outcome = handle_template_analyze(&result, &graph);
        assert_eq!(
            outcome.success_count, 1,
            "Template analysis always succeeds"
        );
    }

    #[test]
    fn test_handler_dispatch_all() {
        let result = make_result();
        let graph = SkillGraphStore::new();

        let names = [
            "skill_merge",
            "fragment_refine",
            "entity_resolution",
            "failure_mining",
            "skill_health",
            "memory_compact",
            "link_recommend",
            "template_analyze",
        ];
        for name in &names {
            let outcome = run_handler(name, &result, &graph);
            assert_eq!(outcome.handler_name, *name);
        }
    }
}
