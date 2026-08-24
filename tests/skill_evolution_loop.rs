use std::sync::Arc;

use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::skill_graph::evolution::{SkillEvolutionEngine, UsageRecord};
use glidinghorse::skill_graph::graph_store::SkillGraphStore;
use glidinghorse::skill_graph::types::SkillGraphNode;
use glidinghorse::tools::skill_registry::SkillRegistry;

#[test]
fn evolution_probe_skill_closes_the_persistent_evidence_loop() {
    let registry = SkillRegistry::new();
    assert_eq!(
        registry
            .load_from_jsonld("skills/evolution_probe/skill.jsonld")
            .unwrap(),
        1
    );
    let probe = registry
        .get_skill("iri://skills/evolution_probe")
        .expect("fixture skill must be loaded");
    assert_eq!(probe.name, "evolution_probe");

    let graph = Arc::new(SkillGraphStore::new());
    graph
        .register_skill(SkillGraphNode::from_skill_meta(&probe))
        .unwrap();
    graph
        .register_skill(SkillGraphNode::new(
            "iri://skills/evolution_companion",
            "evolution_companion",
            "Companion skill for evolution testing",
        ))
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let l0 = Arc::new(L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap());
    let mut engine = SkillEvolutionEngine::new(graph.clone()).with_usage_persistence(l0.clone());

    for (task, success) in [("success-1", true), ("failure-1", false), ("success-2", true)] {
        let mut usage = UsageRecord::new(
            "iri://skills/evolution_probe",
            &format!("iri://task/{task}"),
            "agent:test-evolution",
            success,
        )
        .with_tokens(100)
        .with_context_tag("evolution_probe");
        if !success {
            usage = usage.with_error("deterministic probe failure");
        }
        engine.record_usage(usage).unwrap();
    }

    let stats = engine.get_usage_stats("iri://skills/evolution_probe");
    assert_eq!(stats.total_usage, 3);
    assert_eq!(stats.successful, 2);
    assert_eq!(stats.failed, 1);
    assert!(!engine.get_pending_suggestions().is_empty());

    let restored = SkillEvolutionEngine::new(graph).with_usage_persistence(l0);
    let restored_stats = restored.get_usage_stats("iri://skills/evolution_probe");
    assert_eq!(restored_stats.total_usage, 3);
    assert_eq!(restored_stats.failed, 1);
}
