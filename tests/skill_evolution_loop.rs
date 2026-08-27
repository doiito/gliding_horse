use std::sync::Arc;

use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::skill_graph::discovery::{SkillDiscoveryEngine, Task5W2H};
use glidinghorse::skill_graph::evolution::{
    EvolutionProposalStatus, EvolutionProposalStore, SkillEvolutionEngine, UsageRecord,
};
use glidinghorse::skill_graph::graph_store::SkillGraphStore;
use glidinghorse::skill_graph::types::SkillGraphNode;
use glidinghorse::tools::skill_registry::SkillRegistry;

#[tokio::test]
async fn evolution_probe_skill_closes_the_persistent_evidence_loop() {
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

    let dir = tempfile::tempdir().unwrap();
    let l0 = Arc::new(L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap());
    let graph = Arc::new(SkillGraphStore::new().with_l0_store(l0.clone()));
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

    let mut engine = SkillEvolutionEngine::new(graph.clone()).with_usage_persistence(l0.clone());

    for (task, success) in [
        ("success-1", true),
        ("failure-1", false),
        ("success-2", true),
    ] {
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

    // Evolution is governed: causal analysis creates a typed proposal, while
    // approval, validation and commit remain explicit application actions.
    let suggestion = engine.get_pending_suggestions()[0].clone();
    let proposals = EvolutionProposalStore::new(l0.clone());
    let proposal = proposals
        .create_or_get("evolution-probe-fragment", suggestion, graph.as_ref())
        .unwrap();
    proposals
        .approve(
            &proposal.proposal_id,
            "test-reviewer",
            Some("deterministic fixture evidence".into()),
        )
        .unwrap();
    proposals
        .validate_for_commit(&proposal.proposal_id, graph.as_ref())
        .unwrap();
    let committed = proposals
        .commit_validated_link_patch(&proposal.proposal_id, graph.as_ref())
        .unwrap();
    assert_eq!(committed.status, EvolutionProposalStatus::Committed);
    assert_eq!(
        graph
            .get_fragments_for_skill("iri://skills/evolution_probe")
            .len(),
        1
    );

    // Reconstruct both graph and learning engine to prove that usage evidence
    // and the committed knowledge fragment survive an application restart.
    drop(engine);
    drop(graph);
    let restored_graph = Arc::new(SkillGraphStore::new().with_l0_store(l0.clone()));
    assert_eq!(restored_graph.hydrate_from_l0().unwrap(), 2);
    let restored =
        SkillEvolutionEngine::new(restored_graph.clone()).with_usage_persistence(l0.clone());
    let restored_stats = restored.get_usage_stats("iri://skills/evolution_probe");
    assert_eq!(restored_stats.total_usage, 3);
    assert_eq!(restored_stats.failed, 1);

    // A later similar task discovers the evolved skill and receives the
    // governed fragment as reusable knowledge.
    let discovery = SkillDiscoveryEngine::new(restored_graph.clone());
    let matches = discovery
        .discover_for_task(&Task5W2H::new(
            "Exercise a skill self-evolution cycle",
            "Verify evidence survives restart and produces governed proposals",
        ))
        .await;
    assert!(matches
        .iter()
        .any(|matched| matched.skill.skill_iri == "iri://skills/evolution_probe"));
    let fragments = restored_graph.get_fragments_for_skill("iri://skills/evolution_probe");
    assert_eq!(fragments.len(), 1);
    assert!(fragments[0].recommendation.contains("verified mitigation"));
}
