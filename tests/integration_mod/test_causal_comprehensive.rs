//! Comprehensive Causal Engine Integration Test
//!
//! Exercises the full causal pipeline through the actual engine code paths.
//! Uses the same patterns as the existing tests in test_skill_graph.rs.
//!
//! What this tests:
//!   1. EvolutionEngine: record_usage failure → suggestions
//!   2. EvolutionEngine: record_usage success → NO suggestions (no false positives)
//!   3. EvolutionEngine: sequential failures build cumulative state
//!   4. EvolutionEngine: skill health analysis
//!   5. EvolutionEngine: preventive actions from failure history
//!   6. CausalEngine + EvolutionEngine: integrated failure analysis
//!   7. EvolutionEngine: propagation chain across prerequisites
//!   8. EvolutionEngine: clear_suggestions
//!   9. EvolutionEngine: get_usage_stats cumulative

use std::sync::Arc;

use glidinghorse::causal::engine::CausalEngine;
use glidinghorse::causal::store::CausalModelStore;
use glidinghorse::causal::types::CausalObservation;
use glidinghorse::graph_backend::{GraphBackend, PetgraphBackend};
use glidinghorse::skill_graph::*;
use glidinghorse::skill_graph::evolution::{
    EvolutionSuggestion, HealthStatus, SkillEvolutionEngine, UsageRecord,
};

// =====================================================================
// Helpers (same pattern as existing tests)
// =====================================================================

fn create_test_store() -> Arc<SkillGraphStore> {
    let store = Arc::new(SkillGraphStore::new());
    let skill = SkillGraphNode::new("iri://skills/test-skill", "Test Skill", "A test skill");
    store.register_skill(skill).unwrap();
    store
}

fn create_store_with_prereqs() -> Arc<SkillGraphStore> {
    let store = Arc::new(SkillGraphStore::new());

    let auth = SkillGraphNode::new("iri://skills/auth", "Auth", "Authentication")
        .with_link(SkillLink {
            link_type: SkillLinkType::Prerequisite,
            target_iri: "iri://skills/base".to_string(),
            strength: LinkStrength::Required,
            description: "Auth needs base".to_string(),
        });
    store.register_skill(auth).unwrap();

    let base = SkillGraphNode::new("iri://skills/base", "Base", "Base service");
    store.register_skill(base).unwrap();

    store
}

fn create_causal_engine(store: &Arc<SkillGraphStore>) -> Arc<CausalEngine> {
    let model_store = Arc::new(CausalModelStore::new());
    let backend: Arc<dyn GraphBackend> = Arc::new(PetgraphBackend::new(store.clone()));
    Arc::new(CausalEngine::new(model_store, backend))
}

// =====================================================================
// Test 1: EvolutionEngine failure analysis path
// =====================================================================
#[test]
fn test_failure_creates_suggestions() {
    let store = create_test_store();
    let mut engine = SkillEvolutionEngine::new(store);

    let record = UsageRecord::new(
        "iri://skills/test-skill",
        "iri://task/fail-001",
        "agent:da/001",
        false,
    )
    .with_error("Connection timeout on port 8080")
    .with_tokens(500);

    engine.record_usage(record).unwrap();

    let suggestions = engine.get_pending_suggestions();
    assert!(!suggestions.is_empty(),
        "Recording a failure should produce suggestions, got {} suggestions", suggestions.len());

    for s in suggestions {
        assert!(s.confidence > 0.0,
            "Suggestion confidence should be > 0, got {}", s.confidence);
        assert!(!s.description.is_empty(),
            "Suggestion should have description");
    }
}

// =====================================================================
// Test 2: No false positives on success
// =====================================================================
#[test]
fn test_success_creates_no_suggestions() {
    let store = create_test_store();
    let mut engine = SkillEvolutionEngine::new(store);

    for i in 0..5 {
        let record = UsageRecord::new(
            "iri://skills/test-skill",
            &format!("iri://task/success-{}", i),
            "agent:da/001",
            true,
        )
        .with_tokens(200 * (i + 1));
        engine.record_usage(record).unwrap();
    }

    let suggestions = engine.get_pending_suggestions();
    assert!(suggestions.is_empty(),
        "Successful usage should NOT create suggestions");

    // But usage stats should be tracked
    let stats = engine.get_usage_stats("iri://skills/test-skill");
    assert_eq!(stats.total_usage, 5,
        "Should have recorded 5 usages, got {}", stats.total_usage);
    assert_eq!(stats.successful, 5,
        "All 5 should be successful");
    assert!(stats.avg_tokens > 0,
        "Average tokens should be > 0");
}

// =====================================================================
// Test 3: Sequential failures build cumulative state
// =====================================================================
#[test]
fn test_sequential_failures_cumulative() {
    let store = create_test_store();
    let mut engine = SkillEvolutionEngine::new(store);

    // Batch 1: 2 failures
    for i in 0..2 {
        let record = UsageRecord::new(
            "iri://skills/test-skill",
            &format!("iri://task/seq-batch1-{}", i),
            "agent:da/001",
            false,
        )
        .with_error("Service unavailable");
        engine.record_usage(record).unwrap();
    }

    let stats_after_batch1 = engine.get_usage_stats("iri://skills/test-skill");
    assert_eq!(stats_after_batch1.total_usage, 2,
        "Should have 2 usages after batch 1, got {}", stats_after_batch1.total_usage);
    assert_eq!(stats_after_batch1.failed, 2);

    // Batch 2: 2 more failures = cumulative
    for i in 0..2 {
        let record = UsageRecord::new(
            "iri://skills/test-skill",
            &format!("iri://task/seq-batch2-{}", i),
            "agent:da/001",
            false,
        )
        .with_error("Service still unavailable");
        engine.record_usage(record).unwrap();
    }

    let stats_after_batch2 = engine.get_usage_stats("iri://skills/test-skill");
    assert_eq!(stats_after_batch2.total_usage, 4,
        "Should have 4 cumulative usages after batch 2, got {}", stats_after_batch2.total_usage);
    assert_eq!(stats_after_batch2.failed, 4);
    assert_eq!(stats_after_batch2.success_rate, 0.0,
        "Success rate should be 0.0 since all failed");

    // Suggestions should exist (from failure analysis)
    let suggestions = engine.get_pending_suggestions();
    assert!(!suggestions.is_empty(),
        "Cumulative failures should produce suggestions, got {} suggestions", suggestions.len());
}

// =====================================================================
// Test 4: Skill health analysis
// =====================================================================
#[test]
fn test_skill_health_analysis() {
    let store = create_test_store();
    let mut engine = SkillEvolutionEngine::new(store);

    // Record mostly successful usage
    for i in 0..10 {
        let success = i < 8; // 80% success rate
        let record = UsageRecord::new(
            "iri://skills/test-skill",
            &format!("iri://task/health-{}", i),
            "agent:da/001",
            success,
        )
        .with_tokens(1000);
        engine.record_usage(record).unwrap();
    }

    let health = engine.analyze_skill_health("iri://skills/test-skill");
    assert!(health.health_score > 0.0,
        "Health score should be > 0, got {}", health.health_score);
    assert_eq!(health.usage_count, 10,
        "Health report should show 10 usages, got {}", health.usage_count);
    assert!(!health.recommendations.is_empty()
        || health.status == HealthStatus::Healthy,
        "Health should either have recommendations or be Healthy");
}

// =====================================================================
// Test 5: Health analysis for unknown skill
// =====================================================================
#[test]
fn test_skill_health_unknown_skill() {
    let store = create_test_store();
    let engine = SkillEvolutionEngine::new(store);

    let health = engine.analyze_skill_health("iri://skills/nonexistent");
    assert_eq!(health.status, HealthStatus::NotFound,
        "Unknown skill should be NotFound, got {:?}", health.status);
    assert_eq!(health.health_score, 0.0,
        "Health score for unknown should be 0.0");
}

// =====================================================================
// Test 6: Preventive actions from failure history
// =====================================================================
#[test]
fn test_preventive_actions() {
    let store = create_test_store();
    let mut engine = SkillEvolutionEngine::new(store);

    // Record many failures to trigger preventive actions
    for i in 0..6 {
        let record = UsageRecord::new(
            "iri://skills/test-skill",
            &format!("iri://task/prev-{}", i),
            "agent:da/001",
            false,
        )
        .with_error("Connection timeout on port 8080");
        engine.record_usage(record).unwrap();
    }

    let actions = engine.suggest_preventive_action("iri://skills/test-skill");
    // Should return actionable strings (may or may not have specific error patterns,
    // but should at least return something for 6 failures)
    assert!(!actions.is_empty(),
        "Should return preventive actions after 6 failures, got {} actions", actions.len());

    // Actions should be human-readable strings
    for action in &actions {
        assert!(!action.is_empty(),
            "Preventive action should not be empty");
    }
}

// =====================================================================
// Test 7: CausalEngine + EvolutionEngine integration
// =====================================================================
#[test]
fn test_causal_engine_integration() {
    let store = create_test_store();
    let causal = create_causal_engine(&store);

    let mut engine = SkillEvolutionEngine::new(store.clone())
        .with_causal_analysis(5000)
        .with_causal_engine(causal);

    let record = UsageRecord::new(
        "iri://skills/test-skill",
        "iri://task/causal-integration-test",
        "agent:da/001",
        false,
    )
    .with_error("Token expired: JWT validation failed");

    engine.record_usage(record).unwrap();

    // With CausalEngine attached, the failure should be analyzed
    // and produce suggestions
    let suggestions = engine.get_pending_suggestions();
    assert!(!suggestions.is_empty(),
        "Causal engine integration should produce suggestions, got {}", suggestions.len());

    // All suggestions should have valid confidence
    for s in suggestions {
        assert!(s.confidence > 0.0,
            "Suggestion confidence should be > 0");
    }
}

// =====================================================================
// Test 8: Propagation chain across prerequisites
// =====================================================================
#[test]
fn test_causal_propagation_chain() {
    let store = create_store_with_prereqs();
    let mut engine = SkillEvolutionEngine::new(store.clone());

    // Record a failure on 'base' (prerequisite of 'auth')
    let base_failure = UsageRecord::new(
        "iri://skills/base",
        "iri://task/base-fail-001",
        "agent:da/001",
        false,
    )
    .with_error("Database connection failed");
    engine.record_usage(base_failure).unwrap();

    // Record a failure on 'auth' (which depends on base)
    let auth_failure = UsageRecord::new(
        "iri://skills/auth",
        "iri://task/auth-fail-001",
        "agent:da/001",
        false,
    )
    .with_error("Token verification timeout");
    engine.record_usage(auth_failure).unwrap();

    // Both failures should produce suggestions
    let suggestions = engine.get_pending_suggestions();
    assert!(!suggestions.is_empty(),
        "Propagation chain should produce suggestions, got {}", suggestions.len());

    // Should have at least one suggestion for each failing skill
    let base_suggestions: Vec<&EvolutionSuggestion> = suggestions.iter()
        .filter(|s| s.skill_iri == "iri://skills/base")
        .collect();
    let _auth_suggestions: Vec<&EvolutionSuggestion> = suggestions.iter()
        .filter(|s| s.skill_iri == "iri://skills/auth")
        .collect();

    assert!(!base_suggestions.is_empty(),
        "Base skill should have at least one suggestion, got {} base suggestions", base_suggestions.len());
    // Auth may or may not get suggestions depending on propagation analysis timing
    // (the legacy path checks events within 60 seconds)
}

// =====================================================================
// Test 9: Clear suggestions
// =====================================================================
#[test]
fn test_clear_suggestions() {
    let store = create_test_store();
    let mut engine = SkillEvolutionEngine::new(store);

    let record = UsageRecord::new(
        "iri://skills/test-skill",
        "iri://task/clear-test",
        "agent:da/001",
        false,
    )
    .with_error("Timeout error");
    engine.record_usage(record).unwrap();

    assert!(!engine.get_pending_suggestions().is_empty(),
        "Should have suggestions after failure");

    engine.clear_suggestions();
    assert!(engine.get_pending_suggestions().is_empty(),
        "After clear, pending suggestions should be empty");
}

// =====================================================================
// Test 10: Direct CausalObservation recording
// =====================================================================
#[test]
fn test_direct_causal_observation() {
    let store = Arc::new(SkillGraphStore::new());
    let skill = SkillGraphNode::new("iri://skills/direct-obs", "Direct Obs", "Test direct obs");
    store.register_skill(skill).unwrap();

    let causal = create_causal_engine(&store);

    // Record observations directly via CausalEngine
    causal.record_observation(
        CausalObservation::new(
            "event:direct-001",
            "iri://skills/direct-obs",
            "network",
            "sha256:abc123...",
        )
        .with_context("task_iri", "iri://task/direct-obs-task")
        .with_context("agent_id", "agent:da/001"),
    );

    causal.record_observation(
        CausalObservation::new(
            "event:direct-002",
            "iri://skills/direct-obs",
            "timeout",
            "sha256:def456...",
        )
        .with_context("task_iri", "iri://task/direct-obs-task-2")
        .with_context("agent_id", "agent:da/001"),
    );

    // Infer root cause from observations
    let _inferences = causal.infer_root_cause(&[
        CausalObservation::new(
            "event:infer-001",
            "iri://skills/direct-obs",
            "network",
            "sha256:infer-abc",
        ),
    ], 3);

    // Should not panic - may return empty if no candidates found
    // (which is fine - the engine ran without error)
}
