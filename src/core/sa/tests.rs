use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent_instance::AgentRole;
    use crate::core::agent_runner::AgentRunner;
    use crate::core::event_bus::EventBus;
    use crate::gateway::unified_gateway::UnifiedGateway;
    use crate::memory::memory_manager::MemoryManager;
    use crate::templates::template_engine::TemplateEngine;
    use crate::tools::skill_registry::SkillRegistry;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_sa_with_tempdir() -> (SupervisorAgent, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let l0 = Arc::new(
            crate::memory::l0_store::L0Store::new(dir.path().join("l0").to_string_lossy().as_ref())
                .unwrap(),
        );
        let l2 = Arc::new(crate::memory::l2_blackboard::Blackboard::new().unwrap());
        let proj = Arc::new(crate::memory::l3_projection::ProjectionEngine::new(
            l2.clone(),
            500,
        ));
        let mm = Arc::new(tokio::sync::Mutex::new(MemoryManager::new(
            l0.clone(),
            l2.clone(),
            proj.clone(),
            crate::CoreConfig::default(),
        )));
        let tmpl = Arc::new(TemplateEngine::new(std::path::Path::new("/nonexistent")).unwrap());
        let settings = crate::config::settings::GatewaySettings {
            base_url: "http://localhost:3000".to_string(),
            api_key: "sk-test".to_string(),
            default_model: "deepseek-v4-flash".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            retry_base_ms: 500,
            use_responses_api: false,
            model_mapping: HashMap::new(),
        };
        let gateway = Arc::new(UnifiedGateway::new(&settings).unwrap());
        let skills = Arc::new(SkillRegistry::new());
        let agent_settings = crate::config::settings::AgentSettings::default();
        let runner = Arc::new(AgentRunner::new(
            gateway,
            skills.clone(),
            l2.clone(),
            l0,
            mm,
            tmpl.clone(),
            agent_settings,
        ));
        let sa = SupervisorAgent::new(runner, tmpl, skills, Arc::new(EventBus::new(100)), 10)
            .with_memory(Some(l2), None, None);
        (sa, dir)
    }

    #[test]
    fn test_classify_simple() {
        let (sa, _dir) = make_sa_with_tempdir();
        assert_eq!(
            sa.classify_complexity("What is the weather?"),
            TaskComplexity::Simple
        );
        assert_eq!(
            sa.classify_complexity("Fix this bug in the code"),
            TaskComplexity::Emergency
        );
        assert_eq!(
            sa.classify_complexity("Build a web application with user authentication and database"),
            TaskComplexity::Recursive
        );
    }

    #[test]
    fn test_execution_plan_simple() {
        let (sa, _dir) = make_sa_with_tempdir();
        let plan = sa.analyze_task("Hello");
        assert_eq!(plan.agent_sequence.len(), 1);
        assert_eq!(plan.agent_sequence[0], AgentRole::Do);
    }

    #[test]
    fn test_execution_plan_emergency() {
        let (sa, _dir) = make_sa_with_tempdir();
        let plan = sa.analyze_task("Fix critical security vulnerability");
        assert_eq!(plan.agent_sequence.len(), 3);
        assert_eq!(plan.agent_sequence[0], AgentRole::Do);
        assert!(plan.agent_sequence.contains(&AgentRole::Act));
    }

    #[test]
    fn test_cleanup_expired_cycles() {
        let (mut sa, _dir) = make_sa_with_tempdir();
        sa.active_cycles.insert(
            "old_cycle".to_string(),
            CycleState {
                cycle_id: "old_cycle".to_string(),
                task_iri: "iri://task/1".to_string(),
                phase: CyclePhase::Completed,
                iteration: 1,
                max_iterations: 10,
                started_at: chrono::Utc::now() - chrono::Duration::hours(2),
                phase_history: vec![],
                task_completed: true,
                experience_hints: vec![],
            },
        );
        sa.cleanup_expired_cycles(3600);
        assert!(sa.active_cycles.is_empty());
    }
}
