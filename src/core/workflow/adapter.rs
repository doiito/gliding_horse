//! ExecutionPlan → DAG adapter
//!
//! Automatically converts SA's existing linear ExecutionPlan sequence to DAG internal representation,
//! allowing both old and new workflow definition styles to share the same DAG execution engine.

use super::definition::*;
use super::loader::*;
use crate::core::agent_instance::AgentRole;
use crate::core::sa::{ExecutionPlan, PlanStep};

/// Convert ExecutionPlan to WorkflowDefinition (executable by the DAG engine)
pub fn plan_to_workflow(plan: &ExecutionPlan, _task_iri: &str) -> WorkflowDefinition {
    let plan_id = &plan.plan_id;

    let mut nodes = Vec::new();
    let mut prev_id: Option<String> = None;

    for step in &plan.steps {
        let node_id = format!("wf:{}/{}", plan_id, step.step_id);

        let mut node = WorkflowNodeDef {
            id: node_id.clone(),
            node_type: "AgentNode".to_string(),
            agent_role: format!("{:?}", step.role),
            objective: step.objective.clone(),
            next: None,
            next_nodes: vec![],
            dependencies: step.dependencies.clone(),
            tools: step.tools_allowed.clone(),
            expected_output: step.expected_output.clone(),
            success_criteria: step.success_criteria.clone(),
            approval_prompt: String::new(),
            approval_next_on_approve: None,
            approval_next_on_reject: None,
            input_mapping: None,
            branch_on_failure: None,
            retry_count: 0,
            retry_delay_secs: 0,
            timeout_secs: 0,
            final_node: false,
            extra: Default::default(),
        };

        // Chain linear steps into next links
        if let Some(ref pid) = prev_id {
            // Find predecessor and set next
            if let Some(prev_node) = nodes
                .iter_mut()
                .find(|n: &&mut WorkflowNodeDef| n.id == *pid)
            {
                prev_node.next = Some(node_id.clone());
            }
            // Current node depends on predecessor
            if !node.dependencies.contains(pid) {
                node.dependencies.push(pid.clone());
            }
        }
        prev_id = Some(node_id);
        nodes.push(node);
    }

    // Mark the last as final
    if let Some(last) = nodes.last_mut() {
        last.final_node = true;
    }

    let entry_node = nodes.first().map(|n| n.id.clone()).unwrap_or_default();

    // Handle parallel groups
    let parallel_updates: Vec<(String, Vec<String>)> = plan
        .parallel_groups
        .iter()
        .filter(|g| g.len() > 1)
        .filter_map(|group| {
            let role_strs: Vec<String> = group.iter().map(|r| format!("{:?}", r)).collect();
            let first_idx = nodes
                .iter()
                .position(|n| role_strs.contains(&n.agent_role))?;
            if first_idx == 0 {
                return None;
            }
            let prev_id = nodes[first_idx - 1].id.clone();
            let parallel_ids: Vec<String> = nodes[first_idx..]
                .iter()
                .filter(|n| role_strs.contains(&n.agent_role))
                .map(|n| n.id.clone())
                .collect();
            if parallel_ids.is_empty() {
                None
            } else {
                Some((prev_id, parallel_ids))
            }
        })
        .collect();

    for (prev_id, parallel_ids) in parallel_updates {
        if let Some(prev_node) = nodes.iter_mut().find(|n| n.id == prev_id) {
            prev_node.next_nodes = parallel_ids;
            prev_node.next = None;
        }
    }

    WorkflowDefinition {
        id: format!("iri://workflow/{}", plan_id),
        name: plan.description.clone(),
        description: format!("Auto-converted from ExecutionPlan '{}'", plan.description),
        version: "1.0".to_string(),
        entry_node,
        nodes,
    }
}

/// Convert DAG node (WorkflowNodeDef) to PlanStep (unified iteration interface)
pub fn node_to_planstep(node: &WorkflowNodeDef) -> PlanStep {
    PlanStep {
        step_id: node.id.clone(),
        role: parse_role_from_str(&node.agent_role),
        objective: node.objective.clone(),
        expected_output: if node.expected_output.is_empty() {
            node.success_criteria.clone()
        } else {
            node.expected_output.clone()
        },
        dependencies: node.dependencies.clone(),
        tools_allowed: node.tools.clone(),
        success_criteria: node.success_criteria.clone(),
        branch_on_failure: node.branch_on_failure.is_some(),
        branch_fallback: node.branch_on_failure.as_ref().map(|b| b.target.clone()),
        retry_count: node.retry_count,
        retry_delay_secs: node.retry_delay_secs,
        effect_policy: match parse_role_from_str(&node.agent_role) {
            AgentRole::Plan | AgentRole::Check => crate::core::effect::EffectPolicy::EvidenceOnly,
            AgentRole::Act => crate::core::effect::EffectPolicy::DecisionOnly,
            AgentRole::Do => crate::core::effect::EffectPolicy::None,
        },
    }
}

/// Parse AgentRole from agent_role string
fn parse_role_from_str(role: &str) -> AgentRole {
    match role.to_lowercase().as_str() {
        "plan" | "pa" => AgentRole::Plan,
        "do" | "da" | "executor" => AgentRole::Do,
        "check" | "ca" | "reviewer" => AgentRole::Check,
        "act" | "aa" | "decision" => AgentRole::Act,
        _ => AgentRole::Do,
    }
}

/// Quick check: whether ExecutionPlan is executable by DAG engine (always true, via adapter)
pub fn is_plan_compatible(_plan: &ExecutionPlan) -> bool {
    true
}

/// Convert DAG (WorkflowDag) back to ExecutionPlan (for unified execution path of external workflow.jsonld)
pub fn dag_to_execution_plan(
    dag: &WorkflowDag,
    def: &WorkflowDefinition,
    _task_iri: &str,
) -> ExecutionPlan {
    let order = crate::core::workflow::loader::topological_order(dag)
        .unwrap_or_else(|_| dag.graph.node_indices().collect::<Vec<_>>());

    let steps: Vec<PlanStep> = order
        .iter()
        .map(|&idx| node_to_planstep(&dag.graph[idx].def))
        .collect();

    let agent_sequence: Vec<AgentRole> = steps.iter().map(|s| s.role).collect();

    ExecutionPlan {
        plan_id: def.id.clone(),
        agent_sequence,
        parallel_groups: vec![],
        task_complexity: crate::core::sa::TaskComplexity::Standard,
        description: def.name.clone(),
        steps,
        context_requirements: std::collections::HashMap::new(),
        success_metrics: vec![],
        max_recursion_depth: 0,
        sub_tasks: vec![],
        dag_jsonld: None,
        verify_first: false,
        fallback_steps: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sa::{ExecutionPlan, PlanStep};

    #[test]
    fn test_plan_to_workflow_linear() {
        let plan = ExecutionPlan {
            plan_id: "test_001".to_string(),
            agent_sequence: vec![
                AgentRole::Plan,
                AgentRole::Do,
                AgentRole::Check,
                AgentRole::Act,
            ],
            parallel_groups: vec![],
            task_complexity: crate::core::sa::TaskComplexity::Standard,
            description: "Test plan".to_string(),
            steps: vec![
                PlanStep {
                    step_id: "step_1".to_string(),
                    role: AgentRole::Plan,
                    objective: "Create plan".to_string(),
                    expected_output: "plan".to_string(),
                    dependencies: vec![],
                    tools_allowed: vec!["file_read".to_string()],
                    success_criteria: "Plan complete".to_string(),
                    branch_on_failure: false,
                    branch_fallback: None,
                    retry_count: 0,
                    retry_delay_secs: 0,
                    effect_policy: crate::core::effect::EffectPolicy::EvidenceOnly,
                },
                PlanStep {
                    step_id: "step_2".to_string(),
                    role: AgentRole::Do,
                    objective: "Execute task".to_string(),
                    expected_output: "output".to_string(),
                    dependencies: vec!["step_1".to_string()],
                    tools_allowed: vec!["file_write".to_string(), "bash".to_string()],
                    success_criteria: "Output complete".to_string(),
                    branch_on_failure: false,
                    branch_fallback: None,
                    retry_count: 0,
                    retry_delay_secs: 0,
                    effect_policy: crate::core::effect::EffectPolicy::None,
                },
            ],
            context_requirements: Default::default(),
            success_metrics: vec![],
            max_recursion_depth: 0,
            sub_tasks: vec![],
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: vec![],
        };

        let wf = plan_to_workflow(&plan, "iri://task/test_task");
        assert_eq!(wf.nodes.len(), 2);
        assert_eq!(wf.entry_node, "wf:test_001/step_1");
        assert_eq!(wf.nodes[0].next.as_deref(), Some("wf:test_001/step_2"));
        assert!(wf.nodes[1].final_node);
    }

    #[test]
    fn test_plan_to_workflow_parallel() {
        let plan = ExecutionPlan {
            plan_id: "test_002".to_string(),
            agent_sequence: vec![AgentRole::Do, AgentRole::Check],
            parallel_groups: vec![vec![AgentRole::Do, AgentRole::Do]],
            task_complexity: crate::core::sa::TaskComplexity::Standard,
            description: "Parallel test".to_string(),
            steps: vec![
                PlanStep {
                    step_id: "step_1".to_string(),
                    role: AgentRole::Do,
                    objective: "Module A".to_string(),
                    expected_output: "a".to_string(),
                    dependencies: vec![],
                    tools_allowed: vec![],
                    success_criteria: "".to_string(),
                    branch_on_failure: false,
                    branch_fallback: None,
                    retry_count: 0,
                    retry_delay_secs: 0,
                    effect_policy: crate::core::effect::EffectPolicy::None,
                },
                PlanStep {
                    step_id: "step_2".to_string(),
                    role: AgentRole::Do,
                    objective: "Module B".to_string(),
                    expected_output: "b".to_string(),
                    dependencies: vec![],
                    tools_allowed: vec![],
                    success_criteria: "".to_string(),
                    branch_on_failure: false,
                    branch_fallback: None,
                    retry_count: 0,
                    retry_delay_secs: 0,
                    effect_policy: crate::core::effect::EffectPolicy::None,
                },
            ],
            context_requirements: Default::default(),
            success_metrics: vec![],
            max_recursion_depth: 0,
            sub_tasks: vec![],
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: vec![],
        };

        let wf = plan_to_workflow(&plan, "iri://task/test2");
        // Two Do nodes in parallel: no entry next, should be next_nodes
        assert_eq!(wf.nodes.len(), 2);
    }
}
