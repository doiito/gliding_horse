//! ExecutionPlan → DAG adapter
//!
//! Automatically converts SA's existing linear ExecutionPlan sequence to DAG internal representation,
//! allowing both old and new workflow definition styles to share the same DAG execution engine.

use super::definition::*;
use super::loader::*;
use crate::core::agent_instance::AgentRole;
use crate::core::context_model::{
    AgentSpecSourceKind, AgentSpecSourceRecord, ExecutionPlanProvenance,
};
use crate::core::sa::{ExecutionPlan, PlanStep};

/// Convert ExecutionPlan to WorkflowDefinition (executable by the DAG engine)
pub fn plan_to_workflow(plan: &ExecutionPlan, _task_iri: &str) -> WorkflowDefinition {
    let plan_id = &plan.plan_id;

    // PlanStep dependencies use the LLM-authored step-id namespace, while
    // WorkflowNodeDef uses a plan-scoped runtime namespace. Compile that
    // mapping explicitly; leaving raw step ids here makes build_dag silently
    // drop the edge and can invert a declared cross-role barrier.
    let runtime_ids = plan
        .steps
        .iter()
        .map(|step| {
            (
                step.step_id.as_str(),
                format!("wf:{}/{}", plan_id, step.step_id),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let has_explicit_step_dependencies =
        plan.steps.iter().any(|step| !step.dependencies.is_empty());

    let mut nodes = Vec::new();
    let mut prev_id: Option<String> = None;

    for step in &plan.steps {
        let node_id = runtime_ids
            .get(step.step_id.as_str())
            .expect("runtime id was compiled for every plan step")
            .clone();
        let mut extra = std::collections::HashMap::new();
        if !step.work_packages.is_empty() {
            extra.insert(
                "work_packages".to_string(),
                serde_json::to_value(&step.work_packages).unwrap_or(serde_json::Value::Null),
            );
        }

        let mut node = WorkflowNodeDef {
            id: node_id.clone(),
            node_type: "AgentNode".to_string(),
            agent_role: format!("{:?}", step.role),
            objective: step.objective.clone(),
            next: None,
            next_nodes: vec![],
            dependencies: step
                .dependencies
                .iter()
                .map(|dependency| {
                    runtime_ids
                        .get(dependency.as_str())
                        .cloned()
                        // Fresh generated plans and resume contracts reject
                        // unknown dependencies before this adapter. Preserve
                        // an unexpected raw id so build_dag also fails closed.
                        .unwrap_or_else(|| dependency.clone())
                })
                .collect(),
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
            extra,
        };

        // A dependency-free legacy plan is intentionally linear. Once the
        // plan declares any explicit edge, however, its validated DAG is the
        // sole authority; adding vector-order edges could create an inverse
        // cycle (for example CA listed before the DA it depends on).
        if !has_explicit_step_dependencies {
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
        }
        prev_id = Some(node_id);
        nodes.push(node);
    }

    // A final node is a sink in the compiled dependency graph. Legacy linear
    // plans retain their historical last-node terminal marker.
    if has_explicit_step_dependencies {
        let predecessors = nodes
            .iter()
            .flat_map(|node| node.dependencies.iter().cloned())
            .collect::<std::collections::HashSet<_>>();
        for node in &mut nodes {
            node.final_node = !predecessors.contains(&node.id);
        }
    } else if let Some(last) = nodes.last_mut() {
        last.final_node = true;
    }

    let entry_node = nodes
        .iter()
        .find(|node| node.dependencies.is_empty())
        .or_else(|| nodes.first())
        .map(|node| node.id.clone())
        .unwrap_or_default();

    // `ExecutionPlan.parallel_groups` is a legacy same-role fan-out hint. SA
    // passes it to the owning BizAgent, which has the full role context and a
    // dependency/resource-aware child scheduler. Explicit cross-role DAG
    // parallelism remains available through `dag_jsonld` and bypasses this
    // adapter entirely.

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
        work_packages: node
            .extra
            .get("work_packages")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default(),
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

    let mut plan = ExecutionPlan {
        plan_id: def.id.clone(),
        agent_sequence,
        parallel_groups: vec![],
        task_complexity: crate::core::sa::TaskComplexity::Standard,
        description: def.name.clone(),
        steps,
        agent_spec_provenance: None,
        context_requirements: std::collections::HashMap::new(),
        success_metrics: vec![],
        max_recursion_depth: 0,
        sub_tasks: vec![],
        dag_jsonld: None,
        verify_first: false,
        fallback_steps: vec![],
    };
    plan.set_agent_spec_provenance(ExecutionPlanProvenance::new(
        AgentSpecSourceRecord::new(AgentSpecSourceKind::WorkflowDefinition)
            .with_source_ref(def.id.clone())
            .with_producer("WorkflowAdapter"),
    ))
    .expect("workflow provenance is valid");
    plan
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
                    work_packages: Vec::new(),
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
                    work_packages: vec![
                        crate::core::sa::PlanWorkPackage {
                            id: "a".to_string(),
                            objective: "produce A".to_string(),
                            expected_output: "A".to_string(),
                            success_criteria: "A exists".to_string(),
                            evidence_requirements: vec![
                                crate::core::sa::WorkPackageEvidenceRequirement::ArtifactDelivery {
                                    paths: vec!["A".to_string()],
                                    min_paths: 1,
                                },
                            ],
                            dependencies: Vec::new(),
                        },
                        crate::core::sa::PlanWorkPackage {
                            id: "b".to_string(),
                            objective: "produce B".to_string(),
                            expected_output: "B".to_string(),
                            success_criteria: "B uses A".to_string(),
                            evidence_requirements: vec![
                                crate::core::sa::WorkPackageEvidenceRequirement::ArtifactDelivery {
                                    paths: vec!["B".to_string()],
                                    min_paths: 1,
                                },
                            ],
                            dependencies: vec!["a".to_string()],
                        },
                    ],
                    branch_on_failure: false,
                    branch_fallback: None,
                    retry_count: 0,
                    retry_delay_secs: 0,
                    effect_policy: crate::core::effect::EffectPolicy::None,
                },
            ],
            agent_spec_provenance: None,
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
        assert_eq!(wf.nodes[0].next, None);
        assert_eq!(
            wf.nodes[1].dependencies,
            vec!["wf:test_001/step_1".to_string()]
        );
        assert!(wf.nodes[1].final_node);

        let dag = build_dag(&wf).unwrap();
        let round_trip = dag_to_execution_plan(&dag, &wf, "iri://task/test_task");
        assert_eq!(round_trip.steps[1].work_packages.len(), 2);
        assert_eq!(round_trip.steps[1].work_packages[1].dependencies, vec!["a"]);
        let source = round_trip
            .agent_spec_source_for_step(&round_trip.steps[0].step_id)
            .unwrap()
            .unwrap();
        assert_eq!(source.kind, AgentSpecSourceKind::WorkflowDefinition);
        assert_eq!(source.producer.as_deref(), Some("WorkflowAdapter"));
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
                    work_packages: Vec::new(),
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
                    work_packages: Vec::new(),
                    branch_on_failure: false,
                    branch_fallback: None,
                    retry_count: 0,
                    retry_delay_secs: 0,
                    effect_policy: crate::core::effect::EffectPolicy::None,
                },
            ],
            agent_spec_provenance: None,
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

    #[test]
    fn explicit_step_dependencies_are_namespaced_and_drive_runtime_order() {
        let check = PlanStep {
            step_id: "check".to_string(),
            role: AgentRole::Check,
            objective: "verify the implementation".to_string(),
            expected_output: "verification".to_string(),
            dependencies: vec!["do".to_string()],
            tools_allowed: vec![],
            success_criteria: "verification succeeds".to_string(),
            work_packages: Vec::new(),
            branch_on_failure: false,
            branch_fallback: None,
            retry_count: 0,
            retry_delay_secs: 0,
            effect_policy: crate::core::effect::EffectPolicy::EvidenceOnly,
        };
        let mut do_step = check.clone();
        do_step.step_id = "do".to_string();
        do_step.role = AgentRole::Do;
        do_step.objective = "implement".to_string();
        do_step.dependencies.clear();
        do_step.effect_policy = crate::core::effect::EffectPolicy::None;
        let plan = ExecutionPlan {
            plan_id: "dependency_namespace".to_string(),
            agent_sequence: vec![AgentRole::Check, AgentRole::Do],
            parallel_groups: vec![],
            task_complexity: crate::core::sa::TaskComplexity::Simple,
            description: "out-of-order storage with an explicit DAG".to_string(),
            // Deliberately store the dependent first. Runtime order must come
            // from the dependency, not vector position.
            steps: vec![check, do_step],
            agent_spec_provenance: None,
            context_requirements: Default::default(),
            success_metrics: vec![],
            max_recursion_depth: 0,
            sub_tasks: vec![],
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: vec![],
        };

        let workflow = plan_to_workflow(&plan, "iri://task/dependency-namespace");
        let check_node = workflow
            .nodes
            .iter()
            .find(|node| node.id.ends_with("/check"))
            .unwrap();
        assert_eq!(
            check_node.dependencies,
            vec!["wf:dependency_namespace/do".to_string()]
        );
        assert_eq!(workflow.entry_node, "wf:dependency_namespace/do");

        let dag = build_dag(&workflow).unwrap();
        let order = topological_order(&dag).unwrap();
        let ordered_ids = order
            .iter()
            .map(|index| dag.graph[*index].def.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ordered_ids,
            vec![
                "wf:dependency_namespace/do",
                "wf:dependency_namespace/check"
            ]
        );
    }
}
