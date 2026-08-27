// TEMP VERIFICATION — 验证 MethodologyGate 激活逻辑的实际运行时行为
// 目的: 区分"引擎坏了" vs "接线断了"(报告 2.6 节结论的运行时确认)
// 运行: cargo test --test verify_gate_activation -- --nocapture
use glidinghorse::core::constitution::ConstitutionRegistry;
use glidinghorse::methodology::gate::{MethodologyGate, MethodologyGateHandle};
use glidinghorse::methodology::MethodologyRegistry;
use glidinghorse::tools::hooks::{HookContext, HookManager, HookPoint};
use serde_json::Value;

fn gate() -> MethodologyGate {
    MethodologyGate::new(MethodologyRegistry::new(), 20)
}

fn ctx_with(point: HookPoint, role: &str, tool_name: Option<&str>) -> HookContext {
    let mut ctx = HookContext::new(point, "test_agent", role);
    if let Some(name) = tool_name {
        ctx = ctx.with_data("tool_name", Value::String(name.to_string()));
    }
    ctx
}

#[test]
fn verify_tool_category_activates_when_hooked_with_tool_name() {
    // 结论1: OnToolCategory 引擎逻辑本身正确 —— 只要在带 tool_name 的上下文触发 on_hook_trigger 就能激活
    let mut g = gate();
    let ctx = ctx_with(HookPoint::SkillBefore, "DA", Some("glob"));
    let activated = g.on_hook_trigger(HookPoint::SkillBefore, &ctx);
    let ids: Vec<&str> = activated
        .iter()
        .map(|a| a.methodology_id.as_str())
        .collect();
    println!("[SkillBefore+glob] activated: {:?}", ids);
    assert!(
        ids.contains(&"methodology:index-priority"),
        "index-priority(OnToolCategory file_search) 应激活 —— 证明引擎可用"
    );
}

#[test]
fn verify_hook_point_string_mismatch_never_activates() {
    // 修复后: OnHookPoint("skill_before") / OnHookPoint("phase_start") 与真实 HookPoint 对齐
    let mut g = gate();
    // cost-awareness → OnHookPoint("skill_before") 只应在 SkillBefore 激活
    for point in [
        HookPoint::AgentInit,
        HookPoint::TaskStart,
        HookPoint::TaskError,
        HookPoint::SkillAfter,
        HookPoint::CycleStart,
        HookPoint::CycleEnd,
        HookPoint::LlmRequest,
        HookPoint::LlmResponse,
    ] {
        let ctx = ctx_with(point, "DA", Some("bash"));
        let activated = g.on_hook_trigger(point, &ctx);
        let ids: Vec<&str> = activated
            .iter()
            .map(|a| a.methodology_id.as_str())
            .collect();
        println!("[{}] activated: {:?}", point.as_str(), ids);
        assert!(
            !ids.contains(&"methodology:cost-awareness"),
            "cost-awareness 不应在 {} 激活 —— 仅对齐 skill_before",
            point.as_str()
        );
    }
    // SkillBefore 应激活 cost-awareness(此前因字符串失配死锁)
    let ctx = ctx_with(HookPoint::SkillBefore, "DA", Some("bash"));
    let activated = g.on_hook_trigger(HookPoint::SkillBefore, &ctx);
    let ids: Vec<&str> = activated
        .iter()
        .map(|a| a.methodology_id.as_str())
        .collect();
    println!("[SkillBefore] activated: {:?}", ids);
    assert!(
        ids.contains(&"methodology:cost-awareness"),
        "cost-awareness 应在 SkillBefore 激活 —— D3 对齐生效"
    );
}

#[test]
fn verify_phase_end_condition_requires_phase_end_point() {
    // 结论3: OnPhaseEnd("ACT") 只会在 point == PhaseEnd 且 data["phase"]=="ACT" 时激活
    let mut g = gate();
    let mut ctx = ctx_with(HookPoint::PhaseEnd, "DA", None);
    ctx = ctx.with_data("phase", Value::String("ACT".to_string()));
    let activated = g.on_hook_trigger(HookPoint::PhaseEnd, &ctx);
    let ids: Vec<&str> = activated
        .iter()
        .map(|a| a.methodology_id.as_str())
        .collect();
    println!("[PhaseEnd+ACT] activated: {:?}", ids);
    assert!(
        ids.contains(&"methodology:verification-before-completion"),
        "verification-before-completion 在 PhaseEnd+ACT 下应激活 —— 证明引擎可用,问题在 PhaseEnd 从不触发"
    );
}

#[test]
fn verify_always_and_task_error_conditions_work() {
    let mut g = gate();
    let ctx = ctx_with(HookPoint::AgentInit, "SA", None);
    let activated = g.on_hook_trigger(HookPoint::AgentInit, &ctx);
    let ids: Vec<&str> = activated
        .iter()
        .map(|a| a.methodology_id.as_str())
        .collect();
    println!("[AgentInit] activated: {:?}", ids);
    for id in [
        "methodology:boundary-enforcement",
        "methodology:using-superpowers",
    ] {
        assert!(ids.contains(&id), "Always 条件 {} 应激活", id);
    }

    let mut ctx2 = ctx_with(HookPoint::TaskError, "DA", None);
    ctx2.error = Some("boom".to_string());
    let activated2 = g.on_hook_trigger(HookPoint::TaskError, &ctx2);
    let ids2: Vec<&str> = activated2
        .iter()
        .map(|a| a.methodology_id.as_str())
        .collect();
    println!("[TaskError] activated: {:?}", ids2);
    assert!(
        ids2.contains(&"methodology:systematic-debugging"),
        "OnTaskError 应激活"
    );
}

#[test]
fn verify_all_builtin_activation_conditions() {
    // 汇总: 每个内置方法论在"其激活条件应有的上下文"下是否可达
    let registry = MethodologyRegistry::new();
    println!("内置方法论数量: {}", registry.count());
    for m in registry.all() {
        println!("  {} → {:?}", m.id, m.activation);
    }
    // 静态断言: 死锁条件集合
    let dead = [
        "methodology:index-priority",
        "methodology:cost-awareness",
        "methodology:least-privilege",
        "methodology:brainstorming",
        "methodology:test-driven-development",
        "methodology:verification-before-completion",
    ];
    let alive = [
        "methodology:boundary-enforcement",
        "methodology:using-superpowers",
        "methodology:complexity-assessment",
        "methodology:systematic-debugging",
    ];
    for id in &dead {
        assert!(registry.get(id).is_some(), "{} 应存在", id);
    }
    for id in &alive {
        assert!(registry.get(id).is_some(), "{} 应存在", id);
    }
}

#[test]
fn verify_ontoolcategory_false_activation_on_empty_tool_name() {
    // D1 修复后: OnToolCategory 在无 tool_name 时必须静默(None),不再因 'cat.contains("")' 恒真误激活
    let mut g = gate();
    let ctx = ctx_with(HookPoint::AgentInit, "DA", None);
    let activated = g.on_hook_trigger(HookPoint::AgentInit, &ctx);
    let ids: Vec<&str> = activated
        .iter()
        .map(|a| a.methodology_id.as_str())
        .collect();
    println!("[AgentInit, no tool_name] activated: {:?}", ids);
    let tool_cat_activated = [
        "methodology:index-priority",
        "methodology:least-privilege",
        "methodology:test-driven-development",
    ]
    .iter()
    .any(|id| ids.contains(id));
    println!(
        "OnToolCategory 方法论在无 tool_name 上下文被误激活: {}",
        tool_cat_activated
    );
    assert!(
        !tool_cat_activated,
        "无 tool_name 时 OnToolCategory 不应误激活 —— D1 空串守卫生效"
    );
}

#[test]
fn verify_all_constitution_bindings_resolve() {
    // D10: register_constitution_bindings 会 warn 每个缺失的 methodology。
    // 若此处所有 binding 引用的 ID 都已在注册表解析,则 warn 计数为 0(死绑定清零)。
    let registry = MethodologyRegistry::new();
    let constitution = ConstitutionRegistry::new();
    let mut missing = Vec::new();
    for entry in constitution.all() {
        if let Some(bindings) = constitution.get_bindings(entry.id) {
            for b in bindings {
                if registry.get(&b.methodology_id).is_none() {
                    missing.push(format!("{} -> {}", entry.id, b.methodology_id));
                }
            }
        }
    }
    println!("constitution bindings missing methodologies: {:?}", missing);
    assert!(
        missing.is_empty(),
        "bindings referencing missing methodologies (would warn): {:?}",
        missing
    );
}

#[test]
fn verify_skill_before_after_pairing_records_usage() {
    // 结论: SkillBefore 激活 → pending 捕获 → SkillAfter 结算为 success,usage_history 落账。
    // 走真实 register_hooks + HookManager::execute 路径(生产接线),非直接调 on_hook_trigger。
    let handle = MethodologyGateHandle::new(gate());
    let hm = HookManager::new();
    handle.register_hooks(&hm);

    let rt = tokio::runtime::Runtime::new().unwrap();

    // SkillBefore: 触发激活,closure 捕获 pending_skill_activation
    let mut before = ctx_with(HookPoint::SkillBefore, "DA", Some("glob"));
    rt.block_on(hm.execute(HookPoint::SkillBefore, &mut before));

    // SkillAfter: closure 结算 pending 为 success
    let mut after = ctx_with(HookPoint::SkillAfter, "DA", Some("glob"));
    rt.block_on(hm.execute(HookPoint::SkillAfter, &mut after));

    let inner = handle.inner();
    let gate = inner.read();
    let usage = gate.usage_history();
    println!("[SkillBefore→SkillAfter] usage records: {:?}", usage);
    assert!(
        !usage.is_empty(),
        "SkillAfter 应结算 usage_history —— 生产接线 SkillBefore→SkillAfter 配对生效"
    );
    assert!(
        usage.iter().any(|r| r.success),
        "SkillAfter 结算应标记 success"
    );
    assert!(
        gate.active_methodologies().is_empty(),
        "结算后 active 应清空 —— 窗口已关闭"
    );
}

/// 端到端治理闭环: 使用记录 → 失败建议 → 提案 → 审批 → 校验 → 落地。
/// 方法论提案走合成 IRI,与技能提案共用同一治理门。
#[test]
fn verify_methodology_suggestion_flows_through_proposal_gate() {
    use glidinghorse::skill_graph::evolution::{
        EvolutionApproval, EvolutionProposalStatus, EvolutionProposalStore,
    };

    let mut gate = gate();
    let evolution = glidinghorse::methodology::evolution::EvolutionEngineHandle::new(
        glidinghorse::methodology::evolution::EvolutionEngine::new(),
    );
    let handle = MethodologyGateHandle::new(gate).with_evolution(evolution);
    let hm = HookManager::new();
    handle.register_hooks(&hm);
    let rt = tokio::runtime::Runtime::new().unwrap();

    // 三个 glob 工具窗口全部失败 → usage_history 成功率为 0
    for _ in 0..3 {
        let mut before = ctx_with(HookPoint::SkillBefore, "DA", Some("glob"));
        rt.block_on(hm.execute(HookPoint::SkillBefore, &mut before));
        let mut error = ctx_with(HookPoint::TaskError, "DA", Some("glob"));
        error.error = Some("boom".to_string());
        rt.block_on(hm.execute(HookPoint::TaskError, &mut error));
    }
    let gate_ref = handle.inner();
    assert!(gate_ref.read().usage_count() >= 3);

    let suggestions = gate_ref.read().suggest_methodology_adjustments(3, 0.5);
    assert!(!suggestions.is_empty(), "低成功率方法论必须生成治理建议");

    let dir = tempfile::tempdir().unwrap();
    let l0 = glidinghorse::memory::l0_store::L0Store::new(dir.path().to_str().unwrap()).unwrap();
    let store = EvolutionProposalStore::new(std::sync::Arc::new(l0));
    let graph = glidinghorse::skill_graph::graph_store::SkillGraphStore::new();

    let mut committed = Vec::new();
    for (i, suggestion) in suggestions.iter().enumerate() {
        let proposal = store
            .create_or_get(
                &format!("methodology-verify-{i}"),
                suggestion.clone(),
                &graph,
            )
            .unwrap();
        assert_eq!(proposal.status, EvolutionProposalStatus::PendingReview);
        assert!(
            proposal.base_revisions.is_empty(),
            "合成 IRI 提案不应要求图谱节点修订"
        );
        let approved = store
            .approve(&proposal.proposal_id, "reviewer:test", None)
            .unwrap();
        let validated = store
            .validate_for_commit(&approved.proposal_id, &graph)
            .unwrap();
        let done = store
            .commit_validated_link_patch(&validated.proposal_id, &graph)
            .unwrap();
        assert_eq!(done.status, EvolutionProposalStatus::Committed);
        assert!(matches!(
            done.suggestion.approval,
            EvolutionApproval::Approved { .. }
        ));
        committed.push(done);
    }
    assert!(
        !committed.is_empty(),
        "方法论治理建议必须完成 提案→审批→校验→落地 全生命周期"
    );
    println!(
        "[Governance Loop] committed {} methodology proposals",
        committed.len()
    );
}
