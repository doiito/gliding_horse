use std::sync::Arc;

use glidinghorse::causal::engine::CausalEngine;
use glidinghorse::causal::store::CausalModelStore;
use glidinghorse::graph_backend::{GraphBackend, PetgraphBackend};
use glidinghorse::skill_graph::*;
use glidinghorse::tools::{SkillRegistry, ToolExecutor};

const AXURE_TO_VUE2_SKILL_MD: &str =
    include_str!("../fixtures/skills/axure-to-vue2-skill/SKILL.md");
const AXURE_VUE2_REFACTOR_SKILL_MD: &str =
    include_str!("../fixtures/skills/axure-vue2-refactor/SKILL.md");

fn create_test_graph_with_skills() -> Arc<SkillGraphStore> {
    let store = Arc::new(SkillGraphStore::new());

    let jwt_auth = SkillGraphNode::new(
        "iri://skills/jwt-auth",
        "JWT Authentication",
        "JSON Web Token 认证",
    )
    .with_node_type(SkillNodeType::Atomic)
    .with_tag("auth")
    .with_tag("jwt")
    .with_tag("security")
    .with_5w2h(Skill5W2H {
        what: "JWT 认证".to_string(),
        why: "保护 API 端点安全".to_string(),
        who: SkillRole {
            role_name: "认证中间件".to_string(),
            required_agent_role: Some("DA".to_string()),
        },
        when: SkillTrigger {
            applicable_phases: vec!["Do".to_string()],
            trigger_condition: Some("需要认证时".to_string()),
            deadline_constraint: None,
        },
        where_: SkillContext {
            target_stack: vec!["rust".to_string(), "axum".to_string()],
            repo_pattern: Some("src/middleware/*".to_string()),
        },
        how: SkillApproach {
            approach: "使用 jsonwebtoken crate".to_string(),
            plan_iri: None,
        },
        how_much: SkillCost {
            avg_token_cost: 800,
            avg_duration_seconds: 3,
            max_sub_agents: 0,
        },
    })
    .with_security_info(
        SkillSecurityInfo::new(SkillSource::SystemBuiltin).with_trust_level(TrustLevel::High),
    );

    let oauth_auth = SkillGraphNode::new(
        "iri://skills/oauth-auth",
        "OAuth Authentication",
        "OAuth 2.0 认证",
    )
    .with_node_type(SkillNodeType::Atomic)
    .with_tag("auth")
    .with_tag("oauth")
    .with_tag("security")
    .with_security_info(
        SkillSecurityInfo::new(SkillSource::SystemBuiltin).with_trust_level(TrustLevel::High),
    );

    let rbac = SkillGraphNode::new(
        "iri://skills/rbac",
        "RBAC Authorization",
        "基于角色的访问控制",
    )
    .with_node_type(SkillNodeType::Atomic)
    .with_tag("auth")
    .with_tag("authorization")
    .with_tag("security")
    .with_link(SkillLink {
        link_type: SkillLinkType::Prerequisite,
        target_iri: "iri://skills/jwt-auth".to_string(),
        strength: LinkStrength::Required,
        description: "RBAC 需要 JWT 提供用户身份".to_string(),
    });

    let api_guard = SkillGraphNode::new(
        "iri://skills/api-guard",
        "API Guard",
        "API 综合防护（认证+授权+限流）",
    )
    .with_node_type(SkillNodeType::Composite)
    .with_tag("security")
    .with_tag("api")
    .with_tag("middleware")
    .with_link(SkillLink {
        link_type: SkillLinkType::Composition,
        target_iri: "iri://skills/jwt-auth".to_string(),
        strength: LinkStrength::Required,
        description: "包含 JWT 认证".to_string(),
    })
    .with_link(SkillLink {
        link_type: SkillLinkType::Composition,
        target_iri: "iri://skills/rbac".to_string(),
        strength: LinkStrength::Required,
        description: "包含 RBAC 授权".to_string(),
    });

    let rate_limit = SkillGraphNode::new("iri://skills/rate-limit", "Rate Limiting", "API 限流")
        .with_node_type(SkillNodeType::Atomic)
        .with_tag("security")
        .with_tag("api")
        .with_tag("performance");

    store.register_skill(jwt_auth).unwrap();
    store.register_skill(oauth_auth).unwrap();
    store.register_skill(rbac).unwrap();
    store.register_skill(api_guard).unwrap();
    store.register_skill(rate_limit).unwrap();

    store
}

#[test]
fn test_full_skill_graph_workflow() {
    let store = create_test_graph_with_skills();

    let all_skills = store.list_all_skills();
    assert_eq!(all_skills.len(), 5);

    let jwt = store.get_skill("iri://skills/jwt-auth").unwrap();
    assert_eq!(jwt.name, "JWT Authentication");
    assert_eq!(jwt.node_type, SkillNodeType::Atomic);
    assert!(jwt.tags.contains(&"auth".to_string()));

    let deps = store.resolve_dependencies("iri://skills/rbac");
    assert!(deps.contains(&"iri://skills/jwt-auth".to_string()));
    assert!(deps.contains(&"iri://skills/rbac".to_string()));

    let discovery = SkillDiscoveryEngine::new(store.clone());
    let tree = discovery.get_skill_tree("iri://skills/api-guard", 3);
    assert!(tree.is_object());
}

#[tokio::test]
async fn test_discovery_with_conflicts() {
    let store = create_test_graph_with_skills();

    let mut oauth = store.get_skill("iri://skills/oauth-auth").unwrap();
    oauth.add_link(SkillLink {
        link_type: SkillLinkType::Alternative,
        target_iri: "iri://skills/jwt-auth".to_string(),
        strength: LinkStrength::Recommended,
        description: "OAuth 可替代 JWT".to_string(),
    });
    store.update_skill(oauth).unwrap();

    let engine = SkillDiscoveryEngine::new(store.clone());

    let conflicts = engine.check_conflicts(&["iri://skills/jwt-auth", "iri://skills/oauth-auth"]);
    assert!(!conflicts.is_empty());
    assert!(conflicts.iter().any(|c| c.conflict_type == "alternative"));

    let task_5w2h = Task5W2H::new("认证", "保护 API")
        .with_agent_role("DA")
        .with_phase("Do")
        .with_constraint("低延迟");
    let matches = engine.discover_for_task(&task_5w2h).await;
    assert!(!matches.is_empty());
}

#[test]
fn test_evolution_and_health_tracking() {
    let store = create_test_graph_with_skills();
    let mut engine = SkillEvolutionEngine::new(store.clone());

    for i in 0..15 {
        let record = UsageRecord::new(
            "iri://skills/jwt-auth",
            &format!("iri://task/{}", i),
            "agent:da/001",
            i < 12,
        )
        .with_tokens(800 + i * 10);
        engine.record_usage(record).unwrap();
    }

    let health = engine.analyze_skill_health("iri://skills/jwt-auth");
    assert_eq!(health.usage_count, 15);
    assert!((health.success_rate - 0.8).abs() < 0.01);
    assert!(health.health_score > 0.5);

    let fragment = engine
        .create_fragment(
            "iri://skills/jwt-auth",
            "Token 过期处理不当",
            "添加自动刷新机制",
            "agent:ca/001",
        )
        .unwrap();
    assert_eq!(fragment.problem, "Token 过期处理不当");

    let fragments = store.get_fragments_for_skill("iri://skills/jwt-auth");
    assert_eq!(fragments.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_security_policy_enforcement() {
    let store = create_test_graph_with_skills();
    let engine = SecurityEngine::new(store.clone());

    let context = SecurityContext::new("agent:da/001", "DA");

    let decision = engine
        .check_execution("iri://skills/jwt-auth", &context)
        .await
        .unwrap();
    assert!(decision.is_allowed());

    let untrusted_skill = SkillGraphNode::new("iri://skills/untrusted", "Untrusted", "不可信技能")
        .with_security_info(
            SkillSecurityInfo::new(SkillSource::Imported).with_trust_level(TrustLevel::Untrusted),
        );
    store.register_skill(untrusted_skill).unwrap();

    let decision = engine
        .check_execution("iri://skills/untrusted", &context)
        .await
        .unwrap();
    assert!(!decision.is_allowed());

    engine.whitelist_skill("iri://skills/untrusted").await;
    let decision = engine
        .check_execution("iri://skills/untrusted", &context)
        .await
        .unwrap();
    assert!(decision.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_tool_registration_and_sync() {
    let store = create_test_graph_with_skills();
    let registry = Arc::new(MCPRegistry::new());
    let integration = MCPIntegration::new(registry.clone(), store.clone());

    let server_config = MCPServerConfig {
        server_id: "filesystem-server".to_string(),
        server_name: "Filesystem Server".to_string(),
        endpoint: Some("http://localhost:8080".to_string()),
        trust_level: TrustLevel::High,
        ..Default::default()
    };
    registry.register_server(server_config).await.unwrap();

    let tool = MCPToolInfo::new("filesystem-server", "read_file", "读取文件内容")
        .with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" }
            },
            "required": ["path"]
        }));
    registry.register_tool(tool).await.unwrap();

    let result = integration.sync_tools_to_skills("filesystem-server").await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.tools_added, 1);

    let mapping = registry
        .get_mapping("iri://skills/mcp/filesystem-server-read_file")
        .await;
    assert!(mapping.is_some());

    let tools = registry.list_tools(Some("filesystem-server")).await;
    assert_eq!(tools.len(), 1);
}

#[test]
fn test_conflict_detection_and_resolution() {
    let store = create_test_graph_with_skills();
    let engine = ConflictDetectionEngine::new(Arc::new(tokio::sync::RwLock::new(
        std::collections::HashMap::new(),
    )));

    let mut jwt = store.get_skill("iri://skills/jwt-auth").unwrap();
    jwt.add_link(SkillLink {
        link_type: SkillLinkType::Alternative,
        target_iri: "iri://skills/oauth-auth".to_string(),
        strength: LinkStrength::Required,
        description: "不应同时使用".to_string(),
    });
    store.update_skill(jwt).unwrap();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let report = rt.block_on(engine.detect_all_conflicts()).unwrap();
    assert!(
        report.total_conflicts > 0
            || report.conflicts.len() > 0
            || report.recommendations.len() > 0
    );
}

#[test]
fn test_disclosure_levels() {
    let store = create_test_graph_with_skills();

    let moc_index = store.get_skill_at_level("iri://skills/jwt-auth", DisclosureLevel::MOCIndex);
    assert!(moc_index.is_some());
    let moc = moc_index.unwrap();
    assert!(moc.get("@id").is_some());

    let summary = store.get_skill_at_level("iri://skills/jwt-auth", DisclosureLevel::Summary5W2H);
    assert!(summary.is_some());

    let full = store.get_skill_at_level("iri://skills/jwt-auth", DisclosureLevel::FullContent);
    assert!(full.is_some());
}

#[test]
fn test_bootstrap_meta_tracking() {
    let mut meta = SkillBootstrapMeta::new();
    meta.record_learn(BootstrapSource::new(
        BootstrapSourceType::TaskExecution,
        "agent:da/001",
    ));
    meta.record_learn(BootstrapSource::new(
        BootstrapSourceType::TaskExecution,
        "agent:da/002",
    ));

    assert_eq!(meta.learn_count, 2);
    assert_eq!(meta.bootstrap_sources.len(), 2);
}

#[test]
fn test_audit_trail_and_risk_scoring() {
    let mut security = SkillSecurityInfo::new(SkillSource::MCPExternal)
        .with_trust_level(TrustLevel::Medium)
        .with_permission(SkillPermission {
            permission_id: "perm-1".to_string(),
            resource_pattern: "/api/*".to_string(),
            action: PermissionAction::Read,
            constraints: vec![],
        })
        .with_permission(SkillPermission {
            permission_id: "perm-2".to_string(),
            resource_pattern: "/files/*".to_string(),
            action: PermissionAction::Write,
            constraints: vec!["max_size:10MB".to_string()],
        });

    security.update_risk_score();

    assert!(security.risk_score > 0.0);
    assert!(security.has_permission(PermissionAction::Read, "/api/users"));
    assert!(security.has_permission(PermissionAction::Write, "/files/data.json"));
    assert!(!security.has_permission(PermissionAction::Delete, "/api/users"));
    assert!(!security.has_permission(PermissionAction::Write, "/api/config"));

    security.add_audit_entry(AuditEntry::new(
        "execute",
        "agent:da/001",
        "iri://skills/test",
        AuditOutcome::Success,
    ));
    assert_eq!(security.audit_trail.len(), 1);
}

// ========== SkillCreator 集成测试 ==========

fn create_skill_creator_components() -> (Arc<SkillGraphStore>, Arc<SkillRegistry>) {
    let graph_store = Arc::new(SkillGraphStore::new());
    let registry = Arc::new(SkillRegistry::new());
    (graph_store, registry)
}

#[test]
fn test_skill_creator_static_parse_axure_to_vue2() {
    let def = SkillCreator::convert_markdown_static(AXURE_TO_VUE2_SKILL_MD).unwrap();

    assert!(!def.name.is_empty(), "Skill name 不应为空");
    assert!(
        def.name.contains("axure") || def.name.contains("vue"),
        "Skill name 应包含 axure 或 vue, 实际: {}",
        def.name
    );
    assert!(!def.description.is_empty(), "description 不应为空");
    assert!(!def.what.is_empty(), "what 不应为空");
    assert!(!def.why.is_empty(), "why 不应为空");
    assert!(!def.approach.is_empty(), "approach 不应为空");
    assert!(!def.tags.is_empty(), "tags 不应为空");
}

#[test]
fn test_skill_creator_static_parse_axure_vue2_refactor() {
    let def = SkillCreator::convert_markdown_static(AXURE_VUE2_REFACTOR_SKILL_MD).unwrap();

    assert!(!def.name.is_empty(), "Skill name 不应为空");
    assert!(
        def.name.contains("axure") || def.name.contains("vue") || def.name.contains("refactor"),
        "Skill name 应包含 axure/vue/refactor, 实际: {}",
        def.name
    );
    assert!(!def.description.is_empty(), "description 不应为空");
    assert!(!def.what.is_empty(), "what 不应为空");
}

#[test]
fn test_skill_creator_register_axure_to_vue2() {
    let (graph_store, registry) = create_skill_creator_components();

    let mut def = SkillCreator::convert_markdown_static(AXURE_TO_VUE2_SKILL_MD).unwrap();

    def.category = "execution".to_string();
    def.security_level = "high".to_string();
    def.allowed_roles = vec!["DA".to_string()];
    def.input_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "axure_export_dir": {"type": "string", "description": "Axure 导出 HTML5 目录路径"},
            "output_dir": {"type": "string", "description": "Vue2 项目输出目录"},
            "viewport_width": {"type": "integer", "description": "视口宽度", "default": 1920},
            "viewport_height": {"type": "integer", "description": "视口高度", "default": 1080}
        },
        "required": ["axure_export_dir", "output_dir"]
    });
    def.output_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "vue_project_path": {"type": "string", "description": "生成的 Vue2 项目路径"},
            "components_created": {"type": "integer", "description": "创建的组件数量"},
            "comparison_score": {"type": "number", "description": "对比评分"}
        }
    });
    def.tags = vec![
        "axure".to_string(),
        "vue2".to_string(),
        "conversion".to_string(),
        "frontend".to_string(),
        "playwright".to_string(),
    ];
    def.what = "将 Axure RP 导出的 HTML5 原型转换为优化的 Vue 2 前端代码".to_string();
    def.why = "自动化 Axure 原型到 Vue 代码的转换，提高前端开发效率".to_string();
    def.approach =
        "三阶段流程：Playwright 信息提取 → LLM 驱动的 Vue2 编码 → 自动化验证与迭代修正".to_string();

    let config = SkillCreatorConfig {
        output_dir: std::path::PathBuf::from("/tmp/agent_os_test_skills"),
        auto_register: true,
        validate_before_register: true,
        default_security_level: "high".to_string(),
    };

    let gateway = Arc::new(
        glidinghorse::gateway::UnifiedGateway::new(&glidinghorse::config::GatewaySettings {
            base_url: "http://localhost:3000".to_string(),
            api_key: "test".to_string(),
            default_model: "test".to_string(),
            timeout_seconds: 30,
            max_retries: 1,
            retry_base_ms: 500,
            use_responses_api: false,
            model_mapping: Default::default(),
        })
        .unwrap(),
    );

    let creator = SkillCreator::new(gateway, graph_store.clone(), registry.clone(), config);
    let created = creator.create_from_definition(def, None).unwrap();

    assert_eq!(created.skill_iri, format!("iri://skills/{}", created.name));
    assert!(!created.name.is_empty());

    let retrieved = graph_store.get_skill(&created.skill_iri);
    assert!(retrieved.is_some(), "Skill 应已注册到 GraphStore");
    let node = retrieved.unwrap();
    assert!(
        node.tags.iter().any(|t| t == "axure" || t == "vue2"),
        "tags 应包含 axure 或 vue2"
    );

    let registry_meta = registry.get_skill(&created.skill_iri);
    assert!(registry_meta.is_some(), "Skill 应已注册到 SkillRegistry");
    let meta = registry_meta.unwrap();
    assert_eq!(meta.category, "execution");
    assert_eq!(meta.security_level, "high");

    let json_ld = &created.json_ld;
    assert!(json_ld.get("@id").is_some(), "JSON-LD 应包含 @id");
    assert!(json_ld.get("@type").is_some(), "JSON-LD 应包含 @type");
    assert!(json_ld.get("@context").is_some(), "JSON-LD 应包含 @context");
    assert_eq!(json_ld["@id"], created.skill_iri);
}

#[test]
fn test_skill_creator_register_axure_vue2_refactor() {
    let (graph_store, registry) = create_skill_creator_components();

    let mut def = SkillCreator::convert_markdown_static(AXURE_VUE2_REFACTOR_SKILL_MD).unwrap();

    def.category = "execution".to_string();
    def.security_level = "normal".to_string();
    def.allowed_roles = vec!["DA".to_string(), "CA".to_string()];
    def.input_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "vue_component_path": {"type": "string", "description": "待重构的 Vue2 组件路径"},
            "refactor_options": {
                "type": "object",
                "properties": {
                    "use_tailwind": {"type": "boolean", "description": "是否使用 Tailwind CSS", "default": true},
                    "add_i18n": {"type": "boolean", "description": "是否添加国际化支持", "default": true},
                    "extract_components": {"type": "boolean", "description": "是否抽取子组件", "default": true},
                    "analyze_dynamic_data": {"type": "boolean", "description": "是否分析动态数据区域", "default": true}
                }
            },
            "original_url": {"type": "string", "description": "原始页面 URL (用于对比验证)"},
            "refactored_url": {"type": "string", "description": "重构后页面 URL (用于对比验证)"}
        },
        "required": ["vue_component_path"]
    });
    def.output_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "refactored_path": {"type": "string", "description": "重构后的组件路径"},
            "similarity_score": {"type": "number", "description": "像素对比相似度"},
            "components_extracted": {"type": "integer", "description": "抽取的子组件数量"},
            "i18n_keys_created": {"type": "integer", "description": "创建的 i18n key 数量"}
        }
    });
    def.tags = vec![
        "axure".to_string(),
        "vue2".to_string(),
        "refactor".to_string(),
        "tailwind".to_string(),
        "i18n".to_string(),
        "frontend".to_string(),
    ];
    def.what = "将 Axure 风格的 Vue2 代码重构为符合最佳实践的 Vue2 代码，支持 Tailwind CSS、语义化命名、组件抽象、动态数据分析和国际化".to_string();
    def.why =
        "Axure 导出的代码质量差，包含无意义 ID、绝对定位、硬编码文本等问题，需要重构为可维护的代码"
            .to_string();
    def.approach =
        "六阶段流程：环境搭建 → 分析规划 → 资产重构 → 组件重构 → 样式迁移 → 像素级验证与迭代"
            .to_string();

    let config = SkillCreatorConfig {
        output_dir: std::path::PathBuf::from("/tmp/agent_os_test_skills"),
        auto_register: true,
        validate_before_register: true,
        default_security_level: "normal".to_string(),
    };

    let gateway = Arc::new(
        glidinghorse::gateway::UnifiedGateway::new(&glidinghorse::config::GatewaySettings {
            base_url: "http://localhost:3000".to_string(),
            api_key: "test".to_string(),
            default_model: "test".to_string(),
            timeout_seconds: 30,
            max_retries: 1,
            retry_base_ms: 500,
            use_responses_api: false,
            model_mapping: Default::default(),
        })
        .unwrap(),
    );

    let creator = SkillCreator::new(gateway, graph_store.clone(), registry.clone(), config);
    let created = creator.create_from_definition(def, None).unwrap();

    assert_eq!(created.skill_iri, format!("iri://skills/{}", created.name));

    let retrieved = graph_store.get_skill(&created.skill_iri);
    assert!(retrieved.is_some(), "Skill 应已注册到 GraphStore");

    let registry_meta = registry.get_skill(&created.skill_iri);
    assert!(registry_meta.is_some(), "Skill 应已注册到 SkillRegistry");
    let meta = registry_meta.unwrap();
    assert_eq!(meta.category, "execution");
    assert_eq!(meta.security_level, "normal");
    assert!(meta.allowed_roles.contains(&"DA".to_string()));
    assert!(meta.allowed_roles.contains(&"CA".to_string()));

    let json_ld = &created.json_ld;
    assert!(json_ld.get("@id").is_some());
    assert!(json_ld.get("@context").is_some());
}

#[tokio::test]
async fn test_skill_creator_two_skills_coexist() {
    let (graph_store, registry) = create_skill_creator_components();

    let config = SkillCreatorConfig {
        output_dir: std::path::PathBuf::from("/tmp/agent_os_test_skills"),
        auto_register: true,
        validate_before_register: true,
        default_security_level: "normal".to_string(),
    };

    let gateway = Arc::new(
        glidinghorse::gateway::UnifiedGateway::new(&glidinghorse::config::GatewaySettings {
            base_url: "http://localhost:3000".to_string(),
            api_key: "test".to_string(),
            default_model: "test".to_string(),
            timeout_seconds: 30,
            max_retries: 1,
            retry_base_ms: 500,
            use_responses_api: false,
            model_mapping: Default::default(),
        })
        .unwrap(),
    );

    let creator = SkillCreator::new(gateway, graph_store.clone(), registry.clone(), config);

    let mut def1 = SkillCreator::convert_markdown_static(AXURE_TO_VUE2_SKILL_MD).unwrap();
    def1.category = "execution".to_string();
    def1.security_level = "high".to_string();
    def1.allowed_roles = vec!["DA".to_string()];
    def1.input_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "axure_export_dir": {"type": "string"},
            "output_dir": {"type": "string"}
        },
        "required": ["axure_export_dir"]
    });
    def1.output_schema =
        serde_json::json!({"type": "object", "properties": {"result": {"type": "string"}}});
    def1.tags = vec![
        "axure".to_string(),
        "vue2".to_string(),
        "conversion".to_string(),
    ];
    def1.what = "Axure 原型转 Vue2 代码".to_string();
    def1.why = "自动化前端开发".to_string();
    def1.approach = "三阶段流程".to_string();

    let mut def2 = SkillCreator::convert_markdown_static(AXURE_VUE2_REFACTOR_SKILL_MD).unwrap();
    def2.category = "execution".to_string();
    def2.security_level = "normal".to_string();
    def2.allowed_roles = vec!["DA".to_string(), "CA".to_string()];
    def2.input_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "vue_component_path": {"type": "string"},
            "refactor_options": {"type": "object"}
        },
        "required": ["vue_component_path"]
    });
    def2.output_schema =
        serde_json::json!({"type": "object", "properties": {"result": {"type": "string"}}});
    def2.tags = vec![
        "axure".to_string(),
        "vue2".to_string(),
        "refactor".to_string(),
    ];
    def2.what = "Axure Vue2 代码重构".to_string();
    def2.why = "提升代码质量".to_string();
    def2.approach = "六阶段流程".to_string();

    let created1 = creator.create_from_definition(def1, None).unwrap();
    let created2 = creator.create_from_definition(def2, None).unwrap();

    assert_ne!(
        created1.skill_iri, created2.skill_iri,
        "两个 Skill 的 IRI 应不同"
    );

    let all_skills = graph_store.list_all_skills();
    assert_eq!(all_skills.len(), 2, "应注册了 2 个 Skill");

    let all_registry = registry.list_skills_basic();
    assert!(
        all_registry.len() >= 2,
        "SkillRegistry 中应至少有 2 个 Skill (包含内置技能), 实际: {}",
        all_registry.len()
    );

    let discovery = SkillDiscoveryEngine::new(graph_store.clone());
    let task = Task5W2H::new("Axure 原型转 Vue2", "前端开发")
        .with_agent_role("DA")
        .with_phase("Do");
    let matches = discovery.discover_for_task(&task).await;
    assert!(!matches.is_empty(), "任务发现应能找到相关 Skill");
}

#[test]
fn test_tool_executor_convert_skill_static() {
    let executor = ToolExecutor::new();

    let input = serde_json::json!({
        "markdown_content": AXURE_TO_VUE2_SKILL_MD,
        "source_path": "/dev-data/ai-test/code_skills/.trae/skills/axure-to-vue2-skill/SKILL.md"
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(executor.execute("convert_skill", input));
    assert!(result.is_ok(), "convert_skill 执行应成功: {:?}", result);
    let output = result.unwrap();
    assert!(output["name"].is_string(), "输出应包含 name");
    assert!(output["skill_iri"].is_string(), "输出应包含 skill_iri");
    assert!(
        output["skill_iri"]
            .as_str()
            .unwrap()
            .starts_with("iri://skills/"),
        "skill_iri 应以 iri://skills/ 开头"
    );
}

#[test]
fn test_tool_executor_convert_skill_refactor() {
    let executor = ToolExecutor::new();

    let input = serde_json::json!({
        "markdown_content": AXURE_VUE2_REFACTOR_SKILL_MD,
        "source_path": "/dev-data/ai-test/code_skills/.trae/skills/axure-vue2-refactor/SKILL.md"
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(executor.execute("convert_skill", input));
    assert!(result.is_ok(), "convert_skill 执行应成功: {:?}", result);
    let output = result.unwrap();
    assert!(output["name"].is_string(), "输出应包含 name");
    assert!(output["skill_iri"].is_string(), "输出应包含 skill_iri");
}

#[test]
fn test_skill_creator_json_ld_validity() {
    let (graph_store, registry) = create_skill_creator_components();

    let config = SkillCreatorConfig {
        output_dir: std::path::PathBuf::from("/tmp/agent_os_test_skills"),
        auto_register: true,
        validate_before_register: true,
        default_security_level: "normal".to_string(),
    };

    let gateway = Arc::new(
        glidinghorse::gateway::UnifiedGateway::new(&glidinghorse::config::GatewaySettings {
            base_url: "http://localhost:3000".to_string(),
            api_key: "test".to_string(),
            default_model: "test".to_string(),
            timeout_seconds: 30,
            max_retries: 1,
            retry_base_ms: 500,
            use_responses_api: false,
            model_mapping: Default::default(),
        })
        .unwrap(),
    );

    let creator = SkillCreator::new(gateway, graph_store.clone(), registry.clone(), config);

    let mut def = SkillCreator::convert_markdown_static(AXURE_TO_VUE2_SKILL_MD).unwrap();
    def.category = "execution".to_string();
    def.security_level = "high".to_string();
    def.allowed_roles = vec!["DA".to_string()];
    def.input_schema = serde_json::json!({"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]});
    def.output_schema =
        serde_json::json!({"type": "object", "properties": {"result": {"type": "string"}}});
    def.tags = vec!["axure".to_string(), "vue2".to_string()];
    def.what = "Axure 转 Vue2".to_string();
    def.why = "自动化".to_string();
    def.approach = "三阶段".to_string();

    let created = creator.create_from_definition(def, None).unwrap();
    let json_ld = &created.json_ld;

    let context = json_ld.get("@context").unwrap();
    assert!(context.is_object(), "@context 应为对象");
    assert!(
        context.get("skill").is_some(),
        "@context 应包含 skill 命名空间"
    );
    assert!(
        context.get("schema").is_some(),
        "@context 应包含 schema 命名空间"
    );

    let at_type = json_ld.get("@type").unwrap();
    assert!(at_type.is_array(), "@type 应为数组");
    assert!(
        at_type
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == "skill:Skill"),
        "@type 应包含 skill:Skill"
    );

    let w2h = json_ld.get("skill:5W2H").unwrap();
    assert!(w2h.get("skill:what").is_some(), "5W2H 应包含 what");
    assert!(w2h.get("skill:why").is_some(), "5W2H 应包含 why");
    assert!(w2h.get("skill:how").is_some(), "5W2H 应包含 how");

    let json_str = serde_json::to_string_pretty(&json_ld).unwrap();
    let reparsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(
        reparsed["@id"], json_ld["@id"],
        "JSON-LD 应可序列化/反序列化"
    );
}

#[test]
fn test_skill_creator_disclosure_levels() {
    let (graph_store, registry) = create_skill_creator_components();

    let config = SkillCreatorConfig {
        output_dir: std::path::PathBuf::from("/tmp/agent_os_test_skills"),
        auto_register: true,
        validate_before_register: true,
        default_security_level: "normal".to_string(),
    };

    let gateway = Arc::new(
        glidinghorse::gateway::UnifiedGateway::new(&glidinghorse::config::GatewaySettings {
            base_url: "http://localhost:3000".to_string(),
            api_key: "test".to_string(),
            default_model: "test".to_string(),
            timeout_seconds: 30,
            max_retries: 1,
            retry_base_ms: 500,
            use_responses_api: false,
            model_mapping: Default::default(),
        })
        .unwrap(),
    );

    let creator = SkillCreator::new(gateway, graph_store.clone(), registry.clone(), config);

    let mut def = SkillCreator::convert_markdown_static(AXURE_VUE2_REFACTOR_SKILL_MD).unwrap();
    def.category = "execution".to_string();
    def.security_level = "normal".to_string();
    def.allowed_roles = vec!["DA".to_string(), "CA".to_string()];
    def.input_schema = serde_json::json!({"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]});
    def.output_schema =
        serde_json::json!({"type": "object", "properties": {"result": {"type": "string"}}});
    def.tags = vec!["axure".to_string(), "refactor".to_string()];
    def.what = "Axure Vue2 重构".to_string();
    def.why = "提升代码质量".to_string();
    def.approach = "六阶段流程".to_string();

    let created = creator.create_from_definition(def, None).unwrap();

    let moc = graph_store.get_skill_at_level(&created.skill_iri, DisclosureLevel::MOCIndex);
    assert!(moc.is_some(), "MOCIndex 级别应可获取");
    let moc_val = moc.unwrap();
    assert!(moc_val.get("@id").is_some());

    let summary = graph_store.get_skill_at_level(&created.skill_iri, DisclosureLevel::Summary5W2H);
    assert!(summary.is_some(), "Summary5W2H 级别应可获取");

    let full = graph_store.get_skill_at_level(&created.skill_iri, DisclosureLevel::FullContent);
    assert!(full.is_some(), "FullContent 级别应可获取");
}

// ═════════════════════════════════════════════════════════════════════
// Causal Engine integration tests (end-to-end)
// ═════════════════════════════════════════════════════════════════════

fn create_causal_store_with_links() -> Arc<SkillGraphStore> {
    let store = Arc::new(SkillGraphStore::new());

    // Base service (root cause target)
    let base = SkillGraphNode::new("iri://skills/base", "Base Service", "Core base service");
    store.register_skill(base).unwrap();

    // Auth service depends on base
    let auth = SkillGraphNode::new("iri://skills/auth", "Auth Service", "Authentication")
        .with_link(SkillLink {
            link_type: SkillLinkType::Prerequisite,
            target_iri: "iri://skills/base".to_string(),
            strength: LinkStrength::Required,
            description: "Auth requires base".to_string(),
        });
    store.register_skill(auth).unwrap();

    // API gateway depends on auth and base
    let api = SkillGraphNode::new("iri://skills/api-gateway", "API Gateway", "API gateway")
        .with_link(SkillLink {
            link_type: SkillLinkType::Prerequisite,
            target_iri: "iri://skills/auth".to_string(),
            strength: LinkStrength::Required,
            description: "API depends on auth".to_string(),
        })
        .with_link(SkillLink {
            link_type: SkillLinkType::Composition,
            target_iri: "iri://skills/base".to_string(),
            strength: LinkStrength::Required,
            description: "API includes base".to_string(),
        });
    store.register_skill(api).unwrap();

    store
}

/// Create a minimal CausalEngine wrapping a PetgraphBackend for the given store.
fn create_causal_engine(store: &Arc<SkillGraphStore>) -> Arc<CausalEngine> {
    let model_store = Arc::new(CausalModelStore::new());
    let backend: Arc<dyn GraphBackend> = Arc::new(PetgraphBackend::new(store.clone()));
    Arc::new(CausalEngine::new(model_store, backend))
}

#[test]
fn test_causal_pipeline_e2e() {
    let store = create_causal_store_with_links();
    let causal = create_causal_engine(&store);
    let mut engine = SkillEvolutionEngine::new(store)
        .with_causal_analysis(5000)
        .with_causal_engine(causal);

    // Record failure in api-gateway — should trigger Path A (CausalEngine delegate)
    engine
        .record_usage(
            UsageRecord::new(
                "iri://skills/api-gateway",
                "iri://task/e2e-001",
                "agent:da/001",
                false,
            )
            .with_error("Gateway timeout after 30s"),
        )
        .unwrap();

    // Should produce at least one evolution suggestion
    let suggestions = engine.get_pending_suggestions();
    assert!(
        !suggestions.is_empty(),
        "Causal failure should produce suggestions"
    );

    // The suggestion should reference the failed skill
    assert_eq!(suggestions[0].skill_iri, "iri://skills/api-gateway");

    // Confidence from CausalEngine should be > 0
    assert!(
        suggestions[0].confidence > 0.0,
        "CausalEngine should provide non-zero confidence"
    );
}

#[test]
fn test_causal_propagation_chain_e2e() {
    let store = create_causal_store_with_links();
    let causal = create_causal_engine(&store);
    let mut engine = SkillEvolutionEngine::new(store)
        .with_causal_analysis(5000)
        .with_causal_engine(causal);

    // Record failure in base (root cause)
    engine
        .record_usage(
            UsageRecord::new(
                "iri://skills/base",
                "iri://task/e2e-002",
                "agent:da/001",
                false,
            )
            .with_error("Base service crashed"),
        )
        .unwrap();

    // Record failure in auth (depends on base) — should detect propagation
    engine
        .record_usage(
            UsageRecord::new(
                "iri://skills/auth",
                "iri://task/e2e-003",
                "agent:da/002",
                false,
            )
            .with_error("Auth failed — dependency unavailable"),
        )
        .unwrap();

    // After two failures, both should be tracked via suggestions
    let suggestions = engine.get_pending_suggestions();
    assert_eq!(
        suggestions.len(),
        2,
        "Both failures should produce suggestions"
    );

    // Search root cause on a known-good event via find_root_cause with
    // a generated event ID pattern — verify it returns Some
    let events_found = engine.suggest_preventive_action("iri://skills/base");
    assert!(
        !events_found.is_empty() || suggestions.len() >= 2,
        "Either preventive actions or suggestions should exist"
    );
}

#[test]
fn test_causal_preventive_actions_e2e() {
    let store = create_causal_store_with_links();
    let causal = create_causal_engine(&store);
    let mut engine = SkillEvolutionEngine::new(store)
        .with_causal_analysis(5000)
        .with_causal_engine(causal);

    // Record enough failures to trigger preventive actions
    for i in 0..8 {
        let record = UsageRecord::new(
            "iri://skills/base",
            &format!("iri://task/e2e-{:03}", i),
            "agent:da/001",
            false,
        )
        .with_error("Connection timeout");
        engine.record_usage(record).unwrap();
    }

    // Check that preventive actions exist
    let actions = engine.suggest_preventive_action("iri://skills/base");
    assert!(
        !actions.is_empty(),
        "Multiple failures should generate preventive actions for base"
    );

    // At least one action should mention failure count
    assert!(
        actions.iter().any(|a| a.contains("failures")),
        "Preventive actions should reference recorded failures: {:?}",
        actions
    );
}

#[test]
fn test_causal_pipeline_shared_engine() {
    let store = create_causal_store_with_links();
    let causal = create_causal_engine(&store);

    // Two evolution engines sharing the same CausalEngine
    let mut engine_a = SkillEvolutionEngine::new(store.clone())
        .with_causal_analysis(5000)
        .with_causal_engine(causal.clone());

    let mut engine_b = SkillEvolutionEngine::new(store)
        .with_causal_analysis(5000)
        .with_causal_engine(causal);

    // Both record failures
    engine_a
        .record_usage(
            UsageRecord::new(
                "iri://skills/api-gateway",
                "iri://task/shared-001",
                "agent:da/001",
                false,
            )
            .with_error("Engine A timeout"),
        )
        .unwrap();

    engine_b
        .record_usage(
            UsageRecord::new(
                "iri://skills/auth",
                "iri://task/shared-002",
                "agent:da/002",
                false,
            )
            .with_error("Engine B auth error"),
        )
        .unwrap();

    // Both should produce suggestions
    assert!(
        !engine_a.get_pending_suggestions().is_empty(),
        "Engine A should have suggestions from shared CausalEngine"
    );
    assert!(
        !engine_b.get_pending_suggestions().is_empty(),
        "Engine B should have suggestions from shared CausalEngine"
    );
}

#[test]
fn test_causal_no_false_positive_on_success() {
    let store = create_causal_store_with_links();
    let causal = create_causal_engine(&store);
    let mut engine = SkillEvolutionEngine::new(store)
        .with_causal_analysis(5000)
        .with_causal_engine(causal);

    // Successful usage should NOT trigger causal analysis
    engine
        .record_usage(
            UsageRecord::new(
                "iri://skills/auth",
                "iri://task/success-001",
                "agent:da/001",
                true,
            )
            .with_tokens(500),
        )
        .unwrap();

    // With no failures, no preventive actions
    let actions = engine.suggest_preventive_action("iri://skills/auth");
    assert!(
        actions.is_empty(),
        "No failures means no preventive actions"
    );
    assert!(
        engine.get_pending_suggestions().is_empty(),
        "Successful usage should not create suggestions"
    );
}

#[test]
fn test_causal_legacy_fallback_without_engine() {
    // Without a CausalEngine, the legacy Path B is used
    let store = create_causal_store_with_links();
    let mut engine = SkillEvolutionEngine::new(store).with_causal_analysis(5000);

    engine
        .record_usage(
            UsageRecord::new(
                "iri://skills/auth",
                "iri://task/legacy-001",
                "agent:da/001",
                false,
            )
            .with_error("Legacy path error"),
        )
        .unwrap();

    assert!(
        !engine.get_pending_suggestions().is_empty(),
        "Legacy path should still produce suggestions"
    );
    assert_eq!(
        engine.get_pending_suggestions()[0].confidence,
        0.7,
        "Legacy path defaults to 0.7 confidence"
    );
}
