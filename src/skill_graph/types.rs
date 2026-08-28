use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillNodeType {
    Atomic,
    Composite,
    MOC,
    KnowledgeFragment,
    MCPTool,
    Bootstrap,
}

impl Default for SkillNodeType {
    fn default() -> Self {
        Self::Atomic
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrustLevel {
    #[default]
    Untrusted = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    System = 4,
}

impl TrustLevel {
    pub fn can_execute(&self, required: TrustLevel) -> bool {
        *self as u8 >= required as u8
    }

    pub fn from_success_rate(rate: f32) -> Self {
        if rate >= 0.95 {
            TrustLevel::High
        } else if rate >= 0.80 {
            TrustLevel::Medium
        } else if rate >= 0.60 {
            TrustLevel::Low
        } else {
            TrustLevel::Untrusted
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillSource {
    SystemBuiltin,
    UserDefined,
    MCPExternal,
    BootstrapLearn,
    BootstrapReduce,
    Imported,
}

impl Default for SkillSource {
    fn default() -> Self {
        Self::SystemBuiltin
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPermission {
    pub permission_id: String,
    pub resource_pattern: String,
    pub action: PermissionAction,
    pub constraints: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionAction {
    Read,
    Write,
    Execute,
    Delete,
    Admin,
}

impl Default for PermissionAction {
    fn default() -> Self {
        Self::Read
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillSecurityInfo {
    pub trust_level: TrustLevel,
    pub source: SkillSource,
    pub permissions: Vec<SkillPermission>,
    pub signature: Option<String>,
    pub signature_algorithm: Option<String>,
    pub cert_chain: Vec<String>,
    pub audit_trail: Vec<AuditEntry>,
    pub risk_score: f32,
}

impl SkillSecurityInfo {
    pub fn new(source: SkillSource) -> Self {
        Self {
            trust_level: TrustLevel::default(),
            source,
            permissions: Vec::new(),
            signature: None,
            signature_algorithm: None,
            cert_chain: Vec::new(),
            audit_trail: Vec::new(),
            risk_score: 0.0,
        }
    }

    pub fn with_trust_level(mut self, level: TrustLevel) -> Self {
        self.trust_level = level;
        self
    }

    /// Sets the risk of the operation independently of the provenance trust.
    /// A system-provided skill may be authentic while still requiring stricter
    /// handling because it can perform a destructive operation.
    pub fn with_risk_score(mut self, score: f32) -> Self {
        self.risk_score = score.clamp(0.0, 1.0);
        self
    }

    pub fn with_permission(mut self, permission: SkillPermission) -> Self {
        self.permissions.push(permission);
        self
    }

    pub fn with_signature(mut self, signature: &str, algorithm: &str) -> Self {
        self.signature = Some(signature.to_string());
        self.signature_algorithm = Some(algorithm.to_string());
        self
    }

    pub fn add_audit_entry(&mut self, entry: AuditEntry) {
        self.audit_trail.push(entry);
    }

    pub fn has_permission(&self, action: PermissionAction, resource: &str) -> bool {
        self.permissions.iter().any(|p| {
            if p.action != action {
                return false;
            }
            if p.resource_pattern == "*" {
                return true;
            }
            if p.resource_pattern.ends_with('*') {
                let prefix = &p.resource_pattern[..p.resource_pattern.len() - 1];
                return resource.starts_with(prefix);
            }
            resource == p.resource_pattern
        })
    }

    pub fn update_risk_score(&mut self) {
        let base_score = match self.trust_level {
            TrustLevel::System => 0.0,
            TrustLevel::High => 0.1,
            TrustLevel::Medium => 0.3,
            TrustLevel::Low => 0.5,
            TrustLevel::Untrusted => 1.0,
        };

        let source_modifier = match self.source {
            SkillSource::SystemBuiltin => 0.0,
            SkillSource::UserDefined => 0.1,
            SkillSource::BootstrapLearn => 0.15,
            SkillSource::BootstrapReduce => 0.1,
            SkillSource::MCPExternal => 0.2,
            SkillSource::Imported => 0.25,
        };

        let signature_modifier = if self.signature.is_some() {
            -0.1f32
        } else {
            0.1f32
        };

        self.risk_score = (base_score + source_modifier + signature_modifier).clamp(0.0, 1.0);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub action: String,
    pub agent_id: String,
    pub resource: String,
    pub outcome: AuditOutcome,
    pub details: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditOutcome {
    Success,
    Failure,
    Denied,
    Warning,
}

impl Default for AuditOutcome {
    fn default() -> Self {
        Self::Success
    }
}

impl AuditEntry {
    pub fn new(action: &str, agent_id: &str, resource: &str, outcome: AuditOutcome) -> Self {
        Self {
            timestamp: Utc::now(),
            action: action.to_string(),
            agent_id: agent_id.to_string(),
            resource: resource.to_string(),
            outcome,
            details: None,
        }
    }

    pub fn with_details(mut self, details: &str) -> Self {
        self.details = Some(details.to_string());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillLinkType {
    Prerequisite,
    Composition,
    Related,
    Alternative,
    Extends,
    Generalization,
}

impl SkillLinkType {
    pub fn as_sparql_uri(&self) -> &'static str {
        match self {
            Self::Prerequisite => "skill:Prerequisite",
            Self::Composition => "skill:Composition",
            Self::Related => "skill:Related",
            Self::Alternative => "skill:Alternative",
            Self::Extends => "skill:Extends",
            Self::Generalization => "skill:Generalization",
        }
    }
}

impl Default for SkillLinkType {
    fn default() -> Self {
        Self::Related
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkStrength {
    Required,
    Recommended,
    Optional,
    Navigation,
}

impl Default for LinkStrength {
    fn default() -> Self {
        Self::Recommended
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillLink {
    pub link_type: SkillLinkType,
    pub target_iri: String,
    pub strength: LinkStrength,
    pub description: String,
}

impl SkillLink {
    pub fn new(link_type: SkillLinkType, target_iri: String) -> Self {
        Self {
            link_type,
            target_iri,
            strength: LinkStrength::default(),
            description: String::new(),
        }
    }

    pub fn with_strength(mut self, strength: LinkStrength) -> Self {
        self.strength = strength;
        self
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillRole {
    pub role_name: String,
    pub required_agent_role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillTrigger {
    pub applicable_phases: Vec<String>,
    pub trigger_condition: Option<String>,
    pub deadline_constraint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillContext {
    pub target_stack: Vec<String>,
    pub repo_pattern: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillApproach {
    pub approach: String,
    pub plan_iri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillCost {
    pub avg_token_cost: u32,
    pub avg_duration_seconds: u32,
    pub max_sub_agents: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Skill5W2H {
    pub what: String,
    pub why: String,
    pub who: SkillRole,
    pub when: SkillTrigger,
    pub where_: SkillContext,
    pub how: SkillApproach,
    pub how_much: SkillCost,
}

impl Skill5W2H {
    pub fn new(what: &str, why: &str) -> Self {
        Self {
            what: what.to_string(),
            why: why.to_string(),
            ..Default::default()
        }
    }

    pub fn with_phase(mut self, phase: &str) -> Self {
        self.when.applicable_phases.push(phase.to_string());
        self
    }

    pub fn with_agent_role(mut self, role: &str) -> Self {
        self.who.required_agent_role = Some(role.to_string());
        self
    }

    pub fn with_token_cost(mut self, cost: u32) -> Self {
        self.how_much.avg_token_cost = cost;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FailureMode {
    pub mode: String,
    pub discovered_by: Option<String>,
    pub mitigation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillGraphMeta {
    pub created_at: Option<DateTime<Utc>>,
    pub last_verified_at: Option<DateTime<Utc>>,
    pub usage_count: u32,
    pub success_rate: f32,
    pub avg_token_consumption: u32,
    pub known_failure_modes: Vec<FailureMode>,
}

impl SkillGraphMeta {
    pub fn new() -> Self {
        Self {
            created_at: Some(Utc::now()),
            usage_count: 0,
            success_rate: 1.0,
            ..Default::default()
        }
    }

    pub fn record_usage(&mut self, success: bool) {
        let old_count = self.usage_count;
        self.usage_count += 1;

        let old_rate = self.success_rate;
        self.success_rate = if success {
            (old_rate * old_count as f32 + 1.0) / self.usage_count as f32
        } else {
            (old_rate * old_count as f32) / self.usage_count as f32
        };

        self.last_verified_at = Some(Utc::now());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisclosureLevel {
    MOCIndex = 1,
    Summary5W2H = 2,
    LinksExpanded = 3,
    SchemaSteps = 4,
    FullContent = 5,
}

impl Default for DisclosureLevel {
    fn default() -> Self {
        Self::Summary5W2H
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillStep {
    pub step_id: String,
    pub order: u32,
    pub action: String,
    pub code: Option<String>,
    pub references: Vec<String>,
}

impl SkillStep {
    pub fn new(step_id: &str, order: u32, action: &str) -> Self {
        Self {
            step_id: step_id.to_string(),
            order,
            action: action.to_string(),
            code: None,
            references: Vec::new(),
        }
    }

    pub fn with_code(mut self, code: &str) -> Self {
        self.code = Some(code.to_string());
        self
    }

    pub fn with_reference(mut self, ref_iri: &str) -> Self {
        self.references.push(ref_iri.to_string());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillValidation {
    pub method: String,
    pub success_condition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillContent {
    pub summary: String,
    pub steps: Vec<SkillStep>,
    pub validation: Option<SkillValidation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillGraphNode {
    pub skill_iri: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub node_type: SkillNodeType,
    pub maturity: String,
    pub tags: Vec<String>,
    pub w2h: Skill5W2H,
    pub links: Vec<SkillLink>,
    pub graph_meta: SkillGraphMeta,
    pub content: Option<SkillContent>,
    pub attached_to: Option<String>,
    pub security_info: Option<SkillSecurityInfo>,
    pub mcp_server_id: Option<String>,
    pub storage_tier: StorageTier,
    /// When this skill node was created
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    /// When this skill node was last modified
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,
    /// When this skill node was last used/activated, None if never used
    #[serde(default)]
    pub last_used_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StorageTier {
    #[default]
    L0Permanent,
    L1Session,
    L2Blackboard,
    L3Projection,
}

impl SkillGraphNode {
    pub fn new(skill_iri: &str, name: &str, description: &str) -> Self {
        let now = Utc::now();
        Self {
            skill_iri: skill_iri.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            version: "1.0.0".to_string(),
            node_type: SkillNodeType::Atomic,
            maturity: "stable".to_string(),
            tags: Vec::new(),
            w2h: Skill5W2H::default(),
            links: Vec::new(),
            graph_meta: SkillGraphMeta::new(),
            content: None,
            attached_to: None,
            security_info: None,
            mcp_server_id: None,
            storage_tier: StorageTier::default(),
            created_at: now,
            updated_at: now,
            last_used_at: None,
        }
    }

    pub fn with_node_type(mut self, node_type: SkillNodeType) -> Self {
        self.node_type = node_type;
        self
    }

    pub fn with_5w2h(mut self, w2h: Skill5W2H) -> Self {
        self.w2h = w2h;
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn with_link(mut self, link: SkillLink) -> Self {
        self.links.push(link);
        self
    }

    pub fn with_content(mut self, content: SkillContent) -> Self {
        self.content = Some(content);
        self
    }

    pub fn with_security_info(mut self, security_info: SkillSecurityInfo) -> Self {
        self.security_info = Some(security_info);
        self
    }

    pub fn with_mcp_server(mut self, server_id: &str) -> Self {
        self.mcp_server_id = Some(server_id.to_string());
        self.node_type = SkillNodeType::MCPTool;
        self
    }

    pub fn with_storage_tier(mut self, tier: StorageTier) -> Self {
        self.storage_tier = tier;
        self
    }

    /// Mark this skill as just used (sets `last_used_at` to now).
    pub fn with_last_used(mut self) -> Self {
        self.last_used_at = Some(Utc::now());
        self
    }

    /// Update `updated_at` to now (used after modifications / re-index).
    pub fn with_updated_at(mut self) -> Self {
        self.updated_at = Utc::now();
        self
    }

    /// Convert a `SkillMeta` (from SkillRegistry) into a `SkillGraphNode`.
    /// Tags are populated from `category` and `skill_types`. Provenance trust
    /// and operation risk are deliberately separate: `security_level` affects
    /// the latter and must not silently turn a critical operation into a
    /// zero-risk system operation.
    pub fn from_skill_meta(meta: &crate::tools::skill_registry::SkillMeta) -> Self {
        let mut tags: Vec<String> = meta.skill_types.clone();
        if !tags.contains(&meta.category) {
            tags.push(meta.category.clone());
        }
        for role in &meta.allowed_roles {
            let role_tag = format!("allowed-role:{role}");
            if !tags.contains(&role_tag) {
                tags.push(role_tag);
            }
        }

        let operation_risk = match meta.security_level.as_str() {
            "critical" => 0.9,
            "high" => 0.7,
            "normal" => 0.3,
            "low" => 0.1,
            _ => 0.5,
        };
        let security = SkillSecurityInfo::new(SkillSource::SystemBuiltin)
            .with_trust_level(TrustLevel::System)
            .with_risk_score(operation_risk);

        let w2h = meta
            .discovery_5w2h
            .clone()
            .unwrap_or_else(|| Skill5W2H::new(&meta.name, &meta.description).with_agent_role("DA"));

        Self {
            skill_iri: meta.skill_iri.clone(),
            name: meta.name.clone(),
            description: meta.description.clone(),
            version: meta.version.clone(),
            node_type: SkillNodeType::Atomic,
            maturity: "stable".to_string(),
            tags,
            w2h,
            links: Vec::new(),
            graph_meta: SkillGraphMeta::new(),
            content: None,
            attached_to: None,
            security_info: Some(security),
            mcp_server_id: None,
            storage_tier: StorageTier::L0Permanent,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_used_at: None,
        }
    }

    pub fn is_mcp_tool(&self) -> bool {
        self.node_type == SkillNodeType::MCPTool || self.mcp_server_id.is_some()
    }

    pub fn is_bootstrap(&self) -> bool {
        matches!(self.node_type, SkillNodeType::Bootstrap)
            || self.security_info.as_ref().map_or(false, |s| {
                matches!(
                    s.source,
                    SkillSource::BootstrapLearn | SkillSource::BootstrapReduce
                )
            })
    }

    pub fn get_trust_level(&self) -> TrustLevel {
        self.security_info
            .as_ref()
            .map(|s| s.trust_level)
            .unwrap_or(TrustLevel::Medium)
    }

    pub fn can_execute(&self, required_trust: TrustLevel) -> bool {
        self.get_trust_level().can_execute(required_trust)
    }

    pub fn add_prerequisite(&mut self, target_iri: &str, description: &str) {
        self.links.push(SkillLink {
            link_type: SkillLinkType::Prerequisite,
            target_iri: target_iri.to_string(),
            strength: LinkStrength::Required,
            description: description.to_string(),
        });
    }

    pub fn add_related(&mut self, target_iri: &str, description: &str) {
        self.links.push(SkillLink {
            link_type: SkillLinkType::Related,
            target_iri: target_iri.to_string(),
            strength: LinkStrength::Recommended,
            description: description.to_string(),
        });
    }

    pub fn add_alternative(&mut self, target_iri: &str, description: &str) {
        self.links.push(SkillLink {
            link_type: SkillLinkType::Alternative,
            target_iri: target_iri.to_string(),
            strength: LinkStrength::Optional,
            description: description.to_string(),
        });
    }

    pub fn add_link(&mut self, link: SkillLink) {
        self.links.push(link);
    }

    pub fn get_prerequisites(&self) -> Vec<&SkillLink> {
        self.links
            .iter()
            .filter(|l| l.link_type == SkillLinkType::Prerequisite)
            .collect()
    }

    pub fn get_alternatives(&self) -> Vec<&SkillLink> {
        self.links
            .iter()
            .filter(|l| l.link_type == SkillLinkType::Alternative)
            .collect()
    }

    pub fn get_related(&self) -> Vec<&SkillLink> {
        self.links
            .iter()
            .filter(|l| l.link_type == SkillLinkType::Related)
            .collect()
    }

    /// Build an embedding text combining semantic + structural fields (P0-3).
    pub fn to_embedding_text(&self) -> String {
        format!(
            "{}: {} | What: {} | Why: {} | How: {} | Tags: {}",
            self.name,
            self.description,
            self.w2h.what,
            self.w2h.why,
            self.w2h.how.approach,
            self.tags.join(", ")
        )
    }

    /// Build a structural embedding emphasising hierarchy and links (P1-2).
    pub fn to_structural_embedding_text(&self) -> String {
        let hierarchy = self
            .attached_to
            .as_ref()
            .map(|p| format!(" under {}", p))
            .unwrap_or_default();

        let prereqs: Vec<&str> = self
            .links
            .iter()
            .filter(|l| l.link_type == SkillLinkType::Prerequisite)
            .map(|l| l.target_iri.as_str())
            .collect();
        let compositors: Vec<&str> = self
            .links
            .iter()
            .filter(|l| l.link_type == SkillLinkType::Composition)
            .map(|l| l.target_iri.as_str())
            .collect();

        format!(
            "{} type:{:?} maturity:{}{} prerequisites:{} compositors:{}",
            self.name,
            self.node_type,
            self.maturity,
            hierarchy,
            prereqs.join(","),
            compositors.join(","),
        )
    }

    /// Generate SPARQL INSERT DATA for persisting this node to Oxigraph (P0-1).
    pub fn to_sparql_insert(&self, graph: &str) -> String {
        use std::fmt::Write;
        let mut triples = String::new();

        let _ = write!(
            triples,
            "<{}> a skill:CognitiveSkill ;\n  skill:name '{}' ;\n  skill:description '{}' .\n",
            self.skill_iri,
            self.name.replace('\'', "\\'"),
            self.description.replace('\'', "\\'"),
        );

        let _ = write!(
            triples,
            "<{}> skill:what '{}' ;\n  skill:why '{}' .\n",
            self.skill_iri,
            self.w2h.what.replace('\'', "\\'"),
            self.w2h.why.replace('\'', "\\'"),
        );

        for link in &self.links {
            let _ = writeln!(
                triples,
                "<{}> {} <{}> .",
                self.skill_iri,
                link.link_type.as_sparql_uri(),
                link.target_iri
            );
        }

        let _ = write!(
            triples,
            "<{}> skill:usageCount {} ;\n  skill:successRate \"{}\"^^<http://www.w3.org/2001/XMLSchema#float> ;\n  skill:maturity '{}' .\n",
            self.skill_iri,
            self.graph_meta.usage_count,
            self.graph_meta.success_rate,
            self.maturity,
        );

        format!("PREFIX skill: <https://agent-os.org/ontology/skill#>\nINSERT DATA {{ GRAPH <{}> {{ {} }} }}", graph, triples)
    }

    pub fn to_json_ld(&self) -> serde_json::Value {
        use serde_json::json;

        let mut link_json = Vec::new();
        for link in &self.links {
            link_json.push(json!({
                "@type": format!("skill:{:?}Link", link.link_type),
                "skill:target": link.target_iri,
                "skill:strength": format!("{:?}", link.strength),
                "skill:description": link.description
            }));
        }

        json!({
            "@context": {
                "skill": "https://agent-os.org/ontology/skill#",
                "schema": "https://schema.org/"
            },
            "@id": self.skill_iri,
            "@type": ["skill:Skill", format!("skill:{:?}", self.node_type)],
            "schema:name": self.name,
            "schema:description": self.description,
            "skill:version": self.version,
            "skill:maturity": self.maturity,
            "skill:tags": self.tags,
            "skill:5W2H": {
                "skill:what": self.w2h.what,
                "skill:why": self.w2h.why,
                "skill:who": {
                    "skill:roleName": self.w2h.who.role_name,
                    "skill:requiredRole": self.w2h.who.required_agent_role
                },
                "skill:when": {
                    "skill:applicablePhase": self.w2h.when.applicable_phases,
                    "skill:triggerCondition": self.w2h.when.trigger_condition
                },
                "skill:how": {
                    "skill:approach": self.w2h.how.approach
                },
                "skill:howMuch": {
                    "skill:avgTokenCost": self.w2h.how_much.avg_token_cost,
                    "skill:avgDuration": self.w2h.how_much.avg_duration_seconds
                }
            },
            "skill:links": link_json,
            "skill:graphMeta": {
                "skill:usageCount": self.graph_meta.usage_count,
                "skill:successRate": self.graph_meta.success_rate,
                "skill:avgTokenConsumption": self.graph_meta.avg_token_consumption
            }
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MOCNode {
    pub moc_iri: String,
    pub name: String,
    pub description: String,
    pub skill_count: u32,
    pub entry_points: Vec<String>,
    pub sub_categories: Vec<String>,
    /// When this MOC was created
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
}

impl MOCNode {
    pub fn new(moc_iri: &str, name: &str, description: &str) -> Self {
        Self {
            moc_iri: moc_iri.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            skill_count: 0,
            entry_points: Vec::new(),
            sub_categories: Vec::new(),
            created_at: Utc::now(),
        }
    }

    pub fn add_entry_point(&mut self, skill_iri: &str) {
        self.entry_points.push(skill_iri.to_string());
        self.skill_count += 1;
    }

    pub fn add_sub_category(&mut self, moc_iri: &str) {
        self.sub_categories.push(moc_iri.to_string());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeFragment {
    pub fragment_iri: String,
    pub name: String,
    pub description: String,
    pub attached_to: String,
    pub problem: String,
    pub recommendation: String,
    pub discovered_by: Option<String>,
    pub discovered_at: Option<DateTime<Utc>>,
    /// `failure_mitigation` for legacy fragments, or
    /// `ca_validated_task_knowledge` for accumulated task evidence.
    #[serde(default = "default_fragment_kind")]
    pub kind: String,
    /// Stable normalized family used to prevent cross-task evidence leakage.
    #[serde(default)]
    pub task_family: Option<String>,
    /// Original task that contributed the most recent evidence.
    #[serde(default)]
    pub source_task_iri: Option<String>,
    /// Structured CA verdict (`pass`, `conditional`, or `fail`).
    #[serde(default)]
    pub ca_verdict: Option<String>,
    /// Bounded procedure steps observed from successful execution actions.
    #[serde(default)]
    pub procedure: Vec<String>,
    /// Bounded direct checks observed from CA tool actions.
    #[serde(default)]
    pub successful_checks: Vec<String>,
    /// Failed checks/findings define when this knowledge must not be reused.
    #[serde(default)]
    pub counterexamples: Vec<String>,
    #[serde(default)]
    pub evidence_count: u32,
    #[serde(default)]
    pub success_count: u32,
    #[serde(default)]
    pub failure_count: u32,
    #[serde(default)]
    pub last_verified_at: Option<DateTime<Utc>>,
}

fn default_fragment_kind() -> String {
    "failure_mitigation".to_string()
}

impl KnowledgeFragment {
    pub fn new(fragment_iri: &str, attached_to: &str, problem: &str, recommendation: &str) -> Self {
        Self {
            fragment_iri: fragment_iri.to_string(),
            name: format!("Fragment: {}", problem),
            description: problem.to_string(),
            attached_to: attached_to.to_string(),
            problem: problem.to_string(),
            recommendation: recommendation.to_string(),
            discovered_by: None,
            discovered_at: Some(Utc::now()),
            kind: default_fragment_kind(),
            task_family: None,
            source_task_iri: None,
            ca_verdict: None,
            procedure: Vec::new(),
            successful_checks: Vec::new(),
            counterexamples: Vec::new(),
            evidence_count: 0,
            success_count: 0,
            failure_count: 0,
            last_verified_at: None,
        }
    }

    pub fn with_discoverer(mut self, discoverer: &str) -> Self {
        self.discovered_by = Some(discoverer.to_string());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    Resource,
    Dependency,
    Permission,
    Semantic,
    Temporal,
    Version,
}

impl Default for ConflictType {
    fn default() -> Self {
        Self::Resource
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

impl Default for ConflictSeverity {
    fn default() -> Self {
        Self::Medium
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillConflict {
    pub conflict_id: String,
    pub conflict_type: ConflictType,
    pub severity: ConflictSeverity,
    pub skill_iris: Vec<String>,
    pub description: String,
    pub detected_at: DateTime<Utc>,
    pub resolution: Option<ConflictResolution>,
    pub auto_resolvable: bool,
}

impl SkillConflict {
    pub fn new(conflict_type: ConflictType, skill_iris: Vec<String>, description: &str) -> Self {
        Self {
            conflict_id: format!("conflict:{}", uuid::Uuid::new_v4()),
            conflict_type,
            severity: ConflictSeverity::default(),
            skill_iris,
            description: description.to_string(),
            detected_at: Utc::now(),
            resolution: None,
            auto_resolvable: false,
        }
    }

    pub fn with_severity(mut self, severity: ConflictSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_resolution(mut self, resolution: ConflictResolution) -> Self {
        self.resolution = Some(resolution);
        self
    }

    pub fn mark_auto_resolvable(mut self) -> Self {
        self.auto_resolvable = true;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolution {
    pub strategy: ResolutionStrategy,
    pub description: String,
    pub resolved_by: Option<String>,
    pub resolved_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    PreferNewer,
    PreferHigherTrust,
    PreferSystem,
    Merge,
    KeepBoth,
    RemoveConflict,
    RequireManual,
}

impl ConflictResolution {
    pub fn new(strategy: ResolutionStrategy, description: &str) -> Self {
        Self {
            strategy,
            description: description.to_string(),
            resolved_by: None,
            resolved_at: Utc::now(),
        }
    }

    pub fn with_resolver(mut self, resolver: &str) -> Self {
        self.resolved_by = Some(resolver.to_string());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapSource {
    pub source_type: BootstrapSourceType,
    pub task_iri: Option<String>,
    pub agent_id: String,
    pub timestamp: DateTime<Utc>,
    pub quality_score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapSourceType {
    TaskExecution,
    ErrorRecovery,
    UserFeedback,
    CodeReview,
    KnowledgeExtraction,
}

impl BootstrapSource {
    pub fn new(source_type: BootstrapSourceType, agent_id: &str) -> Self {
        Self {
            source_type,
            task_iri: None,
            agent_id: agent_id.to_string(),
            timestamp: Utc::now(),
            quality_score: 0.0,
        }
    }

    pub fn with_task(mut self, task_iri: &str) -> Self {
        self.task_iri = Some(task_iri.to_string());
        self
    }

    pub fn with_quality_score(mut self, score: f32) -> Self {
        self.quality_score = score;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillBootstrapMeta {
    pub learn_count: u32,
    pub reduce_count: u32,
    pub last_bootstrap_at: Option<DateTime<Utc>>,
    pub bootstrap_sources: Vec<BootstrapSource>,
    pub parent_skill_iri: Option<String>,
    pub derived_skills: Vec<String>,
}

impl SkillBootstrapMeta {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_learn(&mut self, source: BootstrapSource) {
        self.learn_count += 1;
        self.last_bootstrap_at = Some(Utc::now());
        self.bootstrap_sources.push(source);
    }

    pub fn record_reduce(&mut self, derived_iri: &str) {
        self.reduce_count += 1;
        self.last_bootstrap_at = Some(Utc::now());
        self.derived_skills.push(derived_iri.to_string());
    }

    pub fn set_parent(&mut self, parent_iri: &str) {
        self.parent_skill_iri = Some(parent_iri.to_string());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPSkillMapping {
    pub skill_iri: String,
    pub mcp_server_id: String,
    pub mcp_tool_name: String,
    pub parameter_mapping: std::collections::HashMap<String, String>,
    pub result_mapping: std::collections::HashMap<String, String>,
    pub enabled: bool,
}

impl MCPSkillMapping {
    pub fn new(skill_iri: &str, server_id: &str, tool_name: &str) -> Self {
        Self {
            skill_iri: skill_iri.to_string(),
            mcp_server_id: server_id.to_string(),
            mcp_tool_name: tool_name.to_string(),
            parameter_mapping: std::collections::HashMap::new(),
            result_mapping: std::collections::HashMap::new(),
            enabled: true,
        }
    }

    pub fn with_param_mapping(mut self, skill_param: &str, mcp_param: &str) -> Self {
        self.parameter_mapping
            .insert(skill_param.to_string(), mcp_param.to_string());
        self
    }

    pub fn with_result_mapping(mut self, skill_result: &str, mcp_result: &str) -> Self {
        self.result_mapping
            .insert(skill_result.to_string(), mcp_result.to_string());
        self
    }
}

// ─── P1-1: Hyperedge Composition ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperedge {
    pub hyperedge_id: String,
    pub name: String,
    pub description: String,
    pub components: Vec<String>,
    pub target_composite: Option<String>,
    pub composition_type: CompositionType,
    pub weight: f32,
    pub metadata: HashMap<String, String>,
    /// When this hyperedge was created
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionType {
    Conjunction,
    Disjunction,
    Exactly(u32),
    AtLeast(u32),
    Pipeline,
}

impl Hyperedge {
    pub fn new(hyperedge_id: &str, name: &str, description: &str, components: Vec<String>) -> Self {
        Self {
            hyperedge_id: hyperedge_id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            components,
            target_composite: None,
            composition_type: CompositionType::Conjunction,
            weight: 1.0,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    pub fn with_composition_type(mut self, ct: CompositionType) -> Self {
        self.composition_type = ct;
        self
    }

    pub fn with_target(mut self, target: &str) -> Self {
        self.target_composite = Some(target.to_string());
        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }
}

// ─── P1-3: Causal Failure Analysis ────────────────────────────────────────

/// P1-3 causal event type, superseded by `causal::types::CausalObservation`.
/// Use `CausalObservation` for new code.
#[deprecated(
    since = "0.1.2",
    note = "Use crate::causal::types::CausalObservation instead"
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub skill_iri: String,
    pub error_class: String,
    pub error_signature: String,
    pub context: HashMap<String, String>,
    pub propagation_from: Option<String>,
}

/// Superseded by `causal::types::CausalInference`.
#[deprecated(
    since = "0.1.2",
    note = "Use crate::causal::types::CausalInference instead"
)]
#[allow(deprecated)]
#[derive(Debug, Clone)]
pub struct CausalChain {
    pub root_cause: CausalEvent,
    pub propagation_path: Vec<CausalEvent>,
    pub confidence: f32,
}

/// Superseded by `CausalModelStore` + `CausalEngine`.
#[deprecated(
    since = "0.1.2",
    note = "Use crate::causal::store::CausalModelStore instead"
)]
#[allow(deprecated)]
#[derive(Debug, Clone)]
pub struct SkillCausalModel {
    pub error_profiles: HashMap<String, HashMap<String, u32>>,
    pub propagation_edges: HashMap<String, HashMap<String, u32>>,
    pub root_cause_probability: HashMap<String, f32>,
}

#[allow(deprecated)]
impl Default for SkillCausalModel {
    fn default() -> Self {
        Self {
            error_profiles: HashMap::new(),
            propagation_edges: HashMap::new(),
            root_cause_probability: HashMap::new(),
        }
    }
}

#[allow(deprecated)]
impl SkillCausalModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_failure(&mut self, skill_iri: &str, error_sig: &str) {
        let profile = self
            .error_profiles
            .entry(skill_iri.to_string())
            .or_default();
        *profile.entry(error_sig.to_string()).or_insert(0) += 1;
    }

    pub fn record_propagation(&mut self, from: &str, to: &str) {
        let edges = self.propagation_edges.entry(from.to_string()).or_default();
        *edges.entry(to.to_string()).or_insert(0) += 1;
    }

    pub fn propagation_probability(&self, from: &str, to: &str) -> f32 {
        self.propagation_edges
            .get(from)
            .and_then(|edges| edges.get(to))
            .copied()
            .unwrap_or(0) as f32
    }
}

// ─── P1-4: Temporal Versioning ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRecord {
    pub snapshot_id: String,
    pub label: String,
    pub timestamp: DateTime<Utc>,
    pub skill_count: usize,
}

// ─── P2-1: Hybrid Search ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FusedHit {
    pub iri: String,
    pub score: f32,
}

// ─── P2-2: Graph Algorithms ────────────────────────────────────────────────

/// A fused node used by graph algorithm results when the caller wants
/// an IRI + computed score without coupling to petgraph types.
#[derive(Debug, Clone)]
pub struct ScoredNode {
    pub iri: String,
    pub score: f64,
}

// ─── P2-3: Formal Verification ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphInvariant {
    Acyclicity,
    LinkTargetExists,
    CompositeReachability,
    NoDeprecatedPrerequisites,
    Valid5W2H,
    ValidSecurityLevels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub severity: ViolationSeverity,
    pub description: String,
    pub affected_iris: Vec<String>,
    pub suggestion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub invariant: GraphInvariant,
    pub passed: bool,
    pub violations: Vec<Violation>,
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_graph_node_creation() {
        let node = SkillGraphNode::new(
            "iri://skills/rust-jwt-auth",
            "Rust JWT Authentication",
            "Implement JWT authentication in Rust",
        );

        assert_eq!(node.skill_iri, "iri://skills/rust-jwt-auth");
        assert_eq!(node.name, "Rust JWT Authentication");
        assert_eq!(node.node_type, SkillNodeType::Atomic);
        assert!(node.links.is_empty());
    }

    #[test]
    fn test_skill_5w2h() {
        let w2h = Skill5W2H::new("JWT Auth", "Secure authentication")
            .with_phase("Do")
            .with_agent_role("DA")
            .with_token_cost(2500);

        assert_eq!(w2h.what, "JWT Auth");
        assert_eq!(w2h.why, "Secure authentication");
        assert!(w2h.when.applicable_phases.contains(&"Do".to_string()));
        assert_eq!(w2h.who.required_agent_role, Some("DA".to_string()));
        assert_eq!(w2h.how_much.avg_token_cost, 2500);
    }

    #[test]
    fn test_skill_links() {
        let mut node = SkillGraphNode::new(
            "iri://skills/rust-jwt-auth",
            "JWT Auth",
            "JWT authentication",
        );

        node.add_prerequisite("iri://skills/rust-basics", "Requires Rust basics");
        node.add_related("iri://skills/rust-middleware", "Often used with middleware");
        node.add_alternative("iri://skills/session-auth", "Alternative approach");

        assert_eq!(node.get_prerequisites().len(), 1);
        assert_eq!(node.get_related().len(), 1);
        assert_eq!(node.get_alternatives().len(), 1);
    }

    #[test]
    fn test_skill_graph_meta() {
        let mut meta = SkillGraphMeta::new();

        meta.record_usage(true);
        assert_eq!(meta.usage_count, 1);
        assert_eq!(meta.success_rate, 1.0);

        meta.record_usage(false);
        assert_eq!(meta.usage_count, 2);
        assert_eq!(meta.success_rate, 0.5);
    }

    #[test]
    fn test_moc_node() {
        let mut moc = MOCNode::new(
            "iri://moc/authentication",
            "Authentication Domain",
            "All authentication related skills",
        );

        moc.add_entry_point("iri://skills/rust-jwt-auth");
        moc.add_entry_point("iri://skills/oauth2-flow");
        moc.add_sub_category("iri://moc/jwt-implementation");

        assert_eq!(moc.skill_count, 2);
        assert_eq!(moc.entry_points.len(), 2);
        assert_eq!(moc.sub_categories.len(), 1);
    }

    #[test]
    fn test_knowledge_fragment() {
        let fragment = KnowledgeFragment::new(
            "iri://fragment/jwt-key-rotation",
            "iri://skills/rust-jwt-auth",
            "Key rotation causes token invalidation",
            "Use JWKS for graceful key rotation",
        )
        .with_discoverer("agent:ca/inst-001");

        assert_eq!(fragment.attached_to, "iri://skills/rust-jwt-auth");
        assert!(fragment.discovered_by.is_some());
    }

    #[test]
    fn test_to_json_ld() {
        let node = SkillGraphNode::new("iri://skills/test", "Test Skill", "A test skill")
            .with_tag("testing");

        let json = node.to_json_ld();

        assert_eq!(
            json.get("@id").and_then(|v| v.as_str()),
            Some("iri://skills/test")
        );
        assert!(json.get("skill:5W2H").is_some());
        assert!(json
            .get("skill:tags")
            .and_then(|v| v.as_array())
            .unwrap()
            .contains(&serde_json::json!("testing")));
    }

    #[test]
    fn test_trust_level() {
        assert!(TrustLevel::System.can_execute(TrustLevel::High));
        assert!(TrustLevel::High.can_execute(TrustLevel::Medium));
        assert!(!TrustLevel::Low.can_execute(TrustLevel::High));

        assert_eq!(TrustLevel::from_success_rate(0.96), TrustLevel::High);
        assert_eq!(TrustLevel::from_success_rate(0.85), TrustLevel::Medium);
        assert_eq!(TrustLevel::from_success_rate(0.65), TrustLevel::Low);
        assert_eq!(TrustLevel::from_success_rate(0.50), TrustLevel::Untrusted);
    }

    #[test]
    fn test_skill_security_info() {
        let mut security = SkillSecurityInfo::new(SkillSource::BootstrapLearn)
            .with_trust_level(TrustLevel::Medium)
            .with_permission(SkillPermission {
                permission_id: "perm-001".to_string(),
                resource_pattern: "/files/*".to_string(),
                action: PermissionAction::Read,
                constraints: vec![],
            });

        security.update_risk_score();

        assert!(security.has_permission(PermissionAction::Read, "/files/test.txt"));
        assert!(!security.has_permission(PermissionAction::Write, "/files/test.txt"));
        assert!(security.risk_score > 0.0);
    }

    #[test]
    fn test_audit_entry() {
        let entry = AuditEntry::new(
            "skill_execute",
            "agent:da/inst-001",
            "iri://skills/test",
            AuditOutcome::Success,
        )
        .with_details("Execution completed successfully");

        assert_eq!(entry.action, "skill_execute");
        assert_eq!(entry.outcome, AuditOutcome::Success);
        assert!(entry.details.is_some());
    }

    #[test]
    fn test_skill_conflict() {
        let conflict = SkillConflict::new(
            ConflictType::Resource,
            vec!["iri://skills/a".to_string(), "iri://skills/b".to_string()],
            "Both skills access the same resource",
        )
        .with_severity(ConflictSeverity::High)
        .mark_auto_resolvable();

        assert_eq!(conflict.conflict_type, ConflictType::Resource);
        assert_eq!(conflict.severity, ConflictSeverity::High);
        assert!(conflict.auto_resolvable);
        assert!(conflict.resolution.is_none());
    }

    #[test]
    fn test_conflict_resolution() {
        let resolution = ConflictResolution::new(
            ResolutionStrategy::PreferHigherTrust,
            "Selected skill with higher trust level",
        )
        .with_resolver("agent:sa/inst-001");

        assert_eq!(resolution.strategy, ResolutionStrategy::PreferHigherTrust);
        assert!(resolution.resolved_by.is_some());
    }

    #[test]
    fn test_bootstrap_source() {
        let source = BootstrapSource::new(BootstrapSourceType::TaskExecution, "agent:da/inst-001")
            .with_task("iri://task/abc")
            .with_quality_score(0.85);

        assert_eq!(source.source_type, BootstrapSourceType::TaskExecution);
        assert!(source.task_iri.is_some());
        assert_eq!(source.quality_score, 0.85);
    }

    #[test]
    fn test_skill_bootstrap_meta() {
        let mut meta = SkillBootstrapMeta::new();
        let source = BootstrapSource::new(BootstrapSourceType::TaskExecution, "agent:da/inst-001");

        meta.record_learn(source);
        meta.record_reduce("iri://skills/derived");
        meta.set_parent("iri://skills/parent");

        assert_eq!(meta.learn_count, 1);
        assert_eq!(meta.reduce_count, 1);
        assert!(meta.last_bootstrap_at.is_some());
        assert!(meta.parent_skill_iri.is_some());
        assert!(meta
            .derived_skills
            .contains(&"iri://skills/derived".to_string()));
    }

    #[test]
    fn test_mcp_skill_mapping() {
        let mapping = MCPSkillMapping::new(
            "iri://skills/file-read",
            "mcp-server-filesystem",
            "read_file",
        )
        .with_param_mapping("file_path", "path")
        .with_result_mapping("content", "data");

        assert_eq!(mapping.skill_iri, "iri://skills/file-read");
        assert_eq!(mapping.mcp_server_id, "mcp-server-filesystem");
        assert!(mapping.enabled);
        assert!(mapping.parameter_mapping.contains_key("file_path"));
        assert!(mapping.result_mapping.contains_key("content"));
    }

    #[test]
    fn test_skill_node_with_security() {
        let security =
            SkillSecurityInfo::new(SkillSource::SystemBuiltin).with_trust_level(TrustLevel::System);

        let node = SkillGraphNode::new(
            "iri://skills/system-skill",
            "System Skill",
            "A system skill",
        )
        .with_security_info(security);

        assert!(node.can_execute(TrustLevel::High));
        assert_eq!(node.get_trust_level(), TrustLevel::System);
        assert!(!node.is_mcp_tool());
        assert!(!node.is_bootstrap());
    }

    #[test]
    fn test_skill_node_mcp() {
        let node = SkillGraphNode::new("iri://skills/mcp-skill", "MCP Skill", "An MCP tool skill")
            .with_mcp_server("mcp-server-001");

        assert!(node.is_mcp_tool());
        assert_eq!(node.node_type, SkillNodeType::MCPTool);
        assert!(node.mcp_server_id.is_some());
    }

    #[test]
    fn test_storage_tier() {
        let node = SkillGraphNode::new("iri://skills/test", "Test", "Test skill")
            .with_storage_tier(StorageTier::L2Blackboard);

        assert_eq!(node.storage_tier, StorageTier::L2Blackboard);
    }

    // ── P1-1: Hyperedge tests ──────────────────────────────────────────

    #[test]
    fn test_hyperedge_creation() {
        let he = Hyperedge::new(
            "hyperedge:auth",
            "Auth Pipeline",
            "Authentication pipeline",
            vec![
                "iri://skills/login".to_string(),
                "iri://skills/jwt".to_string(),
            ],
        )
        .with_composition_type(CompositionType::Conjunction)
        .with_weight(0.9);

        assert_eq!(he.hyperedge_id, "hyperedge:auth");
        assert_eq!(he.components.len(), 2);
        assert_eq!(he.composition_type, CompositionType::Conjunction);
        assert!((he.weight - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_hyperedge_pipeline() {
        let he = Hyperedge::new(
            "hyperedge:deploy",
            "Deploy Pipeline",
            "Ordered deployment steps",
            vec![
                "iri://skills/build".to_string(),
                "iri://skills/test".to_string(),
                "iri://skills/deploy".to_string(),
            ],
        )
        .with_composition_type(CompositionType::Pipeline)
        .with_target("iri://skills/deploy-pipeline");

        assert_eq!(he.composition_type, CompositionType::Pipeline);
        assert_eq!(
            he.target_composite,
            Some("iri://skills/deploy-pipeline".to_string())
        );
    }

    #[test]
    fn test_hyperedge_at_least() {
        let he = Hyperedge::new(
            "hyperedge:any-2",
            "Any 2 Skills",
            "Any 2 of 5 skills required",
            vec![
                "iri://skills/a".to_string(),
                "iri://skills/b".to_string(),
                "iri://skills/c".to_string(),
            ],
        )
        .with_composition_type(CompositionType::AtLeast(2));

        assert_eq!(he.composition_type, CompositionType::AtLeast(2));
    }

    // ── P1-3: Causal event tests ───────────────────────────────────────

    #[test]
    fn test_causal_model_record_failure() {
        let mut model = SkillCausalModel::new();
        model.record_failure("iri://skills/token-gen", "timeout_error");
        model.record_failure("iri://skills/token-gen", "timeout_error");

        let profile = model.error_profiles.get("iri://skills/token-gen").unwrap();
        assert_eq!(profile.get("timeout_error"), Some(&2));
    }

    #[test]
    fn test_causal_model_propagation() {
        let mut model = SkillCausalModel::new();
        model.record_propagation("iri://skills/db-conn", "iri://skills/query-exec");

        let prob = model.propagation_probability("iri://skills/db-conn", "iri://skills/query-exec");
        assert!((prob - 1.0).abs() < 1e-6);
    }

    // ── P1-4: Snapshot record tests ─────────────────────────────────────

    #[test]
    fn test_snapshot_record() {
        let record = SnapshotRecord {
            snapshot_id: "snapshot:test_20260629".to_string(),
            label: "test".to_string(),
            timestamp: Utc::now(),
            skill_count: 42,
        };
        assert_eq!(record.skill_count, 42);
    }

    // ── P2-1: FusedHit tests ───────────────────────────────────────────

    #[test]
    fn test_fused_hit_ordering() {
        let mut hits = vec![
            FusedHit {
                iri: "iri://skills/a".to_string(),
                score: 0.5,
            },
            FusedHit {
                iri: "iri://skills/b".to_string(),
                score: 0.8,
            },
            FusedHit {
                iri: "iri://skills/c".to_string(),
                score: 0.3,
            },
        ];
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        assert_eq!(hits[0].iri, "iri://skills/b");
    }

    // ── P2-3: Verification types tests ──────────────────────────────────

    #[test]
    fn test_graph_invariant_debug() {
        let inv = GraphInvariant::Acyclicity;
        assert_eq!(format!("{:?}", inv), "Acyclicity");
    }

    #[test]
    fn test_verification_result() {
        let result = VerificationResult {
            invariant: GraphInvariant::LinkTargetExists,
            passed: true,
            violations: vec![],
            duration_ms: 5,
        };
        assert!(result.passed);
        assert_eq!(result.duration_ms, 5);
    }

    #[test]
    fn test_violation_severity() {
        let v = Violation {
            severity: ViolationSeverity::Error,
            description: "Missing link target".to_string(),
            affected_iris: vec!["iri://skills/missing".to_string()],
            suggestion: "Add the missing skill or remove the link".to_string(),
        };
        assert!(matches!(v.severity, ViolationSeverity::Error));
    }

    #[test]
    fn test_skill_link_type_sparql_uri() {
        assert_eq!(
            SkillLinkType::Prerequisite.as_sparql_uri(),
            "skill:Prerequisite"
        );
        assert_eq!(
            SkillLinkType::Composition.as_sparql_uri(),
            "skill:Composition"
        );
        assert_eq!(SkillLinkType::Related.as_sparql_uri(), "skill:Related");
        assert_eq!(
            SkillLinkType::Alternative.as_sparql_uri(),
            "skill:Alternative"
        );
        assert_eq!(SkillLinkType::Extends.as_sparql_uri(), "skill:Extends");
        assert_eq!(
            SkillLinkType::Generalization.as_sparql_uri(),
            "skill:Generalization"
        );
    }

    #[test]
    fn test_to_embedding_text() {
        let skill = SkillGraphNode::new("iri://skills/test", "Test", "A test skill")
            .with_tag("rust")
            .with_tag("auth");
        let text = skill.to_embedding_text();
        assert!(text.contains("Test"));
        assert!(text.contains("rust"));
        assert!(text.contains("auth"));
    }

    #[test]
    fn test_to_structural_embedding_text() {
        let mut skill = SkillGraphNode::new("iri://skills/test", "Test", "A test skill");
        skill.add_prerequisite("iri://skills/rust-basics", "Need Rust basics");
        let text = skill.to_structural_embedding_text();
        assert!(text.contains("Test"));
        assert!(text.contains("rust-basics"));
    }

    #[test]
    fn test_from_skill_meta_converts_fields() {
        let meta = crate::tools::skill_registry::SkillMeta {
            skill_iri: "iri://skills/test_read".to_string(),
            name: "test_read".to_string(),
            description: "Test read operation".to_string(),
            version: "1.0.0".to_string(),
            category: "file".to_string(),
            security_level: "normal".to_string(),
            allowed_roles: vec!["PA".to_string(), "DA".to_string()],
            input_schema: serde_json::json!({}),
            output_schema: serde_json::json!({}),
            compiled_template: String::new(),
            signature: None,
            signature_algorithm: None,
            input_mapping: std::collections::HashMap::new(),
            output_mapping: std::collections::HashMap::new(),
            skill_types: vec!["iri://skill-types/FileOperation".to_string()],
            discovery_5w2h: None,
        };
        let node = SkillGraphNode::from_skill_meta(&meta);
        assert_eq!(node.skill_iri, "iri://skills/test_read");
        assert_eq!(node.name, "test_read");
        assert_eq!(node.description, "Test read operation");
        assert_eq!(node.version, "1.0.0");
        assert_eq!(node.node_type, SkillNodeType::Atomic);
        assert!(node.tags.contains(&"file".to_string()));
        assert!(node
            .tags
            .contains(&"iri://skill-types/FileOperation".to_string()));
        assert!(node.tags.contains(&"allowed-role:PA".to_string()));
        assert!(node.tags.contains(&"allowed-role:DA".to_string()));
        let sec = node.security_info.expect("security_info should be set");
        assert_eq!(sec.source, SkillSource::SystemBuiltin);
        assert_eq!(sec.trust_level, TrustLevel::System);
        assert!((sec.risk_score - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_skill_meta_security_level_mapping() {
        let check = |level: &str, expected_risk: f32| {
            let meta = crate::tools::skill_registry::SkillMeta {
                skill_iri: format!("iri://skills/{}", level),
                name: level.to_string(),
                description: String::new(),
                version: "1.0.0".to_string(),
                category: String::new(),
                security_level: level.to_string(),
                allowed_roles: vec![],
                input_schema: serde_json::json!({}),
                output_schema: serde_json::json!({}),
                compiled_template: String::new(),
                signature: None,
                signature_algorithm: None,
                input_mapping: std::collections::HashMap::new(),
                output_mapping: std::collections::HashMap::new(),
                skill_types: vec![],
                discovery_5w2h: None,
            };
            let node = SkillGraphNode::from_skill_meta(&meta);
            let security = node.security_info.unwrap();
            assert_eq!(security.trust_level, TrustLevel::System);
            assert!((security.risk_score - expected_risk).abs() < f32::EPSILON);
        };
        check("critical", 0.9);
        check("high", 0.7);
        check("normal", 0.3);
        check("low", 0.1);
        check("unknown", 0.5);
    }
}
