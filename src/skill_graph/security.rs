use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::types::*;
use crate::CoreError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureInfo {
    pub algorithm: String,
    pub public_key: String,
    pub signature: String,
    pub signed_at: DateTime<Utc>,
    pub signer_id: Option<String>,
    pub certificate_chain: Vec<String>,
}

impl SignatureInfo {
    pub fn new(algorithm: &str, public_key: &str, signature: &str) -> Self {
        Self {
            algorithm: algorithm.to_string(),
            public_key: public_key.to_string(),
            signature: signature.to_string(),
            signed_at: Utc::now(),
            signer_id: None,
            certificate_chain: Vec::new(),
        }
    }

    pub fn with_signer(mut self, signer_id: &str) -> Self {
        self.signer_id = Some(signer_id.to_string());
        self
    }

    pub fn with_certificate(mut self, cert: &str) -> Self {
        self.certificate_chain.push(cert.to_string());
        self
    }

    pub fn verify(&self, content: &str) -> Result<bool, CoreError> {
        debug!(
            "Verifying signature: algorithm={}, signer={:?}",
            self.algorithm, self.signer_id
        );

        // Only ED25519 is supported; reject unknown/forged algorithm claims.
        if self.algorithm.to_lowercase() != "ed25519" {
            return Err(CoreError::ValidationFailed {
                message: format!("Unsupported signature algorithm: {}", self.algorithm),
            });
        }

        // Empty signature -> not signed.
        if self.signature.is_empty() {
            return Ok(false);
        }

        // Delegate to the canonical ring ED25519 verifier used by SyscallGate
        // (core::validation::SignatureVerifier), so skill signatures share the
        // exact same cryptographic verification path instead of a stub.
        use base64::Engine;
        let public_key_bytes = base64::engine::general_purpose::STANDARD
            .decode(&self.public_key)
            .map_err(|e| CoreError::ValidationFailed {
                message: format!("Invalid public key encoding: {}", e),
            })?;

        let verifier =
            crate::core::validation::SignatureVerifier::new().with_public_key(public_key_bytes);

        verifier.verify(content, &self.signature)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    pub policy_id: String,
    pub name: String,
    pub description: String,
    pub min_trust_level: TrustLevel,
    pub allowed_sources: Vec<SkillSource>,
    pub required_permissions: Vec<String>,
    pub max_risk_score: f32,
    pub require_signature: bool,
    pub audit_enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl SecurityPolicy {
    pub fn new(policy_id: &str, name: &str) -> Self {
        Self {
            policy_id: policy_id.to_string(),
            name: name.to_string(),
            description: String::new(),
            min_trust_level: TrustLevel::Low,
            allowed_sources: vec![
                SkillSource::SystemBuiltin,
                SkillSource::UserDefined,
                SkillSource::BootstrapLearn,
                SkillSource::BootstrapReduce,
            ],
            required_permissions: Vec::new(),
            max_risk_score: 0.5,
            require_signature: false,
            audit_enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn with_min_trust_level(mut self, level: TrustLevel) -> Self {
        self.min_trust_level = level;
        self
    }

    pub fn with_allowed_sources(mut self, sources: Vec<SkillSource>) -> Self {
        self.allowed_sources = sources;
        self
    }

    pub fn with_max_risk_score(mut self, score: f32) -> Self {
        self.max_risk_score = score;
        self
    }

    pub fn with_require_signature(mut self, require: bool) -> Self {
        self.require_signature = require;
        self
    }

    pub fn check_skill(&self, skill: &SkillGraphNode) -> SecurityDecision {
        let mut violations = Vec::new();

        if let Some(ref security_info) = skill.security_info {
            if (security_info.trust_level as u8) < self.min_trust_level as u8 {
                violations.push(format!(
                    "Insufficient trust level: {:?} < {:?}",
                    security_info.trust_level, self.min_trust_level
                ));
            }

            if !self.allowed_sources.contains(&security_info.source) {
                violations.push(format!("Source not allowed: {:?}", security_info.source));
            }

            if security_info.risk_score > self.max_risk_score {
                violations.push(format!(
                    "Risk score too high: {:.2} > {:.2}",
                    security_info.risk_score, self.max_risk_score
                ));
            }

            if self.require_signature && security_info.signature.is_none() {
                violations.push("Required signature missing".to_string());
            }
        } else if self.min_trust_level != TrustLevel::Untrusted {
            violations.push("Security info missing".to_string());
        }

        if violations.is_empty() {
            SecurityDecision::Allowed
        } else {
            SecurityDecision::Denied {
                reasons: violations,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum SecurityDecision {
    Allowed,
    Denied { reasons: Vec<String> },
    RequiresApproval { approver: String, reason: String },
}

impl SecurityDecision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, SecurityDecision::Allowed)
    }
}

#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub agent_id: String,
    pub agent_role: String,
    pub task_iri: Option<String>,
    pub requested_permissions: Vec<SkillPermission>,
    pub timestamp: DateTime<Utc>,
}

impl SecurityContext {
    pub fn new(agent_id: &str, agent_role: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            agent_role: agent_role.to_string(),
            task_iri: None,
            requested_permissions: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_task(mut self, task_iri: &str) -> Self {
        self.task_iri = Some(task_iri.to_string());
        self
    }

    pub fn with_permission(mut self, permission: SkillPermission) -> Self {
        self.requested_permissions.push(permission);
        self
    }
}

pub struct SecurityEngine {
    graph_store: Arc<SkillGraphStore>,
    policies: RwLock<HashMap<String, SecurityPolicy>>,
    approval_queue: RwLock<Vec<(String, SecurityContext, String)>>,
    audit_log: RwLock<Vec<AuditEntry>>,
    whitelisted_skills: RwLock<HashSet<String>>,
}

impl SecurityEngine {
    pub fn new(graph_store: Arc<SkillGraphStore>) -> Self {
        let mut policies = HashMap::new();
        Self::init_default_policies(&mut policies);

        Self {
            graph_store,
            policies: RwLock::new(policies),
            approval_queue: RwLock::new(Vec::new()),
            audit_log: RwLock::new(Vec::new()),
            whitelisted_skills: RwLock::new(HashSet::new()),
        }
    }

    /// Construct an execution engine with an explicit, immutable-at-startup
    /// allowlist of trusted built-in capabilities. User-defined skills are
    /// never included implicitly.
    pub fn with_whitelisted_skills(
        graph_store: Arc<SkillGraphStore>,
        whitelisted_skills: HashSet<String>,
    ) -> Self {
        let mut engine = Self::new(graph_store);
        engine.whitelisted_skills = RwLock::new(whitelisted_skills);
        engine
    }

    fn init_default_policies(policies: &mut HashMap<String, SecurityPolicy>) {
        policies.insert(
            "default".to_string(),
            SecurityPolicy::new("default", "Default Policy")
                .with_min_trust_level(TrustLevel::Low)
                .with_max_risk_score(0.5),
        );

        policies.insert(
            "strict".to_string(),
            SecurityPolicy::new("strict", "Strict Policy")
                .with_min_trust_level(TrustLevel::High)
                .with_max_risk_score(0.2)
                .with_require_signature(true),
        );

        policies.insert(
            "system".to_string(),
            SecurityPolicy::new("system", "System Policy")
                .with_min_trust_level(TrustLevel::System)
                .with_allowed_sources(vec![SkillSource::SystemBuiltin])
                .with_max_risk_score(0.0),
        );
    }

    pub async fn check_execution(
        &self,
        skill_iri: &str,
        context: &SecurityContext,
    ) -> Result<SecurityDecision, CoreError> {
        info!(
            "Checking skill execution permission: skill={}, agent={}",
            skill_iri, context.agent_id
        );

        let skill =
            self.graph_store
                .get_skill(skill_iri)
                .ok_or_else(|| CoreError::SkillNotFound {
                    iri: format!("Skill not found: {}", skill_iri),
                })?;

        let whitelisted = self.whitelisted_skills.read().await;
        if whitelisted.contains(skill_iri) {
            self.add_audit_entry(
                skill_iri,
                &context.agent_id,
                "execute_whitelisted",
                AuditOutcome::Success,
            )
            .await;
            return Ok(SecurityDecision::Allowed);
        }

        let policies = self.policies.read().await;
        let policy = policies
            .get("default")
            .ok_or_else(|| CoreError::Internal {
                message: "Default policy not found".to_string(),
            })?
            .clone();
        drop(policies);

        let decision = policy.check_skill(&skill);

        match &decision {
            SecurityDecision::Allowed => {
                self.add_audit_entry(
                    skill_iri,
                    &context.agent_id,
                    "execute_allowed",
                    AuditOutcome::Success,
                )
                .await;
            }
            SecurityDecision::Denied { reasons } => {
                self.add_audit_entry(
                    skill_iri,
                    &context.agent_id,
                    "execute_denied",
                    AuditOutcome::Denied,
                )
                .await;
                warn!("Skill execution denied: {} - {:?}", skill_iri, reasons);
            }
            SecurityDecision::RequiresApproval { .. } => {
                self.add_audit_entry(
                    skill_iri,
                    &context.agent_id,
                    "execute_pending_approval",
                    AuditOutcome::Warning,
                )
                .await;
            }
        }

        Ok(decision)
    }

    pub async fn check_permission(
        &self,
        skill_iri: &str,
        context: &SecurityContext,
        action: PermissionAction,
        resource: &str,
    ) -> Result<bool, CoreError> {
        let skill =
            self.graph_store
                .get_skill(skill_iri)
                .ok_or_else(|| CoreError::SkillNotFound {
                    iri: format!("Skill not found: {}", skill_iri),
                })?;

        if let Some(ref security_info) = skill.security_info {
            let has_permission = security_info.has_permission(action, resource);

            self.add_audit_entry(
                skill_iri,
                &context.agent_id,
                &format!("permission_check_{:?}", action),
                if has_permission {
                    AuditOutcome::Success
                } else {
                    AuditOutcome::Denied
                },
            )
            .await;

            return Ok(has_permission);
        }

        Ok(false)
    }

    pub async fn add_policy(&self, policy: SecurityPolicy) -> Result<(), CoreError> {
        let policy_id = policy.policy_id.clone();
        info!("Adding security policy: {} ({})", policy.name, policy_id);

        let mut policies = self.policies.write().await;
        policies.insert(policy_id, policy);
        Ok(())
    }

    pub async fn get_policy(&self, policy_id: &str) -> Option<SecurityPolicy> {
        let policies = self.policies.read().await;
        policies.get(policy_id).cloned()
    }

    pub async fn remove_policy(&self, policy_id: &str) -> bool {
        let mut policies = self.policies.write().await;
        policies.remove(policy_id).is_some()
    }

    pub async fn whitelist_skill(&self, skill_iri: &str) {
        info!("Whitelisting skill: {}", skill_iri);
        let mut whitelist = self.whitelisted_skills.write().await;
        whitelist.insert(skill_iri.to_string());
    }

    pub async fn remove_from_whitelist(&self, skill_iri: &str) -> bool {
        let mut whitelist = self.whitelisted_skills.write().await;
        whitelist.remove(skill_iri)
    }

    pub async fn is_whitelisted(&self, skill_iri: &str) -> bool {
        let whitelist = self.whitelisted_skills.read().await;
        whitelist.contains(skill_iri)
    }

    async fn add_audit_entry(
        &self,
        skill_iri: &str,
        agent_id: &str,
        action: &str,
        outcome: AuditOutcome,
    ) {
        let entry = AuditEntry::new(action, agent_id, skill_iri, outcome);
        let mut log = self.audit_log.write().await;
        log.push(entry);

        if log.len() > 10000 {
            log.drain(0..1000);
        }
    }

    pub async fn get_audit_log(
        &self,
        skill_iri: Option<&str>,
        agent_id: Option<&str>,
        limit: usize,
    ) -> Vec<AuditEntry> {
        let log = self.audit_log.read().await;

        let filtered: Vec<AuditEntry> = log
            .iter()
            .filter(|entry| {
                if let Some(iri) = skill_iri {
                    if entry.resource != iri {
                        return false;
                    }
                }
                if let Some(id) = agent_id {
                    if entry.agent_id != id {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .take(limit)
            .collect();

        filtered
    }

    pub async fn request_approval(&self, skill_iri: &str, context: SecurityContext, reason: &str) {
        info!(
            "Requesting approval: skill={}, agent={}",
            skill_iri, context.agent_id
        );

        let mut queue = self.approval_queue.write().await;
        queue.push((skill_iri.to_string(), context, reason.to_string()));
    }

    pub async fn get_pending_approvals(&self) -> Vec<(String, SecurityContext, String)> {
        let queue = self.approval_queue.read().await;
        queue.clone()
    }

    pub async fn approve_request(&self, skill_iri: &str) -> bool {
        let mut queue = self.approval_queue.write().await;
        let initial_len = queue.len();
        queue.retain(|(iri, _, _)| iri != skill_iri);

        if queue.len() < initial_len {
            self.whitelist_skill(skill_iri).await;
            true
        } else {
            false
        }
    }

    pub async fn reject_request(&self, skill_iri: &str) -> bool {
        let mut queue = self.approval_queue.write().await;
        let initial_len = queue.len();
        queue.retain(|(iri, _, _)| iri != skill_iri);
        queue.len() < initial_len
    }

    pub async fn calculate_risk_score(&self, skill_iri: &str) -> Result<f32, CoreError> {
        let skill =
            self.graph_store
                .get_skill(skill_iri)
                .ok_or_else(|| CoreError::SkillNotFound {
                    iri: format!("Skill not found: {}", skill_iri),
                })?;

        let mut risk_score = 0.0f32;

        if let Some(ref security_info) = skill.security_info {
            risk_score = security_info.risk_score;
        }

        if skill.is_mcp_tool() {
            risk_score += 0.1;
        }

        if skill.is_bootstrap() {
            risk_score += 0.05;
        }

        if skill.graph_meta.success_rate < 0.7 {
            risk_score += 0.1;
        }

        if skill.graph_meta.known_failure_modes.len() > 3 {
            risk_score += 0.1;
        }

        Ok(risk_score.min(1.0))
    }

    pub async fn validate_signature(
        &self,
        skill_iri: &str,
        signature_info: &SignatureInfo,
    ) -> Result<bool, CoreError> {
        let skill =
            self.graph_store
                .get_skill(skill_iri)
                .ok_or_else(|| CoreError::SkillNotFound {
                    iri: skill_iri.to_string(),
                })?;

        let content = serde_json::to_string(&skill.to_json_ld()).map_err(|e| {
            CoreError::ValidationFailed {
                message: format!("Failed to serialize skill: {}", e),
            }
        })?;

        signature_info.verify(&content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_skill(iri: &str, trust_level: TrustLevel) -> SkillGraphNode {
        let security_info =
            SkillSecurityInfo::new(SkillSource::UserDefined).with_trust_level(trust_level);

        SkillGraphNode::new(iri, "Test Skill", "A test skill").with_security_info(security_info)
    }

    #[test]
    fn test_signature_info() {
        let sig = SignatureInfo::new("ed25519", "public_key_123", "signature_abc")
            .with_signer("agent:sa/001")
            .with_certificate("cert_1");

        assert_eq!(sig.algorithm, "ed25519");
        assert!(sig.signer_id.is_some());
        assert_eq!(sig.certificate_chain.len(), 1);
    }

    #[test]
    fn test_signature_verify_ed25519_real() {
        use base64::Engine;
        use ring::signature::Ed25519KeyPair;
        use ring::signature::KeyPair;

        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng).expect("keygen");
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).expect("parse");
        let public_key_b64 =
            base64::engine::general_purpose::STANDARD.encode(key_pair.public_key().as_ref());

        let content = "skill-graph-node-payload";
        let signature = key_pair.sign(content.as_bytes());
        let signature_b64 = base64::engine::general_purpose::STANDARD.encode(signature.as_ref());

        let sig = SignatureInfo::new("ed25519", &public_key_b64, &signature_b64);

        // Valid signature verifies true.
        assert!(sig.verify(content).expect("verify"));

        // Tampered content fails verification.
        assert!(!sig.verify("tampered-content").expect("verify"));

        // Empty signature is treated as unsigned.
        let unsigned = SignatureInfo::new("ed25519", &public_key_b64, "");
        assert!(!unsigned.verify(content).expect("verify"));
    }

    #[test]
    fn test_signature_verify_rejects_unknown_algorithm() {
        let sig = SignatureInfo::new("rsa", "cHVibGlj", "c2lnbmF0dXJl");
        let result = sig.verify("content");
        assert!(matches!(result, Err(CoreError::ValidationFailed { .. })));
    }

    #[test]
    fn test_security_policy() {
        let policy = SecurityPolicy::new("test-policy", "Test Policy")
            .with_min_trust_level(TrustLevel::High)
            .with_max_risk_score(0.3)
            .with_require_signature(true);

        assert_eq!(policy.min_trust_level, TrustLevel::High);
        assert!(policy.require_signature);
    }

    #[test]
    fn test_security_policy_check_skill() {
        let policy = SecurityPolicy::new("test", "Test").with_min_trust_level(TrustLevel::Medium);

        let allowed_skill = create_test_skill("iri://skills/allowed", TrustLevel::High);
        let denied_skill = create_test_skill("iri://skills/denied", TrustLevel::Low);

        let decision_allowed = policy.check_skill(&allowed_skill);
        assert!(decision_allowed.is_allowed());

        let decision_denied = policy.check_skill(&denied_skill);
        assert!(!decision_denied.is_allowed());
    }

    #[test]
    fn test_security_context() {
        let context = SecurityContext::new("agent:da/001", "DA")
            .with_task("iri://task/abc")
            .with_permission(SkillPermission {
                permission_id: "perm-1".to_string(),
                resource_pattern: "/files/*".to_string(),
                action: PermissionAction::Read,
                constraints: vec![],
            });

        assert_eq!(context.agent_id, "agent:da/001");
        assert!(context.task_iri.is_some());
        assert_eq!(context.requested_permissions.len(), 1);
    }

    #[test]
    fn test_security_decision() {
        let allowed = SecurityDecision::Allowed;
        assert!(allowed.is_allowed());

        let denied = SecurityDecision::Denied {
            reasons: vec!["Test reason".to_string()],
        };
        assert!(!denied.is_allowed());
    }

    #[tokio::test]
    async fn test_security_engine_whitelist() {
        let graph_store = Arc::new(SkillGraphStore::new());
        let engine = SecurityEngine::new(graph_store);

        let skill_iri = "iri://skills/test";
        assert!(!engine.is_whitelisted(skill_iri).await);

        engine.whitelist_skill(skill_iri).await;
        assert!(engine.is_whitelisted(skill_iri).await);

        let removed = engine.remove_from_whitelist(skill_iri).await;
        assert!(removed);
        assert!(!engine.is_whitelisted(skill_iri).await);
    }

    #[tokio::test]
    async fn test_security_engine_audit_log() {
        let graph_store = Arc::new(SkillGraphStore::new());
        let engine = SecurityEngine::new(graph_store);

        engine
            .add_audit_entry(
                "iri://skills/test",
                "agent:da/001",
                "execute",
                AuditOutcome::Success,
            )
            .await;

        let log = engine.get_audit_log(None, None, 10).await;
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].action, "execute");
    }

    #[tokio::test]
    async fn test_security_engine_policy() {
        let graph_store = Arc::new(SkillGraphStore::new());
        let engine = SecurityEngine::new(graph_store);

        let policy =
            SecurityPolicy::new("custom", "Custom Policy").with_min_trust_level(TrustLevel::System);

        engine.add_policy(policy.clone()).await.unwrap();

        let retrieved = engine.get_policy("custom").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Custom Policy");

        let removed = engine.remove_policy("custom").await;
        assert!(removed);

        let retrieved = engine.get_policy("custom").await;
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_high_risk_system_skill_is_not_silently_allowed() {
        let graph_store = Arc::new(SkillGraphStore::new());
        let skill = SkillGraphNode::new("iri://skills/critical", "Critical", "destructive")
            .with_security_info(
                SkillSecurityInfo::new(SkillSource::SystemBuiltin)
                    .with_trust_level(TrustLevel::System)
                    .with_risk_score(0.9),
            );
        graph_store.register_skill(skill).unwrap();
        let engine = SecurityEngine::new(graph_store);

        let decision = engine
            .check_execution(
                "iri://skills/critical",
                &SecurityContext::new("agent:test", "DA"),
            )
            .await
            .unwrap();
        assert!(matches!(decision, SecurityDecision::Denied { .. }));
    }

    #[tokio::test]
    async fn test_security_engine_approval() {
        let graph_store = Arc::new(SkillGraphStore::new());
        let engine = SecurityEngine::new(graph_store);

        let context = SecurityContext::new("agent:da/001", "DA");
        engine
            .request_approval("iri://skills/test", context, "Need approval")
            .await;

        let pending = engine.get_pending_approvals().await;
        assert_eq!(pending.len(), 1);

        let approved = engine.approve_request("iri://skills/test").await;
        assert!(approved);

        let pending = engine.get_pending_approvals().await;
        assert!(pending.is_empty());

        assert!(engine.is_whitelisted("iri://skills/test").await);
    }
}
