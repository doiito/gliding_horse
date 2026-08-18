use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::tools::sharing_audit::SharingAuditLog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShareType {
    FullAccess,
    Projection,
    ReferenceOnly,
    Summary,
}

impl ShareType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FullAccess => "full_access",
            Self::Projection => "projection",
            Self::ReferenceOnly => "reference_only",
            Self::Summary => "summary",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Permission {
    Read,
    Write,
    Admin,
}

impl Permission {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Admin => "admin",
        }
    }

    pub fn allows_read(&self) -> bool {
        matches!(self, Self::Read | Self::Write | Self::Admin)
    }

    pub fn allows_write(&self) -> bool {
        matches!(self, Self::Write | Self::Admin)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedReference {
    pub share_id: String,
    pub source_agent_iri: String,
    pub target_agent_iri: String,
    pub node_iri: String,
    pub share_type: ShareType,
    pub permission: Permission,
    pub expires_at: Option<u64>,
    pub projection_frame: Option<String>,
    pub metadata: HashMap<String, Value>,
}

impl SharedReference {
    pub fn new(
        source_agent_iri: &str,
        target_agent_iri: &str,
        node_iri: &str,
        share_type: ShareType,
        permission: Permission,
    ) -> Self {
        Self {
            share_id: format!("iri://share/{}", Uuid::new_v4().hyphenated()),
            source_agent_iri: source_agent_iri.to_string(),
            target_agent_iri: target_agent_iri.to_string(),
            node_iri: node_iri.to_string(),
            share_type,
            permission,
            expires_at: None,
            projection_frame: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.expires_at = Some(now + ttl_seconds);
        self
    }

    pub fn with_projection_frame(mut self, frame: &str) -> Self {
        self.projection_frame = Some(frame.to_string());
        self
    }

    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            return now > expires_at;
        }
        false
    }

    pub fn to_json_ld(&self) -> Value {
        let mut doc = serde_json::json!({
            "@context": "https://agent-os.org/share",
            "@id": self.share_id,
            "@type": "SharedReference",
            "source_agent": self.source_agent_iri,
            "target_agent": self.target_agent_iri,
            "node_iri": self.node_iri,
            "share_type": self.share_type.as_str(),
            "permission": self.permission.as_str(),
        });

        if let Some(expires_at) = self.expires_at {
            if let Some(obj) = doc.as_object_mut() {
                obj.insert("expires_at".to_string(), Value::Number(expires_at.into()));
            }
        }

        if let Some(frame) = &self.projection_frame {
            if let Some(obj) = doc.as_object_mut() {
                obj.insert("projection_frame".to_string(), Value::String(frame.clone()));
            }
        }

        doc
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareRequest {
    pub request_id: String,
    pub source_agent_iri: String,
    pub target_agent_iri: String,
    pub requested_iris: Vec<String>,
    pub share_type: ShareType,
    pub permission: Permission,
    pub ttl_seconds: Option<u64>,
}

impl ShareRequest {
    pub fn new(
        source_agent_iri: &str,
        target_agent_iri: &str,
        requested_iris: Vec<String>,
        share_type: ShareType,
        permission: Permission,
    ) -> Self {
        Self {
            request_id: Uuid::new_v4().hyphenated().to_string(),
            source_agent_iri: source_agent_iri.to_string(),
            target_agent_iri: target_agent_iri.to_string(),
            requested_iris,
            share_type,
            permission,
            ttl_seconds: None,
        }
    }

    pub fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        self.ttl_seconds = Some(ttl_seconds);
        self
    }

    pub fn to_json_ld(&self) -> Value {
        let mut doc = serde_json::json!({
            "@context": "https://agent-os.org/share",
            "@id": format!("iri://share/request/{}", self.request_id),
            "@type": "ShareRequest",
            "request_id": self.request_id,
            "source_agent": self.source_agent_iri,
            "target_agent": self.target_agent_iri,
            "requested_iris": self.requested_iris,
            "share_type": self.share_type.as_str(),
            "permission": self.permission.as_str(),
        });

        if let Some(ttl) = self.ttl_seconds {
            if let Some(obj) = doc.as_object_mut() {
                obj.insert("ttl_seconds".to_string(), Value::Number(ttl.into()));
            }
        }

        doc
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareResponse {
    pub request_id: String,
    pub success: bool,
    pub shared_nodes: Vec<Value>,
    pub projections: HashMap<String, String>,
    pub error_message: Option<String>,
}

impl ShareResponse {
    pub fn success(request_id: &str) -> Self {
        Self {
            request_id: request_id.to_string(),
            success: true,
            shared_nodes: Vec::new(),
            projections: HashMap::new(),
            error_message: None,
        }
    }

    pub fn failure(request_id: &str, error: &str) -> Self {
        Self {
            request_id: request_id.to_string(),
            success: false,
            shared_nodes: Vec::new(),
            projections: HashMap::new(),
            error_message: Some(error.to_string()),
        }
    }

    pub fn with_shared_node(mut self, node: Value) -> Self {
        self.shared_nodes.push(node);
        self
    }

    pub fn with_projection(mut self, iri: &str, projection: &str) -> Self {
        self.projections
            .insert(iri.to_string(), projection.to_string());
        self
    }

    pub fn to_json_ld(&self) -> Value {
        serde_json::json!({
            "@context": "https://agent-os.org/share",
            "@id": format!("iri://share/response/{}", self.request_id),
            "@type": "ShareResponse",
            "request_id": self.request_id,
            "success": self.success,
            "shared_nodes": self.shared_nodes,
            "projections": self.projections,
            "error_message": self.error_message,
        })
    }
}

pub struct SharingProtocol {
    active_shares: RwLock<HashMap<String, SharedReference>>,
    share_index: RwLock<HashMap<String, HashSet<String>>>,
    audit_log: Option<Arc<SharingAuditLog>>,
}

impl SharingProtocol {
    pub fn new() -> Self {
        Self {
            active_shares: RwLock::new(HashMap::new()),
            share_index: RwLock::new(HashMap::new()),
            audit_log: None,
        }
    }

    /// Inject audit log (optional)
    pub fn with_audit_log(mut self, audit_log: Arc<SharingAuditLog>) -> Self {
        self.audit_log = Some(audit_log);
        self
    }

    pub fn create_share(
        &self,
        source_agent_iri: &str,
        target_agent_iri: &str,
        node_iris: &[String],
        share_type: ShareType,
        permission: Permission,
        ttl_seconds: Option<u64>,
        projection_frame: Option<&str>,
    ) -> Vec<SharedReference> {
        let mut references = Vec::new();
        let mut shares = self.active_shares.write();
        let mut index = self.share_index.write();

        for node_iri in node_iris {
            let mut ref_ = SharedReference::new(
                source_agent_iri,
                target_agent_iri,
                node_iri,
                share_type,
                permission,
            );

            if let Some(ttl) = ttl_seconds {
                ref_ = ref_.with_ttl(ttl);
            }

            if let Some(frame) = projection_frame {
                ref_ = ref_.with_projection_frame(frame);
            }

            let share_id = ref_.share_id.clone();

            index
                .entry(target_agent_iri.to_string())
                .or_default()
                .insert(share_id.clone());

            shares.insert(share_id, ref_.clone());

            // non-blocking write to audit log
            if let Some(ref audit) = self.audit_log {
                audit.log_share_created(
                    &ref_.share_id,
                    source_agent_iri,
                    target_agent_iri,
                    node_iri,
                    share_type.as_str(),
                    permission.as_str(),
                    ttl_seconds,
                );
            }

            references.push(ref_);
        }

        references
    }

    pub fn resolve_iri(&self, iri: &str, agent_iri: &str) -> Option<SharedReference> {
        let shares = self.active_shares.read();

        for (share_id, ref_) in shares.iter() {
            if ref_.node_iri == iri && ref_.target_agent_iri == agent_iri {
                if ref_.is_expired() {
                    let share_id = share_id.clone();
                    drop(shares);
                    self.revoke_share(&share_id);
                    return None;
                }

                if !ref_.permission.allows_read() {
                    return None;
                }

                if let Some(ref audit) = self.audit_log {
                    audit.log_share_resolved(share_id, agent_iri);
                }

                return Some(ref_.clone());
            }
        }

        None
    }

    pub fn get_shares_for_agent(&self, agent_iri: &str) -> Vec<SharedReference> {
        self.cleanup_expired();

        let index = self.share_index.read();
        let shares = self.active_shares.read();

        if let Some(share_ids) = index.get(agent_iri) {
            share_ids
                .iter()
                .filter_map(|id| shares.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn revoke_share(&self, share_id: &str) -> bool {
        let mut shares = self.active_shares.write();

        if let Some(ref_) = shares.remove(share_id) {
            let mut index = self.share_index.write();
            if let Some(share_ids) = index.get_mut(&ref_.target_agent_iri) {
                share_ids.remove(share_id);
            }
            if let Some(ref audit) = self.audit_log {
                audit.log_share_revoked(share_id);
            }
            true
        } else {
            false
        }
    }

    pub fn revoke_all_for_node(&self, node_iri: &str) -> usize {
        let shares = self.active_shares.read();
        let to_remove: Vec<String> = shares
            .iter()
            .filter(|(_, ref_)| ref_.node_iri == node_iri)
            .map(|(id, _)| id.clone())
            .collect();
        drop(shares);

        let mut count = 0;
        for id in to_remove {
            if self.revoke_share(&id) {
                count += 1;
            }
        }
        count
    }

    pub fn cleanup_expired(&self) -> usize {
        let shares = self.active_shares.read();
        let expired: Vec<String> = shares
            .iter()
            .filter(|(_, ref_)| ref_.is_expired())
            .map(|(id, _)| id.clone())
            .collect();
        drop(shares);

        let mut count = 0;
        for id in expired {
            if self.revoke_share(&id) {
                count += 1;
            }
        }
        count
    }

    pub fn active_share_count(&self) -> usize {
        self.active_shares.read().len()
    }
}

impl Default for SharingProtocol {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ContextInjector {
    sharing: Arc<SharingProtocol>,
}

impl ContextInjector {
    pub fn new(sharing: Arc<SharingProtocol>) -> Self {
        Self { sharing }
    }

    pub fn inject_context(
        &self,
        target_agent_iri: &str,
        task_iri: &str,
        required_context: &[String],
    ) -> Value {
        let mut context_nodes = Vec::new();
        let mut total_size = 0;
        let max_size = 500;

        for iri in required_context {
            if total_size >= max_size {
                break;
            }

            if let Some(ref_) = self.sharing.resolve_iri(iri, target_agent_iri) {
                let node_data = ref_.to_json_ld();
                let node_size = node_data.to_string().len();

                if total_size + node_size <= max_size {
                    context_nodes.push(serde_json::json!({
                        "iri": iri,
                        "data": node_data,
                        "size": node_size,
                    }));
                    total_size += node_size;
                }
            }
        }

        serde_json::json!({
            "@context": "https://agent-os.org/context/injection",
            "@type": "ContextInjection",
            "target_agent": target_agent_iri,
            "task_iri": task_iri,
            "context_nodes": context_nodes,
            "total_size": total_size,
            "node_count": context_nodes.len(),
        })
    }

    pub fn create_handoff_context(
        &self,
        from_agent_iri: &str,
        to_agent_iri: &str,
        task_iri: &str,
        artifacts: &[String],
    ) -> Value {
        self.sharing.create_share(
            from_agent_iri,
            to_agent_iri,
            artifacts,
            ShareType::Projection,
            Permission::Read,
            Some(3600),
            Some("handoff"),
        );

        serde_json::json!({
            "@context": "https://agent-os.org/context/handoff",
            "@type": "AgentHandoff",
            "from_agent": from_agent_iri,
            "to_agent": to_agent_iri,
            "task_iri": task_iri,
            "artifacts": artifacts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_reference() {
        let ref_ = SharedReference::new(
            "iri://agent/pa",
            "iri://agent/da",
            "iri://task/123/result",
            ShareType::Projection,
            Permission::Read,
        )
        .with_ttl(3600);

        assert!(ref_.permission.allows_read());
        assert!(!ref_.permission.allows_write());
        assert!(!ref_.is_expired());
    }

    #[test]
    fn test_sharing_protocol() {
        let protocol = SharingProtocol::new();

        let refs = protocol.create_share(
            "iri://agent/pa",
            "iri://agent/da",
            &["iri://task/123/plan".to_string()],
            ShareType::Projection,
            Permission::Read,
            Some(3600),
            None,
        );

        assert_eq!(refs.len(), 1);
        assert_eq!(protocol.active_share_count(), 1);

        let resolved = protocol.resolve_iri("iri://task/123/plan", "iri://agent/da");
        assert!(resolved.is_some());

        let not_authorized = protocol.resolve_iri("iri://task/123/plan", "iri://agent/ca");
        assert!(not_authorized.is_none());
    }

    #[test]
    fn test_context_injector() {
        let sharing = Arc::new(SharingProtocol::new());
        let injector = ContextInjector::new(sharing.clone());

        sharing.create_share(
            "iri://agent/pa",
            "iri://agent/da",
            &["iri://task/123/plan".to_string()],
            ShareType::Projection,
            Permission::Read,
            Some(3600),
            None,
        );

        let context = injector.inject_context(
            "iri://agent/da",
            "iri://task/123",
            &["iri://task/123/plan".to_string()],
        );

        assert!(context.get("context_nodes").is_some());
    }

    #[test]
    fn test_share_expiration() {
        let protocol = SharingProtocol::new();

        let _refs = protocol.create_share(
            "iri://agent/pa",
            "iri://agent/da",
            &["iri://task/123/plan".to_string()],
            ShareType::Projection,
            Permission::Read,
            Some(1),
            None,
        );

        std::thread::sleep(std::time::Duration::from_secs(2));

        let resolved = protocol.resolve_iri("iri://task/123/plan", "iri://agent/da");
        assert!(resolved.is_none());
        assert_eq!(protocol.active_share_count(), 0);
    }
}
