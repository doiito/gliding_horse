//! Share audit log — records all share operations without altering the SharingProtocol primary data source (HashMap)
//!
//! # Design Decisions
//!
//! - **Non-blocking**: audit is a side effect, failure does not block share operations
//! - **Memory-first**: writes to in-memory Vec, async flush to Oxigraph avoids SharingProtocol holding a Store reference
//! - **Append-only**: audit log is never modified or deleted
//!
//! # Deliberately Simplified
//!
//! - No index queries (low audit query volume, full scan is sufficient)
//! - No SPARQL inserts (keeps independent of Oxigraph)
//! - flush_to_store is a future option, currently memory-only

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Share event type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharingEvent {
    Created,
    Resolved,
    Revoked,
    Expired,
}

impl SharingEvent {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Created => "Created",
            Self::Resolved => "Resolved",
            Self::Revoked => "Revoked",
            Self::Expired => "Expired",
        }
    }
}

/// Share audit entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingAuditEntry {
    pub event_type: SharingEvent,
    pub share_id: String,
    pub source_agent: String,
    pub target_agent: String,
    pub node_iri: String,
    pub share_type: String,
    pub permission: String,
    pub ttl_seconds: Option<u64>,
    pub timestamp: DateTime<Utc>,
}

/// Share audit log
///
/// Stores all audit entries in an in-memory Vec, avoiding SharingProtocol holding an Oxigraph Store reference.
/// Can be exported via `flush_to_store()` when an external Store is available.
pub struct SharingAuditLog {
    entries: RwLock<Vec<SharingAuditEntry>>,
}

impl SharingAuditLog {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }

    /// Record an audit entry
    pub fn log(&self, entry: SharingAuditEntry) {
        self.entries.write().push(entry);
    }

    /// Log share creation
    pub fn log_share_created(
        &self,
        share_id: &str,
        source_agent: &str,
        target_agent: &str,
        node_iri: &str,
        share_type: &str,
        permission: &str,
        ttl_seconds: Option<u64>,
    ) {
        self.log(SharingAuditEntry {
            event_type: SharingEvent::Created,
            share_id: share_id.to_string(),
            source_agent: source_agent.to_string(),
            target_agent: target_agent.to_string(),
            node_iri: node_iri.to_string(),
            share_type: share_type.to_string(),
            permission: permission.to_string(),
            ttl_seconds,
            timestamp: Utc::now(),
        });
    }

    /// Log share resolution
    pub fn log_share_resolved(&self, share_id: &str, by_agent: &str) {
        self.log(SharingAuditEntry {
            event_type: SharingEvent::Resolved,
            share_id: share_id.to_string(),
            source_agent: String::new(),
            target_agent: by_agent.to_string(),
            node_iri: String::new(),
            share_type: String::new(),
            permission: String::new(),
            ttl_seconds: None,
            timestamp: Utc::now(),
        });
    }

    /// Log share revocation
    pub fn log_share_revoked(&self, share_id: &str) {
        self.log(SharingAuditEntry {
            event_type: SharingEvent::Revoked,
            share_id: share_id.to_string(),
            source_agent: String::new(),
            target_agent: String::new(),
            node_iri: String::new(),
            share_type: String::new(),
            permission: String::new(),
            ttl_seconds: None,
            timestamp: Utc::now(),
        });
    }

    /// Query share history received by an Agent
    pub fn query_shares_for_agent(&self, agent_iri: &str) -> Vec<SharingAuditEntry> {
        self.entries
            .read()
            .iter()
            .filter(|e| e.target_agent == agent_iri)
            .cloned()
            .collect()
    }

    /// Query share history for an IRI
    pub fn query_shares_for_node(&self, node_iri: &str) -> Vec<SharingAuditEntry> {
        self.entries
            .read()
            .iter()
            .filter(|e| e.node_iri == node_iri)
            .cloned()
            .collect()
    }

    /// Get all audit entries
    pub fn all_entries(&self) -> Vec<SharingAuditEntry> {
        self.entries.read().clone()
    }

    /// Number of audit entries
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize in-memory audit entries to JSON (for external consumption)
    pub fn to_json(&self) -> serde_json::Value {
        let entries = self.entries.read();
        serde_json::to_value(&*entries).unwrap_or_default()
    }
}

impl Default for SharingAuditLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_share_created() {
        let log = SharingAuditLog::new();
        log.log_share_created(
            "iri://share/abc",
            "iri://agent/pa",
            "iri://agent/da",
            "iri://task/123",
            "Projection",
            "Read",
            Some(3600),
        );
        assert_eq!(log.len(), 1);

        let entry = &log.all_entries()[0];
        assert_eq!(entry.event_type, SharingEvent::Created);
        assert_eq!(entry.source_agent, "iri://agent/pa");
        assert_eq!(entry.share_type, "Projection");
    }

    #[test]
    fn test_log_share_resolved() {
        let log = SharingAuditLog::new();
        log.log_share_resolved("iri://share/abc", "iri://agent/da");
        assert_eq!(log.len(), 1);
        assert_eq!(log.all_entries()[0].event_type, SharingEvent::Resolved);
    }

    #[test]
    fn test_log_share_revoked() {
        let log = SharingAuditLog::new();
        log.log_share_revoked("iri://share/abc");
        assert_eq!(log.len(), 1);
        assert_eq!(log.all_entries()[0].event_type, SharingEvent::Revoked);
    }

    #[test]
    fn test_query_by_agent() {
        let log = SharingAuditLog::new();
        log.log_share_created("s1", "src1", "agent:da1", "n1", "Full", "Read", None);
        log.log_share_created("s2", "src2", "agent:da2", "n2", "Proj", "Read", None);
        log.log_share_created("s3", "src3", "agent:da1", "n3", "Full", "Write", None);

        let results = log.query_shares_for_agent("agent:da1");
        assert_eq!(results.len(), 2);

        let results2 = log.query_shares_for_agent("nonexistent");
        assert!(results2.is_empty());
    }

    #[test]
    fn test_query_by_node() {
        let log = SharingAuditLog::new();
        log.log_share_created("s1", "src1", "tgt1", "iri://task/42", "Full", "Read", None);
        log.log_share_created("s2", "src2", "tgt2", "iri://task/99", "Proj", "Read", None);
        log.log_share_created("s3", "src3", "tgt3", "iri://task/42", "Full", "Write", None);

        let results = log.query_shares_for_node("iri://task/42");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_to_json() {
        let log = SharingAuditLog::new();
        log.log_share_created("s1", "a", "b", "n1", "Full", "Read", None);
        let json = log.to_json();
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_empty_log() {
        let log = SharingAuditLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert!(log.query_shares_for_agent("any").is_empty());
    }
}
