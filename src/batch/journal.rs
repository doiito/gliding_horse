use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core::event_bus::Event;
use crate::memory::l0_store::{L0Entry, L0Store, MesiState};
use crate::CoreError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEventEnvelope {
    pub event_id: String,
    pub sequence: u64,
    pub event_type: String,
    pub task_iri: String,
    pub source_agent_iri: String,
    pub payload: String,
    /// SHA-256 of the exact payload used to distinguish a harmless retry from
    /// an event-id collision carrying different input.
    #[serde(default)]
    pub payload_sha256: String,
    pub received_at: DateTime<Utc>,
}

pub struct BatchEventJournal {
    l0: Arc<L0Store>,
}

impl BatchEventJournal {
    pub fn new(l0: Arc<L0Store>) -> Self {
        Self { l0 }
    }
    fn key(event_id: &str) -> String {
        format!("iri://batch-journal/event/{event_id}")
    }

    /// Returns false when this exact event id was already durably recorded.
    pub fn record(&self, event: &Event) -> Result<bool, CoreError> {
        let key = Self::key(&event.event_id);
        let incoming_payload_sha256 = payload_sha256(&event.payload);
        if let Some(existing) = self.l0.retrieve(&key)? {
            let envelope: BatchEventEnvelope =
                serde_json::from_str(&existing.content).map_err(|error| {
                    CoreError::StorageError {
                        message: format!(
                            "Invalid existing batch journal envelope for '{}': {error}",
                            event.event_id
                        ),
                    }
                })?;
            // Older envelopes did not carry the digest; calculate it from the
            // durable payload so upgrades remain retry-compatible.
            let existing_hash = if envelope.payload_sha256.is_empty() {
                payload_sha256(&envelope.payload)
            } else {
                envelope.payload_sha256
            };
            if existing_hash == incoming_payload_sha256 {
                return Ok(false);
            }
            return Err(CoreError::ValidationFailed {
                message: format!(
                    "Batch journal event id '{}' was reused with a different payload",
                    event.event_id
                ),
            });
        }
        let envelope = BatchEventEnvelope {
            event_id: event.event_id.clone(),
            sequence: event.sequence,
            event_type: event.event_type.clone(),
            task_iri: event.task_iri.clone(),
            source_agent_iri: event.source_agent_iri.clone(),
            payload: event.payload.clone(),
            payload_sha256: incoming_payload_sha256,
            received_at: Utc::now(),
        };
        let now = Utc::now();
        self.l0.store_entry(&L0Entry {
            iri: key,
            content: serde_json::to_string(&envelope).map_err(|e| CoreError::StorageError {
                message: e.to_string(),
            })?,
            importance: 0.2,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            tags: vec!["batch-journal".into()],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: Some("system:batch-journal".into()),
            jsonld_context: None,
            jsonld_types: vec!["batch:EventEnvelope".into()],
            hyperspace_point_id: None,
        })?;
        Ok(true)
    }
    pub fn pending(&self, limit: usize) -> Result<Vec<BatchEventEnvelope>, CoreError> {
        Ok(self
            .l0
            .scan_iri_prefix("iri://batch-journal/event/", limit)?
            .into_iter()
            .filter_map(|entry| serde_json::from_str(&entry.content).ok())
            .collect())
    }
    pub fn acknowledge(&self, event_id: &str) -> Result<bool, CoreError> {
        self.l0.delete(&Self::key(event_id))
    }
}

fn payload_sha256(payload: &str) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(payload.as_bytes())))
}

impl BatchEventEnvelope {
    pub fn into_event(self) -> Event {
        Event {
            event_id: self.event_id,
            task_iri: self.task_iri,
            event_type: self.event_type,
            source_agent_iri: self.source_agent_iri,
            payload_json_ld: self.payload.clone(),
            payload: self.payload,
            timestamp: self.received_at,
            sequence: self.sequence,
            type_mask: 0,
            priority: crate::core::event_bus::EventPriority::Normal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::event_bus::EventPriority;

    fn event() -> Event {
        Event {
            event_id: "evt-journal".into(),
            task_iri: "task:1".into(),
            event_type: "CUSTOM".into(),
            source_agent_iri: "agent:1".into(),
            payload: "payload".into(),
            payload_json_ld: "payload".into(),
            timestamp: Utc::now(),
            sequence: 7,
            type_mask: 1,
            priority: EventPriority::Normal,
        }
    }

    #[test]
    fn durable_event_is_deduplicated_and_acknowledged() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let journal = BatchEventJournal::new(l0.clone());
        assert!(journal.record(&event()).unwrap());
        assert!(!journal.record(&event()).unwrap());
        assert_eq!(journal.pending(10).unwrap().len(), 1);
        let rebuilt = BatchEventJournal::new(l0);
        assert_eq!(rebuilt.pending(10).unwrap()[0].sequence, 7);
        assert!(rebuilt.pending(10).unwrap()[0]
            .payload_sha256
            .starts_with("sha256:"));
        assert!(rebuilt.acknowledge("evt-journal").unwrap());
        assert!(rebuilt.pending(10).unwrap().is_empty());
    }

    #[test]
    fn event_id_reused_with_a_different_payload_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let journal = BatchEventJournal::new(Arc::new(
            L0Store::new(dir.path().to_str().unwrap()).unwrap(),
        ));
        let original = event();
        let mut collision = event();
        collision.payload = "different payload".into();
        collision.payload_json_ld = collision.payload.clone();

        assert!(journal.record(&original).unwrap());
        assert!(matches!(
            journal.record(&collision),
            Err(CoreError::ValidationFailed { .. })
        ));
        assert_eq!(journal.pending(10).unwrap().len(), 1);
    }

    #[test]
    fn legacy_envelope_without_digest_remains_retry_compatible() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let legacy = serde_json::json!({
            "event_id": "evt-journal",
            "sequence": 7,
            "event_type": "CUSTOM",
            "task_iri": "task:1",
            "source_agent_iri": "agent:1",
            "payload": "payload",
            "received_at": Utc::now(),
        });
        l0.store("iri://batch-journal/event/evt-journal", &legacy.to_string())
            .unwrap();
        let journal = BatchEventJournal::new(l0);

        assert!(!journal.record(&event()).unwrap());
    }
}
