use std::sync::Arc;

use tokio::sync::broadcast;

use crate::core::event_bus::{Event, EventBus};

pub struct MemoryBus {
    event_bus: Arc<EventBus>,
}

impl MemoryBus {
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        Self { event_bus }
    }

    pub async fn publish_invalidate(&self, node_iri: &str, scope_iri: &str) {
        self.event_bus
            .emit(
                scope_iri,
                "CACHE_INVALIDATE",
                "system:consistency_engine",
                &serde_json::json!({"node_iri": node_iri}).to_string(),
            )
            .await;
    }

    pub async fn emit_prefetch_request(&self, entity_iri: &str, intent: &str) {
        self.event_bus
            .emit(
                entity_iri,
                "PREFETCH_REQUEST",
                "system:prefetch_engine",
                &serde_json::json!({"entity_iri": entity_iri, "intent": intent}).to_string(),
            )
            .await;
    }

    pub async fn publish(&self, event_type: &str, scope_iri: &str, payload: &str) {
        self.event_bus
            .emit(scope_iri, event_type, "system:memory_scheduler", payload)
            .await;
    }

    pub async fn publish_invalidate_batch(&self, node_iris: &[String], scope_iri: &str) {
        if node_iris.is_empty() {
            return;
        }
        if node_iris.len() == 1 {
            self.publish_invalidate(&node_iris[0], scope_iri).await;
            return;
        }
        self.event_bus
            .emit(
                scope_iri,
                "CACHE_INVALIDATE_BATCH",
                "system:consistency_engine",
                &serde_json::json!({"node_iris": node_iris, "count": node_iris.len()}).to_string(),
            )
            .await;
    }

    pub async fn publish_with_priority(
        &self,
        event_type: &str,
        scope_iri: &str,
        payload: &str,
        priority: crate::core::event_bus::EventPriority,
    ) {
        self.event_bus
            .emit_with_priority(
                scope_iri,
                event_type,
                "system:memory_scheduler",
                payload,
                priority,
            )
            .await;
    }

    /// Subscribe to the event bus
    ///
    /// Returns a `broadcast::Receiver` for system event notifications.
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.event_bus.subscribe()
    }
}
