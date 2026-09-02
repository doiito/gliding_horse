use std::sync::Arc;

use serde_json::Value;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use crate::core::event_bus::{Event, EventBus};

const BATCH_EVENT_TYPES: &[&str] = &[
    "BATCH_AGENT_REGISTERED",
    "BATCH_AGENT_STARTED",
    "BATCH_AGENT_STOPPED",
    "BATCH_AGENT_ERROR",
    "BATCH_EXTRACTION_STARTED",
    "BATCH_EXTRACTION_COMPLETED",
    "BATCH_EXTRACTION_FAILED",
    "BATCH_ENTITY_DETECTED",
    "BATCH_RELATION_DETECTED",
    "BATCH_INTENT_DETECTED",
    "BATCH_DECISION_DETECTED",
    "BATCH_CONTEXT_INJECTED",
];

pub type BatchEventCallback = Arc<dyn Fn(Value) + Send + Sync + 'static>;

pub struct BatchEventBridge {
    event_bus: Arc<EventBus>,
    callbacks: Vec<BatchEventCallback>,
    running: bool,
}

impl BatchEventBridge {
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        for et in BATCH_EVENT_TYPES {
            event_bus.register_type(et);
        }
        Self {
            event_bus,
            callbacks: Vec::new(),
            running: false,
        }
    }

    pub fn add_callback<F>(&mut self, callback: F)
    where
        F: Fn(Value) + Send + Sync + 'static,
    {
        self.callbacks.push(Arc::new(callback));
    }

    pub fn start(&mut self) {
        if self.running {
            return;
        }
        self.running = true;

        let callbacks = self.callbacks.clone();

        let event_types: Vec<String> = BATCH_EVENT_TYPES.iter().map(|s| s.to_string()).collect();

        let mut rx = self.event_bus.subscribe();

        tokio::spawn(async move {
            info!("BatchEventBridge started, listening for batch events");

            loop {
                match rx.recv().await {
                    Ok(event) => {
                        if !event_types.contains(&event.event_type) {
                            continue;
                        }
                        Self::dispatch_event(&event, &callbacks);
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("BatchEventBridge lagged by {} events", n);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        info!("BatchEventBridge stopped (EventBus closed)");
                        break;
                    }
                }
            }
        });
    }

    pub(crate) fn dispatch_event(event: &Event, callbacks: &[BatchEventCallback]) {
        let payload: Value = serde_json::from_str(&event.payload).unwrap_or(Value::Null);

        let envelope = serde_json::json!({
            "channel": "batch",
            "event_type": event.event_type,
            "source": event.source_agent_iri,
            "task_iri": event.task_iri,
            "timestamp": event.timestamp.to_rfc3339(),
            "payload": payload,
        });

        debug!(
            event_type = %event.event_type,
            "Batch event dispatched to callbacks"
        );

        for callback in callbacks {
            callback(envelope.clone());
        }
    }

    pub fn stop(&mut self) {
        self.running = false;
        info!("BatchEventBridge stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::event_bus::EventPriority;

    fn make_test_event(event_type: &str) -> Event {
        Event {
            event_id: "test_evt".into(),
            task_iri: "iri://task/test".into(),
            event_type: event_type.into(),
            source_agent_iri: "iri://agent/test".into(),
            payload: r#"{"status":"ok"}"#.into(),
            payload_json_ld: String::new(),
            timestamp: chrono::Utc::now(),
            sequence: 1,
            type_mask: Default::default(),
            priority: EventPriority::Normal,
        }
    }

    #[test]
    fn test_dispatch_event() {
        let (tx, rx) = std::sync::mpsc::channel();

        let callbacks: Vec<BatchEventCallback> = vec![Arc::new(move |envelope| {
            let _ = tx.send(envelope);
        })];

        let event = make_test_event("BATCH_EXTRACTION_COMPLETED");
        BatchEventBridge::dispatch_event(&event, &callbacks);

        let received = rx.recv_timeout(std::time::Duration::from_secs(1));
        assert!(received.is_ok(), "Should have dispatched batch event");
        if let Ok(envelope) = received {
            assert_eq!(envelope["channel"], "batch");
            assert_eq!(envelope["event_type"], "BATCH_EXTRACTION_COMPLETED");
            assert_eq!(envelope["source"], "iri://agent/test");
            assert_eq!(envelope["payload"]["status"], "ok");
        }
    }

    #[test]
    fn test_new_registers_event_types() {
        let bus = Arc::new(EventBus::new(100));
        let _bridge = BatchEventBridge::new(bus.clone());

        for et in BATCH_EVENT_TYPES {
            let mask = bus.register_type(et);
            assert!(!mask.is_empty(), "Type {} should be registered", et);
        }
    }

    #[test]
    fn test_add_callback() {
        let bus = Arc::new(EventBus::new(100));
        let mut bridge = BatchEventBridge::new(bus);

        let called = std::sync::atomic::AtomicBool::new(false);
        bridge.add_callback(move |_envelope| {
            called.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        bridge.add_callback(|_envelope| {});

        assert_eq!(bridge.callbacks.len(), 2);
    }
}
