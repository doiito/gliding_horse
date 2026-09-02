//! Common, observable task termination contract for every product entry point.

use std::sync::Arc;

use serde_json::json;

use crate::core::agent_runner::TaskResult;
use crate::core::event_bus::EventBus;
use crate::core::execution_journal::TaskExecutionJournal;
use crate::memory::l0_store::L0Store;
use tracing::warn;

/// Emits the canonical terminal task event after an executor returns a result.
///
/// Entry points may add UI or product-specific persistence afterwards, but they
/// must all pass through this component so downstream consumers see the same
/// status, errors and action-level evidence irrespective of transport.
pub struct TaskFinalizer {
    event_bus: Arc<EventBus>,
    l0: Option<Arc<L0Store>>,
}

impl TaskFinalizer {
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        Self {
            event_bus,
            l0: None,
        }
    }

    /// Enable task evidence sealing for product entry points that own the L0
    /// store. The legacy constructor stays available for transports without
    /// durable task trace ownership.
    pub fn with_evidence_ledger(event_bus: Arc<EventBus>, l0: Arc<L0Store>) -> Self {
        Self {
            event_bus,
            l0: Some(l0),
        }
    }

    /// Publish a `TASK_FINALIZED` event and return its event ID.
    pub async fn finalize(&self, task_iri: &str, result: &TaskResult) -> String {
        if let Some(l0) = &self.l0 {
            match TaskExecutionJournal::open(l0.clone(), task_iri)
                .and_then(|journal| journal.seal(&result.status))
            {
                Ok(()) => {}
                Err(error) => {
                    warn!(task_iri = %task_iri, %error, "Failed to seal task evidence during finalization")
                }
            }
        }
        let actions = result
            .tracked_actions
            .iter()
            .map(|action| {
                json!({
                    "action_id": action.action_id,
                    "tool_name": action.tool_name,
                    "agent_role": action.agent_role,
                    "status": action.status,
                    "duration_secs": action.duration_secs,
                    "error": action.error,
                })
            })
            .collect::<Vec<_>>();
        let payload = json!({
            "task_iri": task_iri,
            "result_task_iri": result.task_iri,
            "status": result.status,
            "summary": result.summary,
            "errors": result.errors,
            "turn_count": result.turn_count,
            "tool_call_count": result.tool_call_count,
            "tracked_actions": actions,
        });

        self.event_bus
            .emit(
                task_iri,
                "TASK_FINALIZED",
                "system:task-finalizer",
                &payload.to_string(),
            )
            .await
    }

    /// Publish a terminal failure when execution failed before a `TaskResult`
    /// could be constructed.
    pub async fn finalize_error(&self, task_iri: &str, error: &str) -> String {
        let payload = json!({
            "task_iri": task_iri,
            "status": "failed",
            "errors": [error],
            "tracked_actions": [],
        });
        self.event_bus
            .emit(
                task_iri,
                "TASK_FINALIZED",
                "system:task-finalizer",
                &payload.to_string(),
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent_runner::TaskResult;

    #[tokio::test]
    async fn finalizer_emits_canonical_terminal_event() {
        let event_bus = Arc::new(EventBus::new(4));
        let mut rx = event_bus.subscribe();
        let finalizer = TaskFinalizer::new(event_bus);
        let result = TaskResult {
            task_iri: "iri://task/example".to_string(),
            status: "completed".to_string(),
            summary: "done".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: vec![],
            verdict: None,
            archive_iri: None,
        };

        finalizer.finalize("iri://task/example", &result).await;
        let event = rx.recv().await.unwrap();
        assert_eq!(event.event_type, "TASK_FINALIZED");
        assert_eq!(event.task_iri, "iri://task/example");
        assert_eq!(event.payload_json_ld, event.payload);
        assert!(event.payload.contains("completed"));
    }
}
