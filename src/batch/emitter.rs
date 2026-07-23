use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;

use crate::batch::types::{
    DetectedIntent, EmitCondition, ExtractedEntity, ExtractedRelation, ExtractionResult,
};
use crate::core::event_bus::{EventBus, EventPriority};

pub struct BatchEventEmitter {
    event_bus: Arc<EventBus>,
    config: HashMap<String, Vec<EmitCondition>>,
}

impl BatchEventEmitter {
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        // Register Batch Agent event types for bitmap routing
        event_bus.register_type("BATCH_AGENT_REGISTERED");
        event_bus.register_type("BATCH_AGENT_STARTED");
        event_bus.register_type("BATCH_AGENT_STOPPED");
        event_bus.register_type("BATCH_AGENT_ERROR");
        event_bus.register_type("BATCH_EXTRACTION_STARTED");
        event_bus.register_type("BATCH_EXTRACTION_COMPLETED");
        event_bus.register_type("BATCH_EXTRACTION_FAILED");
        event_bus.register_type("BATCH_ENTITY_DETECTED");
        event_bus.register_type("BATCH_RELATION_DETECTED");
        event_bus.register_type("BATCH_INTENT_DETECTED");
        event_bus.register_type("BATCH_DECISION_DETECTED");
        event_bus.register_type("BATCH_CONTEXT_INJECTED");

        Self {
            event_bus,
            config: HashMap::new(),
        }
    }

    pub fn set_agent_config(&mut self, agent_name: &str, emit_on: Vec<EmitCondition>) {
        self.config.insert(agent_name.to_string(), emit_on);
    }

    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }

    pub async fn emit_agent_registered(&self, agent_name: &str, config_summary: &str) {
        let _ = self
            .event_bus
            .emit(
                &format!("batch://{}", agent_name),
                "BATCH_AGENT_REGISTERED",
                &format!("batch:manager"),
                &json!({
                    "agent_name": agent_name,
                    "config": config_summary,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
                .to_string(),
            )
            .await;
    }

    pub async fn emit_agent_started(&self, agent_name: &str) {
        let _ = self
            .event_bus
            .emit(
                &format!("batch://{}", agent_name),
                "BATCH_AGENT_STARTED",
                &format!("batch:manager"),
                &json!({
                    "agent_name": agent_name,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
                .to_string(),
            )
            .await;
    }

    pub async fn emit_agent_stopped(&self, agent_name: &str, reason: &str) {
        let _ = self
            .event_bus
            .emit(
                &format!("batch://{}", agent_name),
                "BATCH_AGENT_STOPPED",
                &format!("batch:manager"),
                &json!({
                    "agent_name": agent_name,
                    "reason": reason,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
                .to_string(),
            )
            .await;
    }

    pub async fn emit_extraction_started(
        &self,
        agent_name: &str,
        batch_id: &str,
        window_size: usize,
    ) {
        let _ = self
            .event_bus
            .emit(
                &format!("batch://{}", agent_name),
                "BATCH_EXTRACTION_STARTED",
                &format!("batch:{}", agent_name),
                &json!({
                    "agent_name": agent_name,
                    "batch_id": batch_id,
                    "window_size": window_size,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
                .to_string(),
            )
            .await;
    }

    pub async fn emit_extraction_complete(&self, agent_name: &str, result: &ExtractionResult) {
        let should_emit = self.should_emit(agent_name, &EmitCondition::Always)
            || self.should_emit(agent_name, &EmitCondition::ConfidenceAbove(0.0));

        if !should_emit && !result.entities.is_empty() {
            // Always emit if we found something meaningful
        } else if !should_emit {
            return;
        }

        let payload = json!({
            "agent_name": agent_name,
            "batch_id": result.batch_id,
            "timestamp": result.extracted_at.to_rfc3339(),
            "entities_found": result.entities.len(),
            "relations_found": result.relations.len(),
            "tokens_used": result.tokens_consumed,
            "context_summary": result.context_summary,
        });

        let _ = self
            .event_bus
            .emit(
                &format!("batch://{}", agent_name),
                "BATCH_EXTRACTION_COMPLETED",
                &format!("batch:{}", agent_name),
                &payload.to_string(),
            )
            .await;
    }

    pub async fn emit_extraction_failed(&self, agent_name: &str, batch_id: &str, error: &str) {
        let _ = self
            .event_bus
            .emit(
                &format!("batch://{}", agent_name),
                "BATCH_EXTRACTION_FAILED",
                &format!("batch:{}", agent_name),
                &json!({
                    "agent_name": agent_name,
                    "batch_id": batch_id,
                    "error": error,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
                .to_string(),
            )
            .await;
    }

    pub async fn emit_new_entity(
        &self,
        agent_name: &str,
        entity: &ExtractedEntity,
        batch_id: Option<&str>,
        message_ids: &[String],
    ) {
        if !self.should_emit(agent_name, &EmitCondition::NewEntity) {
            return;
        }

        let payload = json!({
            "event_type": "BATCH_ENTITY_DETECTED",
            "payload": {
                "agent_name": agent_name,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "entity": {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "confidence": entity.confidence,
                    "description": entity.description,
                    "aliases": entity.aliases,
                },
                "source": {
                    "batch_id": batch_id.unwrap_or(""),
                    "message_ids": message_ids,
                }
            }
        });

        let _ = self
            .event_bus
            .emit_with_priority(
                &format!("batch://{}", agent_name),
                "BATCH_ENTITY_DETECTED",
                &format!("batch:{}", agent_name),
                &payload.to_string(),
                EventPriority::Normal,
            )
            .await;
    }

    pub async fn emit_new_relation(&self, agent_name: &str, relation: &ExtractedRelation) {
        if !self.should_emit(agent_name, &EmitCondition::NewRelation) {
            return;
        }

        let payload = json!({
            "event_type": "BATCH_RELATION_DETECTED",
            "payload": {
                "agent_name": agent_name,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "relation": {
                    "from": relation.from,
                    "type": relation.relation,
                    "to": relation.to,
                    "confidence": relation.confidence,
                }
            }
        });

        let _ = self
            .event_bus
            .emit_with_priority(
                &format!("batch://{}", agent_name),
                "BATCH_RELATION_DETECTED",
                &format!("batch:{}", agent_name),
                &payload.to_string(),
                EventPriority::Normal,
            )
            .await;
    }

    pub async fn emit_intent_detected(&self, agent_name: &str, intent: &DetectedIntent) {
        let matched = match &EmitCondition::IntentDetected(vec![]) {
            EmitCondition::IntentDetected(types) => {
                types.is_empty() || types.contains(&intent.intent_type)
            }
            _ => false,
        };

        let should_emit =
            self.should_emit(agent_name, &EmitCondition::IntentDetected(vec![])) || matched;

        if !should_emit {
            return;
        }

        let payload = json!({
            "event_type": "BATCH_INTENT_DETECTED",
            "payload": {
                "agent_name": agent_name,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "intent": {
                    "type": intent.intent_type,
                    "confidence": intent.confidence,
                    "details": intent.details,
                }
            }
        });

        let _ = self
            .event_bus
            .emit_with_priority(
                &format!("batch://{}", agent_name),
                "BATCH_INTENT_DETECTED",
                &format!("batch:{}", agent_name),
                &payload.to_string(),
                EventPriority::Normal,
            )
            .await;
    }

    pub async fn emit_decision_detected(
        &self,
        agent_name: &str,
        decision: &crate::batch::types::ExtractedDecision,
    ) {
        let payload = json!({
            "event_type": "BATCH_DECISION_DETECTED",
            "payload": {
                "agent_name": agent_name,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "decision": {
                    "content": decision.decision,
                    "rationale": decision.rationale,
                    "confidence": decision.confidence,
                }
            }
        });

        let _ = self
            .event_bus
            .emit_with_priority(
                &format!("batch://{}", agent_name),
                "BATCH_DECISION_DETECTED",
                &format!("batch:{}", agent_name),
                &payload.to_string(),
                EventPriority::Normal,
            )
            .await;
    }

    pub async fn emit_context_injected(
        &self,
        agent_name: &str,
        context_type: &str,
        items_count: usize,
    ) {
        let _ = self
            .event_bus
            .emit(
                &format!("batch://{}", agent_name),
                "BATCH_CONTEXT_INJECTED",
                &format!("batch:{}", agent_name),
                &json!({
                    "agent_name": agent_name,
                    "context_type": context_type,
                    "items_count": items_count,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
                .to_string(),
            )
            .await;
    }

    fn should_emit(&self, agent_name: &str, condition: &EmitCondition) -> bool {
        self.config
            .get(agent_name)
            .map(|conditions| {
                conditions.iter().any(|c| match (c, condition) {
                    (EmitCondition::NewEntity, EmitCondition::NewEntity) => true,
                    (EmitCondition::NewRelation, EmitCondition::NewRelation) => true,
                    (EmitCondition::Always, _) => true,
                    (EmitCondition::ConfidenceAbove(t), EmitCondition::ConfidenceAbove(c)) => {
                        c >= t
                    }
                    (EmitCondition::IntentDetected(types), EmitCondition::IntentDetected(_)) => {
                        types.is_empty()
                    }
                    _ => false,
                })
            })
            .unwrap_or(true) // Default: emit everything
    }
}
