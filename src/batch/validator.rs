use serde_json::Value;

use crate::batch::error::BatchError;
use crate::batch::types::{
    BatchAgentConfig, ExtractedDecision, ExtractedEntity, ExtractedRelation, ExtractionResult,
};

pub struct OutputValidator;

impl OutputValidator {
    pub fn new() -> Self {
        Self
    }

    pub fn validate(
        &self,
        result: &ExtractionResult,
        config: &BatchAgentConfig,
    ) -> Result<(), BatchError> {
        let mut errors: Vec<String> = Vec::new();

        // Validate entities
        let allowed_entity_types: Vec<String> = config.entity_types.effective_types();
        if !allowed_entity_types.is_empty() {
            for entity in &result.entities {
                if !allowed_entity_types.contains(&entity.entity_type) {
                    errors.push(format!(
                        "Entity '{}' uses disallowed type '{}'",
                        entity.name, entity.entity_type
                    ));
                }
                if entity.name.is_empty() {
                    errors.push("Entity with empty name found".to_string());
                }
                if entity.confidence < 0.0 || entity.confidence > 1.0 {
                    errors.push(format!(
                        "Entity '{}' has out-of-range confidence {}",
                        entity.name, entity.confidence
                    ));
                }
            }
        }

        // Validate relations
        let allowed_relation_types: Vec<String> = config.relation_types.effective_types();
        if !allowed_relation_types.is_empty() {
            for rel in &result.relations {
                if !allowed_relation_types.contains(&rel.relation) {
                    errors.push(format!(
                        "Relation '{}' uses disallowed type '{}'",
                        format!("{}->{}", rel.from, rel.to),
                        rel.relation
                    ));
                }
                if rel.from.is_empty() || rel.to.is_empty() {
                    errors.push("Relation with empty from/to found".to_string());
                }
            }
        }

        // Validate intent
        if let Some(ref intent) = result.intent {
            let allowed_intent_types: Vec<String> = config.intent_types.effective_types();
            if !allowed_intent_types.is_empty()
                && !allowed_intent_types.contains(&intent.intent_type)
            {
                errors.push(format!(
                    "Intent type '{}' is not in allowed vocabulary",
                    intent.intent_type
                ));
            }
        }

        // Validate decisions
        for decision in &result.key_decisions {
            match decision.confidence.as_str() {
                "high" | "medium" | "low" => {}
                _ => errors.push(format!(
                    "Decision '{}' has invalid confidence '{}'",
                    decision.decision, decision.confidence
                )),
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(BatchError::ValidationFailed {
                message: errors.join("; "),
            })
        }
    }

    pub fn validate_llm_json_output(
        &self,
        json: &Value,
        config: &BatchAgentConfig,
    ) -> Result<ExtractionResult, BatchError> {
        let entities = json
            .get("entities")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|e| {
                        let name = e.get("name")?.as_str()?;
                        let entity_type = e
                            .get("entity_type")
                            .or_else(|| e.get("type"))
                            .and_then(|v| v.as_str())?;
                        Some(ExtractedEntity {
                            name: name.to_string(),
                            entity_type: entity_type.to_string(),
                            description: e
                                .get("description")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string()),
                            aliases: e
                                .get("aliases")
                                .and_then(|v| v.as_array())
                                .map(|a| {
                                    a.iter()
                                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                        .collect()
                                })
                                .unwrap_or_default(),
                            confidence: e.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5),
                            source_messages: vec![],
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let relations = json
            .get("relations")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| {
                        let from = r.get("from")?.as_str()?;
                        let relation = r
                            .get("relation")
                            .or_else(|| r.get("type"))
                            .and_then(|v| v.as_str())?;
                        let to = r.get("to")?.as_str()?;
                        Some(ExtractedRelation {
                            from: from.to_string(),
                            relation: relation.to_string(),
                            to: to.to_string(),
                            properties: r
                                .get("properties")
                                .and_then(|v| v.as_object())
                                .map(|o| {
                                    o.iter()
                                        .map(|(k, v)| {
                                            (k.clone(), v.as_str().unwrap_or("").to_string())
                                        })
                                        .collect()
                                })
                                .unwrap_or_default(),
                            confidence: r.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let intent = json.get("intent").and_then(|v| {
            if v.is_null() {
                return None;
            }
            Some(crate::batch::types::DetectedIntent {
                intent_type: v
                    .get("intent_type")
                    .or_else(|| v.get("type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string(),
                confidence: v.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0),
                details: v
                    .get("details")
                    .and_then(|v| v.as_object())
                    .map(|o| {
                        o.iter()
                            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
            })
        });

        let key_decisions = json
            .get("key_decisions")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|d| {
                        Some(ExtractedDecision {
                            decision: d.get("decision")?.as_str()?.to_string(),
                            rationale: d
                                .get("rationale")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string()),
                            evidence: d
                                .get("evidence")
                                .and_then(|v| v.as_array())
                                .map(|a| {
                                    a.iter()
                                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                        .collect()
                                })
                                .unwrap_or_default(),
                            confidence: d
                                .get("confidence")
                                .and_then(|v| v.as_str())
                                .unwrap_or("medium")
                                .to_string(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let context_summary = json
            .get("context_summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let result = ExtractionResult {
            batch_id: String::new(),
            extracted_at: chrono::Utc::now(),
            entities,
            relations,
            intent,
            key_decisions,
            context_summary,
            llm_calls: 1,
            tokens_consumed: 0,
            confidence_scores: std::collections::HashMap::new(),
            raw_response: Some(json.to_string()),
        };

        // Validate against vocabulary
        self.validate(&result, config)?;

        Ok(result)
    }
}

impl Default for OutputValidator {
    fn default() -> Self {
        Self::new()
    }
}
