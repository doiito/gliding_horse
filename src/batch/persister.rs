use std::sync::Arc;

use serde_json::json;
use tracing::{debug, info};

use crate::batch::error::BatchError;
use crate::batch::types::{
    BatchAgentConfig, ExtractedEntity, ExtractedRelation, ExtractionResult, PersistReport,
};
use crate::knowledge_graph::store::KnowledgeGraphStore;
use crate::knowledge_graph::types::{RdfQuad, RdfValue};
use crate::memory::l0_store::L0Store;
use crate::memory::memory_manager::MemoryManager;

pub struct KnowledgePersister {
    kg_store: Option<Arc<KnowledgeGraphStore>>,
    #[allow(dead_code)]
    memory_manager: Option<Arc<tokio::sync::Mutex<MemoryManager>>>,
    l0_store: Option<Arc<L0Store>>,
}

impl KnowledgePersister {
    pub fn new(
        kg_store: Option<Arc<KnowledgeGraphStore>>,
        memory_manager: Option<Arc<tokio::sync::Mutex<MemoryManager>>>,
        l0_store: Option<Arc<L0Store>>,
    ) -> Self {
        Self {
            kg_store,
            memory_manager,
            l0_store,
        }
    }

    pub async fn persist(
        &self,
        result: &ExtractionResult,
        config: &BatchAgentConfig,
    ) -> Result<PersistReport, BatchError> {
        let graph = self.resolve_graph(config);
        let mut report = PersistReport {
            entities_persisted: 0,
            relations_persisted: 0,
            new_entities: 0,
            updated_entities: 0,
            named_graph: graph.clone(),
            // A batch agent can produce many extractions. Keep every
            // extraction summary addressable instead of overwriting the last
            // one at `batch://{agent}`. This is an audit identity, not an
            // event-id idempotency key (that belongs at the execution/journal
            // boundary).
            task_iri: Some(format!("batch://{}/{}", config.name, result.batch_id)),
        };

        // Persist entities
        if !result.entities.is_empty() {
            let iris = self.persist_entities(&result.entities, &config.business_domain, &graph)?;
            report.entities_persisted = result.entities.len();
            report.new_entities = iris.len();
            debug!(
                agent = %config.name,
                entities = %result.entities.len(),
                "Entities persisted"
            );
        }

        // Persist relations
        if !result.relations.is_empty() {
            let count =
                self.persist_relations(&result.relations, &config.business_domain, &graph)?;
            report.relations_persisted = count;
            debug!(
                agent = %config.name,
                relations = %count,
                "Relations persisted"
            );
        }

        // Persist to memory
        self.persist_to_memory(result, &report.task_iri.clone().unwrap_or_default())?;

        info!(
            agent = %config.name,
            graph = %graph,
            entities = %report.entities_persisted,
            relations = %report.relations_persisted,
            "Knowledge persist completed"
        );

        Ok(report)
    }

    pub fn persist_entities(
        &self,
        entities: &[ExtractedEntity],
        domain: &str,
        graph: &str,
    ) -> Result<Vec<String>, BatchError> {
        let store = match self.kg_store.as_ref() {
            Some(s) => s,
            None => return Ok(vec![]),
        };

        let mut iris = Vec::new();
        let mut quads = Vec::new();

        for entity in entities {
            // Relations only identify their endpoints by entity name.  Keep
            // the entity IRI in the batch domain namespace (rather than the
            // entity type namespace) so the two write paths address the same
            // RDF resource.  The type remains an RDF assertion below.
            let entity_iri = entity_iri(domain, &entity.name);

            // rdf:type assertion
            quads.push(RdfQuad {
                subject: entity_iri.clone(),
                predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into(),
                object: RdfValue::Iri(format!(
                    "https://agent-os.org/ontology/batch/{}",
                    entity.entity_type
                )),
                graph: Some(graph.into()),
            });

            // rdfs:label
            quads.push(RdfQuad {
                subject: entity_iri.clone(),
                predicate: "http://www.w3.org/2000/01/rdf-schema#label".into(),
                object: RdfValue::Literal(entity.name.clone()),
                graph: Some(graph.into()),
            });

            // confidence
            quads.push(RdfQuad {
                subject: entity_iri.clone(),
                predicate: "https://agent-os.org/ontology/core/confidence".into(),
                object: RdfValue::TypedLiteral(
                    entity.confidence.to_string(),
                    "http://www.w3.org/2001/XMLSchema#float".into(),
                ),
                graph: Some(graph.into()),
            });

            // description (if present)
            if let Some(ref desc) = entity.description {
                quads.push(RdfQuad {
                    subject: entity_iri.clone(),
                    predicate: "http://www.w3.org/2000/01/rdf-schema#comment".into(),
                    object: RdfValue::Literal(desc.clone()),
                    graph: Some(graph.into()),
                });
            }

            // Source batch
            quads.push(RdfQuad {
                subject: entity_iri.clone(),
                predicate: "https://agent-os.org/ontology/core/source".into(),
                object: RdfValue::Iri(format!("batch://source/{}", domain)),
                graph: Some(graph.into()),
            });

            iris.push(entity_iri);
        }

        store
            .write_quads(&quads, graph)
            .map_err(|e| BatchError::KgWriteFailed { message: e })?;

        Ok(iris)
    }

    pub fn persist_relations(
        &self,
        relations: &[ExtractedRelation],
        domain: &str,
        graph: &str,
    ) -> Result<usize, BatchError> {
        let store = match self.kg_store.as_ref() {
            Some(s) => s,
            None => return Ok(0),
        };

        let mut quads = Vec::new();

        for relation in relations {
            let from_iri = entity_iri(domain, &relation.from);
            let to_iri = entity_iri(domain, &relation.to);
            let rel_iri = format!("https://agent-os.org/ontology/relation/{}", relation.relation);

            quads.push(RdfQuad {
                subject: from_iri,
                predicate: rel_iri,
                object: RdfValue::Iri(to_iri),
                graph: Some(graph.into()),
            });
        }

        store
            .write_quads(&quads, graph)
            .map_err(|e| BatchError::KgWriteFailed { message: e })?;

        Ok(relations.len())
    }

    pub fn persist_to_memory(
        &self,
        result: &ExtractionResult,
        task_iri: &str,
    ) -> Result<(), BatchError> {
        // Store extraction summary in L0 for later retrieval
        if let Some(ref l0) = self.l0_store {
            let summary = json!({
                "type": "batch_extraction",
                "task_iri": task_iri,
                "extracted_at": result.extracted_at.to_rfc3339(),
                "batch_id": result.batch_id,
                "entities_count": result.entities.len(),
                "relations_count": result.relations.len(),
                "context_summary": result.context_summary,
                "intent": result.intent.as_ref().map(|i| json!({
                    "type": i.intent_type,
                    "confidence": i.confidence,
                })),
            })
            .to_string();

            l0.store(task_iri, &summary)
                .map_err(|error| BatchError::MemoryOperationFailed {
                    message: error.to_string(),
                })?;
        }

        Ok(())
    }

    fn resolve_graph(&self, config: &BatchAgentConfig) -> String {
        format!("graph:batch/{}/{}", config.name, config.business_domain)
    }
}

/// Stable RDF identity for a batch entity. Entity type is modeled separately
/// with `rdf:type`, because relation payloads carry names but not endpoint
/// types.
fn entity_iri(domain: &str, name: &str) -> String {
    format!(
        "iri://entity/batch/{}/{}",
        sanitize_id(domain),
        sanitize_id(name)
    )
}

fn sanitize_id(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | ':' => c,
            ' ' | '\t' | '\n' => '_',
            _ => '_',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extraction_result() -> ExtractionResult {
        ExtractionResult {
            batch_id: "batch-test".to_string(),
            extracted_at: chrono::Utc::now(),
            entities: vec![ExtractedEntity {
                name: "Alice".to_string(),
                entity_type: "Person".to_string(),
                description: Some("Test entity".to_string()),
                aliases: Vec::new(),
                confidence: 0.95,
                source_messages: Vec::new(),
            }],
            relations: vec![ExtractedRelation {
                from: "Alice".to_string(),
                relation: "knows".to_string(),
                to: "Alice".to_string(),
                properties: Default::default(),
                confidence: 0.9,
            }],
            intent: None,
            key_decisions: Vec::new(),
            context_summary: "test extraction".to_string(),
            llm_calls: 1,
            tokens_consumed: 1,
            confidence_scores: Default::default(),
            raw_response: None,
        }
    }

    #[test]
    fn test_sanitize_id() {
        assert_eq!(sanitize_id("Hello World"), "Hello_World");
        assert_eq!(sanitize_id("test-id_123"), "test-id_123");
        assert_eq!(sanitize_id("special@#$%chars"), "special____chars");
    }

    #[test]
    fn entity_identity_is_domain_scoped_and_independent_of_entity_type() {
        // ExtractedRelation carries endpoint names only. A Person named
        // "Alice" and a relation endpoint named "Alice" must therefore
        // resolve to exactly the same RDF IRI.
        assert_eq!(
            entity_iri("support", "Alice Smith"),
            "iri://entity/batch/support/Alice_Smith"
        );
        assert_ne!(
            entity_iri("support", "Alice Smith"),
            entity_iri("sales", "Alice Smith")
        );
    }

    #[test]
    fn test_resolve_graph() {
        let config = BatchAgentConfig {
            name: "test_agent".to_string(),
            business_domain: "test_domain".to_string(),
            ..Default::default()
        };

        let persister = KnowledgePersister::new(None, None, None);
        let graph = persister.resolve_graph(&config);
        assert_eq!(graph, "graph:batch/test_agent/test_domain");
    }

    #[tokio::test]
    async fn persist_writes_configured_kg_and_l0_before_reporting_success() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let kg = Arc::new(KnowledgeGraphStore::new().unwrap());
        let persister = KnowledgePersister::new(Some(kg.clone()), None, Some(l0.clone()));
        let config = BatchAgentConfig {
            name: "test_agent".to_string(),
            business_domain: "test_domain".to_string(),
            ..Default::default()
        };

        let report = persister
            .persist(&extraction_result(), &config)
            .await
            .unwrap();

        assert_eq!(report.entities_persisted, 1);
        assert_eq!(report.relations_persisted, 1);
        assert!(l0
            .retrieve("batch://test_agent/batch-test")
            .unwrap()
            .is_some());
        let rows = kg
            .query_sparql(
                "SELECT ?s WHERE { ?s <http://www.w3.org/2000/01/rdf-schema#label> \"Alice\" . }",
                Some("graph:batch/test_agent/test_domain"),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[tokio::test]
    async fn batch_summaries_are_not_overwritten_for_the_same_agent() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let persister = KnowledgePersister::new(None, None, Some(l0.clone()));
        let config = BatchAgentConfig {
            name: "test_agent".to_string(),
            business_domain: "test_domain".to_string(),
            ..Default::default()
        };
        let first = extraction_result();
        let mut second = extraction_result();
        second.batch_id = "batch-second".to_string();

        persister.persist(&first, &config).await.unwrap();
        persister.persist(&second, &config).await.unwrap();

        assert!(l0
            .retrieve("batch://test_agent/batch-test")
            .unwrap()
            .is_some());
        assert!(l0
            .retrieve("batch://test_agent/batch-second")
            .unwrap()
            .is_some());
    }
}
