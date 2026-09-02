//! Persisted, operator-triggered ANN health evidence.
//!
//! It deliberately records a diagnosis and recommendation only. Checkpoint,
//! metadata vacuum, and any future full reindex remain explicit operations so
//! a quality probe can never rewrite retrieval topology by itself.

use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::memory::hyperspace_store::{AnnHealthProbeConfig, AnnHealthReport, HyperspaceStore};
use crate::memory::l0_store::{L0Entry, L0RecordKind, L0Store, MesiState, RetentionClass};
use crate::CoreError;

pub const ANN_HEALTH_EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const ANN_HEALTH_EVIDENCE_PREFIX: &str = "iri://audit/ann-health/";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnnHealthEvidence {
    pub schema_version: u32,
    pub evidence_iri: String,
    pub report: AnnHealthReport,
}

#[derive(Clone)]
pub struct AnnHealthMonitor {
    l0: Arc<L0Store>,
    hyperspace: Arc<HyperspaceStore>,
}

impl AnnHealthMonitor {
    pub fn new(l0: Arc<L0Store>, hyperspace: Arc<HyperspaceStore>) -> Self {
        Self { l0, hyperspace }
    }

    /// Run and persist a read-only diagnostic. The durable evidence has only
    /// aggregate metrics and recommendation fields, never sampled text,
    /// embeddings, prompts, tool data, or LLM responses.
    pub async fn inspect_and_record(
        &self,
        config: &AnnHealthProbeConfig,
    ) -> Result<AnnHealthEvidence, CoreError> {
        let report = self.hyperspace.assess_ann_health(config).await?;
        let report_json =
            serde_json::to_string(&report).map_err(|error| CoreError::StorageError {
                message: format!("Failed to serialize ANN health report: {error}"),
            })?;
        let mut hasher = Sha256::new();
        hasher.update(report_json.as_bytes());
        hasher.update(
            Utc::now()
                .timestamp_nanos_opt()
                .unwrap_or_default()
                .to_be_bytes(),
        );
        let evidence_iri = format!(
            "{ANN_HEALTH_EVIDENCE_PREFIX}{}",
            hex::encode(&hasher.finalize()[..16])
        );
        let evidence = AnnHealthEvidence {
            schema_version: ANN_HEALTH_EVIDENCE_SCHEMA_VERSION,
            evidence_iri: evidence_iri.clone(),
            report,
        };
        let content =
            serde_json::to_string(&evidence).map_err(|error| CoreError::StorageError {
                message: format!("Failed to serialize ANN health evidence: {error}"),
            })?;
        let entry = L0Entry {
            iri: evidence_iri,
            content,
            importance: 0.8,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: vec!["ann-health".to_string(), "retrieval-diagnostic".to_string()],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: vec!["AnnHealthEvidence".to_string()],
        };
        self.l0.store_with_policy(
            &entry,
            L0RecordKind::AuditEvidence,
            RetentionClass::Permanent,
            None,
        )?;
        Ok(evidence)
    }
}
