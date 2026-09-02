//! L0 Store - Long-term memory persistence storage
//!
//! This module handles persistent storage of memories and knowledge, supporting MESI cache coherence states and tag secondary indexes.

use chrono::{DateTime, Utc};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::sync::Arc;
use tracing::{debug, info};

use crate::jsonld::registry::{EntityLocation, IriRegistry, StorageLayer};
use crate::CoreError;

/// MESI cache coherence state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MesiState {
    Modified,
    Exclusive,
    Shared,
    Invalid,
}

impl Default for MesiState {
    fn default() -> Self {
        MesiState::Shared
    }
}

/// L0 memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L0Entry {
    pub iri: String,
    pub content: String,
    pub importance: f32,
    pub access_count: u32,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub tags: Vec<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub mesi_state: MesiState,
    #[serde(default)]
    pub content_hash: String,
    #[serde(default)]
    pub named_graph: Option<String>,
    #[serde(default)]
    pub jsonld_context: Option<String>,
    #[serde(default)]
    pub jsonld_types: Vec<String>,
}

/// L0 search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L0SearchResult {
    pub iri: String,
    pub content: String,
    pub relevance_score: f32,
    pub importance: f32,
    pub tags: Vec<String>,
}

/// L0 Store configuration
#[derive(Debug, Clone)]
pub struct L0Config {
    pub path: String,
    pub max_entries: usize,
    pub compression: bool,
    pub blob_inline_threshold: usize,
    pub cache_size_bytes: usize,
    pub quick_repair: bool,
}

/// Logical lifetime/query partition retained behind the unified L0 API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L0RecordKind {
    CanonicalKnowledge,
    AuditEvidence,
    Checkpoint,
    RawInteraction,
    Telemetry,
    Other,
}

impl L0RecordKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CanonicalKnowledge => "canonical_knowledge",
            Self::AuditEvidence => "audit_evidence",
            Self::Checkpoint => "checkpoint",
            Self::RawInteraction => "raw_interaction",
            Self::Telemetry => "telemetry",
            Self::Other => "other",
        }
    }

    fn infer(iri: &str) -> Self {
        let lower = iri.to_ascii_lowercase();
        if lower.contains("/skills/")
            || lower.contains("/knowledge/")
            || lower.contains("/experience/")
            || lower.contains("/policy/")
        {
            Self::CanonicalKnowledge
        } else if lower.contains("/audit/")
            || lower.contains("/governance/")
            || lower.contains("/decision/")
        {
            Self::AuditEvidence
        } else if lower.contains("/checkpoint/") {
            Self::Checkpoint
        } else if lower.contains("/archive/") {
            Self::RawInteraction
        } else if lower.contains("/trace/")
            || lower.contains("/telemetry/")
            || lower.contains("/metrics/")
        {
            Self::Telemetry
        } else {
            Self::Other
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionClass {
    Permanent,
    LongLived,
    TaskScoped,
    Ephemeral,
}

impl RetentionClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Permanent => "permanent",
            Self::LongLived => "long_lived",
            Self::TaskScoped => "task_scoped",
            Self::Ephemeral => "ephemeral",
        }
    }

    fn for_kind(kind: L0RecordKind) -> Self {
        match kind {
            L0RecordKind::CanonicalKnowledge | L0RecordKind::AuditEvidence => Self::Permanent,
            L0RecordKind::Checkpoint => Self::TaskScoped,
            L0RecordKind::RawInteraction => Self::LongLived,
            L0RecordKind::Telemetry => Self::Ephemeral,
            L0RecordKind::Other => Self::LongLived,
        }
    }
}

impl Default for L0Config {
    fn default() -> Self {
        Self {
            path: "./data/l0_store".to_string(),
            max_entries: 1_000_000,
            compression: true,
            blob_inline_threshold: 4_096,
            cache_size_bytes: 128 * 1024 * 1024,
            quick_repair: true,
        }
    }
}

/// Compute SHA-256 content hash for content-addressed deduplication.
/// Uses SHA-256 (same as workspace_monitor) — provides collision resistance
/// and cross-process reproducibility that DefaultHasher cannot guarantee.
fn compute_content_hash(content: &str) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    format!("sha256:{}", hex::encode(result))
}

/// Table definitions for L0 Store redb database.
const ENTRIES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("entries");
const TAG_INDEX_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("tag_index");
const NAMED_GRAPH_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("named_graph");
const TYPE_INDEX_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("type_index");
const RECORD_KIND_INDEX_TABLE: TableDefinition<&str, &[u8]> =
    TableDefinition::new("record_kind_index");
const BLOB_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("content_blobs");
const BLOB_REFCOUNT_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("blob_refcounts");
const TASK_EVIDENCE_FRAME_TABLE: TableDefinition<&str, &[u8]> =
    TableDefinition::new("task_evidence_frames");
const TASK_EVIDENCE_HEAD_TABLE: TableDefinition<&str, &[u8]> =
    TableDefinition::new("task_evidence_heads");
const TASK_EVIDENCE_SEAL_TABLE: TableDefinition<&str, &[u8]> =
    TableDefinition::new("task_evidence_seals");

const META_SCHEMA_VERSION: &str = "_gh_schema_version";
const META_RECORD_KIND: &str = "_gh_record_kind";
const META_RETENTION_CLASS: &str = "_gh_retention_class";
const META_EXPIRES_AT: &str = "_gh_expires_at";
const META_GENERATION: &str = "_gh_generation";
const META_UPDATED_AT: &str = "_gh_updated_at";
const META_BLOB_REF: &str = "_gh_blob_ref";
const CURRENT_RECORD_SCHEMA_VERSION: u64 = 1;
const BLOB_ENVELOPE_MAGIC: &[u8; 4] = b"GHB0";

const ENTRY_ENVELOPE_MAGIC: &[u8; 4] = b"GHE0";
const ENTRY_ENVELOPE_VERSION: u8 = 1;
const CODEC_RAW: u8 = 0;
const CODEC_GZIP: u8 = 1;
const ENVELOPE_HEADER_LEN: usize = 4 + 1 + 1 + 8 + 32;

/// A task-local, hash-linked evidence frame. It is stored in L0's own redb
/// transaction domain so multiple journal instances cannot allocate the same
/// sequence or append after a terminal seal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskEvidenceFrameRecord {
    pub schema_version: u32,
    pub task_key: String,
    pub task_iri: String,
    pub sequence: u64,
    pub event_iri: String,
    pub event_hash: String,
    /// Serialized journal event. Normal production events contain only hashes,
    /// sizes and identifiers; payload capture remains an explicit caller opt-in.
    pub event_json: String,
    pub previous_frame_hash: Option<String>,
    pub frame_hash: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskEvidenceHead {
    pub next_sequence: u64,
    pub last_frame_hash: Option<String>,
    pub sealed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskEvidenceSealRecord {
    pub schema_version: u32,
    pub task_key: String,
    pub task_iri: String,
    pub frame_count: u64,
    pub root_hash: Option<String>,
    pub terminal_status: String,
    pub sealed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskEvidenceAppendOutcome {
    Appended,
    Conflict,
    Sealed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskEvidenceSealOutcome {
    Sealed,
    Conflict,
    AlreadySealed,
}

/// L0 Store
pub struct L0Store {
    db: Database,
    #[allow(dead_code)]
    config: L0Config,
    #[allow(dead_code)]
    entry_count: u64,
    /// Optional IRI registry reference (auto-registers @id after injection)
    iri_registry: Option<Arc<IriRegistry>>,
}

impl L0Store {
    fn encode_entry(entry: &L0Entry, compress: bool) -> Result<Vec<u8>, CoreError> {
        let raw = serde_json::to_vec(entry).map_err(|e| CoreError::StorageError {
            message: format!("Failed to serialize L0 entry: {e}"),
        })?;
        let (codec, payload) = if compress && !raw.is_empty() {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder
                .write_all(&raw)
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to compress L0 entry: {e}"),
                })?;
            let compressed = encoder.finish().map_err(|e| CoreError::StorageError {
                message: format!("Failed to finish L0 entry compression: {e}"),
            })?;
            // Small JSON records can grow under gzip. Keep a raw envelope in
            // that case while preserving the same versioned disk format.
            if compressed.len() < raw.len() {
                (CODEC_GZIP, compressed)
            } else {
                (CODEC_RAW, raw.clone())
            }
        } else {
            (CODEC_RAW, raw.clone())
        };

        use sha2::Digest;
        let digest = sha2::Sha256::digest(&raw);
        let mut encoded = Vec::with_capacity(ENVELOPE_HEADER_LEN + payload.len());
        encoded.extend_from_slice(ENTRY_ENVELOPE_MAGIC);
        encoded.push(ENTRY_ENVELOPE_VERSION);
        encoded.push(codec);
        encoded.extend_from_slice(&(raw.len() as u64).to_be_bytes());
        encoded.extend_from_slice(&digest);
        encoded.extend_from_slice(&payload);
        Ok(encoded)
    }

    fn decode_entry(bytes: &[u8]) -> Result<L0Entry, CoreError> {
        // Existing databases stored plain JSON. This branch is intentionally
        // retained indefinitely so upgrades are online and non-destructive.
        if !bytes.starts_with(ENTRY_ENVELOPE_MAGIC) {
            return serde_json::from_slice(bytes).map_err(|e| CoreError::StorageError {
                message: format!("Failed to deserialize legacy L0 entry: {e}"),
            });
        }
        if bytes.len() < ENVELOPE_HEADER_LEN {
            return Err(CoreError::StorageError {
                message: "Truncated L0 entry envelope".to_string(),
            });
        }
        let version = bytes[4];
        if version != ENTRY_ENVELOPE_VERSION {
            return Err(CoreError::StorageError {
                message: format!("Unsupported L0 entry envelope version: {version}"),
            });
        }
        let codec = bytes[5];
        let expected_len = u64::from_be_bytes(
            bytes[6..14]
                .try_into()
                .expect("fixed envelope length checked above"),
        ) as usize;
        let expected_digest = &bytes[14..46];
        let payload = &bytes[ENVELOPE_HEADER_LEN..];
        let raw = match codec {
            CODEC_RAW => payload.to_vec(),
            CODEC_GZIP => {
                let mut decoder = GzDecoder::new(payload);
                let mut decoded = Vec::with_capacity(expected_len);
                decoder
                    .read_to_end(&mut decoded)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to decompress L0 entry: {e}"),
                    })?;
                decoded
            }
            other => {
                return Err(CoreError::StorageError {
                    message: format!("Unsupported L0 entry codec: {other}"),
                })
            }
        };
        if raw.len() != expected_len {
            return Err(CoreError::StorageError {
                message: format!(
                    "L0 entry length mismatch: expected={expected_len}, actual={}",
                    raw.len()
                ),
            });
        }
        use sha2::Digest;
        let actual_digest = sha2::Sha256::digest(&raw);
        if actual_digest.as_slice() != expected_digest {
            return Err(CoreError::StorageError {
                message: "L0 entry checksum mismatch".to_string(),
            });
        }
        serde_json::from_slice(&raw).map_err(|e| CoreError::StorageError {
            message: format!("Failed to deserialize L0 entry envelope payload: {e}"),
        })
    }

    fn encode_blob(content: &str, compress: bool) -> Result<Vec<u8>, CoreError> {
        let raw = content.as_bytes();
        let (codec, payload) = if compress && !raw.is_empty() {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder
                .write_all(raw)
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to compress L0 blob: {e}"),
                })?;
            let compressed = encoder.finish().map_err(|e| CoreError::StorageError {
                message: format!("Failed to finish L0 blob compression: {e}"),
            })?;
            if compressed.len() < raw.len() {
                (CODEC_GZIP, compressed)
            } else {
                (CODEC_RAW, raw.to_vec())
            }
        } else {
            (CODEC_RAW, raw.to_vec())
        };
        use sha2::Digest;
        let digest = sha2::Sha256::digest(raw);
        let mut encoded = Vec::with_capacity(ENVELOPE_HEADER_LEN + payload.len());
        encoded.extend_from_slice(BLOB_ENVELOPE_MAGIC);
        encoded.push(ENTRY_ENVELOPE_VERSION);
        encoded.push(codec);
        encoded.extend_from_slice(&(raw.len() as u64).to_be_bytes());
        encoded.extend_from_slice(&digest);
        encoded.extend_from_slice(&payload);
        Ok(encoded)
    }

    fn decode_blob(bytes: &[u8]) -> Result<String, CoreError> {
        if bytes.len() < ENVELOPE_HEADER_LEN || !bytes.starts_with(BLOB_ENVELOPE_MAGIC) {
            return Err(CoreError::StorageError {
                message: "Invalid L0 blob envelope".to_string(),
            });
        }
        if bytes[4] != ENTRY_ENVELOPE_VERSION {
            return Err(CoreError::StorageError {
                message: format!("Unsupported L0 blob version: {}", bytes[4]),
            });
        }
        let expected_len = u64::from_be_bytes(
            bytes[6..14]
                .try_into()
                .expect("fixed blob envelope length checked above"),
        ) as usize;
        let payload = &bytes[ENVELOPE_HEADER_LEN..];
        let raw = match bytes[5] {
            CODEC_RAW => payload.to_vec(),
            CODEC_GZIP => {
                let mut decoder = GzDecoder::new(payload);
                let mut decoded = Vec::with_capacity(expected_len);
                decoder
                    .read_to_end(&mut decoded)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to decompress L0 blob: {e}"),
                    })?;
                decoded
            }
            codec => {
                return Err(CoreError::StorageError {
                    message: format!("Unsupported L0 blob codec: {codec}"),
                })
            }
        };
        if raw.len() != expected_len {
            return Err(CoreError::StorageError {
                message: format!(
                    "L0 blob length mismatch: expected={expected_len}, actual={}",
                    raw.len()
                ),
            });
        }
        use sha2::Digest;
        if sha2::Sha256::digest(&raw).as_slice() != &bytes[14..46] {
            return Err(CoreError::StorageError {
                message: "L0 blob checksum mismatch".to_string(),
            });
        }
        String::from_utf8(raw).map_err(|e| CoreError::StorageError {
            message: format!("L0 blob is not valid UTF-8: {e}"),
        })
    }

    fn entry_record_kind(entry: &L0Entry) -> L0RecordKind {
        entry
            .metadata
            .get(META_RECORD_KIND)
            .and_then(serde_json::Value::as_str)
            .and_then(|value| {
                serde_json::from_value(serde_json::Value::String(value.to_string())).ok()
            })
            .unwrap_or_else(|| L0RecordKind::infer(&entry.iri))
    }

    pub fn entry_generation(entry: &L0Entry) -> u64 {
        entry
            .metadata
            .get(META_GENERATION)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    }

    fn entry_blob_ref(entry: &L0Entry) -> Option<&str> {
        entry
            .metadata
            .get(META_BLOB_REF)
            .and_then(serde_json::Value::as_str)
    }

    fn is_expired(entry: &L0Entry, now: DateTime<Utc>) -> bool {
        entry
            .metadata
            .get(META_EXPIRES_AT)
            .and_then(serde_json::Value::as_str)
            .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
            .is_some_and(|expires| expires.with_timezone(&Utc) <= now)
    }

    fn normalize_lifecycle(
        mut entry: L0Entry,
        existing: Option<&L0Entry>,
        explicit_kind: Option<L0RecordKind>,
        explicit_retention: Option<RetentionClass>,
    ) -> L0Entry {
        let kind = explicit_kind
            .or_else(|| {
                entry
                    .metadata
                    .get(META_RECORD_KIND)
                    .and_then(serde_json::Value::as_str)
                    .and_then(|value| {
                        serde_json::from_value(serde_json::Value::String(value.to_string())).ok()
                    })
            })
            .unwrap_or_else(|| L0RecordKind::infer(&entry.iri));
        let retention = explicit_retention
            .or_else(|| {
                entry
                    .metadata
                    .get(META_RETENTION_CLASS)
                    .and_then(serde_json::Value::as_str)
                    .and_then(|value| {
                        serde_json::from_value(serde_json::Value::String(value.to_string())).ok()
                    })
            })
            .unwrap_or_else(|| RetentionClass::for_kind(kind));
        let previous_generation = existing.map(Self::entry_generation).unwrap_or(0);
        let changed = existing.is_none_or(|old| {
            old.content_hash != entry.content_hash
                || old.tags != entry.tags
                || old.named_graph != entry.named_graph
                || old.jsonld_types != entry.jsonld_types
        });
        let computed_generation = if previous_generation == 0 {
            1
        } else if changed {
            previous_generation.saturating_add(1)
        } else {
            previous_generation
        };
        let requested_generation = entry
            .metadata
            .get(META_GENERATION)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let generation = computed_generation.max(requested_generation);
        entry.metadata.insert(
            META_SCHEMA_VERSION.to_string(),
            serde_json::Value::from(CURRENT_RECORD_SCHEMA_VERSION),
        );
        entry.metadata.insert(
            META_RECORD_KIND.to_string(),
            serde_json::Value::String(kind.as_str().to_string()),
        );
        entry.metadata.insert(
            META_RETENTION_CLASS.to_string(),
            serde_json::Value::String(retention.as_str().to_string()),
        );
        entry.metadata.insert(
            META_GENERATION.to_string(),
            serde_json::Value::from(generation),
        );
        entry.metadata.insert(
            META_UPDATED_AT.to_string(),
            serde_json::Value::String(Utc::now().to_rfc3339()),
        );
        entry
    }

    fn hydrate_entry(&self, mut entry: L0Entry) -> Result<L0Entry, CoreError> {
        let Some(blob_ref) = Self::entry_blob_ref(&entry).map(str::to_string) else {
            return Ok(entry);
        };
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Failed to begin L0 blob read: {e}"),
        })?;
        let blobs = read_txn
            .open_table(BLOB_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open L0 blob table: {e}"),
            })?;
        let value = blobs
            .get(blob_ref.as_str())
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to read L0 blob {blob_ref}: {e}"),
            })?
            .ok_or_else(|| CoreError::StorageError {
                message: format!("Missing L0 content blob: {blob_ref}"),
            })?;
        entry.content = Self::decode_blob(value.value())?;
        Ok(entry)
    }

    pub fn new(path: &str) -> Result<Self, CoreError> {
        Self::with_config(L0Config {
            path: path.to_string(),
            ..Default::default()
        })
    }

    pub fn with_config(config: L0Config) -> Result<Self, CoreError> {
        Self::with_config_and_repair_callback(config, None)
    }

    pub fn with_config_and_repair_callback(
        config: L0Config,
        repair_callback: Option<Arc<dyn Fn(f64) + Send + Sync>>,
    ) -> Result<Self, CoreError> {
        info!("Initializing L0 Store: {}", config.path);

        std::fs::create_dir_all(&config.path).map_err(|e| CoreError::StorageError {
            message: format!("Failed to create storage directory: {}", e),
        })?;

        let db_path = std::path::Path::new(&config.path).join("l0.redb");
        let mut builder = Database::builder();
        builder.set_cache_size(config.cache_size_bytes);
        if let Some(callback) = repair_callback {
            builder.set_repair_callback(move |session| callback(session.progress()));
        }
        let db = builder
            .create(&db_path)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open database: {}", e),
            })?;

        // Ensure tables exist by opening them in a write transaction
        {
            let mut write_txn = db.begin_write().map_err(|e| CoreError::StorageError {
                message: format!("Failed to begin write transaction: {}", e),
            })?;
            write_txn.set_quick_repair(config.quick_repair);
            let _ = write_txn.open_table(ENTRIES_TABLE);
            let _ = write_txn.open_table(TAG_INDEX_TABLE);
            let _ = write_txn.open_table(NAMED_GRAPH_TABLE);
            let _ = write_txn.open_table(TYPE_INDEX_TABLE);
            let _ = write_txn.open_table(RECORD_KIND_INDEX_TABLE);
            let _ = write_txn.open_table(BLOB_TABLE);
            let _ = write_txn.open_table(BLOB_REFCOUNT_TABLE);
            let _ = write_txn.open_table(TASK_EVIDENCE_FRAME_TABLE);
            let _ = write_txn.open_table(TASK_EVIDENCE_HEAD_TABLE);
            let _ = write_txn.open_table(TASK_EVIDENCE_SEAL_TABLE);
            write_txn.commit().map_err(|e| CoreError::StorageError {
                message: format!("Failed to commit transaction: {}", e),
            })?;
        }

        let entry_count = {
            let read_txn = db.begin_read().map_err(|e| CoreError::StorageError {
                message: format!("Failed to begin read transaction: {}", e),
            })?;
            let table =
                read_txn
                    .open_table(ENTRIES_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open table: {}", e),
                    })?;
            table.len().map_err(|e| CoreError::StorageError {
                message: format!("Failed to get entry count: {}", e),
            })?
        };

        Ok(Self {
            db,
            config,
            entry_count,
            iri_registry: None,
        })
    }

    fn begin_write(&self, error_context: &str) -> Result<redb::WriteTransaction, CoreError> {
        let mut transaction = self
            .db
            .begin_write()
            .map_err(|error| CoreError::StorageError {
                message: format!("{error_context}: {error}"),
            })?;
        transaction.set_quick_repair(self.config.quick_repair);
        Ok(transaction)
    }

    fn evidence_head_from_table<T>(table: &T, task_key: &str) -> Result<TaskEvidenceHead, CoreError>
    where
        T: ReadableTable<&'static str, &'static [u8]>,
    {
        let value = table
            .get(task_key)
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to read task evidence head: {error}"),
            })?;
        match value {
            Some(value) => {
                serde_json::from_slice(value.value()).map_err(|error| CoreError::StorageError {
                    message: format!("Failed to decode task evidence head: {error}"),
                })
            }
            None => Ok(TaskEvidenceHead {
                next_sequence: 0,
                last_frame_hash: None,
                sealed: false,
            }),
        }
    }

    /// Read a task evidence head without mutating access metadata.
    pub fn task_evidence_head(&self, task_key: &str) -> Result<TaskEvidenceHead, CoreError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to begin task evidence read: {error}"),
            })?;
        let table = read_txn
            .open_table(TASK_EVIDENCE_HEAD_TABLE)
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to open task evidence head table: {error}"),
            })?;
        Self::evidence_head_from_table(&table, task_key)
    }

    /// Atomically append a hash-linked frame if its observed head is still
    /// current. Callers retry a `Conflict` after reading the new head; a
    /// terminal seal is a permanent append barrier.
    pub fn try_append_task_evidence(
        &self,
        frame: &TaskEvidenceFrameRecord,
    ) -> Result<TaskEvidenceAppendOutcome, CoreError> {
        let write_txn = self.begin_write("Failed to begin task evidence append")?;
        {
            let mut heads = write_txn
                .open_table(TASK_EVIDENCE_HEAD_TABLE)
                .map_err(|error| CoreError::StorageError {
                    message: format!("Failed to open task evidence head table: {error}"),
                })?;
            let head = Self::evidence_head_from_table(&heads, &frame.task_key)?;
            if head.sealed {
                return Ok(TaskEvidenceAppendOutcome::Sealed);
            }
            if head.next_sequence != frame.sequence
                || head.last_frame_hash != frame.previous_frame_hash
            {
                return Ok(TaskEvidenceAppendOutcome::Conflict);
            }
            let frame_key = format!("{}/seq_{:020}", frame.task_key, frame.sequence);
            let mut frames = write_txn
                .open_table(TASK_EVIDENCE_FRAME_TABLE)
                .map_err(|error| CoreError::StorageError {
                    message: format!("Failed to open task evidence frame table: {error}"),
                })?;
            if frames
                .get(frame_key.as_str())
                .map_err(|error| CoreError::StorageError {
                    message: format!("Failed to read task evidence frame: {error}"),
                })?
                .is_some()
            {
                return Ok(TaskEvidenceAppendOutcome::Conflict);
            }
            let encoded = serde_json::to_vec(frame).map_err(|error| CoreError::StorageError {
                message: format!("Failed to encode task evidence frame: {error}"),
            })?;
            frames
                .insert(frame_key.as_str(), encoded.as_slice())
                .map_err(|error| CoreError::StorageError {
                    message: format!("Failed to append task evidence frame: {error}"),
                })?;
            let next_head = TaskEvidenceHead {
                next_sequence: frame.sequence.saturating_add(1),
                last_frame_hash: Some(frame.frame_hash.clone()),
                sealed: false,
            };
            let encoded_head =
                serde_json::to_vec(&next_head).map_err(|error| CoreError::StorageError {
                    message: format!("Failed to encode task evidence head: {error}"),
                })?;
            heads
                .insert(frame.task_key.as_str(), encoded_head.as_slice())
                .map_err(|error| CoreError::StorageError {
                    message: format!("Failed to update task evidence head: {error}"),
                })?;
        }
        write_txn
            .commit()
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to commit task evidence append: {error}"),
            })?;
        Ok(TaskEvidenceAppendOutcome::Appended)
    }

    /// Atomically seal a task's evidence chain. The caller must provide the
    /// currently observed count/root; a concurrent append causes `Conflict`.
    pub fn try_seal_task_evidence(
        &self,
        seal: &TaskEvidenceSealRecord,
    ) -> Result<TaskEvidenceSealOutcome, CoreError> {
        let write_txn = self.begin_write("Failed to begin task evidence seal")?;
        {
            let mut heads = write_txn
                .open_table(TASK_EVIDENCE_HEAD_TABLE)
                .map_err(|error| CoreError::StorageError {
                    message: format!("Failed to open task evidence head table: {error}"),
                })?;
            let head = Self::evidence_head_from_table(&heads, &seal.task_key)?;
            if head.sealed {
                return Ok(TaskEvidenceSealOutcome::AlreadySealed);
            }
            if head.next_sequence != seal.frame_count || head.last_frame_hash != seal.root_hash {
                return Ok(TaskEvidenceSealOutcome::Conflict);
            }
            let mut seals = write_txn
                .open_table(TASK_EVIDENCE_SEAL_TABLE)
                .map_err(|error| CoreError::StorageError {
                    message: format!("Failed to open task evidence seal table: {error}"),
                })?;
            if seals
                .get(seal.task_key.as_str())
                .map_err(|error| CoreError::StorageError {
                    message: format!("Failed to read task evidence seal: {error}"),
                })?
                .is_some()
            {
                return Ok(TaskEvidenceSealOutcome::AlreadySealed);
            }
            let encoded = serde_json::to_vec(seal).map_err(|error| CoreError::StorageError {
                message: format!("Failed to encode task evidence seal: {error}"),
            })?;
            seals
                .insert(seal.task_key.as_str(), encoded.as_slice())
                .map_err(|error| CoreError::StorageError {
                    message: format!("Failed to write task evidence seal: {error}"),
                })?;
            let sealed_head = TaskEvidenceHead {
                sealed: true,
                ..head
            };
            let encoded_head =
                serde_json::to_vec(&sealed_head).map_err(|error| CoreError::StorageError {
                    message: format!("Failed to encode sealed evidence head: {error}"),
                })?;
            heads
                .insert(seal.task_key.as_str(), encoded_head.as_slice())
                .map_err(|error| CoreError::StorageError {
                    message: format!("Failed to seal task evidence head: {error}"),
                })?;
        }
        write_txn
            .commit()
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to commit task evidence seal: {error}"),
            })?;
        Ok(TaskEvidenceSealOutcome::Sealed)
    }

    pub fn task_evidence_frames(
        &self,
        task_key: &str,
        limit: usize,
    ) -> Result<Vec<TaskEvidenceFrameRecord>, CoreError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to begin task evidence frame read: {error}"),
            })?;
        let table = read_txn
            .open_table(TASK_EVIDENCE_FRAME_TABLE)
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to open task evidence frame table: {error}"),
            })?;
        let prefix = format!("{task_key}/");
        let mut frames = Vec::new();
        for result in table
            .range(prefix.as_str()..)
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to iterate task evidence frames: {error}"),
            })?
        {
            let (key, value) = result.map_err(|error| CoreError::StorageError {
                message: format!("Failed to read task evidence frame: {error}"),
            })?;
            if !key.value().starts_with(&prefix) {
                break;
            }
            let frame =
                serde_json::from_slice(value.value()).map_err(|error| CoreError::StorageError {
                    message: format!("Failed to decode task evidence frame: {error}"),
                })?;
            frames.push(frame);
            if frames.len() >= limit.max(1) {
                break;
            }
        }
        Ok(frames)
    }

    pub fn task_evidence_seal(
        &self,
        task_key: &str,
    ) -> Result<Option<TaskEvidenceSealRecord>, CoreError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to begin task evidence seal read: {error}"),
            })?;
        let table = read_txn
            .open_table(TASK_EVIDENCE_SEAL_TABLE)
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to open task evidence seal table: {error}"),
            })?;
        table
            .get(task_key)
            .map_err(|error| CoreError::StorageError {
                message: format!("Failed to read task evidence seal: {error}"),
            })?
            .map(|value| {
                serde_json::from_slice(value.value()).map_err(|error| CoreError::StorageError {
                    message: format!("Failed to decode task evidence seal: {error}"),
                })
            })
            .transpose()
    }

    pub fn store(&self, iri: &str, content: &str) -> Result<(), CoreError> {
        let content_hash = compute_content_hash(content);
        self.write_entry_atomic(
            &L0Entry {
                iri: iri.to_string(),
                content: content.to_string(),
                importance: 0.5,
                access_count: 0,
                created_at: Utc::now(),
                last_accessed: Utc::now(),
                tags: Vec::new(),
                metadata: serde_json::Map::new(),
                mesi_state: MesiState::Shared,
                content_hash,
                named_graph: None,
                jsonld_context: None,
                jsonld_types: Vec::new(),
            },
            true,
            None,
            None,
        )
    }

    fn retrieve_without_update(&self, iri: &str) -> Result<Option<L0Entry>, CoreError> {
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table = read_txn
            .open_table(ENTRIES_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open table: {}", e),
            })?;
        match table.get(iri).map_err(|e| CoreError::StorageError {
            message: format!("Failed to retrieve entry: {}", e),
        })? {
            Some(guard) => {
                let entry = self.hydrate_entry(Self::decode_entry(guard.value())?)?;
                if Self::is_expired(&entry, Utc::now()) {
                    Ok(None)
                } else {
                    Ok(Some(entry))
                }
            }
            None => Ok(None),
        }
    }

    fn merge_entries(existing: &L0Entry, new: &L0Entry) -> L0Entry {
        let mut merged_metadata = existing.metadata.clone();
        for (key, value) in &new.metadata {
            merged_metadata.insert(key.clone(), value.clone());
        }

        let mut merged_tags = existing.tags.clone();
        for tag in &new.tags {
            if !merged_tags.contains(tag) {
                merged_tags.push(tag.clone());
            }
        }

        let mut merged_types = existing.jsonld_types.clone();
        for type_iri in &new.jsonld_types {
            if !merged_types.contains(type_iri) {
                merged_types.push(type_iri.clone());
            }
        }

        L0Entry {
            iri: existing.iri.clone(),
            content: new.content.clone(),
            importance: (existing.importance + new.importance) / 2.0,
            access_count: existing.access_count,
            created_at: existing.created_at,
            last_accessed: Utc::now(),
            tags: merged_tags,
            metadata: merged_metadata,
            mesi_state: new.mesi_state.clone(),
            content_hash: new.content_hash.clone(),
            named_graph: existing.named_graph.clone().or(new.named_graph.clone()),

            jsonld_context: new
                .jsonld_context
                .clone()
                .or(existing.jsonld_context.clone()),
            jsonld_types: merged_types,
        }
    }

    fn write_entry_atomic(
        &self,
        entry: &L0Entry,
        merge: bool,
        explicit_kind: Option<L0RecordKind>,
        explicit_retention: Option<RetentionClass>,
    ) -> Result<(), CoreError> {
        let content_hash = compute_content_hash(&entry.content);

        let write_txn = self.begin_write("Write transaction failed")?;
        {
            let mut entries =
                write_txn
                    .open_table(ENTRIES_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open table: {}", e),
                    })?;
            let existing =
                match entries
                    .get(entry.iri.as_str())
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to read existing entry: {}", e),
                    })? {
                    Some(guard) => Some(self.hydrate_entry(Self::decode_entry(guard.value())?)?),
                    None => None,
                };

            if existing.is_none()
                && self.config.max_entries > 0
                && entries.len().map_err(|e| CoreError::StorageError {
                    message: format!("Failed to count entries: {}", e),
                })? as usize
                    >= self.config.max_entries
            {
                return Err(CoreError::StorageError {
                    message: format!(
                        "L0 entry capacity reached: max_entries={}",
                        self.config.max_entries
                    ),
                });
            }

            let incoming = L0Entry {
                content_hash,
                ..entry.clone()
            };
            let final_entry = if merge {
                existing
                    .as_ref()
                    .map(|old| Self::merge_entries(old, &incoming))
                    .unwrap_or(incoming)
            } else {
                incoming
            };
            let mut final_entry = Self::normalize_lifecycle(
                final_entry,
                existing.as_ref(),
                explicit_kind,
                explicit_retention,
            );
            let old_tags = existing
                .as_ref()
                .map(|old| old.tags.clone())
                .unwrap_or_default();
            let old_graph = existing.as_ref().and_then(|old| old.named_graph.clone());
            let old_types = existing
                .as_ref()
                .map(|old| old.jsonld_types.clone())
                .unwrap_or_default();
            let old_kind = existing.as_ref().map(Self::entry_record_kind);
            let final_kind = Self::entry_record_kind(&final_entry);

            let mut tags =
                write_txn
                    .open_table(TAG_INDEX_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open tag index: {}", e),
                    })?;
            for tag in &old_tags {
                let key = format!("tag:{}", tag);
                let mut iris =
                    match tags
                        .get(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to read tag index: {}", e),
                        })? {
                        Some(guard) => {
                            serde_json::from_slice::<Vec<String>>(guard.value()).unwrap_or_default()
                        }
                        None => Vec::new(),
                    };
                iris.retain(|candidate| candidate != &final_entry.iri);
                if iris.is_empty() {
                    tags.remove(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to remove tag index: {}", e),
                        })?;
                } else {
                    let encoded =
                        serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                            message: format!("Failed to serialize tag index: {}", e),
                        })?;
                    tags.insert(key.as_str(), encoded.as_slice()).map_err(|e| {
                        CoreError::StorageError {
                            message: format!("Failed to update tag index: {}", e),
                        }
                    })?;
                }
            }
            for tag in &final_entry.tags {
                let key = format!("tag:{}", tag);
                let mut iris =
                    match tags
                        .get(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to read tag index: {}", e),
                        })? {
                        Some(guard) => {
                            serde_json::from_slice::<Vec<String>>(guard.value()).unwrap_or_default()
                        }
                        None => Vec::new(),
                    };
                if !iris.contains(&final_entry.iri) {
                    iris.push(final_entry.iri.clone());
                }
                let encoded = serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                    message: format!("Failed to serialize tag index: {}", e),
                })?;
                tags.insert(key.as_str(), encoded.as_slice()).map_err(|e| {
                    CoreError::StorageError {
                        message: format!("Failed to update tag index: {}", e),
                    }
                })?;
            }

            let mut graphs =
                write_txn
                    .open_table(NAMED_GRAPH_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open named graph index: {}", e),
                    })?;
            if let Some(graph) = old_graph {
                let key = format!("graph:{}", graph);
                let mut iris =
                    match graphs
                        .get(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to read named graph index: {}", e),
                        })? {
                        Some(guard) => {
                            serde_json::from_slice::<Vec<String>>(guard.value()).unwrap_or_default()
                        }
                        None => Vec::new(),
                    };
                iris.retain(|candidate| candidate != &final_entry.iri);
                if iris.is_empty() {
                    graphs
                        .remove(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to remove named graph index: {}", e),
                        })?;
                } else {
                    let encoded =
                        serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                            message: format!("Failed to serialize named graph index: {}", e),
                        })?;
                    graphs
                        .insert(key.as_str(), encoded.as_slice())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to update named graph index: {}", e),
                        })?;
                }
            }
            if let Some(graph) = &final_entry.named_graph {
                let key = format!("graph:{}", graph);
                let mut iris =
                    match graphs
                        .get(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to read named graph index: {}", e),
                        })? {
                        Some(guard) => {
                            serde_json::from_slice::<Vec<String>>(guard.value()).unwrap_or_default()
                        }
                        None => Vec::new(),
                    };
                if !iris.contains(&final_entry.iri) {
                    iris.push(final_entry.iri.clone());
                }
                let encoded = serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                    message: format!("Failed to serialize named graph index: {}", e),
                })?;
                graphs
                    .insert(key.as_str(), encoded.as_slice())
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to update named graph index: {}", e),
                    })?;
            }

            let mut types =
                write_txn
                    .open_table(TYPE_INDEX_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open type index: {}", e),
                    })?;
            for type_iri in old_types {
                let key = format!("type:{}", type_iri);
                let mut iris =
                    match types
                        .get(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to read type index: {}", e),
                        })? {
                        Some(guard) => {
                            serde_json::from_slice::<Vec<String>>(guard.value()).unwrap_or_default()
                        }
                        None => Vec::new(),
                    };
                iris.retain(|candidate| candidate != &final_entry.iri);
                if iris.is_empty() {
                    types
                        .remove(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to remove type index: {}", e),
                        })?;
                } else {
                    let encoded =
                        serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                            message: format!("Failed to serialize type index: {}", e),
                        })?;
                    types
                        .insert(key.as_str(), encoded.as_slice())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to update type index: {}", e),
                        })?;
                }
            }
            for type_iri in &final_entry.jsonld_types {
                let key = format!("type:{}", type_iri);
                let mut iris =
                    match types
                        .get(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to read type index: {}", e),
                        })? {
                        Some(guard) => {
                            serde_json::from_slice::<Vec<String>>(guard.value()).unwrap_or_default()
                        }
                        None => Vec::new(),
                    };
                if !iris.contains(&final_entry.iri) {
                    iris.push(final_entry.iri.clone());
                }
                let encoded = serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                    message: format!("Failed to serialize type index: {}", e),
                })?;
                types
                    .insert(key.as_str(), encoded.as_slice())
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to update type index: {}", e),
                    })?;
            }

            let mut kind_index = write_txn.open_table(RECORD_KIND_INDEX_TABLE).map_err(|e| {
                CoreError::StorageError {
                    message: format!("Failed to open record-kind index: {e}"),
                }
            })?;
            if let Some(old_kind) = old_kind {
                let key = format!("kind:{}", old_kind.as_str());
                let mut iris =
                    match kind_index
                        .get(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to read record-kind index: {e}"),
                        })? {
                        Some(value) => {
                            serde_json::from_slice::<Vec<String>>(value.value()).unwrap_or_default()
                        }
                        None => Vec::new(),
                    };
                iris.retain(|iri| iri != &final_entry.iri);
                if iris.is_empty() {
                    kind_index
                        .remove(key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to remove record-kind index: {e}"),
                        })?;
                } else {
                    let encoded =
                        serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                            message: format!("Failed to encode record-kind index: {e}"),
                        })?;
                    kind_index
                        .insert(key.as_str(), encoded.as_slice())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to update record-kind index: {e}"),
                        })?;
                }
            }
            let kind_key = format!("kind:{}", final_kind.as_str());
            let mut iris =
                match kind_index
                    .get(kind_key.as_str())
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to read record-kind index: {e}"),
                    })? {
                    Some(value) => {
                        serde_json::from_slice::<Vec<String>>(value.value()).unwrap_or_default()
                    }
                    None => Vec::new(),
                };
            if !iris.contains(&final_entry.iri) {
                iris.push(final_entry.iri.clone());
            }
            let encoded = serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                message: format!("Failed to encode record-kind index: {e}"),
            })?;
            kind_index
                .insert(kind_key.as_str(), encoded.as_slice())
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to update record-kind index: {e}"),
                })?;

            let old_blob_ref = existing
                .as_ref()
                .and_then(Self::entry_blob_ref)
                .map(str::to_string);
            final_entry.metadata.remove(META_BLOB_REF);
            let new_blob_ref = (final_entry.content.len() >= self.config.blob_inline_threshold)
                .then(|| final_entry.content_hash.clone());
            let mut stored_entry = final_entry.clone();
            if let Some(blob_ref) = &new_blob_ref {
                stored_entry.metadata.insert(
                    META_BLOB_REF.to_string(),
                    serde_json::Value::String(blob_ref.clone()),
                );
                stored_entry.content.clear();
            }

            let mut blobs =
                write_txn
                    .open_table(BLOB_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open content blob table: {e}"),
                    })?;
            let mut refs =
                write_txn
                    .open_table(BLOB_REFCOUNT_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open blob refcount table: {e}"),
                    })?;
            if old_blob_ref != new_blob_ref {
                if let Some(old_ref) = old_blob_ref.as_deref() {
                    let count = refs
                        .get(old_ref)
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to read old blob refcount: {e}"),
                        })?
                        .and_then(|value| value.value().try_into().ok().map(u64::from_be_bytes))
                        .unwrap_or(1);
                    if count <= 1 {
                        refs.remove(old_ref).map_err(|e| CoreError::StorageError {
                            message: format!("Failed to remove blob refcount: {e}"),
                        })?;
                        blobs.remove(old_ref).map_err(|e| CoreError::StorageError {
                            message: format!("Failed to remove unreferenced blob: {e}"),
                        })?;
                    } else {
                        let next = (count - 1).to_be_bytes();
                        refs.insert(old_ref, next.as_slice()).map_err(|e| {
                            CoreError::StorageError {
                                message: format!("Failed to decrement blob refcount: {e}"),
                            }
                        })?;
                    }
                }
                if let Some(new_ref) = new_blob_ref.as_deref() {
                    if blobs
                        .get(new_ref)
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to look up content blob: {e}"),
                        })?
                        .is_none()
                    {
                        let blob =
                            Self::encode_blob(&final_entry.content, self.config.compression)?;
                        blobs.insert(new_ref, blob.as_slice()).map_err(|e| {
                            CoreError::StorageError {
                                message: format!("Failed to store content blob: {e}"),
                            }
                        })?;
                    }
                    let count = refs
                        .get(new_ref)
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to read blob refcount: {e}"),
                        })?
                        .and_then(|value| value.value().try_into().ok().map(u64::from_be_bytes))
                        .unwrap_or(0)
                        .saturating_add(1);
                    let encoded = count.to_be_bytes();
                    refs.insert(new_ref, encoded.as_slice()).map_err(|e| {
                        CoreError::StorageError {
                            message: format!("Failed to increment blob refcount: {e}"),
                        }
                    })?;
                }
            } else if let Some(blob_ref) = new_blob_ref.as_deref() {
                // Repair an incomplete blob table without changing refcount.
                if blobs
                    .get(blob_ref)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to verify content blob: {e}"),
                    })?
                    .is_none()
                {
                    let blob = Self::encode_blob(&final_entry.content, self.config.compression)?;
                    blobs.insert(blob_ref, blob.as_slice()).map_err(|e| {
                        CoreError::StorageError {
                            message: format!("Failed to repair content blob: {e}"),
                        }
                    })?;
                }
            }

            let value = Self::encode_entry(&stored_entry, self.config.compression)?;
            entries
                .insert(final_entry.iri.as_str(), value.as_slice())
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to store entry: {}", e),
                })?;
        }
        write_txn.commit().map_err(|e| CoreError::StorageError {
            message: format!("Failed to commit transaction: {}", e),
        })?;

        debug!(iri = %entry.iri, merge, "Entry stored to L0 atomically");
        Ok(())
    }

    /// Merge an entry with an existing entity. This compatibility API retains
    /// accumulated tags/types/metadata. Use `replace_entry` for exact updates.
    pub fn store_entry(&self, entry: &L0Entry) -> Result<(), CoreError> {
        self.write_entry_atomic(entry, true, None, None)
    }

    /// Replace an entry exactly while atomically migrating all secondary indices.
    pub fn replace_entry(&self, entry: &L0Entry) -> Result<(), CoreError> {
        self.write_entry_atomic(entry, false, None, None)
    }

    /// Store through the unified L0 API while explicitly selecting the
    /// lifecycle partition and optional expiration time.
    pub fn store_with_policy(
        &self,
        entry: &L0Entry,
        kind: L0RecordKind,
        retention: RetentionClass,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<(), CoreError> {
        let mut classified = entry.clone();
        match expires_at {
            Some(value) => {
                classified.metadata.insert(
                    META_EXPIRES_AT.to_string(),
                    serde_json::Value::String(value.to_rfc3339()),
                );
            }
            None => {
                classified.metadata.remove(META_EXPIRES_AT);
            }
        }
        self.write_entry_atomic(&classified, false, Some(kind), Some(retention))
    }

    pub fn retrieve(&self, iri: &str) -> Result<Option<L0Entry>, CoreError> {
        self.retrieve_without_update(iri)
    }

    /// Retrieve and synchronously persist access metadata. Normal reads use
    /// `retrieve()` and remain read-only to avoid write amplification.
    pub fn retrieve_and_touch(&self, iri: &str) -> Result<Option<L0Entry>, CoreError> {
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table = read_txn
            .open_table(ENTRIES_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open table: {}", e),
            })?;
        let value = table.get(iri).map_err(|e| CoreError::StorageError {
            message: format!("Failed to retrieve entry: {}", e),
        })?;

        match value {
            Some(guard) => {
                let mut entry = self.hydrate_entry(Self::decode_entry(guard.value())?)?;
                drop(read_txn);

                if Self::is_expired(&entry, Utc::now()) {
                    return Ok(None);
                }

                entry.access_count += 1;
                entry.last_accessed = Utc::now();
                self.replace_entry(&entry)?;

                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    pub fn delete(&self, iri: &str) -> Result<bool, CoreError> {
        let write_txn = self.begin_write("Write transaction failed")?;
        let removed =
            {
                let mut entries =
                    write_txn
                        .open_table(ENTRIES_TABLE)
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to open table: {}", e),
                        })?;
                let existing = match entries.get(iri).map_err(|e| CoreError::StorageError {
                    message: format!("Failed to read entry before delete: {}", e),
                })? {
                    Some(guard) => Some(Self::decode_entry(guard.value())?),
                    None => None,
                };
                let Some(existing) = existing else {
                    return Ok(false);
                };

                let mut tags =
                    write_txn
                        .open_table(TAG_INDEX_TABLE)
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to open tag index: {}", e),
                        })?;
                for tag in &existing.tags {
                    let key = format!("tag:{}", tag);
                    let mut iris =
                        match tags
                            .get(key.as_str())
                            .map_err(|e| CoreError::StorageError {
                                message: format!("Failed to read tag index: {}", e),
                            })? {
                            Some(guard) => serde_json::from_slice::<Vec<String>>(guard.value())
                                .unwrap_or_default(),
                            None => Vec::new(),
                        };
                    iris.retain(|candidate| candidate != iri);
                    if iris.is_empty() {
                        tags.remove(key.as_str())
                            .map_err(|e| CoreError::StorageError {
                                message: format!("Failed to remove tag index: {}", e),
                            })?;
                    } else {
                        let encoded =
                            serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                                message: format!("Failed to serialize tag index: {}", e),
                            })?;
                        tags.insert(key.as_str(), encoded.as_slice()).map_err(|e| {
                            CoreError::StorageError {
                                message: format!("Failed to update tag index: {}", e),
                            }
                        })?;
                    }
                }

                if let Some(graph) = &existing.named_graph {
                    let mut graphs = write_txn.open_table(NAMED_GRAPH_TABLE).map_err(|e| {
                        CoreError::StorageError {
                            message: format!("Failed to open named graph index: {}", e),
                        }
                    })?;
                    let key = format!("graph:{}", graph);
                    let mut iris =
                        match graphs
                            .get(key.as_str())
                            .map_err(|e| CoreError::StorageError {
                                message: format!("Failed to read named graph index: {}", e),
                            })? {
                            Some(guard) => serde_json::from_slice::<Vec<String>>(guard.value())
                                .unwrap_or_default(),
                            None => Vec::new(),
                        };
                    iris.retain(|candidate| candidate != iri);
                    if iris.is_empty() {
                        graphs
                            .remove(key.as_str())
                            .map_err(|e| CoreError::StorageError {
                                message: format!("Failed to remove named graph index: {}", e),
                            })?;
                    } else {
                        let encoded =
                            serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                                message: format!("Failed to serialize named graph index: {}", e),
                            })?;
                        graphs
                            .insert(key.as_str(), encoded.as_slice())
                            .map_err(|e| CoreError::StorageError {
                                message: format!("Failed to update named graph index: {}", e),
                            })?;
                    }
                }

                let mut types = write_txn.open_table(TYPE_INDEX_TABLE).map_err(|e| {
                    CoreError::StorageError {
                        message: format!("Failed to open type index: {}", e),
                    }
                })?;
                for type_iri in &existing.jsonld_types {
                    let key = format!("type:{}", type_iri);
                    let mut iris =
                        match types
                            .get(key.as_str())
                            .map_err(|e| CoreError::StorageError {
                                message: format!("Failed to read type index: {}", e),
                            })? {
                            Some(guard) => serde_json::from_slice::<Vec<String>>(guard.value())
                                .unwrap_or_default(),
                            None => Vec::new(),
                        };
                    iris.retain(|candidate| candidate != iri);
                    if iris.is_empty() {
                        types
                            .remove(key.as_str())
                            .map_err(|e| CoreError::StorageError {
                                message: format!("Failed to remove type index: {}", e),
                            })?;
                    } else {
                        let encoded =
                            serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                                message: format!("Failed to serialize type index: {}", e),
                            })?;
                        types
                            .insert(key.as_str(), encoded.as_slice())
                            .map_err(|e| CoreError::StorageError {
                                message: format!("Failed to update type index: {}", e),
                            })?;
                    }
                }

                let kind = Self::entry_record_kind(&existing);
                let kind_key = format!("kind:{}", kind.as_str());
                let mut kind_index =
                    write_txn.open_table(RECORD_KIND_INDEX_TABLE).map_err(|e| {
                        CoreError::StorageError {
                            message: format!("Failed to open record-kind index: {e}"),
                        }
                    })?;
                let mut kind_iris = match kind_index.get(kind_key.as_str()).map_err(|e| {
                    CoreError::StorageError {
                        message: format!("Failed to read record-kind index: {e}"),
                    }
                })? {
                    Some(value) => {
                        serde_json::from_slice::<Vec<String>>(value.value()).unwrap_or_default()
                    }
                    None => Vec::new(),
                };
                kind_iris.retain(|candidate| candidate != iri);
                if kind_iris.is_empty() {
                    kind_index
                        .remove(kind_key.as_str())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to remove record-kind index: {e}"),
                        })?;
                } else {
                    let encoded =
                        serde_json::to_vec(&kind_iris).map_err(|e| CoreError::StorageError {
                            message: format!("Failed to encode record-kind index: {e}"),
                        })?;
                    kind_index
                        .insert(kind_key.as_str(), encoded.as_slice())
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to update record-kind index: {e}"),
                        })?;
                }

                if let Some(blob_ref) = Self::entry_blob_ref(&existing) {
                    let mut refs = write_txn.open_table(BLOB_REFCOUNT_TABLE).map_err(|e| {
                        CoreError::StorageError {
                            message: format!("Failed to open blob refcount table: {e}"),
                        }
                    })?;
                    let mut blobs =
                        write_txn
                            .open_table(BLOB_TABLE)
                            .map_err(|e| CoreError::StorageError {
                                message: format!("Failed to open content blob table: {e}"),
                            })?;
                    let count = refs
                        .get(blob_ref)
                        .map_err(|e| CoreError::StorageError {
                            message: format!("Failed to read blob refcount: {e}"),
                        })?
                        .and_then(|value| value.value().try_into().ok().map(u64::from_be_bytes))
                        .unwrap_or(1);
                    if count <= 1 {
                        refs.remove(blob_ref).map_err(|e| CoreError::StorageError {
                            message: format!("Failed to remove blob refcount: {e}"),
                        })?;
                        blobs
                            .remove(blob_ref)
                            .map_err(|e| CoreError::StorageError {
                                message: format!("Failed to remove unreferenced blob: {e}"),
                            })?;
                    } else {
                        let next = (count - 1).to_be_bytes();
                        refs.insert(blob_ref, next.as_slice()).map_err(|e| {
                            CoreError::StorageError {
                                message: format!("Failed to decrement blob refcount: {e}"),
                            }
                        })?;
                    }
                }

                let has_removed = match entries.remove(iri) {
                    Ok(Some(_)) => true,
                    Ok(None) => false,
                    Err(e) => {
                        return Err(CoreError::StorageError {
                            message: format!("Failed to delete entry: {}", e),
                        });
                    }
                };
                has_removed
            };
        write_txn.commit().map_err(|e| CoreError::StorageError {
            message: format!("Failed to commit transaction: {}", e),
        })?;
        Ok(removed)
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<L0SearchResult>, CoreError> {
        let mut results = Vec::new();
        let query_lower = query.to_lowercase();

        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table = read_txn
            .open_table(ENTRIES_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open table: {}", e),
            })?;
        for result in table.iter().map_err(|e| CoreError::StorageError {
            message: format!("Iteration failed: {}", e),
        })? {
            let (_, value) = result.map_err(|e| CoreError::StorageError {
                message: format!("Iteration failed: {}", e),
            })?;

            let entry = self.hydrate_entry(Self::decode_entry(value.value())?)?;
            if Self::is_expired(&entry, Utc::now()) {
                continue;
            }

            let content_lower = entry.content.to_lowercase();
            let tag_match = entry
                .tags
                .iter()
                .any(|t| t.to_lowercase().contains(&query_lower));
            let content_match = content_lower.contains(&query_lower);

            if tag_match || content_match {
                let relevance = if content_match { 0.8 } else { 0.5 };
                results.push(L0SearchResult {
                    iri: entry.iri,
                    content: entry.content,
                    relevance_score: relevance,
                    importance: entry.importance,
                    tags: entry.tags,
                });
            }

            if results.len() >= limit {
                break;
            }
        }

        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    /// Scan by IRI prefix — uses redb key-order iteration, more efficient and reliable than search() content matching
    pub fn scan_iri_prefix(&self, prefix: &str, limit: usize) -> Result<Vec<L0Entry>, CoreError> {
        let mut results = Vec::new();
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table = read_txn
            .open_table(ENTRIES_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open table: {}", e),
            })?;
        for result in table.range(prefix..).map_err(|e| CoreError::StorageError {
            message: format!("Prefix scan failed: {}", e),
        })? {
            let (key_guard, value_guard) = result.map_err(|e| CoreError::StorageError {
                message: format!("Iteration failed: {}", e),
            })?;
            let key_str = key_guard.value();
            if !key_str.starts_with(prefix) {
                break;
            }
            let entry = self.hydrate_entry(Self::decode_entry(value_guard.value())?)?;
            if Self::is_expired(&entry, Utc::now()) {
                continue;
            }
            results.push(entry);
            if results.len() >= limit {
                break;
            }
        }
        Ok(results)
    }

    /// Search using tag index, falls back to full table scan on index miss
    pub fn search_with_index(&self, tags: &[String]) -> Result<Vec<L0Entry>, CoreError> {
        if tags.is_empty() {
            return Ok(Vec::new());
        }

        let mut index_hit = true;
        let mut candidate_iris: Vec<String> = Vec::new();

        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table = read_txn
            .open_table(TAG_INDEX_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open tag index table: {}", e),
            })?;

        for tag in tags {
            let index_key = format!("tag:{}", tag);
            match table.get(index_key.as_str()) {
                Ok(Some(guard)) => {
                    let iris: Vec<String> =
                        serde_json::from_slice(guard.value()).unwrap_or_default();
                    if candidate_iris.is_empty() {
                        candidate_iris = iris;
                    } else {
                        let iris_set: std::collections::HashSet<_> = iris.into_iter().collect();
                        candidate_iris.retain(|iri| iris_set.contains(iri));
                    }
                }
                _ => {
                    index_hit = false;
                    break;
                }
            }
        }

        drop(read_txn);

        if index_hit {
            let mut results = Vec::new();
            for iri in &candidate_iris {
                if let Some(entry) = self.retrieve(iri)? {
                    results.push(entry);
                }
            }
            Ok(results)
        } else {
            self.search_by_tags_fallback(tags)
        }
    }

    /// Full table scan tag search (fallback)
    fn search_by_tags_fallback(&self, tags: &[String]) -> Result<Vec<L0Entry>, CoreError> {
        let mut results = Vec::new();

        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table = read_txn
            .open_table(ENTRIES_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open table: {}", e),
            })?;
        for result in table.iter().map_err(|e| CoreError::StorageError {
            message: format!("Iteration failed: {}", e),
        })? {
            let (_, value) = result.map_err(|e| CoreError::StorageError {
                message: format!("Iteration failed: {}", e),
            })?;

            let entry = self.hydrate_entry(Self::decode_entry(value.value())?)?;
            if Self::is_expired(&entry, Utc::now()) {
                continue;
            }

            if tags.iter().all(|t| entry.tags.contains(t)) {
                results.push(entry);
            }
        }

        Ok(results)
    }

    pub fn search_by_tags(&self, tags: &[String]) -> Result<Vec<L0Entry>, CoreError> {
        self.search_with_index(tags)
    }

    pub fn get_by_importance(&self, min_importance: f32) -> Result<Vec<L0Entry>, CoreError> {
        let mut results = Vec::new();

        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table = read_txn
            .open_table(ENTRIES_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open table: {}", e),
            })?;
        for result in table.iter().map_err(|e| CoreError::StorageError {
            message: format!("Iteration failed: {}", e),
        })? {
            let (_, value) = result.map_err(|e| CoreError::StorageError {
                message: format!("Iteration failed: {}", e),
            })?;

            let entry = self.hydrate_entry(Self::decode_entry(value.value())?)?;
            if Self::is_expired(&entry, Utc::now()) {
                continue;
            }

            if entry.importance >= min_importance {
                results.push(entry);
            }
        }

        results.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    /// Update entry MESI cache coherence state
    pub fn update_mesi_state(&self, iri: &str, state: MesiState) -> Result<(), CoreError> {
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table = read_txn
            .open_table(ENTRIES_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open table: {}", e),
            })?;
        let value = table.get(iri).map_err(|e| CoreError::StorageError {
            message: format!("Failed to retrieve entry: {}", e),
        })?;

        match value {
            Some(guard) => {
                let mut entry = Self::decode_entry(guard.value())?;
                drop(read_txn);
                entry.mesi_state = state;
                self.store_entry(&entry)?;
                Ok(())
            }
            None => Err(CoreError::StorageError {
                message: format!("Entry not found: {}", iri),
            }),
        }
    }

    pub fn count(&self) -> Result<u64, CoreError> {
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table = read_txn
            .open_table(ENTRIES_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open table: {}", e),
            })?;
        table.len().map_err(|e| CoreError::StorageError {
            message: format!("Failed to get entry count: {}", e),
        })
    }

    pub fn generation(&self, iri: &str) -> Result<Option<u64>, CoreError> {
        Ok(self.retrieve(iri)?.as_ref().map(Self::entry_generation))
    }

    pub fn query_by_record_kind(&self, kind: L0RecordKind) -> Result<Vec<L0Entry>, CoreError> {
        let key = format!("kind:{}", kind.as_str());
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Failed to begin record-kind query: {e}"),
        })?;
        let table =
            read_txn
                .open_table(RECORD_KIND_INDEX_TABLE)
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to open record-kind index: {e}"),
                })?;
        let iris = match table
            .get(key.as_str())
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to query record-kind index: {e}"),
            })? {
            Some(value) => serde_json::from_slice::<Vec<String>>(value.value()).unwrap_or_default(),
            None => Vec::new(),
        };
        drop(read_txn);
        let mut entries = Vec::with_capacity(iris.len());
        for iri in iris {
            if let Some(entry) = self.retrieve(&iri)? {
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    /// Physically delete expired entries. Expired records are already hidden
    /// from normal reads; this bounded GC reclaims entries, indices and blobs.
    pub fn gc_expired(&self, now: DateTime<Utc>, limit: usize) -> Result<usize, CoreError> {
        if limit == 0 {
            return Ok(0);
        }
        let expired = {
            let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
                message: format!("Failed to begin L0 TTL scan: {e}"),
            })?;
            let table =
                read_txn
                    .open_table(ENTRIES_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open L0 entries for TTL scan: {e}"),
                    })?;
            let mut iris = Vec::new();
            for item in table.iter().map_err(|e| CoreError::StorageError {
                message: format!("Failed to iterate L0 TTL candidates: {e}"),
            })? {
                let (key, value) = item.map_err(|e| CoreError::StorageError {
                    message: format!("Failed to read L0 TTL candidate: {e}"),
                })?;
                let entry = Self::decode_entry(value.value())?;
                if Self::is_expired(&entry, now) {
                    iris.push(key.value().to_string());
                    if iris.len() >= limit {
                        break;
                    }
                }
            }
            iris
        };
        let mut removed = 0;
        for iri in expired {
            if self.delete(&iri)? {
                removed += 1;
            }
        }
        Ok(removed)
    }

    pub fn blob_stats(&self) -> Result<(u64, u64), CoreError> {
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Failed to begin blob stats read: {e}"),
        })?;
        let blobs = read_txn
            .open_table(BLOB_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open blob table: {e}"),
            })?;
        let refs =
            read_txn
                .open_table(BLOB_REFCOUNT_TABLE)
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to open blob refcount table: {e}"),
                })?;
        Ok((
            blobs.len().map_err(|e| CoreError::StorageError {
                message: format!("Failed to count blobs: {e}"),
            })?,
            refs.iter()
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to iterate blob refcounts: {e}"),
                })?
                .try_fold(0u64, |total, item| {
                    let (_, value) = item.map_err(|e| CoreError::StorageError {
                        message: format!("Failed to read blob refcount: {e}"),
                    })?;
                    let count = value
                        .value()
                        .try_into()
                        .ok()
                        .map(u64::from_be_bytes)
                        .unwrap_or(0);
                    Ok::<_, CoreError>(total.saturating_add(count))
                })?,
        ))
    }

    pub fn flush(&self) -> Result<(), CoreError> {
        // redb persists to disk on commit; no explicit flush needed
        Ok(())
    }

    /// Rewrite at most `limit` legacy plain-JSON records into the current
    /// versioned envelope. Reads already support both formats, so migration is
    /// online and optional; zero means no work rather than an unbounded scan.
    pub fn migrate_legacy_entries(&self, limit: usize) -> Result<usize, CoreError> {
        if limit == 0 {
            return Ok(0);
        }
        let write_txn = self.begin_write("Failed to begin L0 migration transaction")?;
        let migrated = {
            let mut entries =
                write_txn
                    .open_table(ENTRIES_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open L0 entries for migration: {e}"),
                    })?;
            let mut legacy = Vec::new();
            {
                let iter = entries.iter().map_err(|e| CoreError::StorageError {
                    message: format!("Failed to scan L0 entries for migration: {e}"),
                })?;
                for item in iter {
                    let (key, value) = item.map_err(|e| CoreError::StorageError {
                        message: format!("Failed to read L0 migration candidate: {e}"),
                    })?;
                    if !value.value().starts_with(ENTRY_ENVELOPE_MAGIC) {
                        legacy.push((key.value().to_string(), Self::decode_entry(value.value())?));
                        if legacy.len() >= limit {
                            break;
                        }
                    }
                }
            }
            for (iri, entry) in &legacy {
                let encoded = Self::encode_entry(entry, self.config.compression)?;
                entries
                    .insert(iri.as_str(), encoded.as_slice())
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to rewrite migrated L0 entry: {e}"),
                    })?;
            }
            legacy.len()
        };
        write_txn.commit().map_err(|e| CoreError::StorageError {
            message: format!("Failed to commit L0 migration: {e}"),
        })?;
        Ok(migrated)
    }

    /// Upgrade legacy logical records to lifecycle metadata, generation and
    /// content-addressed blob storage. This is explicitly bounded to avoid
    /// delaying glidingcode TUI startup on large databases.
    pub fn migrate_records_to_current_schema(&self, limit: usize) -> Result<usize, CoreError> {
        if limit == 0 {
            return Ok(0);
        }
        let candidates = {
            let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
                message: format!("Failed to begin L0 schema scan: {e}"),
            })?;
            let entries =
                read_txn
                    .open_table(ENTRIES_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open L0 entries for schema scan: {e}"),
                    })?;
            let mut candidates = Vec::new();
            for item in entries.iter().map_err(|e| CoreError::StorageError {
                message: format!("Failed to iterate L0 schema candidates: {e}"),
            })? {
                let (_, value) = item.map_err(|e| CoreError::StorageError {
                    message: format!("Failed to read L0 schema candidate: {e}"),
                })?;
                let decoded = Self::decode_entry(value.value())?;
                let current = decoded
                    .metadata
                    .get(META_SCHEMA_VERSION)
                    .and_then(serde_json::Value::as_u64)
                    == Some(CURRENT_RECORD_SCHEMA_VERSION);
                let needs_blob = decoded.content.len() >= self.config.blob_inline_threshold
                    && Self::entry_blob_ref(&decoded).is_none();
                if !current || needs_blob {
                    candidates.push(self.hydrate_entry(decoded)?);
                    if candidates.len() >= limit {
                        break;
                    }
                }
            }
            candidates
        };
        for entry in &candidates {
            self.replace_entry(entry)?;
        }
        Ok(candidates.len())
    }

    /// Query all entries by named graph
    pub fn query_by_named_graph(&self, graph: &str) -> Result<Vec<L0Entry>, CoreError> {
        let key = format!("graph:{}", graph);
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table =
            read_txn
                .open_table(NAMED_GRAPH_TABLE)
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to open named graph index: {}", e),
                })?;
        match table.get(key.as_str()) {
            Ok(Some(guard)) => {
                let iris: Vec<String> =
                    serde_json::from_slice(guard.value()).map_err(|e| CoreError::StorageError {
                        message: format!("Failed to deserialize named graph index: {}", e),
                    })?;
                let mut entries = Vec::new();
                for iri in iris {
                    if let Some(entry) = self.retrieve(&iri)? {
                        entries.push(entry);
                    }
                }
                Ok(entries)
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Delete all entries in a named graph
    pub fn delete_named_graph(&self, graph: &str) -> Result<usize, CoreError> {
        let entries = self.query_by_named_graph(graph)?;
        let count = entries.len();

        for entry in &entries {
            self.delete(&entry.iri)?;
        }

        let key = format!("graph:{}", graph);
        let write_txn = self.begin_write("Write transaction failed")?;
        {
            let mut table =
                write_txn
                    .open_table(NAMED_GRAPH_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open named graph index: {}", e),
                    })?;
            table
                .remove(key.as_str())
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to delete named graph index: {}", e),
                })?;
        }
        write_txn.commit().map_err(|e| CoreError::StorageError {
            message: format!("Failed to commit transaction: {}", e),
        })?;

        Ok(count)
    }

    /// List all named graphs
    pub fn list_named_graphs(&self) -> Result<Vec<String>, CoreError> {
        let mut graphs = Vec::new();
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table =
            read_txn
                .open_table(NAMED_GRAPH_TABLE)
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to open named graph index: {}", e),
                })?;
        for result in table.iter().map_err(|e| CoreError::StorageError {
            message: format!("Failed to iterate named graph index: {}", e),
        })? {
            let (key_guard, _) = result.map_err(|e| CoreError::StorageError {
                message: format!("Failed to iterate named graph index: {}", e),
            })?;
            let key_str = key_guard.value();
            if let Some(graph) = key_str.strip_prefix("graph:") {
                graphs.push(graph.to_string());
            }
        }
        Ok(graphs)
    }

    pub fn query_by_type(&self, type_iri: &str) -> Result<Vec<L0Entry>, CoreError> {
        let key = format!("type:{}", type_iri);
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let type_index =
            read_txn
                .open_table(TYPE_INDEX_TABLE)
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to open type index: {}", e),
                })?;
        let indexed_iris =
            match type_index
                .get(key.as_str())
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to query type index: {}", e),
                })? {
                Some(guard) => Some(
                    serde_json::from_slice::<Vec<String>>(guard.value()).map_err(|e| {
                        CoreError::StorageError {
                            message: format!("Failed to deserialize type index: {}", e),
                        }
                    })?,
                ),
                None => None,
            };
        drop(type_index);
        drop(read_txn);

        if let Some(iris) = indexed_iris {
            let mut entries = Vec::with_capacity(iris.len());
            for iri in iris {
                if let Some(entry) = self.retrieve(&iri)? {
                    entries.push(entry);
                }
            }
            return Ok(entries);
        }

        // Backward-compatible lazy migration path: databases created before
        // TYPE_INDEX_TABLE was introduced have no rows yet. A full scan keeps
        // them readable; the next exact/merged write populates the index.
        self.scan_by_types(std::slice::from_ref(&type_iri.to_string()))
    }

    pub fn query_by_types(&self, type_iris: &[String]) -> Result<Vec<L0Entry>, CoreError> {
        if type_iris.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for type_iri in type_iris {
            for entry in self.query_by_type(type_iri)? {
                if seen.insert(entry.iri.clone()) {
                    results.push(entry);
                }
            }
        }
        Ok(results)
    }

    fn scan_by_types(&self, type_iris: &[String]) -> Result<Vec<L0Entry>, CoreError> {
        let mut results = Vec::new();
        let read_txn = self.db.begin_read().map_err(|e| CoreError::StorageError {
            message: format!("Read transaction failed: {}", e),
        })?;
        let table = read_txn
            .open_table(ENTRIES_TABLE)
            .map_err(|e| CoreError::StorageError {
                message: format!("Failed to open table: {}", e),
            })?;
        for result in table.iter().map_err(|e| CoreError::StorageError {
            message: format!("Iteration failed: {}", e),
        })? {
            let (_, value) = result.map_err(|e| CoreError::StorageError {
                message: format!("Iteration failed: {}", e),
            })?;

            let entry = self.hydrate_entry(Self::decode_entry(value.value())?)?;
            if Self::is_expired(&entry, Utc::now()) {
                continue;
            }

            if type_iris.iter().any(|t| entry.jsonld_types.contains(t)) {
                results.push(entry);
            }
        }

        Ok(results)
    }

    /// Inject IRI registry, auto-registers @id on subsequent node writes
    pub fn set_iri_registry(&mut self, registry: Arc<IriRegistry>) {
        self.iri_registry = Some(registry);
    }

    pub fn store_jsonld_node(&self, node: &serde_json::Value) -> Result<String, CoreError> {
        let node_obj = node.as_object().ok_or_else(|| CoreError::StorageError {
            message: "JSON-LD node must be an object".to_string(),
        })?;

        let iri = node_obj
            .get("@id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CoreError::StorageError {
                message: "JSON-LD node missing @id field".to_string(),
            })?;

        let jsonld_context = node_obj
            .get("@context")
            .and_then(|v| serde_json::to_string(v).ok());

        let jsonld_types = node_obj
            .get("@type")
            .and_then(|v| match v {
                serde_json::Value::String(s) => Some(vec![s.clone()]),
                serde_json::Value::Array(arr) => Some(
                    arr.iter()
                        .filter_map(|item| item.as_str().map(|s| s.to_string()))
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        let content = serde_json::to_string(node).map_err(|e| CoreError::StorageError {
            message: format!("Failed to serialize JSON-LD node: {}", e),
        })?;

        let content_hash = compute_content_hash(&content);

        // Determine namespace and type (for subsequent IRI registration)
        let primary_type = jsonld_types.first().cloned();

        let existing_entry = self.retrieve_without_update(iri)?;

        let entry = if let Some(existing) = existing_entry {
            let mut merged_metadata = existing.metadata.clone();
            for (key, value) in node_obj.iter() {
                if key != "@id" && key != "@type" && key != "@context" {
                    merged_metadata.insert(key.clone(), value.clone());
                }
            }

            let mut merged_types = existing.jsonld_types.clone();
            for type_iri in &jsonld_types {
                if !merged_types.contains(type_iri) {
                    merged_types.push(type_iri.clone());
                }
            }

            L0Entry {
                iri: iri.to_string(),
                content,
                importance: existing.importance,
                access_count: existing.access_count,
                created_at: existing.created_at,
                last_accessed: Utc::now(),
                tags: existing.tags.clone(),
                metadata: merged_metadata,
                mesi_state: existing.mesi_state.clone(),
                content_hash,
                named_graph: existing.named_graph.clone(),

                jsonld_context: jsonld_context.or(existing.jsonld_context.clone()),
                jsonld_types: merged_types,
            }
        } else {
            let mut metadata = serde_json::Map::new();
            for (key, value) in node_obj.iter() {
                if key != "@id" && key != "@type" && key != "@context" {
                    metadata.insert(key.clone(), value.clone());
                }
            }

            L0Entry {
                iri: iri.to_string(),
                content,
                importance: 0.5,
                access_count: 0,
                created_at: Utc::now(),
                last_accessed: Utc::now(),
                tags: Vec::new(),
                metadata,
                mesi_state: MesiState::Shared,
                content_hash,
                named_graph: None,

                jsonld_context,
                jsonld_types,
            }
        };

        self.store_entry(&entry)?;

        // If IRI registry available, auto-register newly written @id
        if let Some(ref registry) = self.iri_registry {
            let ns = primary_type
                .as_ref()
                .map(|t| t.to_lowercase())
                .unwrap_or_else(|| "node".to_string());
            let named_graph = entry
                .named_graph
                .clone()
                .unwrap_or_else(|| format!("graph:{}", ns));
            let location = EntityLocation {
                iri: iri.to_string(),
                namespace: ns,
                named_graph: Some(named_graph),
                storage_layer: StorageLayer::L0Permanent,
                entity_type: primary_type.clone(),
                created_at: Utc::now(),
                metadata: Default::default(),
            };
            registry.register(iri, location);
        }

        Ok(iri.to_string())
    }

    pub fn retrieve_jsonld_node(&self, iri: &str) -> Result<Option<serde_json::Value>, CoreError> {
        match self.retrieve(iri)? {
            Some(entry) => {
                let mut node = serde_json::Map::new();

                node.insert(
                    "@id".to_string(),
                    serde_json::Value::String(entry.iri.clone()),
                );

                if let Some(context) = entry.jsonld_context {
                    if let Ok(context_value) = serde_json::from_str(&context) {
                        node.insert("@context".to_string(), context_value);
                    }
                }

                if !entry.jsonld_types.is_empty() {
                    if entry.jsonld_types.len() == 1 {
                        node.insert(
                            "@type".to_string(),
                            serde_json::Value::String(entry.jsonld_types[0].clone()),
                        );
                    } else {
                        node.insert(
                            "@type".to_string(),
                            serde_json::Value::Array(
                                entry
                                    .jsonld_types
                                    .into_iter()
                                    .map(serde_json::Value::String)
                                    .collect(),
                            ),
                        );
                    }
                }

                for (key, value) in entry.metadata {
                    node.insert(key, value);
                }

                Ok(Some(serde_json::Value::Object(node)))
            }
            None => Ok(None),
        }
    }
}

/// Memory compressor for L2 -> L0 archival
pub struct MemoryCompressor;

impl MemoryCompressor {
    pub fn compress_session(
        session_id: &str,
        task_id: &str,
        agent_role: &str,
        summary: &str,
    ) -> L0Entry {
        let content_hash = compute_content_hash(summary);
        L0Entry {
            iri: format!("iri://memory/{}", uuid::Uuid::new_v4().hyphenated()),
            content: summary.to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: vec![
                format!("session:{}", session_id),
                format!("task:{}", task_id),
                format!("role:{}", agent_role),
            ],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash,
            named_graph: Some(format!("session:{}", session_id)),
            jsonld_context: None,
            jsonld_types: vec!["Memory".to_string()],
        }
    }

    pub fn compress_nodes(nodes: &[String]) -> String {
        format!(
            r#"{{"@type":"Summary","node_count":{},"compressed_at":"{}"}}"#,
            nodes.len(),
            Utc::now().to_rfc3339()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

    const QUICK_REPAIR_CHILD_ENV: &str = "GLIDING_L0_QUICK_REPAIR_TEST_CHILD";
    const QUICK_REPAIR_PATH_ENV: &str = "GLIDING_L0_QUICK_REPAIR_TEST_PATH";

    #[test]
    fn test_l0_store() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();

        store.store("iri://test/1", r#"{"test": true}"#).unwrap();

        let retrieved = store.retrieve("iri://test/1").unwrap();
        assert!(retrieved.is_some());
        let entry = retrieved.unwrap();
        assert_eq!(entry.mesi_state, MesiState::Shared);
        assert!(!entry.content_hash.is_empty());
    }

    #[test]
    fn test_mesi_state_default() {
        assert_eq!(MesiState::default(), MesiState::Shared);
    }

    #[test]
    fn test_update_mesi_state() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();

        store.store("iri://test/mesi", "content").unwrap();
        store
            .update_mesi_state("iri://test/mesi", MesiState::Modified)
            .unwrap();

        let entry = store.retrieve("iri://test/mesi").unwrap().unwrap();
        assert_eq!(entry.mesi_state, MesiState::Modified);
    }

    #[test]
    fn test_tag_index() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();

        let entry = L0Entry {
            iri: "iri://test/tagged".to_string(),
            content: "tagged content".to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: vec!["rust".to_string(), "test".to_string()],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: Vec::new(),
        };
        store.store_entry(&entry).unwrap();

        let results = store.search_by_tags(&["rust".to_string()]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].iri, "iri://test/tagged");
    }

    #[test]
    fn test_delete_cleans_tag_index() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();

        let entry = L0Entry {
            iri: "iri://test/del".to_string(),
            content: "to be deleted".to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: vec!["deleteme".to_string()],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: Vec::new(),
        };
        store.store_entry(&entry).unwrap();
        store.delete("iri://test/del").unwrap();

        let results = store.search_by_tags(&["deleteme".to_string()]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_content_hash() {
        let hash1 = compute_content_hash("hello");
        let hash2 = compute_content_hash("hello");
        let hash3 = compute_content_hash("world");
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_search_with_index_fallback() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();

        store
            .store("iri://test/fallback", "fallback content")
            .unwrap();

        let results = store
            .search_with_index(&["nonexistent".to_string()])
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_entity_alignment() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();

        let mut entry1 = L0Entry {
            iri: "iri://test/entity".to_string(),
            content: r#"{"name": "Alice"}"#.to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: vec!["person".to_string()],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: Some(r#"{"@vocab": "http://example.org/"}"#.to_string()),
            jsonld_types: vec!["Person".to_string()],
        };
        entry1
            .metadata
            .insert("name".to_string(), serde_json::json!("Alice"));

        store.store_entry(&entry1).unwrap();

        let mut entry2 = L0Entry {
            iri: "iri://test/entity".to_string(),
            content: r#"{"name": "Alice", "age": 30}"#.to_string(),
            importance: 0.7,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: vec!["employee".to_string()],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: vec!["Employee".to_string()],
        };
        entry2
            .metadata
            .insert("age".to_string(), serde_json::json!(30));

        store.store_entry(&entry2).unwrap();

        let merged = store.retrieve("iri://test/entity").unwrap().unwrap();

        assert_eq!(merged.iri, "iri://test/entity");
        assert!(merged.tags.contains(&"person".to_string()));
        assert!(merged.tags.contains(&"employee".to_string()));
        assert!(merged.jsonld_types.contains(&"Person".to_string()));
        assert!(merged.jsonld_types.contains(&"Employee".to_string()));
        assert!(merged.metadata.contains_key("name"));
        assert!(merged.metadata.contains_key("age"));
        assert_eq!(merged.importance, 0.6);
    }

    #[test]
    fn test_query_by_type() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();

        let entry1 = L0Entry {
            iri: "iri://test/person1".to_string(),
            content: "Person 1".to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: Vec::new(),
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: vec!["Person".to_string()],
        };
        store.store_entry(&entry1).unwrap();

        let entry2 = L0Entry {
            iri: "iri://test/person2".to_string(),
            content: "Person 2".to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: Vec::new(),
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: vec!["Person".to_string(), "Employee".to_string()],
        };
        store.store_entry(&entry2).unwrap();

        let entry3 = L0Entry {
            iri: "iri://test/organization".to_string(),
            content: "Organization".to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: Vec::new(),
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: vec!["Organization".to_string()],
        };
        store.store_entry(&entry3).unwrap();

        let person_results = store.query_by_type("Person").unwrap();
        assert_eq!(person_results.len(), 2);

        let employee_results = store.query_by_type("Employee").unwrap();
        assert_eq!(employee_results.len(), 1);
        assert_eq!(employee_results[0].iri, "iri://test/person2");

        let org_results = store.query_by_type("Organization").unwrap();
        assert_eq!(org_results.len(), 1);
    }

    #[test]
    fn test_query_by_types() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();

        let entry1 = L0Entry {
            iri: "iri://test/entity1".to_string(),
            content: "Entity 1".to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: Vec::new(),
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: vec!["Person".to_string()],
        };
        store.store_entry(&entry1).unwrap();

        let entry2 = L0Entry {
            iri: "iri://test/entity2".to_string(),
            content: "Entity 2".to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            tags: Vec::new(),
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: vec!["Organization".to_string()],
        };
        store.store_entry(&entry2).unwrap();

        let results = store
            .query_by_types(&["Person".to_string(), "Organization".to_string()])
            .unwrap();
        assert_eq!(results.len(), 2);

        let person_only = store.query_by_types(&["Person".to_string()]).unwrap();
        assert_eq!(person_only.len(), 1);
    }

    #[test]
    fn test_jsonld_node_storage() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();

        let node = serde_json::json!({
            "@id": "iri://test/person/alice",
            "@type": "Person",
            "@context": {
                "@vocab": "http://example.org/"
            },
            "name": "Alice",
            "age": 30
        });

        let iri = store.store_jsonld_node(&node).unwrap();
        assert_eq!(iri, "iri://test/person/alice");

        let retrieved = store
            .retrieve_jsonld_node("iri://test/person/alice")
            .unwrap();
        assert!(retrieved.is_some());

        let retrieved_node = retrieved.unwrap();
        assert_eq!(retrieved_node["@id"], "iri://test/person/alice");
        assert_eq!(retrieved_node["name"], "Alice");
        assert_eq!(retrieved_node["age"], 30);
    }

    #[test]
    fn test_jsonld_node_merge() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();

        let node1 = serde_json::json!({
            "@id": "iri://test/person/bob",
            "@type": "Person",
            "name": "Bob",
            "age": 25
        });

        store.store_jsonld_node(&node1).unwrap();

        let node2 = serde_json::json!({
            "@id": "iri://test/person/bob",
            "@type": "Employee",
            "department": "Engineering"
        });

        store.store_jsonld_node(&node2).unwrap();

        let retrieved = store
            .retrieve_jsonld_node("iri://test/person/bob")
            .unwrap()
            .unwrap();

        assert_eq!(retrieved["@id"], "iri://test/person/bob");
        assert_eq!(retrieved["name"], "Bob");
        assert_eq!(retrieved["age"], 25);
        assert_eq!(retrieved["department"], "Engineering");

        let types = retrieved["@type"].as_array().unwrap();
        assert!(types.contains(&serde_json::json!("Person")));
        assert!(types.contains(&serde_json::json!("Employee")));
    }

    #[test]
    fn retrieve_does_not_create_write_amplification() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();
        store.store("iri://test/read-only", "payload").unwrap();

        store.retrieve("iri://test/read-only").unwrap().unwrap();
        store.retrieve("iri://test/read-only").unwrap().unwrap();

        let raw = store
            .retrieve_without_update("iri://test/read-only")
            .unwrap()
            .unwrap();
        assert_eq!(raw.access_count, 0, "ordinary reads must remain read-only");
    }

    #[test]
    fn quick_repair_avoids_full_scan_after_unclean_exit() {
        if std::env::var_os(QUICK_REPAIR_CHILD_ENV).is_some() {
            let path = std::env::var(QUICK_REPAIR_PATH_ENV).unwrap();
            let store = L0Store::with_config(L0Config {
                path,
                cache_size_bytes: 8 * 1024 * 1024,
                quick_repair: true,
                ..Default::default()
            })
            .unwrap();
            store
                .store("iri://test/unclean-exit", "durable payload")
                .unwrap();

            // Model SIGKILL/power loss: skip Database::drop(), whose clean-close
            // path would otherwise hide whether quick-repair metadata works.
            std::process::exit(0);
        }

        let dir = tempdir().unwrap();
        let status = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("memory::l0_store::tests::quick_repair_avoids_full_scan_after_unclean_exit")
            .arg("--nocapture")
            .env(QUICK_REPAIR_CHILD_ENV, "1")
            .env(QUICK_REPAIR_PATH_ENV, dir.path())
            .status()
            .unwrap();
        assert!(status.success());

        let repair_callbacks = Arc::new(AtomicUsize::new(0));
        let callback_counter = repair_callbacks.clone();
        let store = L0Store::with_config_and_repair_callback(
            L0Config {
                path: dir.path().to_string_lossy().to_string(),
                cache_size_bytes: 8 * 1024 * 1024,
                quick_repair: true,
                ..Default::default()
            },
            Some(Arc::new(move |_| {
                callback_counter.fetch_add(1, Ordering::Relaxed);
            })),
        )
        .unwrap();

        assert_eq!(repair_callbacks.load(Ordering::Relaxed), 0);
        assert_eq!(
            store
                .retrieve("iri://test/unclean-exit")
                .unwrap()
                .unwrap()
                .content,
            "durable payload"
        );
    }

    #[test]
    fn replace_entry_migrates_all_secondary_indices() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();
        let now = Utc::now();
        let original = L0Entry {
            iri: "iri://test/replace".to_string(),
            content: "v1".to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            tags: vec!["old".to_string()],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: Some("iri://graph/old".to_string()),
            jsonld_context: None,
            jsonld_types: vec!["OldType".to_string()],
        };
        store.store_entry(&original).unwrap();

        let replacement = L0Entry {
            content: "v2".to_string(),
            tags: vec!["new".to_string()],
            named_graph: Some("iri://graph/new".to_string()),
            jsonld_types: vec!["NewType".to_string()],
            ..original
        };
        store.replace_entry(&replacement).unwrap();

        assert!(store
            .search_by_tags(&["old".to_string()])
            .unwrap()
            .is_empty());
        assert_eq!(store.search_by_tags(&["new".to_string()]).unwrap().len(), 1);
        assert!(store
            .query_by_named_graph("iri://graph/old")
            .unwrap()
            .is_empty());
        assert_eq!(
            store.query_by_named_graph("iri://graph/new").unwrap().len(),
            1
        );
        assert!(store.query_by_type("OldType").unwrap().is_empty());
        assert_eq!(store.query_by_type("NewType").unwrap().len(), 1);
    }

    #[test]
    fn delete_cleans_named_graph_index_and_reports_absence() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();
        let now = Utc::now();
        store
            .store_entry(&L0Entry {
                iri: "iri://test/graph-delete".to_string(),
                content: "payload".to_string(),
                importance: 0.5,
                access_count: 0,
                created_at: now,
                last_accessed: now,
                tags: Vec::new(),
                metadata: serde_json::Map::new(),
                mesi_state: MesiState::Shared,
                content_hash: String::new(),
                named_graph: Some("iri://graph/delete".to_string()),
                jsonld_context: None,
                jsonld_types: Vec::new(),
            })
            .unwrap();

        assert!(store.delete("iri://test/graph-delete").unwrap());
        assert!(!store.delete("iri://test/graph-delete").unwrap());
        assert!(store
            .query_by_named_graph("iri://graph/delete")
            .unwrap()
            .is_empty());
        assert!(!store
            .list_named_graphs()
            .unwrap()
            .contains(&"iri://graph/delete".to_string()));
    }

    #[test]
    fn configured_entry_capacity_is_enforced() {
        let dir = tempdir().unwrap();
        let store = L0Store::with_config(L0Config {
            path: dir.path().to_string_lossy().to_string(),
            max_entries: 1,
            compression: false,
            blob_inline_threshold: 4_096,
            ..Default::default()
        })
        .unwrap();
        store.store("iri://test/one", "one").unwrap();
        assert!(store.store("iri://test/two", "two").is_err());
        store.store("iri://test/one", "updated").unwrap();
    }

    #[test]
    fn versioned_envelope_compresses_and_detects_corruption() {
        let now = Utc::now();
        let entry = L0Entry {
            iri: "iri://test/envelope".to_string(),
            content: "compressible-content-".repeat(500),
            importance: 0.5,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            tags: Vec::new(),
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: Vec::new(),
        };
        let encoded = L0Store::encode_entry(&entry, true).unwrap();
        assert!(encoded.starts_with(ENTRY_ENVELOPE_MAGIC));
        assert_eq!(encoded[4], ENTRY_ENVELOPE_VERSION);
        assert_eq!(encoded[5], CODEC_GZIP);
        assert_eq!(
            L0Store::decode_entry(&encoded).unwrap().content,
            entry.content
        );

        let mut corrupted = encoded;
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0x01;
        assert!(L0Store::decode_entry(&corrupted).is_err());
    }

    #[test]
    fn legacy_plain_json_is_dual_read_and_explicitly_migrated() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();
        let now = Utc::now();
        let legacy = L0Entry {
            iri: "iri://test/legacy".to_string(),
            content: "legacy-body".to_string(),
            importance: 0.5,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            tags: Vec::new(),
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: compute_content_hash("legacy-body"),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: Vec::new(),
        };
        let raw = serde_json::to_vec(&legacy).unwrap();
        let tx = store.db.begin_write().unwrap();
        {
            let mut table = tx.open_table(ENTRIES_TABLE).unwrap();
            table.insert(legacy.iri.as_str(), raw.as_slice()).unwrap();
        }
        tx.commit().unwrap();

        assert_eq!(
            store.retrieve(&legacy.iri).unwrap().unwrap().content,
            "legacy-body"
        );
        assert_eq!(store.migrate_legacy_entries(10).unwrap(), 1);
        assert_eq!(store.migrate_legacy_entries(10).unwrap(), 0);

        let tx = store.db.begin_read().unwrap();
        let table = tx.open_table(ENTRIES_TABLE).unwrap();
        let stored = table.get(legacy.iri.as_str()).unwrap().unwrap();
        assert!(stored.value().starts_with(ENTRY_ENVELOPE_MAGIC));
    }

    #[test]
    fn logical_schema_migration_classifies_and_deduplicates_legacy_content() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();
        let now = Utc::now();
        let content = "legacy-large-archive-content-".repeat(300);
        let legacy = L0Entry {
            iri: "iri://archive/legacy-large/1".to_string(),
            content: content.clone(),
            importance: 0.5,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            tags: Vec::new(),
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: compute_content_hash(&content),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: Vec::new(),
        };
        let raw = serde_json::to_vec(&legacy).unwrap();
        let tx = store.db.begin_write().unwrap();
        {
            let mut table = tx.open_table(ENTRIES_TABLE).unwrap();
            table.insert(legacy.iri.as_str(), raw.as_slice()).unwrap();
        }
        tx.commit().unwrap();

        assert_eq!(store.migrate_records_to_current_schema(10).unwrap(), 1);
        assert_eq!(store.migrate_records_to_current_schema(10).unwrap(), 0);
        let migrated = store.retrieve(&legacy.iri).unwrap().unwrap();
        assert_eq!(migrated.content, content);
        assert_eq!(
            migrated
                .metadata
                .get(META_SCHEMA_VERSION)
                .and_then(serde_json::Value::as_u64),
            Some(CURRENT_RECORD_SCHEMA_VERSION)
        );
        assert_eq!(store.blob_stats().unwrap(), (1, 1));
        assert_eq!(
            store
                .query_by_record_kind(L0RecordKind::RawInteraction)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn classified_ttl_records_are_hidden_then_reclaimed() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();
        let now = Utc::now();
        let entry = L0Entry {
            iri: "iri://telemetry/expired/1".to_string(),
            content: "expired telemetry".to_string(),
            importance: 0.1,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            tags: vec!["telemetry".to_string()],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: None,
            jsonld_context: None,
            jsonld_types: Vec::new(),
        };
        store
            .store_with_policy(
                &entry,
                L0RecordKind::Telemetry,
                RetentionClass::Ephemeral,
                Some(now - chrono::Duration::seconds(1)),
            )
            .unwrap();

        assert!(store.retrieve(&entry.iri).unwrap().is_none());
        assert!(store
            .query_by_record_kind(L0RecordKind::Telemetry)
            .unwrap()
            .is_empty());
        assert_eq!(store.gc_expired(now, 10).unwrap(), 1);
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn large_duplicate_content_uses_one_reference_counted_blob() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();
        let content = "shared-large-content-".repeat(500);
        store.store("iri://archive/blob/1", &content).unwrap();
        store.store("iri://archive/blob/2", &content).unwrap();

        assert_eq!(store.blob_stats().unwrap(), (1, 2));
        assert_eq!(
            store
                .retrieve("iri://archive/blob/1")
                .unwrap()
                .unwrap()
                .content,
            content
        );
        assert_eq!(
            store
                .query_by_record_kind(L0RecordKind::RawInteraction)
                .unwrap()
                .len(),
            2
        );
        store.delete("iri://archive/blob/1").unwrap();
        assert_eq!(store.blob_stats().unwrap(), (1, 1));
        store.delete("iri://archive/blob/2").unwrap();
        assert_eq!(store.blob_stats().unwrap(), (0, 0));
    }

    #[test]
    fn canonical_generation_changes_only_for_canonical_content() {
        let dir = tempdir().unwrap();
        let store = L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap();
        store.store("iri://knowledge/generation/1", "v1").unwrap();
        assert_eq!(
            store.generation("iri://knowledge/generation/1").unwrap(),
            Some(1)
        );

        store
            .update_mesi_state("iri://knowledge/generation/1", MesiState::Modified)
            .unwrap();
        assert_eq!(
            store.generation("iri://knowledge/generation/1").unwrap(),
            Some(1)
        );

        store.store("iri://knowledge/generation/1", "v2").unwrap();
        assert_eq!(
            store.generation("iri://knowledge/generation/1").unwrap(),
            Some(2)
        );
    }
}
