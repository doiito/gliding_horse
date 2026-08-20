//! L0 Store - Long-term memory persistence storage
//!
//! This module handles persistent storage of memories and knowledge, supporting MESI cache coherence states and tag secondary indexes.

use chrono::{DateTime, Utc};
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
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
}

impl Default for L0Config {
    fn default() -> Self {
        Self {
            path: "./data/l0_store".to_string(),
            max_entries: 1_000_000,
            compression: true,
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
    pub fn new(path: &str) -> Result<Self, CoreError> {
        info!("Initializing L0 Store: {}", path);

        std::fs::create_dir_all(path).map_err(|e| CoreError::StorageError {
            message: format!("Failed to create storage directory: {}", e),
        })?;

        let db_path = std::path::Path::new(path).join("l0.redb");
        let db_path_str = db_path.to_string_lossy();
        let db = Database::create(db_path_str.as_ref()).map_err(|e| CoreError::StorageError {
            message: format!("Failed to open database: {}", e),
        })?;

        // Ensure tables exist by opening them in a write transaction
        {
            let write_txn = db.begin_write().map_err(|e| CoreError::StorageError {
                message: format!("Failed to begin write transaction: {}", e),
            })?;
            let _ = write_txn.open_table(ENTRIES_TABLE);
            let _ = write_txn.open_table(TAG_INDEX_TABLE);
            let _ = write_txn.open_table(NAMED_GRAPH_TABLE);
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
            config: L0Config {
                path: path.to_string(),
                ..Default::default()
            },
            entry_count,
            iri_registry: None,
        })
    }

    /// Update tag index: remove old tag indexes, then insert new tag indexes
    fn update_tag_index(
        &self,
        iri: &str,
        old_tags: &[String],
        new_tags: &[String],
    ) -> Result<(), CoreError> {
        for tag in old_tags {
            let index_key = format!("tag:{}", tag);
            self.remove_iri_from_tag_index(&index_key, iri)?;
        }
        for tag in new_tags {
            let index_key = format!("tag:{}", tag);
            self.add_iri_to_tag_index(&index_key, iri)?;
        }
        Ok(())
    }

    /// Add IRI to tag index
    fn add_iri_to_tag_index(&self, index_key: &str, iri: &str) -> Result<(), CoreError> {
        let write_txn = self.db.begin_write().map_err(|e| CoreError::StorageError {
            message: format!("Write transaction failed: {}", e),
        })?;
        {
            let mut table =
                write_txn
                    .open_table(TAG_INDEX_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open tag index: {}", e),
                    })?;
            let mut iris: Vec<String> = match table.get(index_key) {
                Ok(Some(guard)) => serde_json::from_slice(guard.value()).unwrap_or_default(),
                _ => Vec::new(),
            };
            if !iris.contains(&iri.to_string()) {
                iris.push(iri.to_string());
            }
            let encoded = serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                message: format!("Failed to serialize tag index: {}", e),
            })?;
            table
                .insert(index_key, encoded.as_slice())
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to write tag index: {}", e),
                })?;
        }
        write_txn.commit().map_err(|e| CoreError::StorageError {
            message: format!("Failed to commit transaction: {}", e),
        })?;
        Ok(())
    }

    /// Remove IRI from tag index
    fn remove_iri_from_tag_index(&self, index_key: &str, iri: &str) -> Result<(), CoreError> {
        let write_txn = self.db.begin_write().map_err(|e| CoreError::StorageError {
            message: format!("Write transaction failed: {}", e),
        })?;
        let iris_empty = {
            let mut table =
                write_txn
                    .open_table(TAG_INDEX_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open tag index: {}", e),
                    })?;
            let mut iris: Vec<String> = match table.get(index_key) {
                Ok(Some(guard)) => serde_json::from_slice(guard.value()).unwrap_or_default(),
                _ => return Ok(()),
            };
            iris.retain(|i| i != iri);
            if iris.is_empty() {
                table
                    .remove(index_key)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to delete tag index: {}", e),
                    })?;
                true
            } else {
                let encoded = serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                    message: format!("Failed to serialize tag index: {}", e),
                })?;
                table.insert(index_key, encoded.as_slice()).map_err(|e| {
                    CoreError::StorageError {
                        message: format!("Failed to write tag index: {}", e),
                    }
                })?;
                false
            }
        };
        let _ = iris_empty;
        write_txn.commit().map_err(|e| CoreError::StorageError {
            message: format!("Failed to commit transaction: {}", e),
        })?;
        Ok(())
    }

    pub fn store(&self, iri: &str, content: &str) -> Result<(), CoreError> {
        let content_hash = compute_content_hash(content);

        let existing_entry = self.retrieve_without_update(iri)?;

        let entry = if let Some(existing) = existing_entry {
            let new_entry = L0Entry {
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
            };
            Self::merge_entries(&existing, &new_entry)
        } else {
            L0Entry {
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
            }
        };

        let value = serde_json::to_vec(&entry).map_err(|e| CoreError::StorageError {
            message: format!("Failed to serialize entry: {}", e),
        })?;

        let write_txn = self.db.begin_write().map_err(|e| CoreError::StorageError {
            message: format!("Write transaction failed: {}", e),
        })?;
        {
            let mut table =
                write_txn
                    .open_table(ENTRIES_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open table: {}", e),
                    })?;
            table
                .insert(iri, value.as_slice())
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to store entry: {}", e),
                })?;
        }
        write_txn.commit().map_err(|e| CoreError::StorageError {
            message: format!("Failed to commit transaction: {}", e),
        })?;

        debug!(iri = %entry.iri, "Entry stored to L0");
        Ok(())
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
                let entry: L0Entry =
                    serde_json::from_slice(guard.value()).map_err(|e| CoreError::StorageError {
                        message: format!("Failed to deserialize entry: {}", e),
                    })?;
                Ok(Some(entry))
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

    pub fn store_entry(&self, entry: &L0Entry) -> Result<(), CoreError> {
        let old_tags = self.get_entry_tags(&entry.iri)?;
        let old_named_graph = self.get_entry_named_graph(&entry.iri)?;

        let content_hash = if entry.content_hash.is_empty() {
            compute_content_hash(&entry.content)
        } else {
            entry.content_hash.clone()
        };

        let existing_entry = self.retrieve_without_update(&entry.iri)?;
        let entry = if let Some(existing) = existing_entry {
            let entry_with_hash = L0Entry {
                content_hash,
                ..entry.clone()
            };
            Self::merge_entries(&existing, &entry_with_hash)
        } else {
            L0Entry {
                content_hash,
                ..entry.clone()
            }
        };

        self.update_tag_index(&entry.iri, &old_tags, &entry.tags)?;

        if old_named_graph != entry.named_graph {
            if let Some(ref old_graph) = old_named_graph {
                self.remove_iri_from_named_graph_index(old_graph, &entry.iri)?;
            }
            if let Some(ref new_graph) = entry.named_graph {
                self.add_iri_to_named_graph_index(new_graph, &entry.iri)?;
            }
        }

        let value = serde_json::to_vec(&entry).map_err(|e| CoreError::StorageError {
            message: format!("Failed to serialize entry: {}", e),
        })?;

        let write_txn = self.db.begin_write().map_err(|e| CoreError::StorageError {
            message: format!("Write transaction failed: {}", e),
        })?;
        {
            let mut table =
                write_txn
                    .open_table(ENTRIES_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open table: {}", e),
                    })?;
            table
                .insert(entry.iri.as_str(), value.as_slice())
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to store entry: {}", e),
                })?;
        }
        write_txn.commit().map_err(|e| CoreError::StorageError {
            message: format!("Failed to commit transaction: {}", e),
        })?;

        debug!(iri = %entry.iri, "Entry stored to L0");
        Ok(())
    }

    fn get_entry_named_graph(&self, iri: &str) -> Result<Option<String>, CoreError> {
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
                let entry: L0Entry =
                    serde_json::from_slice(guard.value()).map_err(|e| CoreError::StorageError {
                        message: format!("Failed to deserialize entry: {}", e),
                    })?;
                Ok(entry.named_graph)
            }
            _ => Ok(None),
        }
    }

    fn add_iri_to_named_graph_index(&self, graph: &str, iri: &str) -> Result<(), CoreError> {
        let key = format!("graph:{}", graph);
        let write_txn = self.db.begin_write().map_err(|e| CoreError::StorageError {
            message: format!("Write transaction failed: {}", e),
        })?;
        {
            let mut table =
                write_txn
                    .open_table(NAMED_GRAPH_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open named graph index: {}", e),
                    })?;
            let mut iris: Vec<String> = match table.get(key.as_str()) {
                Ok(Some(guard)) => serde_json::from_slice(guard.value()).unwrap_or_default(),
                _ => Vec::new(),
            };
            if !iris.contains(&iri.to_string()) {
                iris.push(iri.to_string());
            }
            let encoded = serde_json::to_vec(&iris).map_err(|e| CoreError::StorageError {
                message: format!("Failed to serialize named graph index: {}", e),
            })?;
            table
                .insert(key.as_str(), encoded.as_slice())
                .map_err(|e| CoreError::StorageError {
                    message: format!("Failed to write named graph index: {}", e),
                })?;
        }
        write_txn.commit().map_err(|e| CoreError::StorageError {
            message: format!("Failed to commit transaction: {}", e),
        })?;
        Ok(())
    }

    fn remove_iri_from_named_graph_index(&self, graph: &str, iri: &str) -> Result<(), CoreError> {
        let key = format!("graph:{}", graph);
        let write_txn = self.db.begin_write().map_err(|e| CoreError::StorageError {
            message: format!("Write transaction failed: {}", e),
        })?;
        {
            let mut table =
                write_txn
                    .open_table(NAMED_GRAPH_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open named graph index: {}", e),
                    })?;
            let iris: Vec<String> = match table.get(key.as_str()) {
                Ok(Some(guard)) => serde_json::from_slice(guard.value()).unwrap_or_default(),
                _ => return Ok(()),
            };
            if iris.is_empty() {
                return Ok(());
            }
            // rebuild without the target iri
            let filtered: Vec<String> = iris.into_iter().filter(|i| i != iri).collect();
            if filtered.is_empty() {
                table
                    .remove(key.as_str())
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to delete named graph index: {}", e),
                    })?;
            } else {
                let encoded =
                    serde_json::to_vec(&filtered).map_err(|e| CoreError::StorageError {
                        message: format!("Failed to serialize named graph index: {}", e),
                    })?;
                table
                    .insert(key.as_str(), encoded.as_slice())
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to write named graph index: {}", e),
                    })?;
            }
        }
        write_txn.commit().map_err(|e| CoreError::StorageError {
            message: format!("Failed to commit transaction: {}", e),
        })?;
        Ok(())
    }

    /// Get existing tags of an entry (for index updates)
    fn get_entry_tags(&self, iri: &str) -> Result<Vec<String>, CoreError> {
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
                let entry: L0Entry =
                    serde_json::from_slice(guard.value()).map_err(|e| CoreError::StorageError {
                        message: format!("Failed to deserialize entry: {}", e),
                    })?;
                Ok(entry.tags)
            }
            _ => Ok(Vec::new()),
        }
    }

    pub fn retrieve(&self, iri: &str) -> Result<Option<L0Entry>, CoreError> {
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
                let mut entry: L0Entry =
                    serde_json::from_slice(guard.value()).map_err(|e| CoreError::StorageError {
                        message: format!("Failed to deserialize entry: {}", e),
                    })?;
                drop(read_txn);

                entry.access_count += 1;
                entry.last_accessed = Utc::now();
                self.store_entry(&entry)?;

                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    pub fn delete(&self, iri: &str) -> Result<bool, CoreError> {
        let old_tags = self.get_entry_tags(iri)?;

        for tag in &old_tags {
            let index_key = format!("tag:{}", tag);
            self.remove_iri_from_tag_index(&index_key, iri)?;
        }

        let write_txn = self.db.begin_write().map_err(|e| CoreError::StorageError {
            message: format!("Write transaction failed: {}", e),
        })?;
        let removed = {
            let mut table =
                write_txn
                    .open_table(ENTRIES_TABLE)
                    .map_err(|e| CoreError::StorageError {
                        message: format!("Failed to open table: {}", e),
                    })?;
            let has_removed = match table.remove(iri) {
                Ok(_) => true,
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

            let entry: L0Entry =
                serde_json::from_slice(value.value()).map_err(|e| CoreError::StorageError {
                    message: format!("Failed to deserialize entry: {}", e),
                })?;

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
            let entry: L0Entry = serde_json::from_slice(value_guard.value()).map_err(|e| {
                CoreError::StorageError {
                    message: format!("Failed to deserialize entry: {}", e),
                }
            })?;
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

            let entry: L0Entry =
                serde_json::from_slice(value.value()).map_err(|e| CoreError::StorageError {
                    message: format!("Failed to deserialize entry: {}", e),
                })?;

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

            let entry: L0Entry =
                serde_json::from_slice(value.value()).map_err(|e| CoreError::StorageError {
                    message: format!("Failed to deserialize entry: {}", e),
                })?;

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
                let mut entry: L0Entry =
                    serde_json::from_slice(guard.value()).map_err(|e| CoreError::StorageError {
                        message: format!("Failed to deserialize entry: {}", e),
                    })?;
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

    pub fn flush(&self) -> Result<(), CoreError> {
        // redb persists to disk on commit; no explicit flush needed
        Ok(())
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
        let write_txn = self.db.begin_write().map_err(|e| CoreError::StorageError {
            message: format!("Write transaction failed: {}", e),
        })?;
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

            let entry: L0Entry =
                serde_json::from_slice(value.value()).map_err(|e| CoreError::StorageError {
                    message: format!("Failed to deserialize entry: {}", e),
                })?;

            if entry.jsonld_types.contains(&type_iri.to_string()) {
                results.push(entry);
            }
        }

        Ok(results)
    }

    pub fn query_by_types(&self, type_iris: &[String]) -> Result<Vec<L0Entry>, CoreError> {
        if type_iris.is_empty() {
            return Ok(Vec::new());
        }

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

            let entry: L0Entry =
                serde_json::from_slice(value.value()).map_err(|e| CoreError::StorageError {
                    message: format!("Failed to deserialize entry: {}", e),
                })?;

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
    use tempfile::tempdir;

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
}
