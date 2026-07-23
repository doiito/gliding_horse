use std::collections::BTreeMap;
use std::sync::RwLock;

use dashmap::DashMap;
use roaring::RoaringBitmap;
use serde_json::Value;

/// JSON-LD metadata index with DashMap + RoaringBitmap for fast filtering.
///
/// Indexes:
/// - `inverted`: generic tag:value → bitmap
/// - `type_index`: @type → bitmap
/// - `graph_index`: named_graph → bitmap
/// - `context_index`: @context → bitmap
/// - `numeric`: numeric field key → BTreeMap<value, bitmap>
/// - `forward`: id → full JSON-LD payload
pub struct JsonLdMetadataIndex {
    /// tag:value → bitmap of IDs
    pub inverted: DashMap<String, RoaringBitmap>,
    /// @type → bitmap
    pub type_index: DashMap<String, RoaringBitmap>,
    /// named_graph → bitmap
    pub graph_index: DashMap<String, RoaringBitmap>,
    /// @context → bitmap
    pub context_index: DashMap<String, RoaringBitmap>,
    /// Numeric index: key → BTreeMap<i64, RoaringBitmap>
    pub numeric: DashMap<String, RwLock<BTreeMap<i64, RoaringBitmap>>>,
    /// id → full JSON-LD payload
    pub forward: DashMap<u32, Value>,
    /// deleted IDs
    pub deleted: RwLock<RoaringBitmap>,
}

impl Default for JsonLdMetadataIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for JsonLdMetadataIndex {
    fn clone(&self) -> Self {
        let deleted = self
            .deleted
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let numeric = DashMap::new();
        for entry in self.numeric.iter() {
            let tree = entry
                .value()
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            numeric.insert(entry.key().clone(), RwLock::new(tree));
        }
        Self {
            inverted: self.inverted.clone(),
            type_index: self.type_index.clone(),
            graph_index: self.graph_index.clone(),
            context_index: self.context_index.clone(),
            numeric,
            forward: self.forward.clone(),
            deleted: RwLock::new(deleted),
        }
    }
}

impl JsonLdMetadataIndex {
    pub fn new() -> Self {
        Self {
            inverted: DashMap::new(),
            type_index: DashMap::new(),
            graph_index: DashMap::new(),
            context_index: DashMap::new(),
            numeric: DashMap::new(),
            forward: DashMap::new(),
            deleted: RwLock::new(RoaringBitmap::new()),
        }
    }

    /// Index a JSON-LD value for the given ID.
    /// Automatically extracts @type, @context, named_graph, tags, and numeric values.
    pub fn index(&self, id: u32, jsonld: &Value) {
        // @type
        if let Some(types) = jsonld.get("@type").and_then(|v| v.as_array()) {
            for t in types {
                if let Some(type_str) = t.as_str() {
                    self.type_index
                        .entry(type_str.to_string())
                        .or_default()
                        .insert(id);
                    self.inverted
                        .entry(format!("@type:{type_str}"))
                        .or_default()
                        .insert(id);
                }
            }
        }
        // @context
        if let Some(ctx) = jsonld.get("@context").and_then(|v| v.as_str()) {
            self.context_index
                .entry(ctx.to_string())
                .or_default()
                .insert(id);
        }
        // named_graph
        if let Some(graph) = jsonld.get("named_graph").and_then(|v| v.as_str()) {
            self.graph_index
                .entry(graph.to_string())
                .or_default()
                .insert(id);
        }
        // General properties
        if let Some(obj) = jsonld.as_object() {
            for (key, val) in obj {
                if key.starts_with('@') {
                    continue;
                }
                match val {
                    Value::String(s) => {
                        let tag = format!("{key}:{s}");
                        self.inverted.entry(tag).or_default().insert(id);
                    }
                    Value::Number(n) => {
                        if let Some(n64) = n.as_f64() {
                            let entry = self
                                .numeric
                                .entry(key.clone())
                                .or_insert_with(|| RwLock::new(BTreeMap::new()));
                            let quantized = (n64 * 1000.0) as i64; // milliscale quantization
                            if let Ok(mut tree) = entry.write() {
                                tree.entry(quantized).or_default().insert(id);
                            };
                        }
                    }
                    Value::Array(arr) => {
                        for item in arr {
                            if let Some(s) = item.as_str() {
                                let tag = format!("{key}:{s}");
                                self.inverted.entry(tag).or_default().insert(id);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // Store payload
        self.forward.insert(id, jsonld.clone());
    }

    /// Remove ID from all indexes.
    pub fn remove(&self, id: u32) {
        if let Ok(mut del) = self.deleted.write() {
            del.insert(id);
        }
        self.forward.remove(&id);
        // Deletion and upsert replacement must become visible to filters at
        // once. Deferring this cleanup to `vacuum` leaves stale memberships.
        self.remove_id_from_bitmap_map(&self.inverted, id);
        self.remove_id_from_bitmap_map(&self.type_index, id);
        self.remove_id_from_bitmap_map(&self.graph_index, id);
        self.remove_id_from_bitmap_map(&self.context_index, id);
        self.remove_id_from_numeric_indexes(id);
    }

    fn remove_id_from_bitmap_map(&self, map: &DashMap<String, RoaringBitmap>, id: u32) {
        let keys: Vec<String> = map
            .iter()
            .filter(|entry| entry.value().contains(id))
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys {
            if let Some(mut bitmap) = map.get_mut(&key) {
                bitmap.remove(id);
                if bitmap.is_empty() {
                    drop(bitmap);
                    map.remove(&key);
                }
            }
        }
    }

    fn remove_id_from_numeric_indexes(&self, id: u32) {
        let keys: Vec<String> = self
            .numeric
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys {
            let empty = if let Some(entry) = self.numeric.get(&key) {
                let mut tree = entry.value().write().unwrap_or_else(|e| e.into_inner());
                let buckets: Vec<i64> = tree
                    .iter()
                    .filter(|(_, bitmap)| bitmap.contains(id))
                    .map(|(bucket, _)| *bucket)
                    .collect();
                for bucket in buckets {
                    if let Some(bitmap) = tree.get_mut(&bucket) {
                        bitmap.remove(id);
                        if bitmap.is_empty() {
                            tree.remove(&bucket);
                        }
                    }
                }
                tree.is_empty()
            } else {
                false
            };
            if empty {
                self.numeric.remove(&key);
            }
        }
    }

    pub fn undelete(&self, id: u32) {
        if let Ok(mut del) = self.deleted.write() {
            del.remove(id);
        }
    }

    /// Get payload for ID.
    pub fn get_payload(&self, id: u32) -> Option<Value> {
        self.forward.get(&id).map(|r| r.clone())
    }

    /// Get all non-deleted IDs, optionally filtered by a starting bitmap.
    pub fn all_ids_filtered(&self, filter: Option<&RoaringBitmap>) -> Vec<u32> {
        let deleted = self
            .deleted
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let max = self.forward.iter().map(|e| *e.key()).max().unwrap_or(0);
        if let Some(fb) = filter {
            (0..=max)
                .filter(|id| fb.contains(*id) && !deleted.contains(*id))
                .collect()
        } else {
            (0..=max)
                .filter(|id| self.forward.contains_key(id) && !deleted.contains(*id))
                .collect()
        }
    }

    /// Get all non-deleted IDs (legacy).
    pub fn all_ids(&self) -> Vec<u32> {
        self.all_ids_filtered(None)
    }

    /// Count live entries.
    pub fn count(&self) -> u64 {
        let deleted = self
            .deleted
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        self.forward
            .iter()
            .filter(|e| !deleted.contains(*e.key()))
            .count() as u64
    }

    /// Get IDs matching a tag:value.
    pub fn ids_for_tag(&self, key: &str, value: &str) -> Option<RoaringBitmap> {
        let tag = format!("{key}:{value}");
        self.inverted.get(&tag).map(|r| r.clone())
    }

    /// Get IDs matching an @type.
    pub fn ids_for_type(&self, type_name: &str) -> Option<RoaringBitmap> {
        self.type_index.get(type_name).map(|r| r.clone())
    }

    /// Get IDs for a named_graph.
    pub fn ids_for_graph(&self, graph: &str) -> Option<RoaringBitmap> {
        self.graph_index.get(graph).map(|r| r.clone())
    }

    /// Get IDs for a @context.
    pub fn ids_for_context(&self, ctx: &str) -> Option<RoaringBitmap> {
        self.context_index.get(ctx).map(|r| r.clone())
    }

    /// Get IDs matching a numeric range.
    /// gt and lt are in f64, matched against milliscale (i64) buckets.
    pub fn ids_for_numeric_range(
        &self,
        key: &str,
        gte: Option<f64>,
        lte: Option<f64>,
    ) -> Option<RoaringBitmap> {
        let entry = self.numeric.get(key)?;
        let tree = entry.value().read().unwrap_or_else(|e| e.into_inner());
        let gte_q = gte.map(|v| (v * 1000.0) as i64).unwrap_or(i64::MIN);
        let lte_q = lte.map(|v| (v * 1000.0) as i64).unwrap_or(i64::MAX);

        let mut result = RoaringBitmap::new();
        for (_k, bm) in tree.range(gte_q..=lte_q) {
            result |= bm;
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    /// Vacuum: remove all deleted IDs from inverted/type/numeric indexes.
    /// Returns count of IDs that had entries in at least one index.
    pub fn vacuum(&self) -> u64 {
        let deleted_ids: Vec<u32> = {
            let del = self.deleted.read().unwrap_or_else(|e| e.into_inner());
            del.iter().collect()
        };
        for &id in &deleted_ids {
            // Scan all type_index entries for this ID
            let type_keys: Vec<String> = self
                .type_index
                .iter()
                .filter(|e| e.value().contains(id))
                .map(|e| e.key().clone())
                .collect();
            for k in type_keys {
                if let Some(mut bm) = self.type_index.get_mut(&k) {
                    bm.remove(id);
                    if bm.is_empty() {
                        drop(bm);
                        self.type_index.remove(&k);
                    }
                }
            }
            // Scan all inverted entries for this ID
            let inv_keys: Vec<String> = self
                .inverted
                .iter()
                .filter(|e| e.value().contains(id))
                .map(|e| e.key().clone())
                .collect();
            for k in inv_keys {
                if let Some(mut bm) = self.inverted.get_mut(&k) {
                    bm.remove(id);
                    if bm.is_empty() {
                        drop(bm);
                        self.inverted.remove(&k);
                    }
                }
            }
            // Scan numeric index entries for this ID
            for mut entry in self.numeric.iter_mut() {
                if let Ok(mut tree) = entry.value_mut().write() {
                    let to_remove: Vec<i64> = tree
                        .iter()
                        .filter(|(_, bm)| bm.contains(id))
                        .map(|(k, _)| *k)
                        .collect();
                    for k in to_remove {
                        if let Some(bm) = tree.get_mut(&k) {
                            bm.remove(id);
                            if bm.is_empty() {
                                tree.remove(&k);
                            }
                        }
                    }
                }
            }
        }
        // Remove all deleted IDs from forwards and deleted set
        for &id in &deleted_ids {
            self.forward.remove(&id);
        }
        if let Ok(mut del) = self.deleted.write() {
            del.clear();
        }
        deleted_ids.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_index_and_forward() {
        let idx = JsonLdMetadataIndex::new();
        let val = json!({
            "@type": ["Memory", "Episodic"],
            "@context": "https://schema.org",
            "named_graph": "agent_1",
            "tags": ["important", "recent"],
            "text": "Some memory content"
        });
        idx.index(42, &val);

        let payload = idx.get_payload(42);
        assert!(payload.is_some());
        assert_eq!(
            payload
                .unwrap()
                .get("@type")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            2
        );

        assert!(idx.type_index.contains_key("Memory"));
        assert!(idx.type_index.contains_key("Episodic"));
        assert!(idx.inverted.contains_key("tags:important"));
        assert!(idx.inverted.contains_key("tags:recent"));
    }

    #[test]
    fn test_remove_and_deleted() {
        let idx = JsonLdMetadataIndex::new();
        let val = json!({"@type": ["Test"], "text": "hello"});
        idx.index(1, &val);
        assert!(idx.forward.contains_key(&1));
        idx.remove(1);
        assert!(idx.forward.get(&1).is_none());
        assert!(idx.deleted.read().unwrap().contains(1u32));
    }

    #[test]
    fn test_count() {
        let idx = JsonLdMetadataIndex::new();
        idx.index(1, &json!({"text": "a"}));
        idx.index(2, &json!({"text": "b"}));
        assert_eq!(idx.count(), 2);
        idx.remove(1);
        assert_eq!(idx.count(), 1);
    }

    #[test]
    fn test_numeric_index() {
        let idx = JsonLdMetadataIndex::new();
        idx.index(1, &json!({"importance": 0.8, "text": "a"}));
        idx.index(2, &json!({"importance": 0.3, "text": "b"}));
        idx.index(3, &json!({"importance": 0.6, "text": "c"}));

        let result = idx.ids_for_numeric_range("importance", Some(0.5), None);
        assert!(result.is_some());
        let bm = result.unwrap();
        assert!(bm.contains(1));
        assert!(bm.contains(3));
        assert!(!bm.contains(2));
    }

    #[test]
    fn test_ids_for_type() {
        let idx = JsonLdMetadataIndex::new();
        idx.index(10, &json!({"@type": ["Document", "Report"], "text": "doc"}));
        idx.index(20, &json!({"@type": ["Document"], "text": "other"}));

        let docs = idx.ids_for_type("Document").unwrap();
        assert!(docs.contains(10));
        assert!(docs.contains(20));

        let reports = idx.ids_for_type("Report").unwrap();
        assert!(reports.contains(10));
        assert!(!reports.contains(20));
    }

    #[test]
    fn test_ids_for_graph() {
        let idx = JsonLdMetadataIndex::new();
        idx.index(1, &json!({"named_graph": "agent_a", "text": "x"}));
        let ids = idx.ids_for_graph("agent_a").unwrap();
        assert!(ids.contains(1));
        assert!(idx.ids_for_graph("agent_b").is_none());
    }

    #[test]
    fn test_vacuum_cleans_inverted() {
        let idx = JsonLdMetadataIndex::new();
        let val = json!({"@type": ["Test"], "tag": "value", "text": "x"});
        idx.index(1, &val);
        idx.remove(1);
        // Vacuum cleans all indexes with at least one matching entry
        let cleaned = idx.vacuum();
        assert_eq!(cleaned, 1);
        // After vacuum: type_index entry fully removed
        assert!(!idx.type_index.contains_key("Test"));
    }
}
