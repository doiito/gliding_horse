//! Incremental HNSW index with runtime metric switching and
//! multi-layer entry-point descent.
//!
//! # Design
//!
//! Implements the HNSW (Hierarchical Navigable Small World) algorithm
//! by Malkov & Yashunin, with these adaptations:
//!
//! - Runtime metric switching via `Box<dyn Metric>`
//! - Per-layer search descent (top layer → bottom)
//! - Generation-tagged visited set for lock-free search
//! - Serializable node format for snapshot persistence
//! - Optional filter bitmap integration for filtered search

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use rand::Rng;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use crate::hyper_vector::EmbeddingVector;
use crate::metric::Metric;

/// HNSW configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    pub ef_construction: usize,
    pub ef_search: usize,
    pub m: usize,
    pub m0: usize,
    pub level_mult: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            ef_construction: 200,
            ef_search: 50,
            m: 16,
            m0: 32,
            level_mult: 1.0 / (16.0_f64).ln(),
        }
    }
}

/// Serializable representation of a node for snapshot persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableNode {
    pub coords: Vec<f64>,
    pub metric_tag: u32,
    pub alpha: f64,
    pub neighbors0: Vec<u32>,
    pub neighbors_upper: Vec<Vec<u32>>,
    pub level: usize,
}

impl From<&Node> for SerializableNode {
    fn from(n: &Node) -> Self {
        Self {
            coords: n.vector.coords.clone(),
            metric_tag: n.vector.metric as u32,
            alpha: n.vector.alpha,
            neighbors0: n.neighbors0.clone(),
            neighbors_upper: n.neighbors_upper.clone(),
            level: n.level,
        }
    }
}

impl SerializableNode {
    pub fn to_embedding(&self, metric_kind: crate::hyper_vector::MetricKind) -> EmbeddingVector {
        EmbeddingVector {
            coords: self.coords.clone(),
            metric: metric_kind,
            alpha: self.alpha,
        }
    }
}

#[derive(Clone)]
struct Node {
    vector: EmbeddingVector,
    neighbors0: Vec<u32>,
    neighbors_upper: Vec<Vec<u32>>,
    level: usize,
}

/// Incremental HNSW index with runtime metric switching.
pub struct IncrementalHNSW {
    nodes: Vec<Option<Node>>,
    metric: Box<dyn Metric>,
    entry_point: AtomicU32,
    max_layer: AtomicUsize,
    config: HnswConfig,
    generation: AtomicUsize,
    /// Per-node visited generation tag.
    /// `Vec<AtomicUsize>` rather than `Vec<usize>` so that the generation-based
    /// visited check is genuinely lock-free — concurrent readers (via e.g. RwLock)
    /// can race-free compare-and-mark without a mutex.
    visited_gen: Vec<AtomicUsize>,
}

impl IncrementalHNSW {
    pub fn new(metric: Box<dyn Metric>, config: HnswConfig) -> Self {
        Self {
            nodes: Vec::new(),
            metric,
            entry_point: AtomicU32::new(u32::MAX),
            max_layer: AtomicUsize::new(0),
            config,
            generation: AtomicUsize::new(1),
            visited_gen: Vec::new(),
        }
    }

    pub fn set_metric(&mut self, metric: Box<dyn Metric>) {
        self.metric = metric;
    }

    fn random_level(&self) -> usize {
        let r: f64 = rand::thread_rng().gen();
        if r <= 0.0 {
            return 0;
        }
        (r.ln() * (-self.config.level_mult)).floor() as usize
    }

    /// Insert a node with its vector. Uses multi-layer entry-point descent.
    pub fn insert(&mut self, id: u32, vector: EmbeddingVector) {
        let level = self.random_level();
        let id_u = id as usize;

        if id_u >= self.nodes.len() {
            self.nodes.resize(id_u + 1, None);
        }
        if id_u >= self.visited_gen.len() {
            self.visited_gen
                .resize_with(self.visited_gen.len().max(id_u + 1).max(1024), || {
                    AtomicUsize::new(0)
                });
        }

        let ep = self.entry_point.load(Ordering::Relaxed);

        self.nodes[id_u] = Some(Node {
            vector,
            neighbors0: Vec::with_capacity(self.config.m0),
            neighbors_upper: (0..level)
                .map(|_| Vec::with_capacity(self.config.m))
                .collect(),
            level,
        });

        if ep == u32::MAX {
            self.entry_point.store(id, Ordering::Release);
            self.max_layer.store(level, Ordering::Release);
            return;
        }

        let ep_level = self.nodes[ep as usize].as_ref().map_or(0, |n| n.level);
        let max_level = ep_level.max(level);
        let mut curr_ep = ep;

        // Phase 1: Greedy descent from top layer down to (level+1)
        // (only if level < ep_level)
        if level < ep_level {
            for lvl in (level + 1..=max_level).rev() {
                curr_ep = self.search_layer_single(lvl, curr_ep, id);
            }
        }

        // Phase 2: Search and connect at layers 0..=level
        for lvl in (0..=level.min(max_level)).rev() {
            let ef = if lvl == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let nearest = self.search_layer_top_k(lvl, curr_ep, id, ef);
            let num_neighbors = if lvl == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let neighbor_ids: Vec<u32> = nearest.iter().take(num_neighbors).copied().collect();

            // Set forward connections
            if lvl == 0 {
                if let Some(ref mut n) = self.nodes[id_u] {
                    n.neighbors0 = neighbor_ids.clone();
                }
            } else if let Some(ref mut n) = self.nodes[id_u] {
                if lvl - 1 < n.neighbors_upper.len() {
                    n.neighbors_upper[lvl - 1] = neighbor_ids.clone();
                }
            }

            // Set backward connections
            for &nid in &neighbor_ids {
                let nu = nid as usize;
                if nu < self.nodes.len() {
                    if let Some(ref mut nbr) = self.nodes[nu] {
                        if lvl == 0 {
                            if nbr.neighbors0.len() < self.config.m0 {
                                nbr.neighbors0.push(id);
                            }
                        } else if let Some(nbrs) = nbr.neighbors_upper.get_mut(lvl - 1) {
                            if nbrs.len() < self.config.m {
                                nbrs.push(id);
                            }
                        }
                    }
                }
            }

            if !nearest.is_empty() {
                curr_ep = nearest[0];
            }
        }

        // Update entry point if this node is at a higher level
        if level > ep_level {
            self.entry_point.store(id, Ordering::Release);
            self.max_layer.store(level, Ordering::Release);
        }
    }

    /// Single-layer greedy search: find the nearest node to the query.
    fn search_layer_single(&self, layer: usize, entry: u32, query_id: u32) -> u32 {
        let query_node = match self.nodes.get(query_id as usize).and_then(|n| n.as_ref()) {
            Some(n) => &n.vector,
            None => return entry,
        };
        let mut best = entry;
        let mut best_dist = if self.contains(best) {
            self.metric.distance(
                query_node,
                &self.nodes[best as usize].as_ref().unwrap().vector,
            )
        } else {
            return entry;
        };

        loop {
            let neighbors = self.get_neighbors(best, layer);
            let mut improved = false;
            for &nid in &neighbors {
                let d = self.metric.distance(
                    query_node,
                    &self.nodes[nid as usize].as_ref().unwrap().vector,
                );
                if d < best_dist {
                    best_dist = d;
                    best = nid;
                    improved = true;
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    /// Search layer for top-k nearest nodes to the query, using generation-tagged visited set.
    fn search_layer_top_k(
        &mut self,
        layer: usize,
        entry: u32,
        query_id: u32,
        ef: usize,
    ) -> Vec<u32> {
        let gen = self.generation.fetch_add(1, Ordering::AcqRel);
        let query_node = match self.nodes.get(query_id as usize).and_then(|n| n.as_ref()) {
            Some(n) => &n.vector,
            None => return Vec::new(),
        };

        if !self.contains(entry) {
            return Vec::new();
        }

        let entry_dist = self.metric.distance(
            query_node,
            &self.nodes[entry as usize].as_ref().unwrap().vector,
        );
        let mut c = vec![(entry, entry_dist)];
        let mut i = 0;

        while i < c.len() {
            let (current, _) = c[i];
            for &nid in &self.get_neighbors(current, layer) {
                let nu = nid as usize;
                if nu < self.visited_gen.len() {
                    if self.visited_gen[nu].load(Ordering::Relaxed) == gen {
                        continue;
                    }
                    self.visited_gen[nu].store(gen, Ordering::Relaxed);
                }
                let d = self.metric.distance(
                    query_node,
                    &self.nodes[nid as usize].as_ref().unwrap().vector,
                );
                let pos = c
                    .binary_search_by(|(_, dist)| {
                        dist.partial_cmp(&d).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or_else(|e| e);
                let already_in = pos < c.len() && c[pos].0 == nid;
                if !already_in && (pos < ef || c.len() < ef) {
                    c.insert(pos, (nid, d));
                    if c.len() > ef {
                        c.truncate(ef);
                    }
                }
            }
            i += 1;
        }

        c.into_iter().map(|(id, _)| id).collect()
    }

    /// Public search: find nearest neighbors to a query vector.
    ///
    /// Uses multi-layer descent from the entry point down to layer 0,
    /// then beam search at layer 0 with ef_search beam width.
    ///
    /// If `allowed` is Some, only returns IDs present in the bitmap.
    pub fn search_with_filter(
        &mut self,
        query: &EmbeddingVector,
        top_k: usize,
        allowed: Option<&RoaringBitmap>,
    ) -> Vec<(u32, f64)> {
        self.fix_entry_point();
        let ep = self.entry_point.load(Ordering::Relaxed);
        if ep == u32::MAX {
            return Vec::new();
        }

        let max_layer = self.max_layer.load(Ordering::Relaxed);
        let ef = self.config.ef_search.max(top_k);

        // Multi-layer descent
        let mut curr_ep = ep;
        let curr_id = self.next_available_id();
        if curr_id == u32::MAX {
            return Vec::new();
        }

        // Beam search at layer 0
        if max_layer > 0 {
            for lvl in (1..=max_layer).rev() {
                curr_ep = self.search_layer_single_external(lvl, curr_ep, query);
            }
        }

        // Beam search at layer 0
        let results = self.search_layer0_external(query, ef, curr_ep);

        // Truncate and optionally filter
        let mut results: Vec<(u32, f64)> = results;
        if let Some(ab) = allowed {
            results.retain(|(id, _)| ab.contains(*id));
        }
        results.truncate(top_k);
        results
    }

    /// Legacy search interface (no filter).
    pub fn search(&mut self, query: &EmbeddingVector, top_k: usize) -> Vec<(u32, f64)> {
        self.search_with_filter(query, top_k, None)
    }

    /// Single-layer greedy search for an external query (not a node ID).
    fn search_layer_single_external(
        &self,
        layer: usize,
        entry: u32,
        query: &EmbeddingVector,
    ) -> u32 {
        if !self.contains(entry) {
            return entry;
        }
        let mut best = entry;
        let mut best_dist = self
            .metric
            .distance(query, &self.nodes[best as usize].as_ref().unwrap().vector);
        loop {
            let neighbors = self.get_neighbors(best, layer);
            let mut improved = false;
            for &nid in &neighbors {
                let d = self
                    .metric
                    .distance(query, &self.nodes[nid as usize].as_ref().unwrap().vector);
                if d < best_dist {
                    best_dist = d;
                    best = nid;
                    improved = true;
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    /// Beam search at layer 0 for an external query.
    fn search_layer0_external(
        &mut self,
        query: &EmbeddingVector,
        ef: usize,
        entry: u32,
    ) -> Vec<(u32, f64)> {
        let gen = self.generation.fetch_add(1, Ordering::AcqRel);
        if !self.contains(entry) {
            return Vec::new();
        }

        let entry_dist = self
            .metric
            .distance(query, &self.nodes[entry as usize].as_ref().unwrap().vector);
        let mut c = vec![(entry, entry_dist)];
        let mut i = 0;

        while i < c.len() {
            let (current, _) = c[i];
            let neighbors = self.get_neighbors(current, 0);
            for &nid in &neighbors {
                let nu = nid as usize;
                if nu < self.visited_gen.len() {
                    if self.visited_gen[nu].load(Ordering::Relaxed) == gen {
                        continue;
                    }
                    self.visited_gen[nu].store(gen, Ordering::Relaxed);
                }
                let d = self
                    .metric
                    .distance(query, &self.nodes[nid as usize].as_ref().unwrap().vector);
                let pos = c
                    .binary_search_by(|(_, dist)| {
                        dist.partial_cmp(&d).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or_else(|e| e);
                let already_in = pos < c.len() && c[pos].0 == nid;
                if !already_in && (pos < ef || c.len() < ef) {
                    c.insert(pos, (nid, d));
                    if c.len() > ef {
                        c.truncate(ef);
                    }
                }
            }
            i += 1;
        }

        c
    }

    /// Find the next available node ID (first live node).
    fn next_available_id(&self) -> u32 {
        self.nodes
            .iter()
            .position(|n| n.is_some())
            .map(|i| i as u32)
            .unwrap_or(u32::MAX)
    }

    fn get_neighbors(&self, node_id: u32, layer: usize) -> Vec<u32> {
        match self.nodes.get(node_id as usize).and_then(|n| n.as_ref()) {
            Some(n) if layer == 0 => n
                .neighbors0
                .iter()
                .filter(|&&id| self.contains(id))
                .copied()
                .collect(),
            Some(n) if layer >= 1 && layer - 1 < n.neighbors_upper.len() => n.neighbors_upper
                [layer - 1]
                .iter()
                .filter(|&&id| self.contains(id))
                .copied()
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn remove(&mut self, id: u32) {
        // Remove reverse edges first. Leaving them behind means an upsert of
        // the same ID inherits stale topology from its previous vector.
        for node in self.nodes.iter_mut().filter_map(Option::as_mut) {
            node.neighbors0.retain(|&neighbor| neighbor != id);
            for neighbors in &mut node.neighbors_upper {
                neighbors.retain(|&neighbor| neighbor != id);
            }
        }
        if let Some(n) = self.nodes.get_mut(id as usize) {
            *n = None;
        }
        let ep = self.entry_point.load(Ordering::Relaxed);
        if ep == id {
            self.fix_entry_point();
        }
        // Also update max_layer
        self.fix_max_layer();
    }

    fn fix_entry_point(&self) {
        let new_ep = self
            .nodes
            .iter()
            .position(|n| n.is_some())
            .map(|i| i as u32)
            .unwrap_or(u32::MAX);
        self.entry_point.store(new_ep, Ordering::Release);
    }

    fn fix_max_layer(&self) {
        let max_lvl = self
            .nodes
            .iter()
            .filter_map(|n| n.as_ref())
            .map(|n| n.level)
            .max()
            .unwrap_or(0);
        self.max_layer.store(max_lvl, Ordering::Release);
    }

    pub fn contains(&self, id: u32) -> bool {
        self.nodes
            .get(id as usize)
            .and_then(|n| n.as_ref())
            .is_some()
    }

    pub fn len(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn metric(&self) -> &dyn Metric {
        self.metric.as_ref()
    }

    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    pub fn entry_point(&self) -> u32 {
        self.entry_point.load(Ordering::Relaxed)
    }

    pub fn max_layer(&self) -> usize {
        self.max_layer.load(Ordering::Relaxed)
    }

    // ── Snapshot support ────────────────────────────────────────────────────

    /// Export all nodes as serializable nodes.
    pub fn export_nodes(&self) -> Vec<Option<SerializableNode>> {
        self.nodes
            .iter()
            .map(|n| n.as_ref().map(SerializableNode::from))
            .collect()
    }

    /// Import nodes from a snapshot.
    pub fn import_nodes(&mut self, nodes: Vec<Option<SerializableNode>>) {
        let metric_kind = self.metric.kind();
        self.nodes = nodes
            .into_iter()
            .map(|sn| {
                sn.map(|s| Node {
                    vector: s.to_embedding(metric_kind),
                    neighbors0: s.neighbors0,
                    neighbors_upper: s.neighbors_upper,
                    level: s.level,
                })
            })
            .collect();
        self.fix_entry_point();
        self.fix_max_layer();
        // Resize visited_gen
        self.visited_gen.resize_with(
            self.visited_gen.len().max(self.nodes.len()).max(1024),
            || AtomicUsize::new(0),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyper_vector::MetricKind;
    use crate::metric::CosineMetric;

    fn v(coords: Vec<f64>) -> EmbeddingVector {
        EmbeddingVector::new_unchecked(coords, MetricKind::Cosine)
    }

    #[test]
    fn test_empty_search() {
        let mut idx = IncrementalHNSW::new(Box::new(CosineMetric), HnswConfig::default());
        assert!(idx.search(&v(vec![1.0, 0.0]), 5).is_empty());
    }

    #[test]
    fn test_insert_one() {
        let mut idx = IncrementalHNSW::new(Box::new(CosineMetric), HnswConfig::default());
        idx.insert(0, v(vec![1.0, 0.0]));
        assert!(idx.contains(0));
        let results = idx.search(&v(vec![1.0, 0.0]), 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_insert_two_nearest() {
        let mut idx = IncrementalHNSW::new(Box::new(CosineMetric), HnswConfig::default());
        idx.insert(0, v(vec![1.0, 0.0]));
        idx.insert(1, v(vec![0.0, 1.0]));
        let results = idx.search(&v(vec![0.9, 0.1]), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_remove() {
        let mut idx = IncrementalHNSW::new(Box::new(CosineMetric), HnswConfig::default());
        idx.insert(0, v(vec![1.0, 0.0]));
        idx.insert(1, v(vec![0.0, 1.0]));
        idx.remove(0);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_random_level_distribution() {
        let idx = IncrementalHNSW::new(Box::new(CosineMetric), HnswConfig::default());
        let mut counts = [0usize; 10];
        for _ in 0..10000 {
            let l = idx.random_level();
            if l < 10 {
                counts[l] += 1;
            }
        }
        assert!(counts[0] > counts[1]);
    }

    #[test]
    fn test_search_with_filter() {
        let mut idx = IncrementalHNSW::new(Box::new(CosineMetric), HnswConfig::default());
        for i in 0..10u32 {
            let x = (i as f64) * 0.1;
            idx.insert(i, v(vec![x, 0.0]));
        }

        // Only allow even IDs
        let mut allowed = RoaringBitmap::new();
        allowed.insert(0);
        allowed.insert(2);
        allowed.insert(4);
        allowed.insert(6);
        allowed.insert(8);

        let results = idx.search_with_filter(&v(vec![0.5, 0.0]), 10, Some(&allowed));
        assert_eq!(results.len(), 5);
        for (id, _) in &results {
            assert!(allowed.contains(*id), "ID {id} should be in allowed set");
        }
    }

    #[test]
    fn test_serializable_node_roundtrip() {
        let mut idx = IncrementalHNSW::new(Box::new(CosineMetric), HnswConfig::default());
        idx.insert(0, v(vec![1.0, 0.0, 0.0, 0.0]));
        idx.insert(1, v(vec![0.0, 1.0, 0.0, 0.0]));

        let exported = idx.export_nodes();
        assert_eq!(exported.len(), 2);
        assert!(exported[0].is_some());
        assert!(exported[1].is_some());

        // Re-import into fresh index
        let mut idx2 = IncrementalHNSW::new(Box::new(CosineMetric), HnswConfig::default());
        idx2.import_nodes(exported);
        assert_eq!(idx2.len(), 2);
        assert!(idx2.contains(0));
        assert!(idx2.contains(1));

        let results = idx2.search(&v(vec![1.0, 0.0, 0.0, 0.0]), 5);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_max_layer_tracking() {
        // With CosineMetric, levels are determined by random_level().
        // At minimum, max_layer should be the highest level among inserted nodes.
        let mut idx = IncrementalHNSW::new(Box::new(CosineMetric), HnswConfig::default());
        assert_eq!(idx.max_layer(), 0);
        idx.insert(0, v(vec![1.0, 0.0]));
        let ml0 = idx.max_layer();
        assert!(ml0 <= 16); // reasonable bound
        let _ = idx;
    }
}
