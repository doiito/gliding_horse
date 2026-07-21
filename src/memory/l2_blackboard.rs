use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use oxigraph::store::Store;
use oxigraph::sparql::QueryResults;
use parking_lot::RwLock;
use tracing::{debug, info, instrument, warn};

use crate::core::agent_instance::AgentRole;
use crate::memory::l0_store::MesiState;
use crate::{CoreConfig, CoreError};

/// Filter criteria for query_nodes_filtered()
///
/// When multiple fields are set, results are the INTERSECTION of all filters.
/// When no fields are set, returns ALL nodes (same as query_nodes()).
#[derive(Debug, Clone, Default)]
pub struct QueryFilter {
    pub role: Option<AgentRole>,
    pub cycle_id: Option<String>,
    pub node_type: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Node {
    pub iri: String,
    pub json_ld: String,
    pub size: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub created_by: Option<String>,
    pub tags: Vec<String>,
    pub node_type: Option<String>,
    pub dirty: bool,
    pub mesi_state: MesiState,
    pub parent_task: Option<String>,
    pub named_graph: Option<String>,
    pub jsonld_types: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TaskTreeNode {
    pub task_iri: String,
    pub parent: Option<String>,
    pub children: Vec<String>,
    pub dependencies: Vec<String>,
    pub dependents: Vec<String>,
    pub status: String,
    pub node_iris: Vec<String>,
}

pub struct Blackboard {
    store: Arc<Store>,
    node_cache: DashMap<String, Arc<Node>>,
    task_nodes: RwLock<HashMap<String, Vec<String>>>,
    task_tree: RwLock<HashMap<String, TaskTreeNode>>,

    // ── Secondary indices for multi-dimensional query ──
    // Index: (task_iri, role) → Vec<node_iri>
    role_index: RwLock<HashMap<(String, AgentRole), Vec<String>>>,
    // Index: (task_iri, cycle_id) → Vec<node_iri>
    cycle_index: RwLock<HashMap<(String, String), Vec<String>>>,
    // Index: (task_iri, node_type) → Vec<node_iri>
    type_index: RwLock<HashMap<(String, String), Vec<String>>>,

    node_count: AtomicU64,
    total_bytes: AtomicU64,
    permission_matrix: PermissionMatrix,
    // Battle map: Agent situational awareness
    agent_registry: RwLock<HashMap<String, AgentStatus>>,
    // Battle map: Resource lock table
    resource_locks: RwLock<Vec<ResourceLock>>,

    /// Pending Oxigraph sync thread handles — joined on flush_oxigraph()
    pending_syncs: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Blackboard {
    /// Create Blackboard with a shared unified store
    pub fn with_store(store: Arc<Store>) -> Result<Self, CoreError> {
        info!("Initializing L2 Blackboard with shared store");
        Ok(Self {
            store,
            node_cache: DashMap::new(),
            task_nodes: RwLock::new(HashMap::new()),
            task_tree: RwLock::new(HashMap::new()),
            role_index: RwLock::new(HashMap::new()),
            cycle_index: RwLock::new(HashMap::new()),
            type_index: RwLock::new(HashMap::new()),
            node_count: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            permission_matrix: PermissionMatrix::new(),
            agent_registry: RwLock::new(HashMap::new()),
            resource_locks: RwLock::new(Vec::new()),
            pending_syncs: std::sync::Mutex::new(Vec::new()),
        })
    }

    pub fn new() -> Result<Self, CoreError> {
        info!("Initializing L2 Blackboard");
        Ok(Self {
            store: Arc::new(Store::new()?),
            node_cache: DashMap::new(),
            task_nodes: RwLock::new(HashMap::new()),
            task_tree: RwLock::new(HashMap::new()),
            role_index: RwLock::new(HashMap::new()),
            cycle_index: RwLock::new(HashMap::new()),
            type_index: RwLock::new(HashMap::new()),
            node_count: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            permission_matrix: PermissionMatrix::new(),
            agent_registry: RwLock::new(HashMap::new()),
            resource_locks: RwLock::new(Vec::new()),
            pending_syncs: std::sync::Mutex::new(Vec::new()),
        })
    }

    #[instrument(skip(self, json_ld, config))]
    pub fn write_node(
        &self,
        node_iri: &str,
        json_ld: &str,
        config: &CoreConfig,
    ) -> Result<(), CoreError> {
        let task_iri = extract_task_iri(node_iri);
        let graph_name = task_iri.as_deref().unwrap_or("system:default");
        if !self.permission_matrix.check_permission("system", graph_name, GraphPermission::Write) {
            return Err(CoreError::PermissionDenied {
                agent: "system".to_string(),
                resource: node_iri.to_string(),
                action: "write_node".to_string(),
            });
        }

        let size = json_ld.as_bytes().len();

        if size > config.max_node_size {
            return Err(CoreError::NodeTooLarge {
                size,
                max: config.max_node_size,
            });
        }

        let parsed: serde_json::Value = serde_json::from_str(json_ld)
            .map_err(|e| CoreError::InvalidJsonLd {
                message: format!("JSON parse error: {}", e),
            })?;

        let task_iri = extract_task_iri(node_iri);

        let is_update = self.node_cache.contains_key(node_iri);

        let jsonld_types = extract_jsonld_types(&parsed);

        let node = Node {
            iri: node_iri.to_string(),
            json_ld: json_ld.to_string(),
            size,
            created_at: chrono::Utc::now(),
            created_by: parsed.get("created_by").and_then(|v| v.as_str()).map(String::from),
            tags: parsed.get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            node_type: jsonld_types.first().cloned(),
            dirty: is_update,
            mesi_state: if is_update { MesiState::Modified } else { MesiState::Shared },
            parent_task: task_iri.clone(),
            named_graph: parsed.get("named_graph").and_then(|v| v.as_str()).map(String::from),
            jsonld_types: jsonld_types.clone(),
        };

        self.node_cache.insert(node_iri.to_string(), Arc::new(node));

        // Defer Oxigraph sync to background thread to avoid blocking the write path.
        // The in-memory cache returns immediately; sync errors are logged but not propagated
        // since the node is already available via cache + indices.
        let store = self.store.clone();
        let iri = node_iri.to_string();
        let parsed_json = parsed.clone();
        let handle = std::thread::spawn(move || {
            if let Err(e) = Self::sync_to_oxigraph_sync(&store, &iri, &parsed_json) {
                warn!(error = %e, node_iri = %iri, "Background Oxigraph sync failed");
            }
        });
        self.pending_syncs.lock().unwrap().push(handle);

        if let Some(task_iri) = &task_iri {
            let mut task_nodes = self.task_nodes.write();
            let entry = task_nodes.entry(task_iri.clone()).or_default();
            if !entry.contains(&node_iri.to_string()) {
                entry.push(node_iri.to_string());
            }

            let mut tree = self.task_tree.write();
            let tree_node = tree.entry(task_iri.clone()).or_insert_with(|| TaskTreeNode {
                task_iri: task_iri.clone(),
                parent: None,
                children: Vec::new(),
                dependencies: Vec::new(),
                dependents: Vec::new(),
                status: "running".to_string(),
                node_iris: Vec::new(),
            });
            if !tree_node.node_iris.contains(&node_iri.to_string()) {
                tree_node.node_iris.push(node_iri.to_string());
            }
        }

        // ── Update secondary indices ──
        if let Some(task_iri) = &task_iri {
            // Role index
            if let Some(role_str) = parsed.get("role").and_then(|v| v.as_str()) {
                if let Ok(role) = role_str.parse::<AgentRole>() {
                    let mut idx = self.role_index.write();
                    let entry = idx.entry((task_iri.clone(), role)).or_default();
                    if !entry.contains(&node_iri.to_string()) {
                        entry.push(node_iri.to_string());
                    }
                }
            }

            // Cycle index
            if let Some(cycle_id) = parsed.get("cycle_id").and_then(|v| v.as_str()) {
                let mut idx = self.cycle_index.write();
                let entry = idx.entry((task_iri.clone(), cycle_id.to_string())).or_default();
                if !entry.contains(&node_iri.to_string()) {
                    entry.push(node_iri.to_string());
                }
            }

            // Type index
            if let Some(node_type) = jsonld_types.first() {
                let mut idx = self.type_index.write();
                let entry = idx.entry((task_iri.clone(), node_type.clone())).or_default();
                if !entry.contains(&node_iri.to_string()) {
                    entry.push(node_iri.to_string());
                }
            }
        }

        if !is_update {
            self.node_count.fetch_add(1, Ordering::Relaxed);
        }
        self.total_bytes.fetch_add(size as u64, Ordering::Relaxed);

        debug!(node_iri = %node_iri, size = size, is_update = is_update, "Node written to blackboard (cache + oxigraph)");
        Ok(())
    }

    fn build_triples(node_iri: &str, parsed: &serde_json::Value) -> Vec<String> {
        let subject = format!("<{}>", node_iri);
        let mut triples = Vec::new();

        if let Some(obj) = parsed.as_object() {
            for (key, value) in obj {
                if key.starts_with('@') {
                    if key == "@type" {
                        if let Some(t) = value.as_str() {
                            let type_uri = if t.contains("://") { format!("<{}>", t) } else { format!("<http://agent-os.org/ontology/{}>", t) };
                            triples.push(format!("{} a {} .", subject, type_uri));
                        } else if let Some(arr) = value.as_array() {
                            for t in arr.iter().filter_map(|v| v.as_str()) {
                                let type_uri = if t.contains("://") { format!("<{}>", t) } else { format!("<http://agent-os.org/ontology/{}>", t) };
                                triples.push(format!("{} a {} .", subject, type_uri));
                            }
                        }
                    }
                    continue;
                }

                let escaped_key = key.replace(' ', "_");
                let predicate = format!("<http://agent-os.org/ontology/{}>", escaped_key);

                match value {
                    serde_json::Value::String(s) => {
                        let escaped = Self::escape_sparql_string(s);
                        triples.push(format!(r#"{} {} "{}" ."#, subject, predicate, escaped));
                    }
                    serde_json::Value::Number(n) => {
                        triples.push(format!("{} {} {} .", subject, predicate, n));
                    }
                    serde_json::Value::Bool(b) => {
                        triples.push(format!("{} {} {} .", subject, predicate, if *b { "true" } else { "false" }));
                    }
                    serde_json::Value::Null => {}
                    _ => {
                        let escaped = Self::escape_sparql_string(&value.to_string());
                        triples.push(format!(r#"{} {} "{}" ."#, subject, predicate, escaped));
                    }
                }
            }
        }

        triples
    }

    /// Background-sync a node to Oxigraph (called from std::thread::spawn).
    /// Uses build_triples + SPARQL UPDATE to keep ogx in sync with the cache.
    fn sync_to_oxigraph_sync(store: &Store, node_iri: &str, parsed: &serde_json::Value) -> Result<(), CoreError> {
        let triples = Self::build_triples(node_iri, parsed);

        if !triples.is_empty() {
            let subject = format!("<{}>", node_iri);
            let combined_sparql = format!(
                "DELETE WHERE {{ {} ?p ?o . }}; INSERT DATA {{ {} }}",
                subject,
                triples.join("\n")
            );
            store.update(&combined_sparql)
                .map_err(|e| CoreError::SparqlError {
                    message: format!("Failed to execute atomic DELETE+INSERT: {}", e),
                })?;
        }

        Ok(())
    }

    fn escape_sparql_string(s: &str) -> String {
        let mut escaped = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '\\' => escaped.push_str("\\\\"),
                '"' => escaped.push_str("\\\""),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                '\x08' => escaped.push_str("\\b"),
                '\x0c' => escaped.push_str("\\f"),
                c if c.is_control() => {
                    escaped.push_str(&format!("\\u{:04x}", c as u32));
                }
                c => escaped.push(c),
            }
        }
        escaped
    }

    pub fn read_node(&self, node_iri: &str) -> Result<Option<Arc<Node>>, CoreError> {
        Ok(self.node_cache.get(node_iri).map(|n| n.clone()))
    }

    pub fn delete_node(&self, node_iri: &str) -> Result<bool, CoreError> {
        if let Some((_, node)) = self.node_cache.remove(node_iri) {
            self.node_count.fetch_sub(1, Ordering::Relaxed);
            self.total_bytes.fetch_sub(node.size as u64, Ordering::Relaxed);

            let subject = format!("<{}>", node_iri);
            let delete_sparql = if let Some(ref graph_name) = node.named_graph {
                let graph = format!("<{}>", graph_name);
                format!("DELETE WHERE {{ GRAPH {} {{ {} ?p ?o . }} }}", graph, subject)
            } else {
                format!("DELETE WHERE {{ {} ?p ?o . }}", subject)
            };
            if let Err(e) = self.store.update(&delete_sparql) {
                warn!(node_iri = %node_iri, error = %e, "Failed to delete triples from oxigraph");
            }

            if let Some(task_iri) = extract_task_iri(node_iri) {
                let mut task_nodes = self.task_nodes.write();
                if let Some(nodes) = task_nodes.get_mut(&task_iri) {
                    nodes.retain(|iri| iri != node_iri);
                }

                // Clean up secondary indices
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&node.json_ld) {
                    // Role index
                    if let Some(role_str) = parsed.get("role").and_then(|v| v.as_str()) {
                        if let Ok(role) = role_str.parse::<AgentRole>() {
                            if let Some(role_nodes) = self.role_index.write().get_mut(&(task_iri.clone(), role)) {
                                role_nodes.retain(|iri| iri != node_iri);
                            }
                        }
                    }

                    // Cycle index
                    if let Some(cycle_id) = parsed.get("cycle_id").and_then(|v| v.as_str()) {
                        let key = (task_iri.clone(), cycle_id.to_string());
                        if let Some(cycle_nodes) = self.cycle_index.write().get_mut(&key) {
                            cycle_nodes.retain(|iri| iri != node_iri);
                        }
                    }

                    // Type index (for all jsonld_types)
                    for node_type in &node.jsonld_types {
                        let key = (task_iri.clone(), node_type.clone());
                        if let Some(type_nodes) = self.type_index.write().get_mut(&key) {
                            type_nodes.retain(|iri| iri != node_iri);
                        }
                    }
                }
            }

            debug!(node_iri = %node_iri, named_graph = ?node.named_graph, "Node deleted from blackboard (cache + oxigraph)");
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn mark_dirty(&self, node_iri: &str) {
        if let Some(mut arc_node) = self.node_cache.get_mut(node_iri) {
            let mut node = (**arc_node).clone();
            node.dirty = true;
            node.mesi_state = MesiState::Modified;
            *arc_node = Arc::new(node);
        }
    }

    pub fn flush_dirty_nodes(&self, l0_store: &crate::memory::l0_store::L0Store) -> Result<usize, CoreError> {
        let mut flushed = 0;
        let dirty_iris: Vec<String> = self.node_cache.iter()
            .filter(|e| e.value().dirty)
            .map(|e| e.key().clone())
            .collect();

        for iri in dirty_iris {
            if let Some(mut arc_node) = self.node_cache.get_mut(&iri) {
                let entry = crate::memory::l0_store::L0Entry {
                    iri: arc_node.iri.clone(),
                    content: arc_node.json_ld.clone(),
                    importance: 0.5,
                    access_count: 0,
                    created_at: arc_node.created_at,
                    last_accessed: chrono::Utc::now(),
                    tags: arc_node.tags.clone(),
                    metadata: serde_json::Map::new(),
                    mesi_state: crate::memory::l0_store::MesiState::Shared,
                    content_hash: String::new(),
                    named_graph: arc_node.named_graph.clone(),
                    jsonld_context: None,
                    jsonld_types: arc_node.jsonld_types.clone(),
                    hyperspace_point_id: None,
                };
                l0_store.store_entry(&entry)?;
                let mut node = (**arc_node).clone();
                node.dirty = false;
                node.mesi_state = MesiState::Shared;
                *arc_node = Arc::new(node);
                flushed += 1;
            }
        }

        Ok(flushed)
    }

    pub fn release_subtree(&self, task_iri: &str) -> Result<usize, CoreError> {
        let mut released = 0;

        let mut all_node_iris = Vec::new();
        let mut tasks_to_release = vec![task_iri.to_string()];

        {
            let tree = self.task_tree.read();
            let mut idx = 0;
            while idx < tasks_to_release.len() {
                if let Some(node) = tree.get(&tasks_to_release[idx]) {
                    all_node_iris.extend(node.node_iris.iter().cloned());
                    tasks_to_release.extend(node.children.iter().cloned());
                }
                idx += 1;
            }
        }

        for iri in &all_node_iris {
            if self.delete_node(iri).unwrap_or(false) {
                released += 1;
            }
        }

        {
            let mut tree = self.task_tree.write();
            for task in &tasks_to_release {
                tree.remove(task);
            }
        }

        Ok(released)
    }

    pub fn register_task(&self, task_iri: &str, parent: Option<String>) {
        let mut tree = self.task_tree.write();
        let tree_node = tree.entry(task_iri.to_string()).or_insert_with(|| TaskTreeNode {
            task_iri: task_iri.to_string(),
            parent: parent.clone(),
            children: Vec::new(),
            dependencies: Vec::new(),
            dependents: Vec::new(),
            status: "running".to_string(),
            node_iris: Vec::new(),
        });
        tree_node.parent = parent.clone();

        if let Some(parent_iri) = &parent {
            if let Some(parent_node) = tree.get_mut(parent_iri) {
                if !parent_node.children.contains(&task_iri.to_string()) {
                    parent_node.children.push(task_iri.to_string());
                }
            }
        }
    }

    pub fn complete_task(&self, task_iri: &str, status: &str) {
        let mut tree = self.task_tree.write();
        if let Some(node) = tree.get_mut(task_iri) {
            node.status = status.to_string();
        }
    }

    pub fn sparql_update(&self, sparql: &str) -> Result<(), CoreError> {
        self.store
            .update(sparql)
            .map_err(|e| CoreError::SparqlError {
                message: format!("SPARQL UPDATE failed: {}", e),
            })?;
        debug!(sparql_len = sparql.len(), "SPARQL UPDATE");
        Ok(())
    }

    #[allow(deprecated)]
    pub fn query(&self, sparql: &str) -> Result<Vec<serde_json::Value>, CoreError> {
        let results = self.store.query(sparql)?;
        let mut values = Vec::new();

        match results {
            QueryResults::Solutions(solutions) => {
                for solution in solutions {
                    let solution = solution?;
                    let mut obj = serde_json::Map::new();
                    for (var, value) in solution.iter() {
                        obj.insert(var.to_string(), serde_json::Value::String(value.to_string()));
                    }
                    values.push(serde_json::Value::Object(obj));
                }
            }
            QueryResults::Graph(graph) => {
                for triple in graph {
                    let triple = triple?;
                    let mut obj = serde_json::Map::new();
                    obj.insert("subject".to_string(), serde_json::Value::String(triple.subject.to_string()));
                    obj.insert("predicate".to_string(), serde_json::Value::String(triple.predicate.to_string()));
                    obj.insert("object".to_string(), serde_json::Value::String(triple.object.to_string()));
                    values.push(serde_json::Value::Object(obj));
                }
            }
            QueryResults::Boolean(b) => {
                values.push(serde_json::json!({"result": b}));
            }
        }

        Ok(values)
    }

    pub fn query_nodes(&self, task_iri: &str) -> Result<Vec<Arc<Node>>, CoreError> {
        let node_iris = self.get_task_nodes(task_iri);
        let mut nodes = Vec::new();
        for iri in node_iris {
            if let Some(node) = self.node_cache.get(&iri) {
                nodes.push(node.clone());
            }
        }
        Ok(nodes)
    }

    /// Query nodes with optional multi-dimensional filtering.
    ///
    /// When filter fields are set, returns only nodes matching ALL criteria (intersection).
    /// When no filter fields are set, returns ALL nodes (same as query_nodes()).
    /// Unrecognized role strings are silently ignored (no match returned).
    pub fn query_nodes_filtered(
        &self,
        task_iri: &str,
        filter: &QueryFilter,
    ) -> Result<Vec<Arc<Node>>, CoreError> {
        let all_iris: Vec<String> = self.task_nodes.read()
            .get(task_iri)
            .cloned()
            .unwrap_or_default();

        if all_iris.is_empty() {
            return Ok(Vec::new());
        }

        let mut filtered: Option<Vec<String>> = None;

        if let Some(role) = &filter.role {
            let role_iris = self.role_index.read()
                .get(&(task_iri.to_string(), role.clone()))
                .cloned()
                .unwrap_or_default();
            filtered = Some(filtered.map_or(role_iris.clone(), |f| {
                f.into_iter().filter(|iri| role_iris.contains(iri)).collect()
            }));
        }

        if let Some(cycle_id) = &filter.cycle_id {
            let cycle_iris = self.cycle_index.read()
                .get(&(task_iri.to_string(), cycle_id.clone()))
                .cloned()
                .unwrap_or_default();
            filtered = Some(filtered.map_or(cycle_iris.clone(), |f| {
                f.into_iter().filter(|iri| cycle_iris.contains(iri)).collect()
            }));
        }

        if let Some(node_type) = &filter.node_type {
            let type_iris = self.type_index.read()
                .get(&(task_iri.to_string(), node_type.clone()))
                .cloned()
                .unwrap_or_default();
            filtered = Some(filtered.map_or(type_iris.clone(), |f| {
                f.into_iter().filter(|iri| type_iris.contains(iri)).collect()
            }));
        }

        let target_iris = filtered.unwrap_or(all_iris);

        let mut nodes = Vec::with_capacity(target_iris.len());
        for iri in target_iris {
            if let Some(node) = self.node_cache.get(&iri) {
                nodes.push(node.clone());
            }
        }
        Ok(nodes)
    }

    pub fn get_task_nodes(&self, task_iri: &str) -> Vec<String> {
        self.task_nodes.read()
            .get(task_iri)
            .cloned()
            .unwrap_or_default()
    }

    pub fn node_count(&self) -> u64 {
        self.node_count.load(Ordering::Relaxed)
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Auto garbage collection: release completed task nodes with no remaining references
    pub fn gc_completed_tasks(&self) -> Result<usize, CoreError> {
        let mut released = 0;
        let mut tasks_to_release = Vec::new();

        {
            let tree = self.task_tree.read();
            for (task_iri, node) in tree.iter() {
                if node.status == "completed" || node.status == "failed" {
                    if node.children.is_empty() {
                        tasks_to_release.push(task_iri.clone());
                    }
                }
            }
        }

        for task_iri in &tasks_to_release {
            match self.release_subtree(task_iri) {
                Ok(count) => released += count,
                Err(e) => {
                    warn!(task_iri = %task_iri, error = %e, "Failed to release completed task");
                }
            }
        }

        if released > 0 {
            debug!(released = released, tasks = tasks_to_release.len(), "Auto GC completed");
        }

        Ok(released)
    }

    /// Check and return tasks that need garbage collection
    pub fn get_gc_candidates(&self) -> Vec<(String, String)> {
        let tree = self.task_tree.read();
        tree.iter()
            .filter(|(_, node)| {
                (node.status == "completed" || node.status == "failed") && node.children.is_empty()
            })
            .map(|(iri, node)| (iri.clone(), node.status.clone()))
            .collect()
    }

    pub fn clear(&self) {
        self.node_cache.clear();
        self.task_nodes.write().clear();
        self.task_tree.write().clear();
        self.role_index.write().clear();
        self.cycle_index.write().clear();
        self.type_index.write().clear();
        self.node_count.store(0, Ordering::Relaxed);
        self.total_bytes.store(0, Ordering::Relaxed);

        if let Err(e) = self.store.update("DELETE WHERE { ?s ?p ?o . }") {
            warn!(error = %e, "Failed to clear oxigraph store");
        }
    }

    pub fn write_node_to_graph(
        &self,
        node_iri: &str,
        json_ld: &str,
        graph_name: &str,
        config: &CoreConfig,
    ) -> Result<(), CoreError> {
        let size = json_ld.as_bytes().len();

        if size > config.max_node_size {
            return Err(CoreError::NodeTooLarge {
                size,
                max: config.max_node_size,
            });
        }

        let parsed: serde_json::Value = serde_json::from_str(json_ld)
            .map_err(|e| CoreError::InvalidJsonLd {
                message: format!("JSON parse error: {}", e),
            })?;

        let task_iri = extract_task_iri(node_iri);
        let is_update = self.node_cache.contains_key(node_iri);
        let jsonld_types = extract_jsonld_types(&parsed);

        let node = Node {
            iri: node_iri.to_string(),
            json_ld: json_ld.to_string(),
            size,
            created_at: chrono::Utc::now(),
            created_by: parsed.get("created_by").and_then(|v| v.as_str()).map(String::from),
            tags: parsed.get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            node_type: jsonld_types.first().cloned(),
            dirty: is_update,
            mesi_state: if is_update { MesiState::Modified } else { MesiState::Shared },
            parent_task: task_iri.clone(),
            named_graph: Some(graph_name.to_string()),
            jsonld_types: jsonld_types.clone(),
        };

        self.node_cache.insert(node_iri.to_string(), Arc::new(node));

        // Defer Oxigraph sync to background thread (avoids blocking the write path)
        let store = self.store.clone();
        let iri = node_iri.to_string();
        let gname = graph_name.to_string();
        let parsed_json = parsed.clone();
        let handle = std::thread::spawn(move || {
            if let Err(e) = Self::sync_to_oxigraph_with_graph_sync(&store, &iri, &parsed_json, &gname) {
                warn!(error = %e, node_iri = %iri, graph = %gname, "Background Oxigraph sync failed");
            }
        });
        self.pending_syncs.lock().unwrap().push(handle);

        if let Some(task_iri) = &task_iri {
            let mut task_nodes = self.task_nodes.write();
            let entry = task_nodes.entry(task_iri.clone()).or_default();
            if !entry.contains(&node_iri.to_string()) {
                entry.push(node_iri.to_string());
            }

            let mut tree = self.task_tree.write();
            let tree_node = tree.entry(task_iri.clone()).or_insert_with(|| TaskTreeNode {
                task_iri: task_iri.clone(),
                parent: None,
                children: Vec::new(),
                dependencies: Vec::new(),
                dependents: Vec::new(),
                status: "running".to_string(),
                node_iris: Vec::new(),
            });
                if !tree_node.node_iris.contains(&node_iri.to_string()) {
                    tree_node.node_iris.push(node_iri.to_string());
                }
            }

            // Secondary indices for named-graph write (same logic as write_node)
            if let Some(task_iri) = &task_iri {
                if let Some(role_str) = parsed.get("role").and_then(|v| v.as_str()) {
                    if let Ok(role) = role_str.parse::<AgentRole>() {
                        let mut idx = self.role_index.write();
                        let entry = idx.entry((task_iri.clone(), role)).or_default();
                        if !entry.contains(&node_iri.to_string()) {
                            entry.push(node_iri.to_string());
                        }
                    }
                }
                if let Some(cycle_id) = parsed.get("cycle_id").and_then(|v| v.as_str()) {
                    let mut idx = self.cycle_index.write();
                    let entry = idx.entry((task_iri.clone(), cycle_id.to_string())).or_default();
                    if !entry.contains(&node_iri.to_string()) {
                        entry.push(node_iri.to_string());
                    }
                }
                if let Some(node_type) = jsonld_types.first() {
                    let mut idx = self.type_index.write();
                    let entry = idx.entry((task_iri.clone(), node_type.clone())).or_default();
                    if !entry.contains(&node_iri.to_string()) {
                        entry.push(node_iri.to_string());
                    }
                }
            }

            if !is_update {
            self.node_count.fetch_add(1, Ordering::Relaxed);
        }
        self.total_bytes.fetch_add(size as u64, Ordering::Relaxed);

        debug!(node_iri = %node_iri, graph = %graph_name, size = size, "Node written to named graph");
        Ok(())
    }

    /// Background-sync to Oxigraph with named graph (called from std::thread::spawn).
    fn sync_to_oxigraph_with_graph_sync(
        store: &Store,
        node_iri: &str,
        parsed: &serde_json::Value,
        graph_name: &str,
    ) -> Result<(), CoreError> {
        let triples = Self::build_triples(node_iri, parsed);

        if !triples.is_empty() {
            let subject = format!("<{}>", node_iri);
            let graph = format!("<{}>", graph_name);
            let combined_sparql = format!(
                "DELETE WHERE {{ GRAPH {} {{ {} ?p ?o . }} }}; INSERT DATA {{ GRAPH {} {{ {} }} }}",
                graph, subject, graph, triples.join("\n")
            );
            store.update(&combined_sparql)
                .map_err(|e| CoreError::SparqlError {
                    message: format!("Failed to execute atomic DELETE+INSERT in named graph: {}", e),
                })?;
        }

        Ok(())
    }

    /// Wait for all pending background Oxigraph syncs to complete.
    /// Useful before querying Oxigraph directly after writes.
    pub fn flush_oxigraph(&self) {
        let handles = std::mem::take(&mut *self.pending_syncs.lock().unwrap());
        for h in handles {
            let _ = h.join();
        }
    }

    pub fn query_graph(&self, graph_name: &str, sparql: &str) -> Result<Vec<serde_json::Value>, CoreError> {
        let graph_sparql = if sparql.to_uppercase().contains("GRAPH") {
            sparql.to_string()
        } else {
            let graph = format!("<{}>", graph_name);
            format!("SELECT * WHERE {{ GRAPH {} {{ {} }} }}", graph, sparql)
        };

        self.query(&graph_sparql)
    }

    #[instrument(skip(self, nodes, config))]
    pub fn write_nodes_batch(
        &self,
        nodes: Vec<(String, String)>,
        config: &CoreConfig,
    ) -> Result<Vec<Result<(), CoreError>>, CoreError> {
        let mut validated = Vec::with_capacity(nodes.len());
        for (i, (node_iri, json_ld)) in nodes.iter().enumerate() {
            let size = json_ld.as_bytes().len();
            if size > config.max_node_size {
                return Err(CoreError::NodeTooLarge {
                    size,
                    max: config.max_node_size,
                });
            }

            let parsed: serde_json::Value = serde_json::from_str(json_ld)
                .map_err(|e| CoreError::InvalidJsonLd {
                    message: format!("Batch item {} JSON parse error: {}", i, e),
                })?;

            validated.push((node_iri.clone(), json_ld.clone(), parsed));
        }

        let mut delete_parts = Vec::with_capacity(validated.len());
        let mut insert_parts = Vec::new();

        for (node_iri, _, parsed) in &validated {
            let subject = format!("<{}>", node_iri);
            let triples = Self::build_triples(node_iri, parsed);
            if !triples.is_empty() {
                delete_parts.push(format!("DELETE WHERE {{ {} ?p ?o . }}", subject));
                insert_parts.push(format!("INSERT DATA {{ {} }}", triples.join("\n")));
            }
        }

        if !delete_parts.is_empty() || !insert_parts.is_empty() {
            let mut combined_parts = delete_parts;
            combined_parts.extend(insert_parts);
            let combined_sparql = combined_parts.join("; ");
            self.store.update(&combined_sparql)
                .map_err(|e| CoreError::SparqlError {
                    message: format!("Failed to execute batch atomic DELETE+INSERT: {}", e),
                })?;
        }

        let mut results = Vec::with_capacity(validated.len());
        for (node_iri, json_ld, parsed) in &validated {
            let size = json_ld.as_bytes().len();
            let task_iri = extract_task_iri(node_iri);
            let is_update = self.node_cache.contains_key(node_iri);
            let jsonld_types = extract_jsonld_types(parsed);

            let node = Node {
                iri: node_iri.clone(),
                json_ld: json_ld.clone(),
                size,
                created_at: chrono::Utc::now(),
                created_by: parsed.get("created_by").and_then(|v| v.as_str()).map(String::from),
                tags: parsed.get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
                node_type: jsonld_types.first().cloned(),
                dirty: is_update,
                mesi_state: if is_update { MesiState::Modified } else { MesiState::Shared },
                parent_task: task_iri.clone(),
                named_graph: parsed.get("named_graph").and_then(|v| v.as_str()).map(String::from),
                jsonld_types: jsonld_types.clone(),
            };

            self.node_cache.insert(node_iri.clone(), Arc::new(node));

            if let Some(task_iri) = &task_iri {
                let mut task_nodes = self.task_nodes.write();
                let entry = task_nodes.entry(task_iri.clone()).or_default();
                if !entry.contains(&node_iri.to_string()) {
                    entry.push(node_iri.to_string());
                }

                let mut tree = self.task_tree.write();
                let tree_node = tree.entry(task_iri.clone()).or_insert_with(|| TaskTreeNode {
                    task_iri: task_iri.clone(),
                    parent: None,
                    children: Vec::new(),
                    dependencies: Vec::new(),
                    dependents: Vec::new(),
                    status: "running".to_string(),
                    node_iris: Vec::new(),
                });
                if !tree_node.node_iris.contains(&node_iri.to_string()) {
                    tree_node.node_iris.push(node_iri.to_string());
                }
            }

            if !is_update {
                self.node_count.fetch_add(1, Ordering::Relaxed);
            }
            self.total_bytes.fetch_add(size as u64, Ordering::Relaxed);

            results.push(Ok(()));
        }

        debug!(count = results.len(), "Batch write nodes completed");
        Ok(results)
    }

    #[instrument(skip(self, nodes, config))]
    pub fn write_batch_to_graphs(
        &self,
        nodes: Vec<(String, String, String)>,
        config: &CoreConfig,
    ) -> Result<usize, CoreError> {
        let mut validated = Vec::with_capacity(nodes.len());
        for (i, (node_iri, json_ld, graph_name)) in nodes.iter().enumerate() {
            let size = json_ld.as_bytes().len();
            if size > config.max_node_size {
                return Err(CoreError::NodeTooLarge {
                    size,
                    max: config.max_node_size,
                });
            }

            let parsed: serde_json::Value = serde_json::from_str(json_ld)
                .map_err(|e| CoreError::InvalidJsonLd {
                    message: format!("Batch item {} JSON parse error: {}", i, e),
                })?;

            validated.push((node_iri.clone(), json_ld.clone(), graph_name.clone(), parsed));
        }

        let mut delete_parts = Vec::with_capacity(validated.len());
        let mut insert_parts = Vec::new();

        for (node_iri, _, graph_name, parsed) in &validated {
            let subject = format!("<{}>", node_iri);
            let graph = format!("<{}>", graph_name);
            let triples = Self::build_triples(node_iri, parsed);
            if !triples.is_empty() {
                delete_parts.push(format!("DELETE WHERE {{ GRAPH {} {{ {} ?p ?o . }} }}", graph, subject));
                insert_parts.push(format!("INSERT DATA {{ GRAPH {} {{ {} }} }}", graph, triples.join("\n")));
            }
        }

        if !delete_parts.is_empty() || !insert_parts.is_empty() {
            let mut combined_parts = delete_parts;
            combined_parts.extend(insert_parts);
            let combined_sparql = combined_parts.join("; ");
            self.store.update(&combined_sparql)
                .map_err(|e| CoreError::SparqlError {
                    message: format!("Failed to execute batch atomic DELETE+INSERT in named graphs: {}", e),
                })?;
        }

        let mut success_count = 0;
        for (node_iri, json_ld, graph_name, parsed) in &validated {
            let size = json_ld.as_bytes().len();
            let task_iri = extract_task_iri(node_iri);
            let is_update = self.node_cache.contains_key(node_iri);
            let jsonld_types = extract_jsonld_types(parsed);

            let node = Node {
                iri: node_iri.clone(),
                json_ld: json_ld.clone(),
                size,
                created_at: chrono::Utc::now(),
                created_by: parsed.get("created_by").and_then(|v| v.as_str()).map(String::from),
                tags: parsed.get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
                node_type: jsonld_types.first().cloned(),
                dirty: is_update,
                mesi_state: if is_update { MesiState::Modified } else { MesiState::Shared },
                parent_task: task_iri.clone(),
                named_graph: Some(graph_name.clone()),
                jsonld_types: jsonld_types.clone(),
            };

            self.node_cache.insert(node_iri.clone(), Arc::new(node));

            if let Some(task_iri) = &task_iri {
                let mut task_nodes = self.task_nodes.write();
                let entry = task_nodes.entry(task_iri.clone()).or_default();
                if !entry.contains(&node_iri.to_string()) {
                    entry.push(node_iri.to_string());
                }

                let mut tree = self.task_tree.write();
                let tree_node = tree.entry(task_iri.clone()).or_insert_with(|| TaskTreeNode {
                    task_iri: task_iri.clone(),
                    parent: None,
                    children: Vec::new(),
                    dependencies: Vec::new(),
                    dependents: Vec::new(),
                    status: "running".to_string(),
                    node_iris: Vec::new(),
                });
                if !tree_node.node_iris.contains(&node_iri.to_string()) {
                    tree_node.node_iris.push(node_iri.to_string());
                }
            }

            // Secondary indices for write_batch_to_graphs (same logic as write_node)
            if let Some(task_iri) = &task_iri {
                if let Some(role_str) = parsed.get("role").and_then(|v| v.as_str()) {
                    if let Ok(role) = role_str.parse::<AgentRole>() {
                        let mut idx = self.role_index.write();
                        let entry = idx.entry((task_iri.clone(), role)).or_default();
                        if !entry.contains(&node_iri.to_string()) {
                            entry.push(node_iri.to_string());
                        }
                    }
                }
                if let Some(cycle_id) = parsed.get("cycle_id").and_then(|v| v.as_str()) {
                    let mut idx = self.cycle_index.write();
                    let entry = idx.entry((task_iri.clone(), cycle_id.to_string())).or_default();
                    if !entry.contains(&node_iri.to_string()) {
                        entry.push(node_iri.to_string());
                    }
                }
                if let Some(node_type) = jsonld_types.first() {
                    let mut idx = self.type_index.write();
                    let entry = idx.entry((task_iri.clone(), node_type.clone())).or_default();
                    if !entry.contains(&node_iri.to_string()) {
                        entry.push(node_iri.to_string());
                    }
                }
            }

            if !is_update {
                self.node_count.fetch_add(1, Ordering::Relaxed);
            }
            self.total_bytes.fetch_add(size as u64, Ordering::Relaxed);

            success_count += 1;
        }

        debug!(count = success_count, "Batch write to graphs completed");
        Ok(success_count)
    }

    pub fn query_by_types(&self, types: &[String]) -> Result<Vec<Arc<Node>>, CoreError> {
        if types.is_empty() {
            return Ok(Vec::new());
        }

        let type_filters: Vec<String> = types
            .iter()
            .map(|t| {
                if t.contains("://") {
                    format!("<{}>", t)
                } else {
                    format!("<http://agent-os.org/ontology/{}>", t)
                }
            })
            .collect();

        let sparql = format!(
            "SELECT DISTINCT ?s WHERE {{ {} }}",
            type_filters
                .iter()
                .map(|t| format!("{{ ?s a {} }}", t))
                .collect::<Vec<_>>()
                .join(" UNION ")
        );

        let results = self.query(&sparql)?;

        let mut nodes = Vec::new();
        for result in results {
            if let Some(iri_value) = result.get("?s").and_then(|v| v.as_str()) {
                let iri = iri_value.trim_start_matches('<').trim_end_matches('>');
                if let Some(node) = self.node_cache.get(iri) {
                    nodes.push(node.clone());
                }
            }
        }

        debug!(types = ?types, count = nodes.len(), "Query by types completed");
        Ok(nodes)
    }

    pub fn check_permission(
        &self,
        agent_role: &str,
        graph_name: &str,
        permission: GraphPermission,
    ) -> bool {
        self.permission_matrix.check_permission(agent_role, graph_name, permission)
    }

    pub fn get_permission_matrix(&self) -> &PermissionMatrix {
        &self.permission_matrix
    }

    // ========== Agent Situational Awareness ==========

    pub fn register_agent(&self, agent_id: &str, role: &str, task_iri: &str) {
        let mut registry = self.agent_registry.write();
        let now = chrono::Utc::now();
        registry.insert(agent_id.to_string(), AgentStatus {
            agent_id: agent_id.to_string(),
            agent_role: role.to_string(),
            task_iri: task_iri.to_string(),
            status: AgentActivity::Idle,
            started_at: now,
            last_heartbeat: now,
            current_operation: None,
            resource_locks: Vec::new(),
        });
        debug!(agent_id = %agent_id, role = %role, "Agent registered in battle map");
    }

    pub fn update_agent_heartbeat(&self, agent_id: &str) {
        let mut registry = self.agent_registry.write();
        if let Some(status) = registry.get_mut(agent_id) {
            status.last_heartbeat = chrono::Utc::now();
        }
    }

    pub fn update_agent_status(&self, agent_id: &str, status: AgentActivity, operation: Option<&str>) {
        let mut registry = self.agent_registry.write();
        if let Some(s) = registry.get_mut(agent_id) {
            s.status = status;
            s.current_operation = operation.map(|o| o.to_string());
            s.last_heartbeat = chrono::Utc::now();
        }
    }

    pub fn get_agent_status(&self, agent_id: &str) -> Option<AgentStatus> {
        self.agent_registry.read().get(agent_id).cloned()
    }

    pub fn list_active_agents(&self) -> Vec<AgentStatus> {
        self.agent_registry.read().values().cloned().collect()
    }

    pub fn unregister_agent(&self, agent_id: &str) {
        let mut registry = self.agent_registry.write();
        if let Some(status) = registry.remove(agent_id) {
            debug!(agent_id = %agent_id, task_iri = %status.task_iri, "Agent unregistered from battle map");
        }
    }

    pub fn detect_stale_agents(&self, max_idle_seconds: i64) -> Vec<String> {
        let now = chrono::Utc::now();
        let mut stale = Vec::new();
        let registry = self.agent_registry.read();
        for (id, status) in registry.iter() {
            let elapsed = (now - status.last_heartbeat).num_seconds();
            if elapsed > max_idle_seconds {
                stale.push(id.clone());
            }
        }
        stale
    }

    // ========== Resource Lock Management ==========

    pub fn acquire_resource(&self, lock: ResourceLock) -> Result<bool, CoreError> {
        let mut locks = self.resource_locks.write();

        // Check conflict: same resource and mutually exclusive lock types
        let has_conflict = locks.iter().any(|existing| {
            existing.resource_id == lock.resource_id
                && existing.acquired_by != lock.acquired_by
                && (lock.lock_type == LockType::Exclusive
                    || existing.lock_type == LockType::Exclusive
                    || (lock.lock_type == LockType::Write && existing.lock_type == LockType::Write))
        });

        if has_conflict {
            return Ok(false);
        }

        locks.push(lock);
        Ok(true)
    }

    pub fn release_resource(&self, agent_id: &str, resource_id: &str) {
        let mut locks = self.resource_locks.write();
        locks.retain(|l| !(l.acquired_by == agent_id && l.resource_id == resource_id));
    }

    pub fn release_agent_resources(&self, agent_id: &str) {
        let mut locks = self.resource_locks.write();
        locks.retain(|l| l.acquired_by != agent_id);
    }

    pub fn list_resource_locks(&self) -> Vec<ResourceLock> {
        self.resource_locks.read().clone()
    }

    pub fn check_resource_available(&self, resource_id: &str, requested_lock: LockType) -> bool {
        let locks = self.resource_locks.read();
        !locks.iter().any(|existing| {
            existing.resource_id == resource_id
                && (requested_lock == LockType::Exclusive
                    || existing.lock_type == LockType::Exclusive
                    || (requested_lock == LockType::Write && existing.lock_type == LockType::Write))
        })
    }

    // ========== Cross-task Dependencies ==========

    pub fn add_task_dependency(&self, task_iri: &str, depends_on: &str) {
        let mut tree = self.task_tree.write();
        if let Some(node) = tree.get_mut(task_iri) {
            if !node.dependencies.contains(&depends_on.to_string()) {
                node.dependencies.push(depends_on.to_string());
            }
        }
        if let Some(dep_node) = tree.get_mut(depends_on) {
            if !dep_node.dependents.contains(&task_iri.to_string()) {
                dep_node.dependents.push(task_iri.to_string());
            }
        }
    }

    pub fn remove_task_dependency(&self, task_iri: &str, depends_on: &str) {
        let mut tree = self.task_tree.write();
        if let Some(node) = tree.get_mut(task_iri) {
            node.dependencies.retain(|d| d != depends_on);
        }
        if let Some(dep_node) = tree.get_mut(depends_on) {
            dep_node.dependents.retain(|d| d != task_iri);
        }
    }

    pub fn get_task_dependencies(&self, task_iri: &str) -> Vec<String> {
        self.task_tree.read().get(task_iri)
            .map(|n| n.dependencies.clone())
            .unwrap_or_default()
    }

    pub fn get_task_dependents(&self, task_iri: &str) -> Vec<String> {
        self.task_tree.read().get(task_iri)
            .map(|n| n.dependents.clone())
            .unwrap_or_default()
    }

    pub fn get_task_dag(&self, task_iri: &str) -> Result<Vec<Vec<String>>, CoreError> {
        let tree = self.task_tree.read();

        // Build adjacency and collect all reachable nodes via dependencies
        let mut reachable = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(task_iri.to_string());

        while let Some(current) = queue.pop_front() {
            if reachable.contains(&current) {
                continue;
            }
            reachable.insert(current.clone());
            if let Some(node) = tree.get(&current) {
                for dep in &node.dependencies {
                    queue.push_back(dep.clone());
                }
                for dep in &node.dependents {
                    queue.push_back(dep.clone());
                }
                for child in &node.children {
                    queue.push_back(child.clone());
                }
                if let Some(parent) = &node.parent {
                    queue.push_back(parent.clone());
                }
            }
        }

        if reachable.is_empty() {
            return Ok(Vec::new());
        }

        // Kahn's algorithm: compute in-degree within the reachable subgraph
        // Edge direction: dependency -> dependent means "dependency must finish before dependent"
        let mut in_degree: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut adj: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();

        for iri in &reachable {
            in_degree.entry(iri.clone()).or_insert(0);
            adj.entry(iri.clone()).or_default();
        }

        for iri in &reachable {
            if let Some(node) = tree.get(iri) {
                // Dependencies: edge from dep -> current (dep must finish before current)
                for dep in &node.dependencies {
                    if reachable.contains(dep) {
                        adj.entry(dep.clone()).or_default().push(iri.clone());
                        *in_degree.entry(iri.clone()).or_insert(0) += 1;
                    }
                }
                // Parent-child: parent -> child
                if let Some(parent) = &node.parent {
                    if reachable.contains(parent) {
                        adj.entry(parent.clone()).or_default().push(iri.clone());
                        *in_degree.entry(iri.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut layers = Vec::new();
        let mut current_layer: Vec<String> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(iri, _)| iri.clone())
            .collect();
        current_layer.sort();

        let mut remaining: std::collections::HashSet<String> = reachable.iter()
            .filter(|iri| {
                let deg = in_degree.get(*iri).copied().unwrap_or(0);
                deg > 0
            })
            .cloned()
            .collect();

        // Track processed in-degree (mutable)
        let mut current_in_degree = in_degree.clone();

        while !current_layer.is_empty() {
            layers.push(current_layer.clone());

            let mut next_layer = Vec::new();
            for node_iri in &current_layer {
                if let Some(neighbors) = adj.get(node_iri) {
                    for next in neighbors {
                        if let Some(deg) = current_in_degree.get_mut(next) {
                            *deg = deg.saturating_sub(1);
                            if *deg == 0 {
                                next_layer.push(next.clone());
                                remaining.remove(next);
                            }
                        }
                    }
                }
            }
            next_layer.sort();
            current_layer = next_layer;
        }

        Ok(layers)
    }

    // ========== blackboard:shared Coordination Zone ==========

    pub fn publish_coordination(&self, msg: &CoordinationMessage) -> Result<(), CoreError> {
        let iri = format!("iri://coordination/{}", uuid::Uuid::new_v4().hyphenated());
        let json_ld = serde_json::json!({
            "@id": &iri,
            "@type": "CoordinationMessage",
            "from_agent": msg.from_agent,
            "msg_type": format!("{:?}", msg.msg_type),
            "payload": msg.payload,
            "timestamp": msg.timestamp.to_rfc3339(),
        });
        let config = CoreConfig { max_node_size: 65536, ..CoreConfig::default() };
        self.write_node_to_graph(&iri, &json_ld.to_string(), "blackboard:shared", &config)
    }

    pub fn read_coordination_messages(&self) -> Result<Vec<CoordinationMessage>, CoreError> {
        self.read_coordination_messages_since(&chrono::DateTime::UNIX_EPOCH)
    }

    pub fn read_coordination_messages_since(&self, since: &chrono::DateTime<chrono::Utc>) -> Result<Vec<CoordinationMessage>, CoreError> {
        let subjects = self.query(
            "SELECT DISTINCT ?s WHERE { GRAPH <blackboard:shared> { ?s a ?type } } LIMIT 50"
        ).unwrap_or_default();

        let mut msgs = Vec::new();
        for r in &subjects {
            if let Some(s_val) = r.get("?s").and_then(|v| v.as_str()) {
                let iri = s_val.trim_start_matches('<').trim_end_matches('>');
                if iri.starts_with("iri://coordination/") {
                    if let Ok(Some(node)) = self.read_node(iri) {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&node.json_ld) {
                            let from_agent = parsed.get("from_agent").and_then(|v| v.as_str()).unwrap_or("unknown");
                            let msg_type_str = parsed.get("msg_type").and_then(|v| v.as_str()).unwrap_or("TaskAnnouncement");
                            let payload = parsed.get("payload").cloned().unwrap_or(serde_json::Value::Null);
                            let ts = parsed.get("timestamp").and_then(|v| v.as_str()).and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                                .map(|dt| dt.into()).unwrap_or_else(chrono::Utc::now);
                            if ts < *since {
                                continue;
                            }
                            let msg = CoordinationMessage {
                                from_agent: from_agent.to_string(),
                                msg_type: Self::parse_coordination_msg_type(msg_type_str),
                                payload,
                                timestamp: ts,
                            };
                            msgs.push(msg);
                        }
                    }
                }
            }
        }

        Ok(msgs)
    }

    fn parse_coordination_msg_type(s: &str) -> CoordinationMsgType {
        match s {
            "TaskAnnouncement" => CoordinationMsgType::TaskAnnouncement,
            "ProgressUpdate" => CoordinationMsgType::ProgressUpdate,
            "ResourceRequest" => CoordinationMsgType::ResourceRequest,
            "ConflictWarning" => CoordinationMsgType::ConflictWarning,
            "SyncRequest" => CoordinationMsgType::SyncRequest,
            _ => CoordinationMsgType::TaskAnnouncement,
        }
    }




    pub fn publish_agent_snapshot_to_shared(&self, agent_id: &str) -> Result<(), CoreError> {
        let status = match self.get_agent_status(agent_id) {
            Some(s) => s,
            None => return Err(CoreError::NodeNotFound { iri: agent_id.to_string() }),
        };
        let iri = format!("iri://snapshot/{}/{}", agent_id, uuid::Uuid::new_v4().hyphenated());
        let snapshot = serde_json::json!({
            "@id": &iri,
            "@type": "AgentSnapshot",
            "agent_id": status.agent_id,
            "agent_role": status.agent_role,
            "task_iri": status.task_iri,
            "status": status.status.to_string(),
            "started_at": status.started_at.to_rfc3339(),
            "last_heartbeat": status.last_heartbeat.to_rfc3339(),
            "current_operation": status.current_operation,
            "resource_count": status.resource_locks.len(),
        });
        let config = CoreConfig { max_node_size: 65536, ..CoreConfig::default() };
        self.write_node_to_graph(&iri, &snapshot.to_string(), "blackboard:shared", &config)
    }

    pub fn publish_shared_state(&self, task_iri: &str, state: &serde_json::Value) -> Result<(), CoreError> {
        let iri = format!("iri://shared/state/{}", task_iri.trim_start_matches("iri://task/"));
        let payload = serde_json::json!({
            "@id": &iri,
            "@type": "SharedState",
            "task_iri": task_iri,
            "state": state,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        let config = CoreConfig { max_node_size: 65536, ..CoreConfig::default() };
        self.write_node_to_graph(&iri, &payload.to_string(), "blackboard:shared", &config)
    }

    pub fn get_shared_state(&self, task_iri: &str) -> Result<Option<serde_json::Value>, CoreError> {
        let iri = format!("iri://shared/state/{}", task_iri.trim_start_matches("iri://task/"));
        match self.read_node(&iri) {
            Ok(Some(node)) => {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&node.json_ld) {
                    Ok(parsed.get("state").cloned())
                } else {
                    Ok(None)
                }
            }
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

fn extract_task_iri(node_iri: &str) -> Option<String> {
    if node_iri.starts_with("iri://task/") {
        let rest = &node_iri["iri://task/".len()..];
        let task_id = rest.split('/').next().unwrap_or(rest);
        Some(format!("iri://task/{}", task_id))
    } else if node_iri.starts_with("iri://") {
        let rest = &node_iri["iri://".len()..];
        let mut segments = rest.split('/');
        let ns = segments.next().unwrap_or(rest);
        if let Some(id) = segments.next() {
            Some(format!("iri://{}/{}", ns, id))
        } else {
            Some(format!("iri://{}", ns))
        }
    } else {
        None
    }
}

fn extract_jsonld_types(parsed: &serde_json::Value) -> Vec<String> {
    if let Some(type_val) = parsed.get("@type") {
        match type_val {
            serde_json::Value::String(s) => vec![s.clone()],
            serde_json::Value::Array(arr) => {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            }
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    }
}

// ========== Battle Map Type Definitions ==========

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AgentActivity {
    Idle,
    Working,
    Blocked,
    Error,
}

impl std::fmt::Display for AgentActivity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentActivity::Idle => write!(f, "Idle"),
            AgentActivity::Working => write!(f, "Working"),
            AgentActivity::Blocked => write!(f, "Blocked"),
            AgentActivity::Error => write!(f, "Error"),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentStatus {
    pub agent_id: String,
    pub agent_role: String,
    pub task_iri: String,
    pub status: AgentActivity,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub current_operation: Option<String>,
    pub resource_locks: Vec<ResourceLock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LockType {
    Read,
    Write,
    Exclusive,
}

impl std::fmt::Display for LockType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockType::Read => write!(f, "Read"),
            LockType::Write => write!(f, "Write"),
            LockType::Exclusive => write!(f, "Exclusive"),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceLock {
    pub resource_type: String,
    pub resource_id: String,
    pub acquired_at: chrono::DateTime<chrono::Utc>,
    pub acquired_by: String,
    pub lock_type: LockType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CoordinationMsgType {
    TaskAnnouncement,
    ProgressUpdate,
    ResourceRequest,
    ConflictWarning,
    SyncRequest,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoordinationMessage {
    pub from_agent: String,
    pub msg_type: CoordinationMsgType,
    pub payload: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GraphPermission {
    Read,
    Write,
}

pub struct PermissionMatrix {
    permissions: DashMap<String, DashMap<String, Vec<GraphPermission>>>,
}

impl PermissionMatrix {
    pub fn new() -> Self {
        let matrix = Self {
            permissions: DashMap::new(),
        };
        matrix.initialize_default_permissions();
        matrix
    }

    fn initialize_default_permissions(&self) {
        let default_rules = vec![
            ("Plan", "system:plan", vec![GraphPermission::Read, GraphPermission::Write]),
            ("Plan", "system:knowledge", vec![GraphPermission::Read]),
            ("Do", "system:execution", vec![GraphPermission::Read, GraphPermission::Write]),
            ("Do", "system:knowledge", vec![GraphPermission::Read]),
            ("Check", "system:review", vec![GraphPermission::Read, GraphPermission::Write]),
            ("Check", "system:execution", vec![GraphPermission::Read]),
            ("Act", "system:decision", vec![GraphPermission::Read, GraphPermission::Write]),
            ("Act", "system:review", vec![GraphPermission::Read]),
        ];

        for (role, graph, perms) in default_rules {
            self.set_permission(role, graph, perms);
        }
    }

    pub fn set_permission(&self, agent_role: &str, graph_name: &str, permissions: Vec<GraphPermission>) {
        self.permissions
            .entry(agent_role.to_string())
            .or_insert_with(DashMap::new)
            .insert(graph_name.to_string(), permissions);
    }

    pub fn check_permission(&self, agent_role: &str, graph_name: &str, permission: GraphPermission) -> bool {
        if agent_role == "system" {
            return true;
        }
        if let Some(role_perms) = self.permissions.get(agent_role) {
            if let Some(perms) = role_perms.get(graph_name) {
                return perms.contains(&permission);
            }
        }
        false
    }

    pub fn get_permissions(&self, agent_role: &str, graph_name: &str) -> Vec<GraphPermission> {
        if let Some(role_perms) = self.permissions.get(agent_role) {
            if let Some(perms) = role_perms.get(graph_name) {
                return perms.clone();
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blackboard_write_read() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        let json_ld = r#"{"@id":"iri://test/1","@type":"Test","value":42}"#;
        bb.write_node("iri://test/node_1", json_ld, &config).unwrap();
        let result = bb.read_node("iri://test/node_1").unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_size_limit() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig { max_node_size: 100, ..Default::default() };
        let large_json = format!(r#"{{"@id":"iri://test/1","data":"{}"}}"#, "x".repeat(200));
        let result = bb.write_node("iri://test/node_1", &large_json, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_task_iri() {
        assert_eq!(extract_task_iri("iri://task/abc/turn_1"), Some("iri://task/abc".to_string()));
        assert_eq!(extract_task_iri("iri://task/abc"), Some("iri://task/abc".to_string()));
        assert_eq!(extract_task_iri("iri://project/xyz/node_1"), Some("iri://project/xyz".to_string()));
        assert_eq!(extract_task_iri("not_an_iri"), None);
    }

    #[test]
    fn test_sparql_query_after_write() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        let json_ld = r#"{"@id":"iri://task/t1/node_1","@type":"Test","status":"running"}"#;
        bb.write_node("iri://task/t1/node_1", json_ld, &config).unwrap();
        bb.flush_oxigraph();

        let results = bb.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10").unwrap();
        assert!(!results.is_empty(), "SPARQL query should return results after write_node");
    }

    #[test]
    fn test_clear_also_clears_oxigraph() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        let json_ld = r#"{"@id":"iri://task/t1/node_1","@type":"Test","status":"running"}"#;
        bb.write_node("iri://task/t1/node_1", json_ld, &config).unwrap();
        bb.clear();

        let results = bb.query("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
        assert!(results.is_empty(), "Oxigraph store should be cleared after clear()");
    }

    #[test]
    fn test_mesi_state_default() {
        assert_eq!(MesiState::default(), MesiState::Shared);
    }

    #[test]
    fn test_mark_dirty() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        let json_ld = r#"{"@id":"iri://test/1","@type":"Test"}"#;
        bb.write_node("iri://test/node_1", json_ld, &config).unwrap();

        let node = bb.read_node("iri://test/node_1").unwrap().unwrap();
        assert!(!node.dirty);
        assert_eq!(node.mesi_state, MesiState::Shared);

        bb.mark_dirty("iri://test/node_1");
        let node = bb.read_node("iri://test/node_1").unwrap().unwrap();
        assert!(node.dirty);
        assert_eq!(node.mesi_state, MesiState::Modified);
    }

    #[test]
    fn test_write_node_marks_dirty_on_update() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        let json_ld = r#"{"@id":"iri://test/1","@type":"Test"}"#;
        bb.write_node("iri://test/node_1", json_ld, &config).unwrap();

        let node = bb.read_node("iri://test/node_1").unwrap().unwrap();
        assert!(!node.dirty);
        assert_eq!(node.mesi_state, MesiState::Shared);

        let json_ld_v2 = r#"{"@id":"iri://test/1","@type":"Test","version":2}"#;
        bb.write_node("iri://test/node_1", json_ld_v2, &config).unwrap();

        let node = bb.read_node("iri://test/node_1").unwrap().unwrap();
        assert!(node.dirty);
        assert_eq!(node.mesi_state, MesiState::Modified);
    }

    #[test]
    fn test_register_and_complete_task() {
        let bb = Blackboard::new().unwrap();

        bb.register_task("iri://task/t1", None);
        bb.register_task("iri://task/t1/sub1", Some("iri://task/t1".to_string()));

        let tree = bb.task_tree.read();
        let t1 = tree.get("iri://task/t1").unwrap();
        assert_eq!(t1.status, "running");
        assert!(t1.children.contains(&"iri://task/t1/sub1".to_string()));

        let sub1 = tree.get("iri://task/t1/sub1").unwrap();
        assert_eq!(sub1.parent, Some("iri://task/t1".to_string()));
        drop(tree);

        bb.complete_task("iri://task/t1", "completed");
        let tree = bb.task_tree.read();
        let t1 = tree.get("iri://task/t1").unwrap();
        assert_eq!(t1.status, "completed");
    }

    #[test]
    fn test_release_subtree() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();

        bb.register_task("iri://task/t1", None);
        bb.register_task("iri://task/t1/sub1", Some("iri://task/t1".to_string()));

        let json_ld = r#"{"@id":"iri://task/t1/n1","@type":"Test"}"#;
        bb.write_node("iri://task/t1/n1", json_ld, &config).unwrap();
        let json_ld2 = r#"{"@id":"iri://task/t1/sub1/n2","@type":"Test"}"#;
        bb.write_node("iri://task/t1/sub1/n2", json_ld2, &config).unwrap();

        assert_eq!(bb.node_count(), 2);

        let released = bb.release_subtree("iri://task/t1").unwrap();
        assert_eq!(released, 2);
        assert_eq!(bb.node_count(), 0);

        let tree = bb.task_tree.read();
        assert!(tree.get("iri://task/t1").is_none());
        assert!(tree.get("iri://task/t1/sub1").is_none());
    }

    #[test]
    fn test_node_parent_task() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        let json_ld = r#"{"@id":"iri://task/abc/node_1","@type":"Test"}"#;
        bb.write_node("iri://task/abc/node_1", json_ld, &config).unwrap();

        let node = bb.read_node("iri://task/abc/node_1").unwrap().unwrap();
        assert_eq!(node.parent_task, Some("iri://task/abc".to_string()));
    }

    #[test]
    fn test_jsonld_types_extraction() {
        let parsed = serde_json::json!({"@id": "iri://test/1", "@type": "TestType"});
        let types = extract_jsonld_types(&parsed);
        assert_eq!(types, vec!["TestType"]);

        let parsed_multi = serde_json::json!({"@id": "iri://test/2", "@type": ["Type1", "Type2", "Type3"]});
        let types_multi = extract_jsonld_types(&parsed_multi);
        assert_eq!(types_multi, vec!["Type1", "Type2", "Type3"]);

        let parsed_none = serde_json::json!({"@id": "iri://test/3"});
        let types_none = extract_jsonld_types(&parsed_none);
        assert!(types_none.is_empty());
    }

    #[test]
    fn test_write_node_to_graph() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        let json_ld = r#"{"@id":"iri://test/1","@type":"Test","value":42}"#;
        
        bb.write_node_to_graph("iri://test/node_1", json_ld, "system:plan", &config).unwrap();
        
        let node = bb.read_node("iri://test/node_1").unwrap().unwrap();
        assert_eq!(node.named_graph, Some("system:plan".to_string()));
        assert_eq!(node.jsonld_types, vec!["Test"]);
    }

    #[test]
    fn test_named_graph_isolation() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        
        let json_ld1 = r#"{"@id":"iri://test/1","@type":"Plan","status":"draft"}"#;
        let json_ld2 = r#"{"@id":"iri://test/2","@type":"Execution","status":"running"}"#;
        
        bb.write_node_to_graph("iri://test/node_1", json_ld1, "system:plan", &config).unwrap();
        bb.write_node_to_graph("iri://test/node_2", json_ld2, "system:execution", &config).unwrap();
        bb.flush_oxigraph();
        
        let plan_results = bb.query_graph("system:plan", "?s ?p ?o").unwrap();
        assert!(!plan_results.is_empty(), "Plan graph should have nodes");
        
        let exec_results = bb.query_graph("system:execution", "?s ?p ?o").unwrap();
        assert!(!exec_results.is_empty(), "Execution graph should have nodes");
    }

    #[test]
    fn test_write_batch_to_graphs() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        
        let nodes = vec![
            ("iri://test/node_1".to_string(), r#"{"@id":"iri://test/1","@type":"Plan"}"#.to_string(), "system:plan".to_string()),
            ("iri://test/node_2".to_string(), r#"{"@id":"iri://test/2","@type":"Execution"}"#.to_string(), "system:execution".to_string()),
            ("iri://test/node_3".to_string(), r#"{"@id":"iri://test/3","@type":"Review"}"#.to_string(), "system:review".to_string()),
        ];
        
        let count = bb.write_batch_to_graphs(nodes, &config).unwrap();
        assert_eq!(count, 3);
        assert_eq!(bb.node_count(), 3);
    }

    #[test]
    fn test_query_by_types() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        
        let json_ld1 = r#"{"@id":"iri://test/1","@type":"Plan"}"#;
        let json_ld2 = r#"{"@id":"iri://test/2","@type":"Execution"}"#;
        let json_ld3 = r#"{"@id":"iri://test/3","@type":["Plan","Urgent"]}"#;
        
        bb.write_node("iri://test/node_1", json_ld1, &config).unwrap();
        bb.write_node("iri://test/node_2", json_ld2, &config).unwrap();
        bb.write_node("iri://test/node_3", json_ld3, &config).unwrap();
        bb.flush_oxigraph();
        
        let plan_nodes = bb.query_by_types(&["Plan".to_string()]).unwrap();
        assert_eq!(plan_nodes.len(), 2);
        
        let exec_nodes = bb.query_by_types(&["Execution".to_string()]).unwrap();
        assert_eq!(exec_nodes.len(), 1);
    }

    #[test]
    fn test_permission_matrix() {
        let matrix = PermissionMatrix::new();
        
        assert!(matrix.check_permission("Plan", "system:plan", GraphPermission::Read));
        assert!(matrix.check_permission("Plan", "system:plan", GraphPermission::Write));
        assert!(matrix.check_permission("Plan", "system:knowledge", GraphPermission::Read));
        assert!(!matrix.check_permission("Plan", "system:knowledge", GraphPermission::Write));
        
        assert!(matrix.check_permission("Do", "system:execution", GraphPermission::Write));
        assert!(!matrix.check_permission("Do", "system:plan", GraphPermission::Write));
        
        assert!(!matrix.check_permission("Unknown", "system:plan", GraphPermission::Read));
    }

    #[test]
    fn test_blackboard_permission_check() {
        let bb = Blackboard::new();
        assert!(bb.is_ok());
        
        if let Ok(bb) = bb {
            assert!(bb.check_permission("Plan", "system:plan", GraphPermission::Write));
            assert!(!bb.check_permission("Plan", "system:execution", GraphPermission::Write));
        }
    }

    #[test]
    fn test_multi_type_jsonld() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        
        let json_ld = r#"{"@id":"iri://test/1","@type":["Plan","Urgent","Priority"]}"#;
        bb.write_node("iri://test/node_1", json_ld, &config).unwrap();
        
        let node = bb.read_node("iri://test/node_1").unwrap().unwrap();
        assert_eq!(node.jsonld_types, vec!["Plan", "Urgent", "Priority"]);
        assert_eq!(node.node_type, Some("Plan".to_string()));
    }

    #[test]
    fn test_permission_matrix_custom_rules() {
        let matrix = PermissionMatrix::new();
        
        matrix.set_permission("CustomAgent", "custom:graph", vec![GraphPermission::Read]);
        
        assert!(matrix.check_permission("CustomAgent", "custom:graph", GraphPermission::Read));
        assert!(!matrix.check_permission("CustomAgent", "custom:graph", GraphPermission::Write));
    }

    #[test]
    fn test_get_permissions() {
        let matrix = PermissionMatrix::new();
        
        let perms = matrix.get_permissions("Plan", "system:plan");
        assert_eq!(perms.len(), 2);
        assert!(perms.contains(&GraphPermission::Read));
        assert!(perms.contains(&GraphPermission::Write));
        
        let perms_empty = matrix.get_permissions("Plan", "nonexistent:graph");
        assert!(perms_empty.is_empty());
    }

    #[test]
    fn test_delete_node_from_named_graph() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        
        let json_ld = r#"{"@id":"iri://test/1","@type":"Test","value":42}"#;
        bb.write_node_to_graph("iri://test/node_1", json_ld, "system:plan", &config).unwrap();
        bb.flush_oxigraph();
        
        let node = bb.read_node("iri://test/node_1").unwrap().unwrap();
        assert_eq!(node.named_graph, Some("system:plan".to_string()));
        
        let plan_results = bb.query_graph("system:plan", "?s ?p ?o").unwrap();
        assert!(!plan_results.is_empty(), "Plan graph should have nodes before delete");
        
        let deleted = bb.delete_node("iri://test/node_1").unwrap();
        assert!(deleted);
        
        let node_after = bb.read_node("iri://test/node_1").unwrap();
        assert!(node_after.is_none(), "Node should be deleted from cache");
        
        let plan_results_after = bb.query_graph("system:plan", "?s ?p ?o").unwrap();
        assert!(plan_results_after.is_empty(), "Plan graph should be empty after delete");
    }

    #[test]
    fn test_delete_node_from_default_graph() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        
        let json_ld = r#"{"@id":"iri://test/1","@type":"Test","value":42}"#;
        bb.write_node("iri://test/node_1", json_ld, &config).unwrap();
        
        let node = bb.read_node("iri://test/node_1").unwrap().unwrap();
        assert!(node.named_graph.is_none(), "Node should be in default graph");
        
        let deleted = bb.delete_node("iri://test/node_1").unwrap();
        assert!(deleted);
        
        let node_after = bb.read_node("iri://test/node_1").unwrap();
        assert!(node_after.is_none(), "Node should be deleted from cache");
    }

    // ========== Agent Situational Awareness Tests ==========

    #[test]
    fn test_agent_register_and_status() {
        let bb = Blackboard::new().unwrap();
        bb.register_agent("agent_1", "DA", "iri://task/test");
        let status = bb.get_agent_status("agent_1").expect("Agent should be registered");
        assert_eq!(status.agent_id, "agent_1");
        assert_eq!(status.agent_role, "DA");
        assert_eq!(status.task_iri, "iri://task/test");
        assert_eq!(status.status, AgentActivity::Idle);
    }

    #[test]
    fn test_agent_update_status() {
        let bb = Blackboard::new().unwrap();
        bb.register_agent("agent_1", "DA", "iri://task/test");
        bb.update_agent_status("agent_1", AgentActivity::Working, Some("crunching numbers"));
        let status = bb.get_agent_status("agent_1").unwrap();
        assert_eq!(status.status, AgentActivity::Working);
        assert_eq!(status.current_operation, Some("crunching numbers".to_string()));
    }

    #[test]
    fn test_agent_heartbeat_updates_timestamp() {
        let bb = Blackboard::new().unwrap();
        bb.register_agent("agent_1", "DA", "iri://task/test");
        let status_before = bb.get_agent_status("agent_1").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        bb.update_agent_heartbeat("agent_1");
        let status_after = bb.get_agent_status("agent_1").unwrap();
        assert!(status_after.last_heartbeat > status_before.last_heartbeat);
    }

    #[test]
    fn test_agent_unregister() {
        let bb = Blackboard::new().unwrap();
        bb.register_agent("agent_1", "DA", "iri://task/test");
        assert!(bb.get_agent_status("agent_1").is_some());
        bb.unregister_agent("agent_1");
        assert!(bb.get_agent_status("agent_1").is_none());
    }

    #[test]
    fn test_list_active_agents() {
        let bb = Blackboard::new().unwrap();
        bb.register_agent("agent_1", "DA", "iri://task/a");
        bb.register_agent("agent_2", "CA", "iri://task/b");
        let agents = bb.list_active_agents();
        assert_eq!(agents.len(), 2);
    }

    #[test]
    fn test_detect_stale_agents() {
        let bb = Blackboard::new().unwrap();
        bb.register_agent("agent_1", "DA", "iri://task/test");
        // heartbeat just happened, should not be stale
        let stale = bb.detect_stale_agents(3600);
        assert!(stale.is_empty(), "Fresh agent should not be stale");
    }

    // ========== Resource Lock Tests ==========

    #[test]
    fn test_acquire_and_release_lock() {
        let bb = Blackboard::new().unwrap();
        let lock = ResourceLock {
            resource_type: "file".to_string(),
            resource_id: "file:///data/test.csv".to_string(),
            acquired_at: chrono::Utc::now(),
            acquired_by: "agent_1".to_string(),
            lock_type: LockType::Write,
        };
        assert!(bb.acquire_resource(lock).unwrap());
        assert_eq!(bb.list_resource_locks().len(), 1);
        bb.release_resource("agent_1", "file:///data/test.csv");
        assert_eq!(bb.list_resource_locks().len(), 0);
    }

    #[test]
    fn test_write_lock_conflict() {
        let bb = Blackboard::new().unwrap();
        let lock1 = ResourceLock {
            resource_type: "file".to_string(),
            resource_id: "file:///data/test.csv".to_string(),
            acquired_at: chrono::Utc::now(),
            acquired_by: "agent_1".to_string(),
            lock_type: LockType::Write,
        };
        assert!(bb.acquire_resource(lock1).unwrap());
        let lock2 = ResourceLock {
            resource_type: "file".to_string(),
            resource_id: "file:///data/test.csv".to_string(),
            acquired_at: chrono::Utc::now(),
            acquired_by: "agent_2".to_string(),
            lock_type: LockType::Write,
        };
        assert!(!bb.acquire_resource(lock2).unwrap(), "Two Write locks on same resource should conflict");
    }

    #[test]
    fn test_read_lock_no_conflict() {
        let bb = Blackboard::new().unwrap();
        let lock1 = ResourceLock {
            resource_type: "file".to_string(),
            resource_id: "file:///data/test.csv".to_string(),
            acquired_at: chrono::Utc::now(),
            acquired_by: "agent_1".to_string(),
            lock_type: LockType::Read,
        };
        assert!(bb.acquire_resource(lock1).unwrap());
        let lock2 = ResourceLock {
            resource_type: "file".to_string(),
            resource_id: "file:///data/test.csv".to_string(),
            acquired_at: chrono::Utc::now(),
            acquired_by: "agent_2".to_string(),
            lock_type: LockType::Read,
        };
        assert!(bb.acquire_resource(lock2).unwrap(), "Two Read locks should coexist");
    }

    #[test]
    fn test_exclusive_lock_blocks_all() {
        let bb = Blackboard::new().unwrap();
        let lock1 = ResourceLock {
            resource_type: "db".to_string(),
            resource_id: "db://crm".to_string(),
            acquired_at: chrono::Utc::now(),
            acquired_by: "agent_1".to_string(),
            lock_type: LockType::Exclusive,
        };
        assert!(bb.acquire_resource(lock1).unwrap());
        // Read lock should also be blocked by Exclusive
        let lock2 = ResourceLock {
            resource_type: "db".to_string(),
            resource_id: "db://crm".to_string(),
            acquired_at: chrono::Utc::now(),
            acquired_by: "agent_2".to_string(),
            lock_type: LockType::Read,
        };
        assert!(!bb.acquire_resource(lock2).unwrap(), "Exclusive lock should block Read");
    }

    #[test]
    fn test_release_agent_resources() {
        let bb = Blackboard::new().unwrap();
        bb.acquire_resource(ResourceLock {
            resource_type: "file".to_string(), resource_id: "f1".to_string(),
            acquired_at: chrono::Utc::now(), acquired_by: "agent_1".to_string(),
            lock_type: LockType::Read,
        }).unwrap();
        bb.acquire_resource(ResourceLock {
            resource_type: "file".to_string(), resource_id: "f2".to_string(),
            acquired_at: chrono::Utc::now(), acquired_by: "agent_1".to_string(),
            lock_type: LockType::Read,
        }).unwrap();
        assert_eq!(bb.list_resource_locks().len(), 2);
        bb.release_agent_resources("agent_1");
        assert_eq!(bb.list_resource_locks().len(), 0);
    }

    #[test]
    fn test_check_resource_available() {
        let bb = Blackboard::new().unwrap();
        assert!(bb.check_resource_available("file:///data/test.csv", LockType::Write));
        bb.acquire_resource(ResourceLock {
            resource_type: "file".to_string(), resource_id: "file:///data/test.csv".to_string(),
            acquired_at: chrono::Utc::now(), acquired_by: "agent_1".to_string(),
            lock_type: LockType::Write,
        }).unwrap();
        assert!(!bb.check_resource_available("file:///data/test.csv", LockType::Write));
        assert!(bb.check_resource_available("file:///data/other.csv", LockType::Write));
    }

    // ========== Cross-task Dependency Tests ==========

    #[test]
    fn test_task_dependency_tracking() {
        let bb = Blackboard::new().unwrap();
        bb.register_task("iri://task/parent", None);
        bb.register_task("iri://task/child", Some("iri://task/parent".to_string()));
        bb.add_task_dependency("iri://task/child", "iri://task/parent");

        let deps = bb.get_task_dependencies("iri://task/child");
        assert_eq!(deps, vec!["iri://task/parent"]);

        let dependents = bb.get_task_dependents("iri://task/parent");
        assert_eq!(dependents, vec!["iri://task/child"]);
    }

    #[test]
    fn test_remove_task_dependency() {
        let bb = Blackboard::new().unwrap();
        bb.register_task("iri://task/a", None);
        bb.register_task("iri://task/b", None);
        bb.add_task_dependency("iri://task/b", "iri://task/a");
        assert_eq!(bb.get_task_dependencies("iri://task/b").len(), 1);
        bb.remove_task_dependency("iri://task/b", "iri://task/a");
        assert!(bb.get_task_dependencies("iri://task/b").is_empty());
    }

    #[test]
    fn test_task_dependency_no_duplicates() {
        let bb = Blackboard::new().unwrap();
        bb.register_task("iri://task/a", None);
        bb.register_task("iri://task/b", None);
        bb.add_task_dependency("iri://task/b", "iri://task/a");
        bb.add_task_dependency("iri://task/b", "iri://task/a");
        assert_eq!(bb.get_task_dependencies("iri://task/b").len(), 1);
    }

    // ========== Coordination Message Tests ==========

    #[test]
    fn test_publish_and_read_coordination() {
        let bb = Blackboard::new().unwrap();
        let msg = CoordinationMessage {
            from_agent: "agent_1".to_string(),
            msg_type: CoordinationMsgType::TaskAnnouncement,
            payload: serde_json::json!({"task": "analyze"}),
            timestamp: chrono::Utc::now(),
        };
        bb.publish_coordination(&msg).unwrap();
        bb.flush_oxigraph();

        let msgs = bb.read_coordination_messages().unwrap();
        assert!(!msgs.is_empty(), "Should have at least one coordination message, got {} messages", msgs.len());
        assert_eq!(msgs[0].from_agent, "agent_1");
        assert_eq!(msgs[0].msg_type, CoordinationMsgType::TaskAnnouncement);
    }

    #[test]
    fn test_publish_agent_snapshot() {
        let bb = Blackboard::new().unwrap();
        bb.register_agent("agent_x", "DA", "iri://task/snap");
        bb.update_agent_status("agent_x", AgentActivity::Working, Some("testing snapshot"));
        bb.publish_agent_snapshot_to_shared("agent_x").unwrap();
        bb.flush_oxigraph();
        let results = bb.query_graph("blackboard:shared", "?s ?p ?o").unwrap();
        assert!(!results.is_empty(), "blackboard:shared should have snapshot data");
    }

    // ========== DAG Topology Layer Tests ==========

    #[test]
    fn test_get_task_dag_simple_chain() {
        let bb = Blackboard::new().unwrap();
        bb.register_task("iri://task/a", None);
        bb.register_task("iri://task/b", None);
        bb.register_task("iri://task/c", None);
        bb.add_task_dependency("iri://task/b", "iri://task/a");
        bb.add_task_dependency("iri://task/c", "iri://task/b");
        let dag = bb.get_task_dag("iri://task/a").unwrap();
        // 3 layers: [a], [b], [c]
        assert_eq!(dag.len(), 3, "Expected 3 layers, got {:?}", dag);
        assert_eq!(dag[0], vec!["iri://task/a"]);
        assert_eq!(dag[1], vec!["iri://task/b"]);
        assert_eq!(dag[2], vec!["iri://task/c"]);
    }

    #[test]
    fn test_get_task_dag_fan_out() {
        let bb = Blackboard::new().unwrap();
        bb.register_task("iri://task/a", None);
        bb.register_task("iri://task/b", None);
        bb.register_task("iri://task/c", None);
        bb.add_task_dependency("iri://task/b", "iri://task/a");
        bb.add_task_dependency("iri://task/c", "iri://task/a");
        let dag = bb.get_task_dag("iri://task/a").unwrap();
        // 2 layers: [a], [b, c] (b and c can run in parallel after a)
        assert_eq!(dag.len(), 2, "Expected 2 layers, got {:?}", dag);
        assert_eq!(dag[0], vec!["iri://task/a"]);
        assert_eq!(dag[1], vec!["iri://task/b", "iri://task/c"]);
    }

    #[test]
    fn test_get_task_dag_diamond() {
        let bb = Blackboard::new().unwrap();
        bb.register_task("iri://task/a", None);
        bb.register_task("iri://task/b", None);
        bb.register_task("iri://task/c", None);
        bb.register_task("iri://task/d", None);
        bb.add_task_dependency("iri://task/b", "iri://task/a");
        bb.add_task_dependency("iri://task/c", "iri://task/a");
        bb.add_task_dependency("iri://task/d", "iri://task/b");
        bb.add_task_dependency("iri://task/d", "iri://task/c");
        let dag = bb.get_task_dag("iri://task/a").unwrap();
        // 3 layers: [a], [b, c], [d]
        assert_eq!(dag.len(), 3, "Expected 3 layers, got {:?}", dag);
        assert_eq!(dag[0], vec!["iri://task/a"]);
        assert_eq!(dag[1], vec!["iri://task/b", "iri://task/c"]);
        assert_eq!(dag[2], vec!["iri://task/d"]);
    }

    // ========== Shared State Tests ==========

    #[test]
    fn test_publish_and_read_shared_state() {
        let bb = Blackboard::new().unwrap();
        let state = serde_json::json!({"progress": 0.5, "phase": "analysis"});
        bb.publish_shared_state("iri://task/xyz", &state).unwrap();
        let read_back = bb.get_shared_state("iri://task/xyz").unwrap();
        assert!(read_back.is_some(), "Should have shared state");
        let val = read_back.unwrap();
        assert_eq!(val.get("progress").and_then(|v| v.as_f64()), Some(0.5));
        assert_eq!(val.get("phase").and_then(|v| v.as_str()), Some("analysis"));
    }

    #[test]
    fn test_get_shared_state_nonexistent() {
        let bb = Blackboard::new().unwrap();
        let result = bb.get_shared_state("iri://task/nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_read_coordination_messages_since() {
        let bb = Blackboard::new().unwrap();
        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        let msg = CoordinationMessage {
            from_agent: "agent_filter".to_string(),
            msg_type: CoordinationMsgType::ProgressUpdate,
            payload: serde_json::json!({"step": 2}),
            timestamp: chrono::Utc::now(),
        };
        bb.publish_coordination(&msg).unwrap();
        bb.flush_oxigraph();

        // Filter with past timestamp should include the message
        let msgs = bb.read_coordination_messages_since(&past).unwrap();
        assert!(!msgs.is_empty(), "Should find messages after past timestamp");

        // Filter with future timestamp should exclude the message
        let future = chrono::Utc::now() + chrono::Duration::hours(1);
        let msgs_future = bb.read_coordination_messages_since(&future).unwrap();
        assert!(msgs_future.is_empty(), "Should not find messages after future timestamp");
    }
}
