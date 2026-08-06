use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use crate::core::event_bus::EventBus;
use crate::memory::l2_blackboard::Blackboard;
use crate::memory::l3_projection::ProjectionEngine;
use crate::memory::memory_bus::MemoryBus;
use crate::{CoreConfig, CoreError};

struct PrefetchTask {
    entity_iri: String,
    #[allow(dead_code)]
    intent: String,
    #[allow(dead_code)]
    priority: f64,
}

const MAX_RETRIES: u32 = 3;

pub struct PrefetchEngine {
    memory_bus: Arc<MemoryBus>,
    blackboard: Arc<Blackboard>,
    projection: Arc<ProjectionEngine>,
    queue: RwLock<VecDeque<PrefetchTask>>,
    entity_graph: RwLock<HashMap<String, Vec<String>>>,
    pending: RwLock<HashSet<String>>,
    retry_count: RwLock<HashMap<String, u32>>,
    semaphore: Arc<Semaphore>,
    max_hops: usize,
    top_k: usize,
}

impl PrefetchEngine {
    pub fn new(
        memory_bus: Arc<MemoryBus>,
        blackboard: Arc<Blackboard>,
        projection: Arc<ProjectionEngine>,
    ) -> Self {
        Self {
            memory_bus,
            blackboard,
            projection,
            queue: RwLock::new(VecDeque::new()),
            entity_graph: RwLock::new(HashMap::new()),
            pending: RwLock::new(HashSet::new()),
            retry_count: RwLock::new(HashMap::new()),
            semaphore: Arc::new(Semaphore::new(3)),
            max_hops: 2,
            top_k: 5,
        }
    }

    /// Wire a PREFETCH_REQUEST / MEMORY_PREFETCH consumer that refreshes the
    /// entity graph from the blackboard and drains the queue on each event.
    pub fn spawn_consumer(self: &Arc<Self>, event_bus: Arc<EventBus>, blackboard: Arc<Blackboard>) {
        let bb = blackboard.clone();
        let pf = self.clone();
        event_bus.spawn_consumer(
            vec![
                "MEMORY_PREFETCH".to_string(),
                "PREFETCH_REQUEST".to_string(),
            ],
            move |event| {
                let bb = bb.clone();
                let pf = pf.clone();
                async move {
                    let entity_iri = serde_json::from_str::<serde_json::Value>(&event.payload)
                        .ok()
                        .and_then(|v| {
                            v.get("entity_iri")
                                .and_then(|s| s.as_str())
                                .map(String::from)
                        })
                        .unwrap_or_else(|| event.task_iri.clone());
                    if let Ok(related) = bb.query_by_types(&["KnowledgeFragment".to_string()]) {
                        let related_iris: Vec<String> = related
                            .iter()
                            .map(|n| n.iri.clone())
                            .filter(|iri| *iri != entity_iri)
                            .collect();
                        pf.update_entity_graph(&entity_iri, &related_iris);
                    }
                    match pf.execute_prefetch().await {
                        Ok(n) => {
                            info!(entity_iri = %entity_iri, prefetched = n, "prefetch executed on demand");
                        }
                        Err(e) => {
                            warn!(entity_iri = %entity_iri, error = %e, "prefetch execution failed");
                        }
                    }
                }
            },
        );
    }

    pub async fn on_intent_change(&self, new_intent: &str, current_entities: &[String]) {
        let mut candidates: HashMap<String, f64> = HashMap::new();

        for entity_iri in current_entities {
            let related = self.get_related_entities(entity_iri, self.max_hops);
            for (related_iri, score) in related {
                if current_entities.contains(&related_iri) {
                    continue;
                }
                *candidates.entry(related_iri).or_default() += score;
            }
        }

        let mut sorted: Vec<(String, f64)> = candidates.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_candidates: Vec<(String, f64)> = sorted.into_iter().take(self.top_k).collect();

        let to_enqueue: Vec<(String, f64)> = {
            let mut pending_set = self.pending.write();
            top_candidates
                .into_iter()
                .filter(|(iri, _)| {
                    if pending_set.contains(iri) {
                        debug!(entity_iri = %iri, "skipping duplicate prefetch");
                        false
                    } else {
                        pending_set.insert(iri.clone());
                        true
                    }
                })
                .collect()
        };

        {
            let mut queue = self.queue.write();
            for (entity_iri, priority) in &to_enqueue {
                let task = PrefetchTask {
                    entity_iri: entity_iri.clone(),
                    intent: new_intent.to_string(),
                    priority: *priority,
                };
                queue.push_back(task);
            }
        }

        for (entity_iri, _) in &to_enqueue {
            self.memory_bus
                .emit_prefetch_request(entity_iri, new_intent)
                .await;
        }

        debug!(
            intent = %new_intent,
            candidates = to_enqueue.len(),
            "intent change triggered prefetch"
        );
    }

    pub fn on_new_entity(&self, entity_iri: &str) {
        {
            let mut pending_set = self.pending.write();
            if pending_set.contains(entity_iri) {
                return;
            }
            pending_set.insert(entity_iri.to_string());
        }

        let mut main_queue = self.queue.write();
        main_queue.push_back(PrefetchTask {
            entity_iri: entity_iri.to_string(),
            intent: "new_entity".to_string(),
            priority: 0.5,
        });
        drop(main_queue);

        let has_relations = self.entity_graph.read().contains_key(entity_iri);
        if has_relations {
            let related = self.get_related_entities(entity_iri, 1);
            let mut queue = self.queue.write();
            for (related_iri, _) in related {
                let mut pending = self.pending.write();
                if pending.contains(&related_iri) {
                    continue;
                }
                pending.insert(related_iri.clone());
                drop(pending);
                queue.push_back(PrefetchTask {
                    entity_iri: related_iri,
                    intent: "new_entity_cascade".to_string(),
                    priority: 0.3,
                });
            }
        }

        debug!(entity_iri = %entity_iri, "new entity added to prefetch queue");
    }

    pub async fn execute_prefetch(&self) -> Result<usize, CoreError> {
        let tasks: Vec<PrefetchTask> = {
            let mut queue = self.queue.write();
            queue.drain(..).collect()
        };

        if tasks.is_empty() {
            return Ok(0);
        }

        info!(task_count = tasks.len(), "starting prefetch execution");

        let config = CoreConfig::default();
        let mut handles = Vec::new();

        let success_set: Arc<parking_lot::RwLock<HashSet<String>>> =
            Arc::new(parking_lot::RwLock::new(HashSet::new()));
        let fail_set: Arc<parking_lot::RwLock<Vec<(String, u32)>>> =
            Arc::new(parking_lot::RwLock::new(Vec::new()));

        for task in &tasks {
            let semaphore = self.semaphore.clone();
            let blackboard = self.blackboard.clone();
            let projection = self.projection.clone();
            let config = config.clone();
            let entity_iri = task.entity_iri.clone();
            let successes = success_set.clone();
            let failures = fail_set.clone();

            let handle = tokio::spawn(async move {
                let _permit = match semaphore.acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return,
                };

                let result = projection
                    .project(&entity_iri, "reference_only", HashMap::new())
                    .await;

                match result {
                    Ok(projection_json) => {
                        if let Ok(parsed) =
                            serde_json::from_str::<serde_json::Value>(&projection_json)
                        {
                            if let Some(artifacts) =
                                parsed.get("artifacts").and_then(|a| a.as_array())
                            {
                                for artifact in artifacts {
                                    if let Some(iri) = artifact.get("@id").and_then(|v| v.as_str())
                                    {
                                        let node_json =
                                            serde_json::to_string(artifact).unwrap_or_default();
                                        if !node_json.is_empty() {
                                            let _ = blackboard.write_node(iri, &node_json, &config);
                                        }
                                    }
                                }
                            }
                        }
                        successes.write().insert(entity_iri.clone());
                    }
                    Err(e) => {
                        warn!(entity_iri = %entity_iri, error = %e, "prefetch projection failed");
                        failures.write().push((entity_iri.clone(), 1u32));
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.await;
        }

        {
            let successes = success_set.read();
            if !successes.is_empty() {
                let mut pending = self.pending.write();
                for iri in successes.iter() {
                    pending.remove(iri);
                }
            }
        }

        {
            let failures = fail_set.read();
            let mut retries = self.retry_count.write();
            let mut queue = self.queue.write();
            for (entity_iri, _) in failures.iter() {
                let count = retries.entry(entity_iri.clone()).or_insert(0);
                *count += 1;
                if *count < MAX_RETRIES {
                    debug!(entity_iri = %entity_iri, retry = *count, "retrying prefetch");
                    queue.push_back(PrefetchTask {
                        entity_iri: entity_iri.clone(),
                        intent: "retry".to_string(),
                        priority: 0.1,
                    });
                } else {
                    warn!(entity_iri = %entity_iri, "prefetch retries exhausted");
                    self.pending.write().remove(entity_iri);
                    retries.remove(entity_iri);
                }
            }
        }

        let prefetched = success_set.read().len();
        info!(prefetched = prefetched, "prefetch completed");
        Ok(prefetched)
    }

    pub fn update_entity_graph(&self, entity_iri: &str, related: &[String]) {
        let mut graph = self.entity_graph.write();
        graph.insert(entity_iri.to_string(), related.to_vec());
        for rel in related {
            graph.entry(rel.clone()).or_default();
            if let Some(neighbors) = graph.get_mut(rel) {
                if !neighbors.contains(&entity_iri.to_string()) {
                    neighbors.push(entity_iri.to_string());
                }
            }
        }
        debug!(entity_iri = %entity_iri, related_count = related.len(), "entity graph updated");
    }

    pub fn get_related_entities(&self, entity_iri: &str, max_hops: usize) -> Vec<(String, f64)> {
        let graph = self.entity_graph.read();
        let mut visited: HashMap<String, f64> = HashMap::new();
        let mut current_hop: Vec<String> = vec![entity_iri.to_string()];
        visited.insert(entity_iri.to_string(), 1.0);

        for hop in 1..=max_hops {
            let decay = 0.5_f64.powi(hop as i32);
            let mut next_hop = Vec::new();

            for node in &current_hop {
                if let Some(neighbors) = graph.get(node) {
                    for neighbor in neighbors {
                        if !visited.contains_key(neighbor) {
                            visited.insert(neighbor.clone(), decay);
                            next_hop.push(neighbor.clone());
                        }
                    }
                }
            }

            current_hop = next_hop;
        }

        visited.remove(entity_iri);

        let mut result: Vec<(String, f64)> = visited.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    pub fn queue_len(&self) -> usize {
        self.queue.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::l2_blackboard::Blackboard;
    use crate::memory::l3_projection::ProjectionEngine;
    use crate::memory::memory_bus::MemoryBus;
    use crate::core::event_bus::EventBus;

    fn build() -> (Arc<PrefetchEngine>, Arc<Blackboard>) {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let proj = Arc::new(ProjectionEngine::new(blackboard.clone(), 500));
        let bus = Arc::new(MemoryBus::new(Arc::new(EventBus::new(100))));
        let engine = Arc::new(PrefetchEngine::new(bus, blackboard.clone(), proj));
        (engine, blackboard)
    }

    #[test]
    fn update_entity_graph_builds_bidirectional_edges() {
        let (engine, _) = build();
        engine.update_entity_graph(
            "iri://entity/a",
            &["iri://entity/b".to_string(), "iri://entity/c".to_string()],
        );
        // hop 1 衰减 = 0.5
        let related = engine.get_related_entities("iri://entity/a", 1);
        let iris: Vec<String> = related.iter().map(|(i, _)| i.clone()).collect();
        assert!(iris.contains(&"iri://entity/b".to_string()));
        assert!(iris.contains(&"iri://entity/c".to_string()));
        assert!(
            related.iter().all(|(_, s)| (s - 0.5).abs() < f64::EPSILON),
            "all 1-hop neighbors decay to 0.5, got {:?}",
            related
        );
    }

    #[test]
    fn get_related_entities_excludes_self_and_respects_hops() {
        let (engine, _) = build();
        engine.update_entity_graph(
            "iri://entity/root",
            &["iri://entity/hop1".to_string()],
        );
        engine.update_entity_graph(
            "iri://entity/hop1",
            &["iri://entity/hop2".to_string()],
        );
        let related = engine.get_related_entities("iri://entity/root", 2);
        let map: HashMap<String, f64> = related.into_iter().collect();
        assert!(!map.contains_key("iri://entity/root"), "self must be excluded");
        assert!(
            (map["iri://entity/hop1"] - 0.5).abs() < f64::EPSILON,
            "hop1 must decay to 0.5"
        );
        assert!(
            (map["iri://entity/hop2"] - 0.25).abs() < f64::EPSILON,
            "hop2 must decay to 0.25"
        );
    }

    #[test]
    fn on_intent_change_enqueues_and_rejects_duplicates() {
        let (engine, _) = build();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            engine.on_new_entity("iri://entity/a");
            engine.on_intent_change("new intent", &["iri://entity/a".to_string()]).await;
            assert_eq!(engine.queue_len(), 1, "single entity enqueued via on_new_entity");
        });
    }

    #[test]
    fn execute_prefetch_drains_empty_queue() {
        let (engine, _) = build();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async { engine.execute_prefetch().await.unwrap() });
        assert_eq!(result, 0, "empty queue returns 0");
    }

    #[test]
    fn on_new_entity_enqueues_and_cascades() {
        let (engine, _) = build();
        engine.update_entity_graph("iri://entity/a", &["iri://entity/b".to_string()]);
        engine.on_new_entity("iri://entity/a");
        assert!(engine.queue_len() >= 2, "a self + cascade b, got {}", engine.queue_len());
    }

    #[test]
    fn prefetch_consumer_drains_queue_on_event() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let proj = Arc::new(ProjectionEngine::new(blackboard.clone(), 500));
        let event_bus: Arc<EventBus> = Arc::new(EventBus::new(16));
        let bus = Arc::new(MemoryBus::new(event_bus.clone()));
        let engine: Arc<PrefetchEngine> =
            Arc::new(PrefetchEngine::new(bus, blackboard.clone(), proj));

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            engine.spawn_consumer(event_bus.clone(), blackboard.clone());
            engine.on_new_entity("iri://entity/a");
            assert!(engine.queue_len() > 0, "entity enqueued before drain");
            let bus = MemoryBus::new(event_bus.clone());
            bus.emit_prefetch_request("iri://entity/a", "test").await;
            for _ in 0..50 {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                if engine.queue_len() == 0 {
                    break;
                }
            }
        });
        assert_eq!(
            engine.queue_len(),
            0,
            "consumer drains the queue after PREFETCH_REQUEST event"
        );
    }
}
