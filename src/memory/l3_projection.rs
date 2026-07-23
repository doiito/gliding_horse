use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, instrument, warn};

use crate::jsonld::framing::{
    apply_frame, estimate_tokens, fit_to_budget, EmbedDirective, FrameTemplate,
};
use crate::jsonld::JsonLdContext;
use crate::memory::hyperspace_store::HyperspaceStore;
use crate::memory::l2_blackboard::Blackboard;
use crate::CoreError;

#[derive(Debug, Clone)]
pub struct MaterializedView {
    pub cache_key: String,
    pub result_json: String,
    pub dependent_nodes: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub is_valid: bool,
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_views: usize,
    pub valid_views: usize,
    pub invalid_views: usize,
    pub total_size_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionFrame {
    pub name: String,
    pub description: String,
    pub target_role: String,
    pub include_properties: Vec<String>,
    pub max_size: usize,
    pub max_nodes: usize,
    pub sparql_template: Option<String>,
    pub params: Vec<String>,
    pub jsonld_frame: Option<FrameTemplate>,
}

pub struct ProjectionEngine {
    blackboard: Arc<Blackboard>,
    max_size: usize,
    frames: HashMap<String, ProjectionFrame>,
    materialized_cache: RwLock<HashMap<String, MaterializedView>>,
    /// node_iri → Vec<cache_key> reverse index for O(1) node invalidation
    reverse_index: RwLock<HashMap<String, Vec<String>>>,
    vector_store: Option<Arc<HyperspaceStore>>,
}

impl ProjectionEngine {
    pub fn new(blackboard: Arc<Blackboard>, max_size: usize) -> Self {
        Self::with_vector_store(blackboard, max_size, None)
    }

    pub fn with_vector_store(
        blackboard: Arc<Blackboard>,
        max_size: usize,
        vector_store: Option<Arc<HyperspaceStore>>,
    ) -> Self {
        let frames = Self::load_default_frames();
        Self {
            blackboard,
            max_size,
            frames,
            materialized_cache: RwLock::new(HashMap::new()),
            reverse_index: RwLock::new(HashMap::new()),
            vector_store,
        }
    }

    pub fn invalidate_for_node(&self, node_iri: &str) -> usize {
        let index = self.reverse_index.read();
        let cache_keys = match index.get(node_iri) {
            Some(keys) => keys.clone(),
            None => return 0,
        };
        drop(index);

        let mut cache = self.materialized_cache.write();
        let mut invalidated = 0;
        for key in &cache_keys {
            if let Some(view) = cache.get_mut(key) {
                view.is_valid = false;
                invalidated += 1;
                debug!(cache_key = %key, "L3 projection cache invalidated");
            }
        }
        invalidated
    }

    pub fn invalidate_for_nodes(&self, node_iris: &[String]) -> usize {
        let mut total = 0;
        for iri in node_iris {
            total += self.invalidate_for_node(iri);
        }
        total
    }

    pub fn cleanup_invalid(&self) -> usize {
        let mut cache = self.materialized_cache.write();
        let before = cache.len();
        cache.retain(|_, view| view.is_valid);
        let removed = before - cache.len();
        if removed > 0 {
            self.rebuild_reverse_index(&cache);
        }
        removed
    }

    fn load_default_frames() -> HashMap<String, ProjectionFrame> {
        let mut frames = HashMap::new();

        frames.insert(
            "summary_only".to_string(),
            ProjectionFrame {
                name: "summary_only".to_string(),
                description: "SA global situation awareness".to_string(),
                target_role: "SA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "summary".to_string(),
                    "status".to_string(),
                    "confidence".to_string(),
                ],
                max_size: 500,
                max_nodes: 20,
                sparql_template: Some(
                    r#"
                PREFIX ex: <http://agent-os.org/ontology/>
                CONSTRUCT {
                    ?node ex:summary ?summary .
                    ?node ex:status ?status .
                    ?node ex:confidence ?conf .
                    ?node a ?type .
                }
                WHERE {
                    ?node a ?type .
                    OPTIONAL { ?node ex:summary ?summary }
                    OPTIONAL { ?node ex:status ?status }
                    OPTIONAL { ?node ex:confidence ?conf }
                }
            "#
                    .to_string(),
                ),
                params: vec![],
                jsonld_frame: Some(
                    FrameTemplate::new(serde_json::json!({
                        "agent": "https://agent-harness.os/agent#"
                    }))
                    .with_include_properties(vec!["summary".to_string(), "status".to_string()])
                    .with_max_depth(1),
                ),
            },
        );

        frames.insert(
            "pa_init".to_string(),
            ProjectionFrame {
                name: "pa_init".to_string(),
                description: "PA startup input".to_string(),
                target_role: "PA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "summary".to_string(),
                    "goal".to_string(),
                    "constraints".to_string(),
                    "resources".to_string(),
                    "task:what".to_string(),
                    "task:why".to_string(),
                    "task:how".to_string(),
                    "task:where".to_string(),
                    "five_w2h_what".to_string(),
                    "five_w2h_why".to_string(),
                    "five_w2h_deadline".to_string(),
                    "five_w2h_execution_env".to_string(),
                ],
                max_size: 512,
                max_nodes: 10,
                sparql_template: Some(
                    r#"
                PREFIX ex: <http://agent-os.org/ontology/>
                CONSTRUCT {
                    ?task ex:goal ?goal .
                    ?task ex:constraints ?constraints .
                    ?task ex:resources ?resources .
                    ?task ex:summary ?summary .
                }
                WHERE {
                    ?task a ex:Task .
                    ?task ex:summary ?summary .
                    OPTIONAL { ?task ex:goal ?goal }
                    OPTIONAL { ?task ex:constraints ?constraints }
                    OPTIONAL { ?task ex:resources ?resources }
                }
            "#
                    .to_string(),
                ),
                params: vec!["task_iri".to_string()],
                jsonld_frame: Some(
                    FrameTemplate::new(serde_json::json!({
                        "exec": "https://agent-harness.os/exec#",
                        "task": "https://agent-harness.os/task#"
                    }))
                    .with_embed_rule("task:subTasks".to_string(), EmbedDirective::Always)
                    .with_embed_rule("exec:assignedTo".to_string(), EmbedDirective::Link)
                    .with_max_depth(3),
                ),
            },
        );

        frames.insert(
            "da_input".to_string(),
            ProjectionFrame {
                name: "da_input".to_string(),
                description: "DA execution input".to_string(),
                target_role: "DA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "summary".to_string(),
                    "instructions".to_string(),
                    "dependencies".to_string(),
                    "language".to_string(),
                ],
                max_size: 512,
                max_nodes: 10,
                sparql_template: Some(
                    r#"
                PREFIX ex: <http://agent-os.org/ontology/>
                CONSTRUCT {
                    ?node ex:instructions ?instructions .
                    ?node ex:dependencies ?deps .
                    ?node ex:language ?lang .
                    ?node ex:summary ?summary .
                }
                WHERE {
                    ?node a ex:PlanNode .
                    ?node ex:summary ?summary .
                    OPTIONAL { ?node ex:instructions ?instructions }
                    OPTIONAL { ?node ex:dependencies ?deps }
                    OPTIONAL { ?node ex:language ?lang }
                }
            "#
                    .to_string(),
                ),
                params: vec!["plan_iri".to_string()],
                jsonld_frame: Some(
                    FrameTemplate::new(serde_json::json!({
                        "exec": "https://agent-harness.os/exec#",
                        "task": "https://agent-harness.os/task#"
                    }))
                    .with_embed_rule("task:inputData".to_string(), EmbedDirective::Always)
                    .with_embed_rule("task:resources".to_string(), EmbedDirective::Link)
                    .with_max_depth(4),
                ),
            },
        );

        frames.insert(
            "ca_review".to_string(),
            ProjectionFrame {
                name: "ca_review".to_string(),
                description: "CA check input".to_string(),
                target_role: "CA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "summary".to_string(),
                    "storage_path".to_string(),
                    "language".to_string(),
                    "dependencies".to_string(),
                    "task:what".to_string(),
                    "task:why".to_string(),
                    "task:who".to_string(),
                    "task:when".to_string(),
                    "task:where".to_string(),
                    "task:how".to_string(),
                    "task:howMuch".to_string(),
                    "five_w2h_what".to_string(),
                    "five_w2h_why".to_string(),
                    "five_w2h_success_criteria".to_string(),
                    "five_w2h_deadline".to_string(),
                    "five_w2h_execution_env".to_string(),
                    "five_w2h_required_steps".to_string(),
                    "five_w2h_token_budget".to_string(),
                    "five_w2h_forbidden_tools".to_string(),
                    "auditResult".to_string(),
                    "verdict".to_string(),
                ],
                max_size: 256,
                max_nodes: 10,
                sparql_template: None,
                params: vec!["artifact_iri".to_string()],
                jsonld_frame: Some(
                    FrameTemplate::new(serde_json::json!({
                        "exec": "https://agent-harness.os/exec#",
                        "task": "https://agent-harness.os/task#"
                    }))
                    .with_embed_rule("exec:results".to_string(), EmbedDirective::Always)
                    .with_embed_rule("exec:validationRules".to_string(), EmbedDirective::Always)
                    .with_max_depth(3),
                ),
            },
        );

        frames.insert(
            "aa_decision".to_string(),
            ProjectionFrame {
                name: "aa_decision".to_string(),
                description: "AA decision input".to_string(),
                target_role: "AA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "summary".to_string(),
                    "verdict".to_string(),
                    "severity".to_string(),
                    "suggestions".to_string(),
                    "task:what".to_string(),
                    "task:why".to_string(),
                    "task:howMuch".to_string(),
                    "five_w2h_what".to_string(),
                    "five_w2h_why".to_string(),
                    "auditResult".to_string(),
                    "overallVerdict".to_string(),
                    "verdict".to_string(),
                    "suggestions".to_string(),
                    "decision".to_string(),
                ],
                max_size: 512,
                max_nodes: 10,
                sparql_template: None,
                params: vec!["review_iri".to_string()],
                jsonld_frame: Some(
                    FrameTemplate::new(serde_json::json!({
                        "exec": "https://agent-harness.os/exec#",
                        "task": "https://agent-harness.os/task#"
                    }))
                    .with_embed_rule("exec:reviewResults".to_string(), EmbedDirective::Always)
                    .with_embed_rule("exec:alternatives".to_string(), EmbedDirective::Link)
                    .with_max_depth(2),
                ),
            },
        );

        frames.insert(
            "health_check".to_string(),
            ProjectionFrame {
                name: "health_check".to_string(),
                description: "Health status check for SA".to_string(),
                target_role: "SA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "status".to_string(),
                    "confidence".to_string(),
                    "error_count".to_string(),
                ],
                max_size: 256,
                max_nodes: 20,
                sparql_template: None,
                params: vec![],
                jsonld_frame: None,
            },
        );

        frames.insert(
            "error_analysis".to_string(),
            ProjectionFrame {
                name: "error_analysis".to_string(),
                description: "Error analysis view for SA".to_string(),
                target_role: "SA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "error_type".to_string(),
                    "error_message".to_string(),
                    "timestamp".to_string(),
                ],
                max_size: 512,
                max_nodes: 10,
                sparql_template: None,
                params: vec!["agent_id".to_string()],
                jsonld_frame: None,
            },
        );

        frames.insert(
            "reference_only".to_string(),
            ProjectionFrame {
                name: "reference_only".to_string(),
                description: "Minimal IRI reference only".to_string(),
                target_role: "any".to_string(),
                include_properties: vec!["@id".to_string()],
                max_size: 128,
                max_nodes: 50,
                sparql_template: None,
                params: vec![],
                jsonld_frame: Some(FrameTemplate::new(serde_json::json!({})).with_max_depth(1)),
            },
        );

        frames.insert(
            "5w2h_summary".to_string(),
            ProjectionFrame {
                name: "5w2h_summary".to_string(),
                description: "5W2H summary view for SA dashboard".to_string(),
                target_role: "SA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "task:what".to_string(),
                    "task:why".to_string(),
                    "status".to_string(),
                    "summary".to_string(),
                    "five_w2h_what".to_string(),
                    "five_w2h_why".to_string(),
                    "five_w2h_deadline".to_string(),
                    "five_w2h_priority".to_string(),
                ],
                max_size: 300,
                max_nodes: 20,
                sparql_template: Some(
                    r#"
        PREFIX task: <https://pdca-agent.org/ontology/task#>
        CONSTRUCT {
            ?node task:what ?what .
            ?node task:why ?why .
            ?node task:status ?status .
            ?node task:summary ?summary .
        }
        WHERE {
            ?node a task:5W2H .
            ?node task:what ?what .
            OPTIONAL { ?node task:why ?why }
            OPTIONAL { ?node task:status ?status }
            OPTIONAL { ?node task:summary ?summary }
        }
    "#
                    .to_string(),
                ),
                params: vec![],
                jsonld_frame: Some(
                    FrameTemplate::new(serde_json::json!({
                        "task": "https://pdca-agent.org/ontology/task#"
                    }))
                    .with_include_properties(vec![
                        "task:what".to_string(),
                        "task:why".to_string(),
                        "status".to_string(),
                    ])
                    .with_max_depth(2),
                ),
            },
        );

        // ── Workspace Monitor L3 frames ──
        frames.insert(
            "workspace_stale_files".to_string(),
            ProjectionFrame {
                name: "workspace_stale_files".to_string(),
                description: "Lists all workspace files with ReadStale or WrittenUnread state"
                    .to_string(),
                target_role: "DA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "ws:filePath".to_string(),
                    "ws:state".to_string(),
                    "ws:currentVersion".to_string(),
                    "ws:lastReadVersion".to_string(),
                    "ws:mtime".to_string(),
                    "ws:fileExt".to_string(),
                ],
                max_size: 1024,
                max_nodes: 100,
                sparql_template: Some(
                    r#"
        PREFIX ws: <iri://workspace/ontology/>
        CONSTRUCT {
            ?node a ws:File .
            ?node ws:filePath ?path .
            ?node ws:state ?state .
            ?node ws:currentVersion ?ver .
            ?node ws:lastReadVersion ?lver .
            ?node ws:mtime ?mtime .
            ?node ws:fileExt ?ext .
        }
        WHERE {
            ?node a ws:File .
            ?node ws:filePath ?path .
            ?node ws:state ?state .
            FILTER(?state = "ReadStale" || ?state = "WrittenUnread")
            OPTIONAL { ?node ws:currentVersion ?ver }
            OPTIONAL { ?node ws:lastReadVersion ?lver }
            OPTIONAL { ?node ws:mtime ?mtime }
            OPTIONAL { ?node ws:fileExt ?ext }
        }
        ORDER BY DESC(?mtime)
    "#
                    .to_string(),
                ),
                params: vec![],
                jsonld_frame: Some(
                    FrameTemplate::new(serde_json::json!({
                        "ws": "iri://workspace/ontology/"
                    }))
                    .with_max_depth(2),
                ),
            },
        );

        frames.insert(
            "workspace_file_detail".to_string(),
            ProjectionFrame {
                name: "workspace_file_detail".to_string(),
                description: "Detailed view of a specific workspace file by path".to_string(),
                target_role: "DA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "ws:filePath".to_string(),
                    "ws:fileSize".to_string(),
                    "ws:fileExt".to_string(),
                    "ws:language".to_string(),
                    "ws:mtime".to_string(),
                    "ws:contentHash".to_string(),
                    "ws:state".to_string(),
                    "ws:lastReadAt".to_string(),
                    "ws:lastReadVersion".to_string(),
                    "ws:currentVersion".to_string(),
                    "ws:readCount".to_string(),
                    "ws:parentDir".to_string(),
                ],
                max_size: 512,
                max_nodes: 5,
                sparql_template: Some(
                    r#"
        PREFIX ws: <iri://workspace/ontology/>
        CONSTRUCT {
            ?node a ws:File .
            ?node ws:filePath ?path .
            ?node ws:fileSize ?size .
            ?node ws:fileExt ?ext .
            ?node ws:language ?lang .
            ?node ws:mtime ?mtime .
            ?node ws:contentHash ?hash .
            ?node ws:state ?state .
            ?node ws:lastReadAt ?lread .
            ?node ws:lastReadVersion ?lver .
            ?node ws:currentVersion ?ver .
            ?node ws:readCount ?rc .
            ?node ws:parentDir ?parent .
        }
        WHERE {
            ?node a ws:File .
            ?node ws:filePath ?path .
            FILTER(CONTAINS(LCASE(?path), LCASE($target_path)))
            OPTIONAL { ?node ws:fileSize ?size }
            OPTIONAL { ?node ws:fileExt ?ext }
            OPTIONAL { ?node ws:language ?lang }
            OPTIONAL { ?node ws:mtime ?mtime }
            OPTIONAL { ?node ws:contentHash ?hash }
            OPTIONAL { ?node ws:state ?state }
            OPTIONAL { ?node ws:lastReadAt ?lread }
            OPTIONAL { ?node ws:lastReadVersion ?lver }
            OPTIONAL { ?node ws:currentVersion ?ver }
            OPTIONAL { ?node ws:readCount ?rc }
            OPTIONAL { ?node ws:parentDir ?parent }
        }
        LIMIT 5
    "#
                    .to_string(),
                ),
                params: vec!["target_path".to_string()],
                jsonld_frame: Some(
                    FrameTemplate::new(serde_json::json!({
                        "ws": "iri://workspace/ontology/"
                    }))
                    .with_max_depth(2),
                ),
            },
        );

        frames.insert(
            "workspace_overview".to_string(),
            ProjectionFrame {
                name: "workspace_overview".to_string(),
                description: "Aggregated workspace overview: file counts by state, by language"
                    .to_string(),
                target_role: "SA".to_string(),
                include_properties: vec![
                    "@id".to_string(),
                    "@type".to_string(),
                    "ws:filePath".to_string(),
                    "ws:state".to_string(),
                    "ws:language".to_string(),
                    "ws:fileSize".to_string(),
                    "ws:fileExt".to_string(),
                ],
                max_size: 2048,
                max_nodes: 500,
                sparql_template: Some(
                    r#"
        PREFIX ws: <iri://workspace/ontology/>
        CONSTRUCT {
            ?node a ws:File .
            ?node ws:filePath ?path .
            ?node ws:state ?state .
            ?node ws:language ?lang .
            ?node ws:fileSize ?size .
            ?node ws:fileExt ?ext .
        }
        WHERE {
            ?node a ws:File .
            ?node ws:filePath ?path .
            ?node ws:state ?state .
            OPTIONAL { ?node ws:language ?lang }
            OPTIONAL { ?node ws:fileSize ?size }
            OPTIONAL { ?node ws:fileExt ?ext }
        }
        ORDER BY ?path
    "#
                    .to_string(),
                ),
                params: vec![],
                jsonld_frame: Some(
                    FrameTemplate::new(serde_json::json!({
                        "ws": "iri://workspace/ontology/"
                    }))
                    .with_max_depth(1),
                ),
            },
        );

        frames
    }

    #[instrument(skip(self, params))]
    pub async fn project(
        &self,
        task_iri: &str,
        frame_name: &str,
        params: HashMap<String, String>,
    ) -> Result<String, CoreError> {
        debug!(task_iri = %task_iri, frame = %frame_name, "Executing projection");

        let frame = self
            .frames
            .get(frame_name)
            .ok_or_else(|| CoreError::FrameNotFound {
                name: frame_name.to_string(),
            })?;

        let cache_key = format!("{}:{}", task_iri, frame_name);
        if let Some(cached) = self.materialized_cache.read().get(&cache_key) {
            if cached.is_valid {
                debug!(cache_key = %cache_key, "Projection cache hit");
                return Ok(cached.result_json.clone());
            }
        }

        let mut projection = serde_json::Map::new();
        projection.insert(
            "@context".to_string(),
            (*JsonLdContext::context_value()).clone(),
        );
        projection.insert(
            "task_iri".to_string(),
            serde_json::Value::String(task_iri.to_string()),
        );
        projection.insert(
            "frame".to_string(),
            serde_json::Value::String(frame_name.to_string()),
        );

        let artifacts = if let Some(sparql_template) = &frame.sparql_template {
            // Flush pending Oxigraph syncs before SPARQL query to avoid write-lock contention
            // and ensure all recently written nodes are queryable.
            self.blackboard.flush_oxigraph();
            self.execute_sparql_construct(
                sparql_template,
                &frame.include_properties,
                frame.max_nodes,
            )?
        } else {
            self.project_from_cache(task_iri, &frame.include_properties, frame.max_nodes)?
        };

        projection.insert("artifacts".to_string(), serde_json::Value::Array(artifacts));

        if !params.is_empty() {
            let params_obj: serde_json::Map<String, serde_json::Value> = params
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .collect();
            projection.insert("params".to_string(), serde_json::Value::Object(params_obj));
        }

        let result = serde_json::to_string(&serde_json::Value::Object(projection.clone()))
            .map_err(|e| CoreError::Internal {
                message: e.to_string(),
            })?;

        let size = result.as_bytes().len();
        let result = if size > self.max_size {
            let truncated = self.truncate_projection(result, self.max_size)?;
            debug!(
                original_size = size,
                truncated_size = truncated.len(),
                "Projection truncated"
            );
            truncated
        } else {
            let artifacts_count = projection
                .get("artifacts")
                .and_then(|a| a.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            debug!(
                size = size,
                artifacts_count = artifacts_count,
                "Projection complete"
            );
            result
        };

        let dependent_nodes: Vec<String> = self.blackboard.get_task_nodes(task_iri);
        let view = MaterializedView {
            cache_key: cache_key.clone(),
            result_json: result.clone(),
            dependent_nodes: dependent_nodes.clone(),
            created_at: chrono::Utc::now(),
            is_valid: true,
        };
        self.materialized_cache
            .write()
            .insert(cache_key.clone(), view);
        {
            let mut index = self.reverse_index.write();
            for node in &dependent_nodes {
                index
                    .entry(node.clone())
                    .or_default()
                    .push(cache_key.clone());
            }
        }

        Ok(result)
    }

    fn execute_sparql_construct(
        &self,
        sparql: &str,
        include_properties: &[String],
        max_nodes: usize,
    ) -> Result<Vec<serde_json::Value>, CoreError> {
        debug!(sparql_len = sparql.len(), "Executing SPARQL CONSTRUCT");

        let results = self.blackboard.query(sparql)?;

        let mut subject_data: HashMap<String, serde_json::Map<String, serde_json::Value>> =
            HashMap::new();

        for result in results.iter() {
            let subject = result.get("subject").and_then(|s| s.as_str());
            let predicate = result.get("predicate").and_then(|p| p.as_str());
            let object = result.get("object").and_then(|o| o.as_str());

            if let (Some(subj), Some(pred), Some(obj)) = (subject, predicate, object) {
                let prop_name = Self::predicate_to_property_name(pred);

                if include_properties.contains(&prop_name) || prop_name == "@type" {
                    let entry = subject_data.entry(subj.to_string()).or_insert_with(|| {
                        let mut m = serde_json::Map::new();
                        m.insert(
                            "@id".to_string(),
                            serde_json::Value::String(subj.to_string()),
                        );
                        m
                    });

                    let value = Self::parse_object_value(obj);
                    if prop_name == "@type" {
                        if let Some(existing) = entry.get_mut("@type") {
                            if let Some(types) = existing.as_array_mut() {
                                if !types.contains(&value) {
                                    types.push(value);
                                }
                            }
                        } else {
                            entry.insert("@type".to_string(), serde_json::json!([value]));
                        }
                    } else {
                        entry.insert(prop_name, value);
                    }
                }
            }
        }

        let artifacts: Vec<serde_json::Value> = subject_data
            .into_values()
            .filter(|m| m.len() > 1)
            .map(|m| serde_json::Value::Object(m))
            .take(max_nodes)
            .collect();

        debug!(
            artifacts = artifacts.len(),
            "SPARQL CONSTRUCT completed (optimized, no N+1)"
        );
        Ok(artifacts)
    }

    fn predicate_to_property_name(predicate: &str) -> String {
        let predicate = predicate.trim_start_matches('<').trim_end_matches('>');

        if predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
            return "@type".to_string();
        }

        if let Some(prop) = predicate.strip_prefix("http://agent-os.org/prop/") {
            return prop.replace('_', " ");
        }

        if let Some(prop) = predicate.strip_prefix("http://agent-os.org/ontology/") {
            return prop.to_string();
        }

        for prefix in &[
            "https://agent-harness.os/task#",
            "https://agent-harness.os/exec#",
        ] {
            if let Some(prop) = predicate.strip_prefix(prefix) {
                return prop.to_string();
            }
        }

        predicate.to_string()
    }

    fn parse_object_value(object: &str) -> serde_json::Value {
        let object = object.trim_start_matches('<').trim_end_matches('>');

        if object.starts_with('"') && object.ends_with('"') {
            let inner = &object[1..object.len() - 1];
            let unescaped = inner
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
            return serde_json::Value::String(unescaped);
        }

        if let Ok(n) = object.parse::<i64>() {
            return serde_json::Value::Number(n.into());
        }
        if let Ok(n) = object.parse::<f64>() {
            if let Some(num) = serde_json::Number::from_f64(n) {
                return serde_json::Value::Number(num);
            }
        }
        if object == "true" {
            return serde_json::Value::Bool(true);
        }
        if object == "false" {
            return serde_json::Value::Bool(false);
        }

        serde_json::Value::String(object.to_string())
    }

    fn project_from_cache(
        &self,
        task_iri: &str,
        include_properties: &[String],
        max_nodes: usize,
    ) -> Result<Vec<serde_json::Value>, CoreError> {
        let node_iris = self.blackboard.get_task_nodes(task_iri);
        let mut artifacts = Vec::new();

        for node_iri in node_iris.iter().take(max_nodes) {
            if let Some(node) = self.blackboard.read_node(node_iri)? {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&node.json_ld) {
                    let mut artifact = serde_json::Map::new();
                    for prop in include_properties {
                        if let Some(value) = parsed.get(prop) {
                            artifact.insert(prop.clone(), value.clone());
                        }
                    }
                    if !artifact.is_empty() {
                        artifacts.push(serde_json::Value::Object(artifact));
                    }
                }
            }
        }

        Ok(artifacts)
    }

    fn truncate_projection(&self, json: String, max_size: usize) -> Result<String, CoreError> {
        let mut value: serde_json::Value =
            serde_json::from_str(&json).map_err(|e| CoreError::Internal {
                message: e.to_string(),
            })?;

        loop {
            let current = serde_json::to_string(&value).map_err(|e| CoreError::Internal {
                message: e.to_string(),
            })?;

            if current.len() <= max_size {
                return Ok(current);
            }

            if let Some(artifacts) = value.get_mut("artifacts").and_then(|a| a.as_array_mut()) {
                if artifacts.pop().is_none() {
                    break;
                }
            } else {
                break;
            }
        }

        let current = serde_json::to_string(&value).map_err(|e| CoreError::Internal {
            message: e.to_string(),
        })?;

        if current.len() <= max_size {
            return Ok(current);
        }

        if let Some(obj) = value.as_object_mut() {
            obj.retain(|k, _| k == "@context" || k == "task_iri" || k == "frame");
        }

        let current = serde_json::to_string(&value).map_err(|e| CoreError::Internal {
            message: e.to_string(),
        })?;

        if current.len() > max_size {
            if let Some(obj) = value.as_object_mut() {
                obj.insert(
                    "@context".to_string(),
                    Value::String("https://agent-os.org/context".to_string()),
                );
            }
        }

        serde_json::to_string(&value).map_err(|e| CoreError::Internal {
            message: e.to_string(),
        })
    }

    pub fn register_frame(&mut self, frame: ProjectionFrame) {
        self.frames.insert(frame.name.clone(), frame);
    }

    pub fn list_frames(&self) -> Vec<&ProjectionFrame> {
        self.frames.values().collect()
    }

    pub fn get_frame(&self, name: &str) -> Option<&ProjectionFrame> {
        self.frames.get(name)
    }

    fn rebuild_reverse_index(&self, cache: &HashMap<String, MaterializedView>) {
        let mut index = self.reverse_index.write();
        index.clear();
        for (cache_key, view) in cache {
            for node in &view.dependent_nodes {
                index
                    .entry(node.clone())
                    .or_default()
                    .push(cache_key.clone());
            }
        }
    }

    pub fn invalidate_view(&self, frame_name: &str, task_iri: &str) {
        let cache_key = format!("{}:{}", task_iri, frame_name);
        if let Some(view) = self.materialized_cache.write().get_mut(&cache_key) {
            view.is_valid = false;
            debug!(cache_key = %cache_key, "Materialized view invalidated");
        }
    }

    pub fn invalidate_by_node(&self, node_iri: &str) {
        let index = self.reverse_index.read();
        let cache_keys: Vec<String> = match index.get(node_iri) {
            Some(keys) => keys.clone(),
            None => return,
        };
        drop(index);

        let mut cache = self.materialized_cache.write();
        for key in &cache_keys {
            if let Some(view) = cache.get_mut(key) {
                view.is_valid = false;
                debug!(node_iri = %node_iri, cache_key = %key, "View invalidated by node change");
            }
        }
    }

    pub fn clear_cache(&self) {
        let mut cache = self.materialized_cache.write();
        let count = cache.len();
        cache.clear();
        self.reverse_index.write().clear();
        debug!(cleared_count = count, "Projection cache cleared");
    }

    pub fn invalidate_cache_for_task(&self, task_iri: &str) {
        let mut cache = self.materialized_cache.write();
        let keys_to_invalidate: Vec<String> = cache
            .keys()
            .filter(|k| k.starts_with(&format!("{}:", task_iri)))
            .cloned()
            .collect();

        for key in keys_to_invalidate {
            if let Some(view) = cache.get_mut(&key) {
                view.is_valid = false;
                debug!(cache_key = %key, "Cache invalidated for task");
            }
        }
    }

    pub fn remove_invalid_entries(&self) -> usize {
        let mut cache = self.materialized_cache.write();
        let initial_len = cache.len();
        cache.retain(|_, v| v.is_valid);
        let removed = initial_len - cache.len();
        if removed > 0 {
            self.rebuild_reverse_index(&cache);
        }
        removed
    }

    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.materialized_cache.read();
        let total = cache.len();
        let valid = cache.values().filter(|v| v.is_valid).count();
        let total_size: usize = cache.values().map(|v| v.result_json.len()).sum();
        CacheStats {
            total_views: total,
            valid_views: valid,
            invalid_views: total - valid,
            total_size_bytes: total_size,
        }
    }

    async fn vector_enhanced_search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<String>, CoreError> {
        if let Some(ref vs) = self.vector_store {
            match vs.search(query, limit as u64).await {
                Ok(entries) => {
                    let iris: Vec<String> = entries.iter().map(|e| e.iri.clone()).collect();
                    debug!(
                        query_len = query.len(),
                        results = iris.len(),
                        "Vector search completed"
                    );
                    return Ok(iris);
                }
                Err(e) => {
                    warn!("Vector search failed: {}, falling back to empty", e);
                }
            }
        }
        Ok(Vec::new())
    }

    fn project_from_iris(
        &self,
        iris: &[String],
        include_properties: &[String],
        max_nodes: usize,
    ) -> Result<Vec<serde_json::Value>, CoreError> {
        let mut artifacts = Vec::new();

        for node_iri in iris.iter().take(max_nodes) {
            if let Some(node) = self.blackboard.read_node(node_iri)? {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&node.json_ld) {
                    let mut artifact = serde_json::Map::new();
                    for prop in include_properties {
                        if let Some(value) = parsed.get(prop) {
                            artifact.insert(prop.clone(), value.clone());
                        }
                    }
                    if !artifact.is_empty() {
                        artifacts.push(serde_json::Value::Object(artifact));
                    }
                }
            }
        }

        Ok(artifacts)
    }

    pub async fn semantic_project(
        &self,
        query: &str,
        frame_name: &str,
        limit: usize,
    ) -> Result<String, CoreError> {
        let frame = self
            .frames
            .get(frame_name)
            .ok_or_else(|| CoreError::FrameNotFound {
                name: frame_name.to_string(),
            })?;

        let vector_iris = self.vector_enhanced_search(query, limit).await?;

        let artifacts = if !vector_iris.is_empty() {
            self.project_from_iris(&vector_iris, &frame.include_properties, limit)?
        } else {
            Vec::new()
        };

        let mut projection = serde_json::Map::new();
        projection.insert(
            "@context".to_string(),
            (*JsonLdContext::context_value()).clone(),
        );
        projection.insert(
            "query".to_string(),
            serde_json::Value::String(query.to_string()),
        );
        projection.insert(
            "frame".to_string(),
            serde_json::Value::String(frame_name.to_string()),
        );
        projection.insert("artifacts".to_string(), serde_json::Value::Array(artifacts));

        serde_json::to_string(&serde_json::Value::Object(projection)).map_err(|e| {
            CoreError::Internal {
                message: e.to_string(),
            }
        })
    }

    #[instrument(skip(self, frame))]
    pub async fn project_with_frame(
        &self,
        task_iri: &str,
        frame: &FrameTemplate,
    ) -> Result<Value, CoreError> {
        debug!(task_iri = %task_iri, "Executing frame-driven projection");

        let node_iris = self.blackboard.get_task_nodes(task_iri);
        let mut artifacts = Vec::new();
        let max_nodes = 50;

        for node_iri in node_iris.iter().take(max_nodes) {
            if let Some(node) = self.blackboard.read_node(node_iri)? {
                if let Ok(parsed) = serde_json::from_str::<Value>(&node.json_ld) {
                    let framed = apply_frame(&parsed, frame);
                    artifacts.push(framed);
                }
            }
        }

        let mut projection = serde_json::Map::new();
        projection.insert("@context".to_string(), frame.context.clone());
        projection.insert("task_iri".to_string(), Value::String(task_iri.to_string()));
        projection.insert("artifacts".to_string(), Value::Array(artifacts));

        Ok(Value::Object(projection))
    }

    #[instrument(skip(self))]
    pub async fn project_with_budget(
        &self,
        task_iri: &str,
        budget: usize,
    ) -> Result<Value, CoreError> {
        debug!(task_iri = %task_iri, budget = budget, "Executing budget-controlled projection");

        let node_iris = self.blackboard.get_task_nodes(task_iri);
        let mut artifacts = Vec::new();
        let mut current_tokens = 0;

        let default_frame = FrameTemplate::new(serde_json::json!({
            "agent": "https://agent-harness.os/agent#"
        }))
        .with_max_depth(3);

        for node_iri in node_iris {
            if let Some(node) = self.blackboard.read_node(&node_iri)? {
                if let Ok(parsed) = serde_json::from_str::<Value>(&node.json_ld) {
                    let estimated = estimate_tokens(&parsed);

                    if current_tokens + estimated > budget {
                        let remaining_budget = budget.saturating_sub(current_tokens);
                        if remaining_budget > 10 {
                            let fitted = fit_to_budget(&parsed, remaining_budget, &default_frame);
                            artifacts.push(fitted);
                        }
                        break;
                    }

                    let framed = apply_frame(&parsed, &default_frame);
                    current_tokens += estimate_tokens(&framed);
                    artifacts.push(framed);
                }
            }
        }

        let mut projection = serde_json::Map::new();
        projection.insert(
            "@context".to_string(),
            serde_json::json!({
                "agent": "https://agent-harness.os/agent#"
            }),
        );
        projection.insert("task_iri".to_string(), Value::String(task_iri.to_string()));
        projection.insert("artifacts".to_string(), Value::Array(artifacts));
        projection.insert("token_budget".to_string(), Value::Number(budget.into()));
        projection.insert(
            "estimated_tokens".to_string(),
            Value::Number(current_tokens.into()),
        );

        Ok(Value::Object(projection))
    }

    pub fn with_frame_templates(mut self, templates: HashMap<String, FrameTemplate>) -> Self {
        for (name, jsonld_frame) in templates {
            if let Some(projection_frame) = self.frames.get_mut(&name) {
                projection_frame.jsonld_frame = Some(jsonld_frame);
            }
        }
        self
    }

    /// Read a single node from L2 blackboard by node IRI (L3 projection entry point)
    /// Used by agent tool read_agent_output — replaces direct L0 access
    pub fn read_node(&self, node_iri: &str) -> Result<Option<serde_json::Value>, CoreError> {
        match self.blackboard.read_node(node_iri)? {
            Some(node) => {
                let parsed: serde_json::Value =
                    serde_json::from_str(&node.json_ld).map_err(|e| CoreError::Internal {
                        message: format!("Failed to parse L2 node JSON: {}", e),
                    })?;
                Ok(Some(parsed))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_projection() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let engine = ProjectionEngine::new(blackboard.clone(), 500);

        let config = crate::CoreConfig::default();
        let json_ld = r#"{"@id":"iri://task_1/node_1","@type":"Artifact","status":"created"}"#;
        blackboard
            .write_node("iri://task_1/node_1", json_ld, &config)
            .unwrap();

        let result = engine
            .project("iri://task_1", "reference_only", HashMap::new())
            .await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_frame_templates() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let engine = ProjectionEngine::new(blackboard, 1024);

        assert!(engine.get_frame("summary_only").is_some());
        assert!(engine.get_frame("pa_init").is_some());
        assert!(engine.get_frame("da_input").is_some());
        assert!(engine.get_frame("ca_review").is_some());
        assert!(engine.get_frame("aa_decision").is_some());
        assert!(engine.get_frame("health_check").is_some());
        assert!(engine.get_frame("error_analysis").is_some());
        assert!(engine.get_frame("reference_only").is_some());
        assert!(engine.get_frame("5w2h_summary").is_some());
    }

    #[test]
    fn test_jsonld_frame_integration() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let engine = ProjectionEngine::new(blackboard, 1024);

        let summary_frame = engine.get_frame("summary_only").unwrap();
        assert!(summary_frame.jsonld_frame.is_some());

        let jsonld_frame = summary_frame.jsonld_frame.as_ref().unwrap();
        assert!(jsonld_frame.max_depth.is_some());
        assert!(!jsonld_frame.include_properties.is_empty());
    }

    #[test]
    fn test_truncate_projection() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let engine = ProjectionEngine::new(blackboard, 100);

        let large_json = serde_json::json!({
            "@context": "test",
            "task_iri": "iri://task/1",
            "frame": "test",
            "artifacts": [
                {"@id": "a1", "data": "x".repeat(50)},
                {"@id": "a2", "data": "y".repeat(50)},
                {"@id": "a3", "data": "z".repeat(50)},
            ]
        })
        .to_string();

        let result = engine.truncate_projection(large_json, 100).unwrap();
        assert!(
            result.len() <= 100,
            "Truncated result should fit within max_size"
        );
    }

    #[tokio::test]
    async fn test_project_with_frame() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let engine = ProjectionEngine::new(blackboard.clone(), 1024);

        let config = crate::CoreConfig::default();
        let json_ld = r#"{
            "@id": "iri://task/1/node/1",
            "@type": "TaskNode",
            "summary": "Test task",
            "description": "A longer description",
            "status": "running",
            "nested": {
                "@id": "iri://task/1/node/2",
                "value": "nested value"
            }
        }"#;

        let write_result = blackboard.write_node("iri://task/1/node/1", json_ld, &config);
        assert!(
            write_result.is_ok(),
            "Failed to write node: {:?}",
            write_result.err()
        );

        let task_nodes = blackboard.get_task_nodes("iri://task/1");
        assert!(
            !task_nodes.is_empty(),
            "No task nodes found for iri://task/1"
        );

        let frame = FrameTemplate::new(serde_json::json!({
            "task": "https://agent-harness.os/task#"
        }))
        .with_include_properties(vec!["summary".to_string(), "status".to_string()])
        .with_max_depth(2);

        let result = engine.project_with_frame("iri://task/1", &frame).await;
        assert!(result.is_ok());

        let projection = result.unwrap();
        assert!(projection.is_object());
        let obj = projection.as_object().unwrap();
        assert!(obj.contains_key("artifacts"));

        let artifacts = obj.get("artifacts").unwrap().as_array().unwrap();
        assert!(!artifacts.is_empty(), "Artifacts should not be empty");

        let first_artifact = artifacts[0].as_object().unwrap();
        assert!(first_artifact.contains_key("summary"));
        assert!(first_artifact.contains_key("status"));
    }

    #[tokio::test]
    async fn test_project_with_budget() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let engine = ProjectionEngine::new(blackboard.clone(), 1024);

        let config = crate::CoreConfig::default();
        for i in 1..=5 {
            let json_ld = format!(
                r#"{{
                "@id": "iri://task_1/node_{}",
                "@type": "TaskNode",
                "summary": "Task node {}",
                "description": "A longer description for node {}"
            }}"#,
                i, i, i
            );
            blackboard
                .write_node(&format!("iri://task_1/node_{}", i), &json_ld, &config)
                .unwrap();
        }

        let budget = 100;
        let result = engine.project_with_budget("iri://task_1", budget).await;
        assert!(result.is_ok());

        let projection = result.unwrap();
        assert!(projection.is_object());
        let obj = projection.as_object().unwrap();

        assert!(obj.contains_key("token_budget"));
        assert!(obj.contains_key("estimated_tokens"));
        assert!(obj.contains_key("artifacts"));

        let estimated_tokens = obj.get("estimated_tokens").unwrap().as_u64().unwrap() as usize;
        assert!(
            estimated_tokens <= budget,
            "Estimated tokens should be within budget"
        );
    }

    #[test]
    fn test_with_frame_templates() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let engine = ProjectionEngine::new(blackboard, 1024);

        let mut templates = HashMap::new();
        templates.insert(
            "summary_only".to_string(),
            FrameTemplate::new(serde_json::json!({
                "custom": "https://example.org/custom#"
            }))
            .with_max_depth(5),
        );

        let updated_engine = engine.with_frame_templates(templates);

        let frame = updated_engine.get_frame("summary_only").unwrap();
        assert!(frame.jsonld_frame.is_some());
        let jsonld_frame = frame.jsonld_frame.as_ref().unwrap();
        assert_eq!(jsonld_frame.max_depth, Some(5));
    }

    #[test]
    fn test_predefined_frames_have_jsonld() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let engine = ProjectionEngine::new(blackboard, 1024);

        let frames_with_jsonld = vec![
            "summary_only",
            "pa_init",
            "da_input",
            "ca_review",
            "aa_decision",
            "reference_only",
        ];

        for frame_name in frames_with_jsonld {
            let frame = engine.get_frame(frame_name).unwrap();
            assert!(
                frame.jsonld_frame.is_some(),
                "Frame {} should have jsonld_frame",
                frame_name
            );
        }
    }

    #[test]
    fn test_workspace_frames_registered() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let engine = ProjectionEngine::new(blackboard, 2048);

        let frame = engine.get_frame("workspace_overview").unwrap();
        assert_eq!(frame.name, "workspace_overview");
        assert_eq!(frame.target_role, "SA");
        assert!(frame
            .include_properties
            .contains(&"ws:filePath".to_string()));
        assert!(frame.include_properties.contains(&"ws:state".to_string()));
        assert!(frame.sparql_template.is_some());

        let frame = engine.get_frame("workspace_stale_files").unwrap();
        assert_eq!(frame.name, "workspace_stale_files");
        assert_eq!(frame.target_role, "DA");
        assert!(frame
            .include_properties
            .contains(&"ws:filePath".to_string()));
        assert!(frame.sparql_template.is_some());

        let frame = engine.get_frame("workspace_file_detail").unwrap();
        assert_eq!(frame.name, "workspace_file_detail");
        assert!(frame.params.contains(&"target_path".to_string()));
    }

    #[test]
    fn test_workspace_frames_max_nodes() {
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let engine = ProjectionEngine::new(blackboard, 2048);

        let overview = engine.get_frame("workspace_overview").unwrap();
        assert!(
            overview.max_nodes >= 100,
            "workspace_overview should handle at least 100 files"
        );

        let stale = engine.get_frame("workspace_stale_files").unwrap();
        assert!(
            stale.max_nodes >= 50,
            "workspace_stale_files should handle at least 50 files"
        );

        let detail = engine.get_frame("workspace_file_detail").unwrap();
        assert!(
            detail.max_nodes >= 1,
            "workspace_file_detail should handle at least 1 file"
        );
    }
}
