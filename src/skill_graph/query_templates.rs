use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::index::PreAggregatedIndex;
use crate::skill_graph::types::*;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryTemplateId {
    MocScan,
    SkillByTag,
    SkillByStack,
    SkillByRole,
    SkillSummary,
    SkillSteps,
    StepDetail,
    Validation,
    RequiresDirect,
    RequiresTransitive,
}

impl QueryTemplateId {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "q:moc-scan" => Some(Self::MocScan),
            "q:skill-by-tag" => Some(Self::SkillByTag),
            "q:skill-by-stack" => Some(Self::SkillByStack),
            "q:skill-by-role" => Some(Self::SkillByRole),
            "q:skill-summary" => Some(Self::SkillSummary),
            "q:skill-steps" => Some(Self::SkillSteps),
            "q:step-detail" => Some(Self::StepDetail),
            "q:validation" => Some(Self::Validation),
            "q:requires-direct" => Some(Self::RequiresDirect),
            "q:requires-transitive" => Some(Self::RequiresTransitive),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MocScan => "q:moc-scan",
            Self::SkillByTag => "q:skill-by-tag",
            Self::SkillByStack => "q:skill-by-stack",
            Self::SkillByRole => "q:skill-by-role",
            Self::SkillSummary => "q:skill-summary",
            Self::SkillSteps => "q:skill-steps",
            Self::StepDetail => "q:step-detail",
            Self::Validation => "q:validation",
            Self::RequiresDirect => "q:requires-direct",
            Self::RequiresTransitive => "q:requires-transitive",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::MocScan => "Scan MOC index for domain entry points",
            Self::SkillByTag => "Query skill list by tag",
            Self::SkillByStack => "Query skill list by tech stack",
            Self::SkillByRole => "Query skill list by role",
            Self::SkillSummary => "Query skill summary by @id (5W2H)",
            Self::SkillSteps => "Fetch step list by @id",
            Self::StepDetail => "Fetch step details by @id",
            Self::Validation => "Fetch validation conditions",
            Self::RequiresDirect => "Query direct dependencies",
            Self::RequiresTransitive => "Query transitive dependencies (pre-computed)",
        }
    }

    pub fn target_layer(&self) -> &'static str {
        match self {
            Self::MocScan | Self::SkillByTag | Self::SkillByStack | Self::SkillByRole => {
                "L2-pre-aggregated"
            }
            Self::SkillSummary | Self::RequiresDirect | Self::RequiresTransitive => "L2-indexed",
            Self::SkillSteps | Self::StepDetail | Self::Validation => "L3-detailed",
        }
    }

    pub fn all_templates() -> &'static [QueryTemplateId] {
        &[
            Self::MocScan,
            Self::SkillByTag,
            Self::SkillByStack,
            Self::SkillByRole,
            Self::SkillSummary,
            Self::SkillSteps,
            Self::StepDetail,
            Self::Validation,
            Self::RequiresDirect,
            Self::RequiresTransitive,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    pub params: HashMap<String, Value>,
}

impl QueryParams {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    pub fn with(mut self, key: &str, value: Value) -> Self {
        self.params.insert(key.to_string(), value);
        self
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.params.get(key).and_then(|v| v.as_str())
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.params
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
    }

    pub fn get_str_vec(&self, key: &str) -> Vec<String> {
        self.params
            .get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for QueryParams {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub template_id: String,
    pub data: Value,
    pub from_layer: String,
    pub result_count: usize,
    pub cached: bool,
}

pub struct QueryEngine {
    graph_store: Arc<SkillGraphStore>,
    index: Arc<PreAggregatedIndex>,
}

impl QueryEngine {
    pub fn new(graph_store: Arc<SkillGraphStore>, index: Arc<PreAggregatedIndex>) -> Self {
        Self { graph_store, index }
    }

    pub fn execute(&self, template_id: QueryTemplateId, params: QueryParams) -> QueryResult {
        match template_id {
            QueryTemplateId::MocScan => self.query_moc_scan(&params),
            QueryTemplateId::SkillByTag => self.query_skill_by_tag(&params),
            QueryTemplateId::SkillByStack => self.query_skill_by_stack(&params),
            QueryTemplateId::SkillByRole => self.query_skill_by_role(&params),
            QueryTemplateId::SkillSummary => self.query_skill_summary(&params),
            QueryTemplateId::SkillSteps => self.query_skill_steps(&params),
            QueryTemplateId::StepDetail => self.query_step_detail(&params),
            QueryTemplateId::Validation => self.query_validation(&params),
            QueryTemplateId::RequiresDirect => self.query_requires_direct(&params),
            QueryTemplateId::RequiresTransitive => self.query_requires_transitive(&params),
        }
    }

    fn query_moc_scan(&self, params: &QueryParams) -> QueryResult {
        let keyword = params.get_str("keyword").unwrap_or("");
        let mocs = self.graph_store.list_mocs();

        let filtered: Vec<Value> = if keyword.is_empty() {
            mocs.iter()
                .map(|m| {
                    serde_json::json!({
                        "@id": m.moc_iri,
                        "name": m.name,
                        "description": m.description,
                        "skill_count": m.entry_points.len()
                    })
                })
                .collect()
        } else {
            mocs.iter()
                .filter(|m| {
                    m.name.to_lowercase().contains(&keyword.to_lowercase())
                        || m.description
                            .to_lowercase()
                            .contains(&keyword.to_lowercase())
                })
                .map(|m| {
                    serde_json::json!({
                        "@id": m.moc_iri,
                        "name": m.name,
                        "skill_count": m.entry_points.len()
                    })
                })
                .collect()
        };

        let count = filtered.len();
        QueryResult {
            template_id: "q:moc-scan".to_string(),
            data: Value::Array(filtered),
            from_layer: "L2-pre-aggregated".to_string(),
            result_count: count,
            cached: false,
        }
    }

    fn query_skill_by_tag(&self, params: &QueryParams) -> QueryResult {
        let tags = params.get_str_vec("tags");
        let min_rate = params.get_f32("min_success_rate");

        let skill_iris = if tags.len() == 1 {
            self.index.find_by_tag(&tags[0])
        } else {
            let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
            self.index.find_by_tags_intersection(&tag_refs)
        };

        let results: Vec<Value> = skill_iris
            .iter()
            .filter_map(|iri| self.index.get_summary(iri))
            .filter(|entry| min_rate.map_or(true, |r| entry.success_rate >= r))
            .map(|entry| {
                serde_json::json!({
                    "@id": entry.skill_iri,
                    "name": entry.name,
                    "what": entry.what,
                    "tags": entry.tags,
                    "success_rate": entry.success_rate
                })
            })
            .collect();

        let count = results.len();
        QueryResult {
            template_id: "q:skill-by-tag".to_string(),
            data: Value::Array(results),
            from_layer: "L2-pre-aggregated".to_string(),
            result_count: count,
            cached: false,
        }
    }

    fn query_skill_by_stack(&self, params: &QueryParams) -> QueryResult {
        let stack = params.get_str("stack").unwrap_or("");
        let skill_iris = self.index.find_by_stack(stack);

        let results: Vec<Value> = skill_iris
            .iter()
            .filter_map(|iri| self.index.get_summary(iri))
            .map(|entry| {
                serde_json::json!({
                    "@id": entry.skill_iri,
                    "name": entry.name,
                    "stack": entry.stack,
                    "success_rate": entry.success_rate
                })
            })
            .collect();

        let count = results.len();
        QueryResult {
            template_id: "q:skill-by-stack".to_string(),
            data: Value::Array(results),
            from_layer: "L2-pre-aggregated".to_string(),
            result_count: count,
            cached: false,
        }
    }

    fn query_skill_by_role(&self, params: &QueryParams) -> QueryResult {
        let role = params.get_str("role").unwrap_or("");
        let skill_iris = self.index.find_by_role(role);

        let results: Vec<Value> = skill_iris
            .iter()
            .filter_map(|iri| self.index.get_summary(iri))
            .map(|entry| {
                serde_json::json!({
                    "@id": entry.skill_iri,
                    "name": entry.name,
                    "role": entry.role,
                    "success_rate": entry.success_rate
                })
            })
            .collect();

        let count = results.len();
        QueryResult {
            template_id: "q:skill-by-role".to_string(),
            data: Value::Array(results),
            from_layer: "L2-pre-aggregated".to_string(),
            result_count: count,
            cached: false,
        }
    }

    fn query_skill_summary(&self, params: &QueryParams) -> QueryResult {
        let skill_iri = params.get_str("skill_iri").unwrap_or("");
        let level = params.get_str("level").unwrap_or("summary");

        let disclosure = match level {
            "moc" => DisclosureLevel::MOCIndex,
            "summary" => DisclosureLevel::Summary5W2H,
            "links" => DisclosureLevel::LinksExpanded,
            _ => DisclosureLevel::Summary5W2H,
        };

        let data = self.graph_store.get_skill_at_level(skill_iri, disclosure);
        let count = if data.is_some() { 1 } else { 0 };

        QueryResult {
            template_id: "q:skill-summary".to_string(),
            data: data.unwrap_or(Value::Null),
            from_layer: "L2-indexed".to_string(),
            result_count: count,
            cached: false,
        }
    }

    fn query_skill_steps(&self, params: &QueryParams) -> QueryResult {
        let skill_iri = params.get_str("skill_iri").unwrap_or("");
        let data = self
            .graph_store
            .get_skill_at_level(skill_iri, DisclosureLevel::SchemaSteps);
        let count = if data.is_some() { 1 } else { 0 };

        QueryResult {
            template_id: "q:skill-steps".to_string(),
            data: data.unwrap_or(Value::Null),
            from_layer: "L3-detailed".to_string(),
            result_count: count,
            cached: false,
        }
    }

    fn query_step_detail(&self, params: &QueryParams) -> QueryResult {
        let skill_iri = params.get_str("skill_iri").unwrap_or("");
        let data = self
            .graph_store
            .get_skill_at_level(skill_iri, DisclosureLevel::FullContent);
        let count = if data.is_some() { 1 } else { 0 };

        QueryResult {
            template_id: "q:step-detail".to_string(),
            data: data.unwrap_or(Value::Null),
            from_layer: "L3-detailed".to_string(),
            result_count: count,
            cached: false,
        }
    }

    fn query_validation(&self, params: &QueryParams) -> QueryResult {
        let skill_iri = params.get_str("skill_iri").unwrap_or("");
        let skill = self.graph_store.get_skill(skill_iri);

        let data = skill.and_then(|s| {
            s.content.and_then(|c| {
                c.validation.map(|v| {
                    serde_json::json!({
                        "method": v.method,
                        "success_condition": v.success_condition
                    })
                })
            })
        });
        let count = if data.is_some() { 1 } else { 0 };

        QueryResult {
            template_id: "q:validation".to_string(),
            data: data.unwrap_or(Value::Null),
            from_layer: "L3-detailed".to_string(),
            result_count: count,
            cached: false,
        }
    }

    fn query_requires_direct(&self, params: &QueryParams) -> QueryResult {
        let skill_iri = params.get_str("skill_iri").unwrap_or("");
        let skill = self.graph_store.get_skill(skill_iri);

        let deps: Vec<Value> = skill
            .map(|s| {
                s.links
                    .iter()
                    .filter(|l| l.link_type == SkillLinkType::Prerequisite)
                    .map(|l| {
                        serde_json::json!({
                            "target": l.target_iri,
                            "strength": format!("{:?}", l.strength),
                            "description": l.description
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let count = deps.len();
        QueryResult {
            template_id: "q:requires-direct".to_string(),
            data: Value::Array(deps),
            from_layer: "L2-indexed".to_string(),
            result_count: count,
            cached: false,
        }
    }

    fn query_requires_transitive(&self, params: &QueryParams) -> QueryResult {
        let skill_iri = params.get_str("skill_iri").unwrap_or("");
        let deps = self.index.get_transitive_deps(skill_iri);

        let results: Vec<Value> = deps
            .iter()
            .map(|dep| serde_json::json!({ "@id": dep }))
            .collect();

        let count = results.len();
        QueryResult {
            template_id: "q:requires-transitive".to_string(),
            data: Value::Array(results),
            from_layer: "L2-pre-computed".to_string(),
            result_count: count,
            cached: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_query_template_id_from_str() {
        assert_eq!(
            QueryTemplateId::from_str("q:moc-scan"),
            Some(QueryTemplateId::MocScan)
        );
        assert_eq!(
            QueryTemplateId::from_str("q:skill-by-tag"),
            Some(QueryTemplateId::SkillByTag)
        );
        assert_eq!(QueryTemplateId::from_str("q:unknown"), None);
    }

    #[test]
    fn test_query_template_id_as_str() {
        assert_eq!(QueryTemplateId::MocScan.as_str(), "q:moc-scan");
        assert_eq!(
            QueryTemplateId::RequiresTransitive.as_str(),
            "q:requires-transitive"
        );
    }

    #[test]
    fn test_query_template_all() {
        let all = QueryTemplateId::all_templates();
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn test_query_params() {
        let params = QueryParams::new()
            .with("skill_iri", Value::String("iri://skills/test".to_string()))
            .with(
                "tags",
                Value::Array(vec![
                    Value::String("auth".to_string()),
                    Value::String("jwt".to_string()),
                ]),
            );

        assert_eq!(params.get_str("skill_iri"), Some("iri://skills/test"));
        assert_eq!(params.get_str_vec("tags"), vec!["auth", "jwt"]);
    }

    #[test]
    fn test_query_result() {
        let result = QueryResult {
            template_id: "q:moc-scan".to_string(),
            data: Value::Array(vec![]),
            from_layer: "L2-pre-aggregated".to_string(),
            result_count: 0,
            cached: false,
        };
        assert_eq!(result.result_count, 0);
        assert!(!result.cached);
    }

    #[test]
    fn test_query_engine_moc_scan() {
        let store = Arc::new(SkillGraphStore::new());
        let index = Arc::new(PreAggregatedIndex::new());
        let engine = QueryEngine::new(store.clone(), index.clone());

        store
            .register_moc(MOCNode {
                moc_iri: "iri://moc/auth".to_string(),
                name: "Authentication & Authorization".to_string(),
                description: "Authentication-related skills".to_string(),
                entry_points: vec!["iri://skills/jwt".to_string()],
                skill_count: 1,
                sub_categories: vec![],
                created_at: Utc::now(),
            })
            .unwrap();

        let result = engine.execute(
            QueryTemplateId::MocScan,
            QueryParams::new().with("keyword", Value::String("auth".to_string())),
        );
        assert_eq!(result.result_count, 1);
    }

    #[test]
    fn test_query_engine_skill_by_tag() {
        let store = Arc::new(SkillGraphStore::new());
        let index = Arc::new(PreAggregatedIndex::new());
        let engine = QueryEngine::new(store.clone(), index.clone());

        let skill = SkillGraphNode::new("iri://skills/jwt", "JWT Auth", "JWT Authentication")
            .with_tag("auth")
            .with_tag("jwt");
        index.index_skill(&skill);

        let result = engine.execute(
            QueryTemplateId::SkillByTag,
            QueryParams::new().with(
                "tags",
                Value::Array(vec![Value::String("auth".to_string())]),
            ),
        );
        assert_eq!(result.result_count, 1);
    }

    #[test]
    fn test_query_engine_skill_summary() {
        let store = Arc::new(SkillGraphStore::new());
        let index = Arc::new(PreAggregatedIndex::new());
        let engine = QueryEngine::new(store.clone(), index.clone());

        let skill = SkillGraphNode::new("iri://skills/jwt", "JWT Auth", "JWT Authentication");
        store.register_skill(skill).unwrap();

        let result = engine.execute(
            QueryTemplateId::SkillSummary,
            QueryParams::new().with("skill_iri", Value::String("iri://skills/jwt".to_string())),
        );
        assert_eq!(result.result_count, 1);
    }

    #[test]
    fn test_query_engine_requires_transitive() {
        let store = Arc::new(SkillGraphStore::new());
        let index = Arc::new(PreAggregatedIndex::new());
        let engine = QueryEngine::new(store.clone(), index.clone());

        let s1 = SkillGraphNode::new("iri://skills/s1", "S1", "S1");
        let mut s2 = SkillGraphNode::new("iri://skills/s2", "S2", "S2");
        s2.add_link(SkillLink {
            link_type: SkillLinkType::Prerequisite,
            target_iri: "iri://skills/s1".to_string(),
            strength: LinkStrength::Required,
            description: "Requires S1".to_string(),
        });

        index.index_skill(&s1);
        index.index_skill(&s2);

        let result = engine.execute(
            QueryTemplateId::RequiresTransitive,
            QueryParams::new().with("skill_iri", Value::String("iri://skills/s2".to_string())),
        );
        assert_eq!(result.result_count, 1);
    }
}
