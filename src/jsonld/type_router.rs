//! JSON-LD type routing module
//!
//! Polymorphic discovery and routing decisions based on @type:
//! - Determine applicable SPARQL projection templates
//! - Trigger corresponding events
//! - Apply SA monitoring rules

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TypeRoute {
    pub node_type: String,
    pub projection_templates: Vec<String>,
    pub events: Vec<String>,
    pub sa_rules: Vec<String>,
}

impl TypeRoute {
    pub fn new(node_type: String) -> Self {
        Self {
            node_type,
            projection_templates: vec!["summary_only".to_string()],
            events: vec![],
            sa_rules: vec![],
        }
    }

    pub fn with_projection_templates(mut self, templates: Vec<String>) -> Self {
        self.projection_templates = templates;
        self
    }

    pub fn with_events(mut self, events: Vec<String>) -> Self {
        self.events = events;
        self
    }

    pub fn with_sa_rules(mut self, rules: Vec<String>) -> Self {
        self.sa_rules = rules;
        self
    }

    pub fn add_projection_template(&mut self, template: String) {
        if !self.projection_templates.contains(&template) {
            self.projection_templates.push(template);
        }
    }

    pub fn add_event(&mut self, event: String) {
        if !self.events.contains(&event) {
            self.events.push(event);
        }
    }

    pub fn add_sa_rule(&mut self, rule: String) {
        if !self.sa_rules.contains(&rule) {
            self.sa_rules.push(rule);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeRouter {
    routes: HashMap<String, TypeRoute>,
}

impl Default for TypeRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeRouter {
    pub fn new() -> Self {
        let mut router = Self {
            routes: HashMap::new(),
        };
        router.load_default_routes();
        router
    }

    fn load_default_routes(&mut self) {
        self.register_type(
            TypeRoute::new("PlanNode".to_string())
                .with_projection_templates(vec![
                    "summary_only".to_string(),
                    "pa_init".to_string(),
                    "plan_detail".to_string(),
                ])
                .with_events(vec![
                    "PLAN_CREATED".to_string(),
                    "PLAN_UPDATED".to_string(),
                    "PLAN_COMPLETED".to_string(),
                ])
                .with_sa_rules(vec![
                    "check_plan_progress".to_string(),
                    "validate_plan_completeness".to_string(),
                ]),
        );

        self.register_type(
            TypeRoute::new("CodeArtifact".to_string())
                .with_projection_templates(vec![
                    "summary_only".to_string(),
                    "da_input".to_string(),
                    "code_review_input".to_string(),
                ])
                .with_events(vec![
                    "ARTIFACT_CREATED".to_string(),
                    "ARTIFACT_UPDATED".to_string(),
                ])
                .with_sa_rules(vec![
                    "check_code_quality".to_string(),
                    "track_dependencies".to_string(),
                ]),
        );

        self.register_type(
            TypeRoute::new("ReviewResult".to_string())
                .with_projection_templates(vec![
                    "summary_only".to_string(),
                    "ca_review".to_string(),
                ])
                .with_events(vec![
                    "REVIEW_CREATED".to_string(),
                    "REVIEW_FAILED".to_string(),
                ])
                .with_sa_rules(vec![
                    "check_review_verdict".to_string(),
                    "escalate_critical".to_string(),
                ]),
        );

        self.register_type(
            TypeRoute::new("DecisionNode".to_string())
                .with_projection_templates(vec![
                    "summary_only".to_string(),
                    "aa_decision".to_string(),
                ])
                .with_events(vec![
                    "DECISION_MADE".to_string(),
                    "LOOP_DETECTED".to_string(),
                ])
                .with_sa_rules(vec![
                    "check_loop_count".to_string(),
                    "evaluate_efficiency".to_string(),
                ]),
        );

        self.register_type(
            TypeRoute::new("Experience".to_string())
                .with_projection_templates(vec![
                    "summary_only".to_string(),
                    "experience_detail".to_string(),
                ])
                .with_events(vec!["EXPERIENCE_CREATED".to_string()])
                .with_sa_rules(vec!["index_for_retrieval".to_string()]),
        );

        self.register_type(
            TypeRoute::new("AdvisoryNode".to_string())
                .with_projection_templates(vec![
                    "summary_only".to_string(),
                    "advisory_detail".to_string(),
                ])
                .with_events(vec!["ADVISORY_CREATED".to_string()])
                .with_sa_rules(vec!["inject_to_target".to_string()]),
        );

        self.register_type(
            TypeRoute::new("task:5W2H".to_string())
                .with_projection_templates(vec![
                    "5w2h_summary".to_string(),
                    "summary_only".to_string(),
                ])
                .with_events(vec!["5W2H_CREATED".to_string(), "5W2H_UPDATED".to_string()])
                .with_sa_rules(vec![
                    "check_deadline".to_string(),
                    "check_budget".to_string(),
                ]),
        );
    }

    pub fn register_type(&mut self, route: TypeRoute) {
        self.routes.insert(route.node_type.clone(), route);
    }

    pub fn get_route(&self, node_type: &str) -> Option<&TypeRoute> {
        self.routes.get(node_type)
    }

    pub fn get_projection_templates(&self, node_type: &str) -> Vec<String> {
        self.routes
            .get(node_type)
            .map(|r| r.projection_templates.clone())
            .unwrap_or_else(|| vec!["summary_only".to_string()])
    }

    pub fn get_events(&self, node_type: &str) -> Vec<String> {
        self.routes
            .get(node_type)
            .map(|r| r.events.clone())
            .unwrap_or_default()
    }

    pub fn get_sa_rules(&self, node_type: &str) -> Vec<String> {
        self.routes
            .get(node_type)
            .map(|r| r.sa_rules.clone())
            .unwrap_or_default()
    }

    pub fn find_by_capability(&self, capability: &str) -> Vec<String> {
        let capability_lower = capability.to_lowercase();

        self.routes
            .values()
            .filter(|route| {
                route
                    .projection_templates
                    .iter()
                    .any(|t| t.to_lowercase().contains(&capability_lower))
                    || route
                        .events
                        .iter()
                        .any(|e| e.to_lowercase().contains(&capability_lower))
                    || route
                        .sa_rules
                        .iter()
                        .any(|r| r.to_lowercase().contains(&capability_lower))
            })
            .map(|route| route.node_type.clone())
            .collect()
    }

    pub fn match_types(&self, node_types: &[String]) -> Vec<TypeRoute> {
        node_types
            .iter()
            .filter_map(|t| self.routes.get(t).cloned())
            .collect()
    }

    pub fn merge_routes(&self, node_types: &[String]) -> TypeRoute {
        let matched_routes = self.match_types(node_types);

        if matched_routes.is_empty() {
            return TypeRoute::new("Unknown".to_string());
        }

        if matched_routes.len() == 1 {
            return matched_routes
                .into_iter()
                .next()
                .expect("confirmed len == 1 above");
        }

        let first_route = matched_routes.first().expect("reached only when len > 0");
        let mut merged = TypeRoute::new(first_route.node_type.clone());

        let mut templates_set: HashSet<String> = HashSet::new();
        let mut events_set: HashSet<String> = HashSet::new();
        let mut rules_set: HashSet<String> = HashSet::new();

        for route in matched_routes {
            for template in route.projection_templates {
                templates_set.insert(template);
            }
            for event in route.events {
                events_set.insert(event);
            }
            for rule in route.sa_rules {
                rules_set.insert(rule);
            }
        }

        merged.projection_templates = templates_set.into_iter().collect();
        merged.events = events_set.into_iter().collect();
        merged.sa_rules = rules_set.into_iter().collect();

        merged
    }

    pub fn list_types(&self) -> Vec<String> {
        self.routes.keys().cloned().collect()
    }

    pub fn has_type(&self, node_type: &str) -> bool {
        self.routes.contains_key(node_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_route_new() {
        let route = TypeRoute::new("TestNode".to_string());
        assert_eq!(route.node_type, "TestNode");
        assert_eq!(route.projection_templates, vec!["summary_only"]);
        assert!(route.events.is_empty());
        assert!(route.sa_rules.is_empty());
    }

    #[test]
    fn test_type_route_with_methods() {
        let route = TypeRoute::new("TestNode".to_string())
            .with_projection_templates(vec!["template1".to_string(), "template2".to_string()])
            .with_events(vec!["EVENT1".to_string(), "EVENT2".to_string()])
            .with_sa_rules(vec!["rule1".to_string()]);

        assert_eq!(route.projection_templates.len(), 2);
        assert_eq!(route.events.len(), 2);
        assert_eq!(route.sa_rules.len(), 1);
    }

    #[test]
    fn test_type_route_add_methods() {
        let mut route = TypeRoute::new("TestNode".to_string());

        route.add_projection_template("template1".to_string());
        route.add_projection_template("template1".to_string());
        route.add_event("EVENT1".to_string());
        route.add_sa_rule("rule1".to_string());

        assert_eq!(route.projection_templates.len(), 2);
        assert_eq!(route.events.len(), 1);
        assert_eq!(route.sa_rules.len(), 1);
    }

    #[test]
    fn test_type_router_new() {
        let router = TypeRouter::new();
        assert!(router.has_type("PlanNode"));
        assert!(router.has_type("CodeArtifact"));
        assert!(router.has_type("ReviewResult"));
        assert!(router.has_type("DecisionNode"));
        assert!(router.has_type("Experience"));
        assert!(router.has_type("AdvisoryNode"));
    }

    #[test]
    fn test_get_projection_templates() {
        let router = TypeRouter::new();

        let templates = router.get_projection_templates("PlanNode");
        assert!(templates.contains(&"summary_only".to_string()));
        assert!(templates.contains(&"pa_init".to_string()));
        assert!(templates.contains(&"plan_detail".to_string()));

        let templates_unknown = router.get_projection_templates("UnknownType");
        assert_eq!(templates_unknown, vec!["summary_only"]);
    }

    #[test]
    fn test_get_events() {
        let router = TypeRouter::new();

        let events = router.get_events("PlanNode");
        assert!(events.contains(&"PLAN_CREATED".to_string()));
        assert!(events.contains(&"PLAN_UPDATED".to_string()));
        assert!(events.contains(&"PLAN_COMPLETED".to_string()));

        let events_unknown = router.get_events("UnknownType");
        assert!(events_unknown.is_empty());
    }

    #[test]
    fn test_get_sa_rules() {
        let router = TypeRouter::new();

        let rules = router.get_sa_rules("PlanNode");
        assert!(rules.contains(&"check_plan_progress".to_string()));
        assert!(rules.contains(&"validate_plan_completeness".to_string()));

        let rules_unknown = router.get_sa_rules("UnknownType");
        assert!(rules_unknown.is_empty());
    }

    #[test]
    fn test_register_type() {
        let mut router = TypeRouter::new();

        let custom_route = TypeRoute::new("CustomNode".to_string())
            .with_projection_templates(vec!["custom_template".to_string()])
            .with_events(vec!["CUSTOM_EVENT".to_string()])
            .with_sa_rules(vec!["custom_rule".to_string()]);

        router.register_type(custom_route);

        assert!(router.has_type("CustomNode"));
        assert_eq!(
            router.get_projection_templates("CustomNode"),
            vec!["custom_template"]
        );
        assert_eq!(router.get_events("CustomNode"), vec!["CUSTOM_EVENT"]);
        assert_eq!(router.get_sa_rules("CustomNode"), vec!["custom_rule"]);
    }

    #[test]
    fn test_find_by_capability() {
        let router = TypeRouter::new();

        let found = router.find_by_capability("plan");
        assert!(found.contains(&"PlanNode".to_string()));

        let found_review = router.find_by_capability("review");
        assert!(found_review.contains(&"ReviewResult".to_string()));
        assert!(found_review.contains(&"CodeArtifact".to_string()));

        let found_none = router.find_by_capability("nonexistent");
        assert!(found_none.is_empty());
    }

    #[test]
    fn test_match_types() {
        let router = TypeRouter::new();

        let matched = router.match_types(&["PlanNode".to_string(), "CodeArtifact".to_string()]);

        assert_eq!(matched.len(), 2);
        assert!(matched.iter().any(|r| r.node_type == "PlanNode"));
        assert!(matched.iter().any(|r| r.node_type == "CodeArtifact"));

        let matched_unknown = router.match_types(&["UnknownType".to_string()]);
        assert!(matched_unknown.is_empty());
    }

    #[test]
    fn test_merge_routes() {
        let router = TypeRouter::new();

        let merged = router.merge_routes(&["PlanNode".to_string(), "CodeArtifact".to_string()]);

        assert!(merged.projection_templates.contains(&"pa_init".to_string()));
        assert!(merged
            .projection_templates
            .contains(&"da_input".to_string()));
        assert!(merged
            .projection_templates
            .contains(&"summary_only".to_string()));

        assert!(merged.events.contains(&"PLAN_CREATED".to_string()));
        assert!(merged.events.contains(&"ARTIFACT_CREATED".to_string()));

        assert!(merged.sa_rules.contains(&"check_plan_progress".to_string()));
        assert!(merged.sa_rules.contains(&"check_code_quality".to_string()));
    }

    #[test]
    fn test_merge_routes_single_type() {
        let router = TypeRouter::new();

        let merged = router.merge_routes(&["PlanNode".to_string()]);

        assert_eq!(merged.node_type, "PlanNode");
        assert_eq!(merged.projection_templates.len(), 3);
        assert_eq!(merged.events.len(), 3);
        assert_eq!(merged.sa_rules.len(), 2);
    }

    #[test]
    fn test_merge_routes_empty() {
        let router = TypeRouter::new();

        let merged = router.merge_routes(&[]);

        assert_eq!(merged.node_type, "Unknown");
        assert_eq!(merged.projection_templates, vec!["summary_only"]);
    }

    #[test]
    fn test_list_types() {
        let router = TypeRouter::new();

        let types = router.list_types();
        assert!(types.contains(&"PlanNode".to_string()));
        assert!(types.contains(&"CodeArtifact".to_string()));
        assert!(types.contains(&"ReviewResult".to_string()));
        assert!(types.contains(&"DecisionNode".to_string()));
        assert!(types.contains(&"Experience".to_string()));
        assert!(types.contains(&"AdvisoryNode".to_string()));
        assert_eq!(types.len(), 7);
    }

    #[test]
    fn test_all_default_types() {
        let router = TypeRouter::new();

        let code_artifact = router.get_route("CodeArtifact").unwrap();
        assert_eq!(code_artifact.projection_templates.len(), 3);
        assert_eq!(code_artifact.events.len(), 2);
        assert_eq!(code_artifact.sa_rules.len(), 2);

        let review_result = router.get_route("ReviewResult").unwrap();
        assert_eq!(review_result.projection_templates.len(), 2);
        assert_eq!(review_result.events.len(), 2);
        assert_eq!(review_result.sa_rules.len(), 2);

        let decision_node = router.get_route("DecisionNode").unwrap();
        assert_eq!(decision_node.projection_templates.len(), 2);
        assert_eq!(decision_node.events.len(), 2);
        assert_eq!(decision_node.sa_rules.len(), 2);

        let experience = router.get_route("Experience").unwrap();
        assert_eq!(experience.projection_templates.len(), 2);
        assert_eq!(experience.events.len(), 1);
        assert_eq!(experience.sa_rules.len(), 1);

        let advisory = router.get_route("AdvisoryNode").unwrap();
        assert_eq!(advisory.projection_templates.len(), 2);
        assert_eq!(advisory.events.len(), 1);
        assert_eq!(advisory.sa_rules.len(), 1);
    }

    #[test]
    fn test_type_route_serialization() {
        let route = TypeRoute::new("TestNode".to_string())
            .with_projection_templates(vec!["template1".to_string()])
            .with_events(vec!["EVENT1".to_string()])
            .with_sa_rules(vec!["rule1".to_string()]);

        let json = serde_json::to_string(&route).unwrap();
        assert!(json.contains("\"node_type\":\"TestNode\""));
        assert!(json.contains("\"projection_templates\":[\"template1\"]"));
        assert!(json.contains("\"events\":[\"EVENT1\"]"));
        assert!(json.contains("\"sa_rules\":[\"rule1\"]"));

        let deserialized: TypeRoute = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.node_type, "TestNode");
        assert_eq!(deserialized.projection_templates, vec!["template1"]);
        assert_eq!(deserialized.events, vec!["EVENT1"]);
        assert_eq!(deserialized.sa_rules, vec!["rule1"]);
    }
}
