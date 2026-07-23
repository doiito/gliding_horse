//! Ontology pipeline tools (SHACL, lint, diff, Turtle validation).
//!
//! Gated behind `#[cfg(feature = "ontology")]` – these tools are only
//! available when the OO fusion feature is enabled.
//!
//! Each tool follows the pattern `async fn(Value) -> Result<Value, String>`
//! expected by the ToolExecutor.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::ontology::OntologyPipeline;

// ─── Input structs ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct OntologyValidateTurtleInput {
    pub ttl: String,
}

#[derive(Debug, Deserialize)]
pub struct OntologyLintTurtleInput {
    pub ttl: String,
}

#[derive(Debug, Deserialize)]
pub struct OntologyDiffTurtleInput {
    pub old_ttl: String,
    pub new_ttl: String,
}

#[derive(Debug, Deserialize)]
pub struct OntologyValidateShaclInput {
    pub shapes_ttl: String,
    pub data_ttl: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OntologyReasonInput {
    pub ttl: String,
    pub profile: Option<String>,
    pub materialize: Option<bool>,
}

// ─── Tool implementations ───────────────────────────────────

/// Validate Turtle syntax without loading into a store.
pub async fn execute_ontology_validate_turtle(input: Value) -> Result<Value, String> {
    let params: OntologyValidateTurtleInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    let result = crate::ontology::validate_turtle_content(&params.ttl)
        .map_err(|e| format!("Turtle validation failed: {}", e))?;

    Ok(json!({
        "success": true,
        "result": result,
    }))
}

/// Lint Turtle content (checks labels, comments, domain/range).
pub async fn execute_ontology_lint_turtle(input: Value) -> Result<Value, String> {
    let params: OntologyLintTurtleInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    let result = crate::ontology::lint_turtle_content(&params.ttl)
        .map_err(|e| format!("Turtle lint failed: {}", e))?;

    Ok(json!({
        "success": true,
        "result": result,
    }))
}

/// Diff two Turtle documents and report changes.
pub async fn execute_ontology_diff_turtle(input: Value) -> Result<Value, String> {
    let params: OntologyDiffTurtleInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    let result = crate::ontology::diff_turtle(&params.old_ttl, &params.new_ttl)
        .map_err(|e| format!("Turtle diff failed: {}", e))?;

    Ok(json!({
        "success": true,
        "result": result,
    }))
}

/// Validate data Turtle against SHACL shape definitions.
pub async fn execute_ontology_validate_shacl(input: Value) -> Result<Value, String> {
    let params: OntologyValidateShaclInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    // Create a shared store and load the data
    let shared = crate::ontology::new_shared_store()
        .map_err(|e| format!("Failed to create ontology store: {}", e))?;

    // If data_ttl is provided, load it into the store first
    if let Some(ref data_ttl) = params.data_ttl {
        shared
            .load_turtle(data_ttl, None)
            .map_err(|e| format!("Failed to load data Turtle: {}", e))?;
    }

    let result = shared
        .validate_shacl(&params.shapes_ttl)
        .map_err(|e| format!("SHACL validation failed: {}", e))?;

    Ok(json!({
        "success": true,
        "result": result,
    }))
}

/// Run RDFS/OWL-RL reasoning over Turtle data and return inferred triples.
pub async fn execute_ontology_reason(input: Value) -> Result<Value, String> {
    let params: OntologyReasonInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    let profile = params.profile.as_deref().unwrap_or("owl-rl");
    let materialize = params.materialize.unwrap_or(true);

    let shared = crate::ontology::new_shared_store()
        .map_err(|e| format!("Failed to create ontology store: {}", e))?;

    shared
        .load_turtle(&params.ttl, None)
        .map_err(|e| format!("Failed to load Turtle data: {}", e))?;

    let result = shared
        .reason(profile, materialize)
        .map_err(|e| format!("Reasoning failed: {}", e))?;

    Ok(json!({
        "success": true,
        "profile": profile,
        "materialize": materialize,
        "result": result,
    }))
}

#[cfg(test)]
#[cfg(feature = "ontology")]
mod tests {
    use super::*;

    const VALID_TTL: &str = r#"
        @prefix ex: <http://example.org/> .
        ex:Alice a ex:Person ;
            ex:name "Alice" .
    "#;

    const INVALID_TTL: &str = "this is not turtle @@@";

    #[tokio::test]
    async fn test_validate_turtle_valid() {
        let input = json!({"ttl": VALID_TTL});
        let result = execute_ontology_validate_turtle(input).await.unwrap();
        assert!(result["success"].as_bool().unwrap());
        // validate_string returns a JSON string with "valid" and "triple_count"
        let report: Value = serde_json::from_str(result["result"].as_str().unwrap()).unwrap();
        assert!(report["valid"].as_bool().unwrap());
        assert!(report["triple_count"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_validate_turtle_invalid() {
        let input = json!({"ttl": INVALID_TTL});
        let result = execute_ontology_validate_turtle(input).await.unwrap();
        assert!(result["success"].as_bool().unwrap());
        // OO never returns Err for validation; errors are in the JSON report
        let report: Value = serde_json::from_str(result["result"].as_str().unwrap()).unwrap();
        assert!(!report["valid"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_lint_turtle() {
        let input = json!({"ttl": VALID_TTL});
        let result = execute_ontology_lint_turtle(input).await.unwrap();
        assert!(result["success"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_diff_turtle() {
        let new_ttl = format!("{}\n\nex:Bar a ex:Thing .", VALID_TTL);
        let input = json!({"old_ttl": VALID_TTL, "new_ttl": new_ttl});
        let result = execute_ontology_diff_turtle(input).await.unwrap();
        assert!(result["success"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_validate_shacl() {
        let shapes = r#"
            @prefix sh: <http://www.w3.org/ns/shacl#> .
            @prefix ex: <http://example.org/> .
            ex:PersonShape a sh:NodeShape ;
                sh:targetClass ex:Person .
        "#;
        let input = json!({
            "shapes_ttl": shapes,
            "data_ttl": VALID_TTL,
        });
        let result = execute_ontology_validate_shacl(input).await.unwrap();
        assert!(result["success"].as_bool().unwrap());
    }
}
