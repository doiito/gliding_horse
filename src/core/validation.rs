//! Validation Engine - JSON-LD and Schema validation
//!
//! This module provides validation for JSON-LD documents and schemas.
//! Includes MetaValidator for LLM output "one roundtrip, double harvest" mechanism.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::jsonld::JsonLdContext;
use crate::CoreError;

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub normalized: Option<String>,
}

/// Validation error detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
    pub severity: String,
}

/// Validation engine wrapper
pub struct ValidationEngine {
    validator: JsonLdValidator,
    meta_validator: MetaValidator,
}

impl ValidationEngine {
    pub fn new(max_size: usize) -> Self {
        Self {
            validator: JsonLdValidator::new(max_size, true),
            meta_validator: MetaValidator::new(),
        }
    }

    pub fn validate_json_ld(&self, json_ld: &str) -> Result<(), CoreError> {
        let result = self.validator.validate(json_ld);
        if result.valid {
            Ok(())
        } else {
            Err(CoreError::InvalidJsonLd {
                message: result.errors.join("; "),
            })
        }
    }

    pub fn validate(&self, json_ld: &str) -> ValidationResult {
        self.validator.validate(json_ld)
    }

    pub fn validate_schema(&self, json_ld: &str, schema: &serde_json::Value) -> ValidationResult {
        self.validator.validate_schema(json_ld, schema)
    }

    pub fn validate_and_convert_meta(
        &self,
        content_type: &str,
        meta: &serde_json::Value,
    ) -> Result<serde_json::Value, CoreError> {
        self.meta_validator.validate_and_convert(content_type, meta)
    }
}

impl Default for ValidationEngine {
    fn default() -> Self {
        Self::new(2048)
    }
}

/// JSON-LD Validator
#[derive(Clone)]
pub struct JsonLdValidator {
    max_size: usize,
    strict: bool,
}

impl JsonLdValidator {
    pub fn new(max_size: usize, strict: bool) -> Self {
        Self { max_size, strict }
    }

    pub fn validate(&self, json_ld: &str) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        if json_ld.len() > self.max_size {
            errors.push(format!(
                "Document size {} exceeds maximum {}",
                json_ld.len(),
                self.max_size
            ));
            return ValidationResult {
                valid: false,
                errors,
                warnings,
                normalized: None,
            };
        }

        let value: serde_json::Value = match serde_json::from_str(json_ld) {
            Ok(v) => v,
            Err(e) => {
                errors.push(format!("Invalid JSON: {}", e));
                return ValidationResult {
                    valid: false,
                    errors,
                    warnings,
                    normalized: None,
                };
            }
        };

        if self.strict {
            if value.get("@id").is_none() {
                warnings.push("Document missing @id field".to_string());
            }

            if value.get("@type").is_none() {
                warnings.push("Document missing @type field".to_string());
            }
        }

        if let Some(context) = value.get("@context") {
            if !self.validate_context(context) {
                warnings.push("Invalid or missing @context".to_string());
            }
        }

        let normalized = serde_json::to_string_pretty(&value).ok();

        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
            normalized,
        }
    }

    fn validate_context(&self, context: &serde_json::Value) -> bool {
        match context {
            serde_json::Value::String(s) => !s.is_empty(),
            serde_json::Value::Object(_) => true,
            serde_json::Value::Array(arr) => !arr.is_empty(),
            _ => false,
        }
    }

    pub fn validate_schema(&self, json_ld: &str, schema: &serde_json::Value) -> ValidationResult {
        let base_result = self.validate(json_ld);
        if !base_result.valid {
            return base_result;
        }

        let mut errors = Vec::new();
        let warnings = base_result.warnings;

        let value: serde_json::Value = match serde_json::from_str(json_ld) {
            Ok(v) => v,
            Err(e) => {
                errors.push(format!("JSON parse error: {}", e));
                return ValidationResult {
                    valid: false,
                    errors,
                    warnings,
                    normalized: None,
                };
            }
        };

        let compiled = match jsonschema::JSONSchema::options().compile(schema) {
            Ok(c) => c,
            Err(e) => {
                errors.push(format!("Invalid schema: {}", e));
                return ValidationResult {
                    valid: false,
                    errors,
                    warnings,
                    normalized: None,
                };
            }
        };

        if let Err(validation_errors) = compiled.validate(&value) {
            for error in validation_errors {
                errors.push(format!("{}: {}", error.instance_path, error));
            }
        }

        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
            normalized: base_result.normalized,
        }
    }
}

impl Default for JsonLdValidator {
    fn default() -> Self {
        Self::new(2048, true)
    }
}

/// Signature verifier
#[derive(Clone)]
pub struct SignatureVerifier {
    public_key: Option<Vec<u8>>,
}

impl SignatureVerifier {
    pub fn new() -> Self {
        Self { public_key: None }
    }

    pub fn with_public_key(mut self, key: Vec<u8>) -> Self {
        self.public_key = Some(key);
        self
    }

    pub fn verify(&self, data: &str, signature: &str) -> Result<bool, CoreError> {
        let public_key = self
            .public_key
            .as_ref()
            .ok_or_else(|| CoreError::Internal {
                message: "No public key configured".to_string(),
            })?;

        use base64::Engine;
        let signature_bytes = base64::engine::general_purpose::STANDARD
            .decode(signature)
            .map_err(|e| CoreError::Internal {
                message: format!("Invalid signature encoding: {}", e),
            })?;

        let peer_public_key =
            ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, public_key);

        match peer_public_key.verify(data.as_bytes(), &signature_bytes) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    #[cfg(test)]
    pub fn sign(&self, _data: &str) -> String {
        String::new()
    }
}

impl Default for SignatureVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Meta Validator for LLM output "one roundtrip, double harvest" mechanism.
///
/// Validates metadata from LLM output and converts to JSON-LD format.
///
/// Provides generic schemas applicable to any task type, not just coding tasks.
pub struct MetaValidator {
    schemas: HashMap<String, serde_json::Value>,
}

impl MetaValidator {
    pub fn new() -> Self {
        let mut schemas = HashMap::new();

        // Plan - task plan, applicable to any task type
        schemas.insert("plan".to_string(), serde_json::json!({
            "type": "object",
            "required": ["summary"],
            "properties": {
                "summary": {"type": "string", "maxLength": 200},
                "goal": {"type": "string", "description": "Task goal"},
                "approach": {"type": "string", "description": "Execution method/strategy"},
                "sub_tasks": {"type": "array", "items": {"type": "string"}, "description": "Sub-task list"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                "estimated_complexity": {"type": "string", "enum": ["simple", "medium", "complex"]},
                "risks": {"type": "array", "items": {"type": "string"}, "description": "Potential risks"},
                "constraints": {"type": "array", "items": {"type": "string"}, "description": "Constraints"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "key_entities": {"type": "array", "items": {"type": "string"}, "description": "Key entities"},
                "dependencies": {"type": "array", "items": {"type": "string"}, "description": "Dependencies"},
                "tags": {"type": "array", "items": {"type": "string"}},
            }
        }));

        // Execution - execution result, applicable to any task type
        schemas.insert("execution".to_string(), serde_json::json!({
            "type": "object",
            "required": ["summary"],
            "properties": {
                "summary": {"type": "string", "maxLength": 200},
                "result_type": {"type": "string", "description": "Result type (e.g., document, data, artifact, report, etc.)"},
                "output_location": {"type": "string", "description": "Output location (file path, URL, etc.)"},
                "output_format": {"type": "string", "description": "Output format"},
                "steps_completed": {"type": "array", "items": {"type": "string"}, "description": "Completed steps"},
                "artifacts": {"type": "array", "items": {"type": "string"}, "description": "Generated artifacts"},
                "metrics": {"type": "object", "description": "Execution metrics"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "key_entities": {"type": "array", "items": {"type": "string"}},
                "dependencies": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
            }
        }));

        // Check - check/validation result, applicable to any type of quality check
        schemas.insert("check".to_string(), serde_json::json!({
            "type": "object",
            "required": ["summary", "verdict"],
            "properties": {
                "summary": {"type": "string", "maxLength": 200},
                "verdict": {"type": "string", "enum": ["pass", "fail", "partial", "inconclusive"]},
                "quality_score": {"type": "number", "minimum": 0, "maximum": 100, "description": "Quality score"},
                "issues": {"type": "array", "items": {"type": "object"}, "description": "Issues found"},
                "strengths": {"type": "array", "items": {"type": "string"}, "description": "Strengths"},
                "weaknesses": {"type": "array", "items": {"type": "string"}, "description": "Weaknesses"},
                "recommendations": {"type": "array", "items": {"type": "string"}, "description": "Recommendations"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "key_entities": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
            }
        }));

        // Analysis - analysis/research result, applicable to any type of information gathering and analysis
        schemas.insert("analysis".to_string(), serde_json::json!({
            "type": "object",
            "required": ["summary"],
            "properties": {
                "summary": {"type": "string", "maxLength": 200},
                "findings": {"type": "array", "items": {"type": "string"}, "description": "Findings/conclusions"},
                "data_sources": {"type": "array", "items": {"type": "string"}, "description": "Data sources"},
                "methodology": {"type": "string", "description": "Analysis methodology"},
                "coverage": {"type": "string", "description": "Coverage scope"},
                "gaps": {"type": "array", "items": {"type": "string"}, "description": "Information gaps"},
                "reliability": {"type": "string", "enum": ["high", "medium", "low"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "key_entities": {"type": "array", "items": {"type": "string"}},
                "dependencies": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
            }
        }));

        // Decision - decision result, applicable to any type of decision making
        schemas.insert("decision".to_string(), serde_json::json!({
            "type": "object",
            "required": ["summary", "action"],
            "properties": {
                "summary": {"type": "string", "maxLength": 200},
                "action": {"type": "string", "enum": ["continue", "retry", "stop", "escalate", "pivot"]},
                "reasoning": {"type": "string", "description": "Decision reasoning"},
                "alternatives_considered": {"type": "array", "items": {"type": "string"}, "description": "Alternatives considered"},
                "iteration_count": {"type": "integer", "minimum": 0},
                "next_steps": {"type": "array", "items": {"type": "string"}, "description": "Next steps"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "key_entities": {"type": "array", "items": {"type": "string"}},
                "dependencies": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
            }
        }));

        // Compatibility aliases for old names
        schemas.insert(
            "code".to_string(),
            schemas.get("execution").cloned().unwrap(),
        );
        schemas.insert("review".to_string(), schemas.get("check").cloned().unwrap());
        schemas.insert(
            "research".to_string(),
            schemas.get("analysis").cloned().unwrap(),
        );

        Self { schemas }
    }

    pub fn validate_and_convert(
        &self,
        content_type: &str,
        meta: &serde_json::Value,
    ) -> Result<serde_json::Value, CoreError> {
        let schema = self
            .schemas
            .get(content_type)
            .ok_or_else(|| CoreError::ValidationFailed {
                message: format!("Unknown content type: {}", content_type),
            })?;

        let compiled = jsonschema::JSONSchema::options()
            .compile(schema)
            .map_err(|e| CoreError::ValidationFailed {
                message: format!("Schema compilation error: {}", e),
            })?;

        if let Err(validation_errors) = compiled.validate(meta) {
            let errors: Vec<String> = validation_errors
                .map(|e| format!("{}: {}", e.instance_path, e))
                .collect();
            return Err(CoreError::ValidationFailed {
                message: errors.join("; "),
            });
        }

        let json_ld = self.convert_to_json_ld(content_type, meta)?;

        Ok(json_ld)
    }

    fn convert_to_json_ld(
        &self,
        content_type: &str,
        meta: &serde_json::Value,
    ) -> Result<serde_json::Value, CoreError> {
        let node_iri = format!("iri://node_{}", uuid::Uuid::new_v4().hyphenated());

        // Generic type name mapping
        let type_name = match content_type {
            "plan" => "PlanNode",
            "execution" | "code" => "ExecutionResult",
            "check" | "review" => "CheckResult",
            "analysis" | "research" => "AnalysisReport",
            "decision" => "DecisionNode",
            _ => "Node",
        };

        let mut json_ld = serde_json::json!({
            "@id": node_iri,
            "@type": type_name,
        });
        JsonLdContext::inject(&mut json_ld);

        if let Some(obj) = json_ld.as_object_mut() {
            if let Some(meta_obj) = meta.as_object() {
                for (key, value) in meta_obj {
                    let prefixed_key = if ["summary", "confidence", "@id", "@type", "@context"]
                        .contains(&key.as_str())
                    {
                        key.clone()
                    } else {
                        format!("ex:{}", key)
                    };
                    obj.insert(prefixed_key, value.clone());
                }
            }

            obj.insert(
                "validated_at".to_string(),
                serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
            );
        }

        Ok(json_ld)
    }

    pub fn register_schema(&mut self, content_type: String, schema: serde_json::Value) {
        self.schemas.insert(content_type, schema);
    }

    pub fn get_schema(&self, content_type: &str) -> Option<&serde_json::Value> {
        self.schemas.get(content_type)
    }

    pub fn list_content_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self
            .schemas
            .keys()
            .filter(|k| !["code", "review", "research"].contains(&k.as_str()))
            .cloned()
            .collect();
        types.sort();
        types
    }
}

impl Default for MetaValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_json_ld() {
        let validator = JsonLdValidator::default();

        let valid = r#"{"@id":"iri://test/1","@type":"Test","value":42}"#;
        let result = validator.validate(valid);
        assert!(result.valid);

        let invalid = r#"not json"#;
        let result = validator.validate(invalid);
        assert!(!result.valid);
    }

    #[test]
    fn test_size_limit() {
        let validator = JsonLdValidator::new(100, true);

        let large = format!(r#"{{"@id":"iri://test/1","data":"{}"}}"#, "x".repeat(200));
        let result = validator.validate(&large);
        assert!(!result.valid);
    }

    #[test]
    fn test_meta_validator_plan() {
        let validator = MetaValidator::new();

        let plan = serde_json::json!({
            "summary": "Create a marketing strategy",
            "goal": "Increase brand awareness",
            "approach": "Multi-channel campaign",
            "sub_tasks": ["Research market", "Design assets", "Launch campaign"],
            "priority": "high",
            "confidence": 0.85
        });

        let result = validator.validate_and_convert("plan", &plan);
        assert!(result.is_ok());

        let json_ld = result.unwrap();
        assert_eq!(
            json_ld.get("@type").and_then(|t| t.as_str()),
            Some("PlanNode")
        );
    }

    #[test]
    fn test_meta_validator_execution() {
        let validator = MetaValidator::new();

        let execution = serde_json::json!({
            "summary": "Marketing campaign launched",
            "result_type": "campaign",
            "output_location": "https://campaign.example.com",
            "steps_completed": ["Research", "Design", "Launch"],
            "confidence": 0.9
        });

        let result = validator.validate_and_convert("execution", &execution);
        assert!(result.is_ok());

        let json_ld = result.unwrap();
        assert_eq!(
            json_ld.get("@type").and_then(|t| t.as_str()),
            Some("ExecutionResult")
        );
    }

    #[test]
    fn test_meta_validator_check() {
        let validator = MetaValidator::new();

        let check = serde_json::json!({
            "summary": "Campaign quality check passed",
            "verdict": "pass",
            "quality_score": 92,
            "strengths": ["Clear messaging", "Good visuals"],
            "recommendations": ["Add more CTAs"],
            "confidence": 0.88
        });

        let result = validator.validate_and_convert("check", &check);
        assert!(result.is_ok());

        let json_ld = result.unwrap();
        assert_eq!(
            json_ld.get("@type").and_then(|t| t.as_str()),
            Some("CheckResult")
        );
    }

    #[test]
    fn test_meta_validator_analysis() {
        let validator = MetaValidator::new();

        let analysis = serde_json::json!({
            "summary": "Market analysis completed",
            "findings": ["Growing demand", "Competitor weakness"],
            "data_sources": ["Industry report", "Survey data"],
            "reliability": "high",
            "confidence": 0.82
        });

        let result = validator.validate_and_convert("analysis", &analysis);
        assert!(result.is_ok());

        let json_ld = result.unwrap();
        assert_eq!(
            json_ld.get("@type").and_then(|t| t.as_str()),
            Some("AnalysisReport")
        );
    }

    #[test]
    fn test_meta_validator_decision() {
        let validator = MetaValidator::new();

        let decision = serde_json::json!({
            "summary": "Proceed with campaign",
            "action": "continue",
            "reasoning": "All checks passed",
            "next_steps": ["Monitor metrics", "Optimize"],
            "confidence": 0.9
        });

        let result = validator.validate_and_convert("decision", &decision);
        assert!(result.is_ok());

        let json_ld = result.unwrap();
        assert_eq!(
            json_ld.get("@type").and_then(|t| t.as_str()),
            Some("DecisionNode")
        );
    }

    #[test]
    fn test_backward_compatibility() {
        let validator = MetaValidator::new();

        // Old name "code" should map to "execution" schema
        let code = serde_json::json!({
            "summary": "Task completed",
            "confidence": 0.9
        });
        let result = validator.validate_and_convert("code", &code);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().get("@type").and_then(|t| t.as_str()),
            Some("ExecutionResult")
        );

        // Old name "review" should map to "check" schema
        let review = serde_json::json!({
            "summary": "Review completed",
            "verdict": "pass",
            "confidence": 0.85
        });
        let result = validator.validate_and_convert("review", &review);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().get("@type").and_then(|t| t.as_str()),
            Some("CheckResult")
        );

        // Old name "research" should map to "analysis" schema
        let research = serde_json::json!({
            "summary": "Research completed",
            "confidence": 0.8
        });
        let result = validator.validate_and_convert("research", &research);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().get("@type").and_then(|t| t.as_str()),
            Some("AnalysisReport")
        );
    }

    #[test]
    fn test_list_content_types() {
        let validator = MetaValidator::new();
        let types = validator.list_content_types();

        assert!(types.contains(&"plan".to_string()));
        assert!(types.contains(&"execution".to_string()));
        assert!(types.contains(&"check".to_string()));
        assert!(types.contains(&"analysis".to_string()));
        assert!(types.contains(&"decision".to_string()));

        // Aliases should not appear in the list
        assert!(!types.contains(&"code".to_string()));
        assert!(!types.contains(&"review".to_string()));
        assert!(!types.contains(&"research".to_string()));
    }
}
