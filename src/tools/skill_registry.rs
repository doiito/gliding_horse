//! Skill Registry - On-demand skill loading with JSON-LD support
//!
//! This module manages skill definitions with progressive disclosure:
//! - Layer 1: Basic metadata (name, description, category)
//! - Layer 2: Input/Output schemas
//! - Layer 3: Compiled templates and execution details
//!
//! Ported from rust-core (pdca-core v2.0.0).
//! Skills are loaded from JSON-LD files on demand.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, info};

use crate::jsonld::JsonLdContext;
use crate::memory::l2_blackboard::Blackboard;
use crate::CoreError;

/// Skill disclosure layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisclosureLayer {
    Basic = 1,
    Schema = 2,
    Full = 3,
}

/// Layer 1: Basic skill metadata (always loaded)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillBasic {
    pub skill_iri: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub category: String,
    pub security_level: String,
    pub allowed_roles: Vec<String>,
}

/// Layer 2: Schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSchema {
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
}

/// Layer 3: Full skill details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillFull {
    pub compiled_template: String,
    pub execution_handler: Option<String>,
    pub timeout_seconds: u32,
    pub retry_count: u32,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// Complete skill metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMeta {
    pub skill_iri: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub category: String,
    pub security_level: String,
    pub allowed_roles: Vec<String>,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
    pub compiled_template: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature_algorithm: Option<String>,
    #[serde(default)]
    pub input_mapping: HashMap<String, String>,
    #[serde(default)]
    pub output_mapping: HashMap<String, String>,
    #[serde(default)]
    pub skill_types: Vec<String>,
}

/// Cached skill with disclosure level
#[allow(dead_code)]
struct CachedSkill {
    basic: SkillBasic,
    schema: Option<SkillSchema>,
    full: Option<SkillFull>,
    loaded_layer: DisclosureLayer,
    source_path: Option<PathBuf>,
    input_mapping: HashMap<String, String>,
    output_mapping: HashMap<String, String>,
    skill_types: Vec<String>,
}

/// Skill registry with progressive disclosure
#[allow(dead_code)]
pub struct SkillRegistry {
    skills: RwLock<HashMap<String, CachedSkill>>,
    skills_by_role: RwLock<HashMap<String, Vec<String>>>,
    skills_by_category: RwLock<HashMap<String, Vec<String>>>,
    jsonld_path: Option<PathBuf>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        let registry = Self {
            skills: RwLock::new(HashMap::new()),
            skills_by_role: RwLock::new(HashMap::new()),
            skills_by_category: RwLock::new(HashMap::new()),
            jsonld_path: None,
        };

        registry.load_default_skills();
        registry
    }

    pub fn with_jsonld_path<P: AsRef<Path>>(path: P) -> Self {
        let registry = Self {
            skills: RwLock::new(HashMap::new()),
            skills_by_role: RwLock::new(HashMap::new()),
            skills_by_category: RwLock::new(HashMap::new()),
            jsonld_path: Some(path.as_ref().to_path_buf()),
        };

        registry.load_default_skills();
        registry
    }

    fn load_default_skills(&self) {
        info!("Loading default skills");
        for skill in Self::default_skills() {
            self.register_skill(skill);
        }
    }

    fn default_skills() -> Vec<SkillMeta> {
        vec![
            SkillMeta {
                skill_iri: "iri://skills/file_read".to_string(),
                name: "file_read".to_string(),
                description: "Read content from a file".to_string(),
                version: "1.0.0".to_string(),
                category: "file".to_string(),
                security_level: "normal".to_string(),
                allowed_roles: vec!["PA","DA","CA","AA"].into_iter().map(String::from).collect(),
                input_schema: json!({"type":"object","properties":{"path":{"type":"string"},"encoding":{"type":"string","default":"utf-8"}},"required":["path"]}),
                output_schema: json!({"type":"object","properties":{"content":{"type":"string"},"size":{"type":"integer"}}}),
                compiled_template: r#"{"path":"___","encoding":"utf-8"}"#.into(),
                signature: None,
                signature_algorithm: None,
                input_mapping: vec![
                    ("path", "iri://schema/file/path"),
                    ("encoding", "iri://schema/file/encoding"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                output_mapping: vec![
                    ("content", "iri://schema/file/content"),
                    ("size", "iri://schema/file/size"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                skill_types: vec![
                    "iri://skill-types/FileOperation".to_string(),
                    "iri://skill-types/ReadOperation".to_string(),
                    "iri://skill-types/IOOperation".to_string(),
                ],
            },
            SkillMeta {
                skill_iri: "iri://skills/file_write".to_string(),
                name: "file_write".to_string(),
                description: "Write content to a file".to_string(),
                version: "1.0.0".to_string(),
                category: "file".to_string(),
                security_level: "high".to_string(),
                allowed_roles: vec!["DA"].into_iter().map(String::from).collect(),
                input_schema: json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string","enum":["w","a"],"default":"w"}},"required":["path","content"]}),
                output_schema: json!({"type":"object","properties":{"success":{"type":"boolean"},"bytes_written":{"type":"integer"}}}),
                compiled_template: r#"{"path":"___","content":"___","mode":"w"}"#.into(),
                signature: None,
                signature_algorithm: None,
                input_mapping: vec![
                    ("path", "iri://schema/file/path"),
                    ("content", "iri://schema/file/content"),
                    ("mode", "iri://schema/file/write-mode"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                output_mapping: vec![
                    ("success", "iri://schema/operation/success"),
                    ("bytes_written", "iri://schema/file/bytes-written"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                skill_types: vec![
                    "iri://skill-types/FileOperation".to_string(),
                    "iri://skill-types/WriteOperation".to_string(),
                    "iri://skill-types/IOOperation".to_string(),
                ],
            },
            SkillMeta {
                skill_iri: "iri://skills/http_request".to_string(),
                name: "http_request".to_string(),
                description: "Make an HTTP request".to_string(),
                version: "1.0.0".to_string(),
                category: "network".to_string(),
                security_level: "normal".to_string(),
                allowed_roles: vec!["DA","CA"].into_iter().map(String::from).collect(),
                input_schema: json!({"type":"object","properties":{"url":{"type":"string"},"method":{"type":"string","enum":["GET","POST","PUT","DELETE","PATCH"],"default":"GET"},"headers":{"type":"object"},"body":{"type":"object"},"timeout":{"type":"integer","default":30}},"required":["url"]}),
                output_schema: json!({"type":"object","properties":{"status_code":{"type":"integer"},"headers":{"type":"object"},"body":{"type":"string"}}}),
                compiled_template: r#"{"url":"___","method":"GET","headers":{},"body":null}"#.into(),
                signature: None,
                signature_algorithm: None,
                input_mapping: vec![
                    ("url", "iri://schema/http/url"),
                    ("method", "iri://schema/http/method"),
                    ("headers", "iri://schema/http/headers"),
                    ("body", "iri://schema/http/body"),
                    ("timeout", "iri://schema/http/timeout"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                output_mapping: vec![
                    ("status_code", "iri://schema/http/status-code"),
                    ("headers", "iri://schema/http/headers"),
                    ("body", "iri://schema/http/body"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                skill_types: vec![
                    "iri://skill-types/NetworkOperation".to_string(),
                    "iri://skill-types/HTTPOperation".to_string(),
                    "iri://skill-types/RemoteOperation".to_string(),
                ],
            },
            SkillMeta {
                skill_iri: "iri://skills/llm_chat".to_string(),
                name: "llm_chat".to_string(),
                description: "Send a chat completion request to LLM".to_string(),
                version: "1.0.0".to_string(),
                category: "ai".to_string(),
                security_level: "normal".to_string(),
                allowed_roles: vec!["PA","DA","CA","AA"].into_iter().map(String::from).collect(),
                input_schema: json!({"type":"object","properties":{"messages":{"type":"array"},"model":{"type":"string","default":"deepseek-v4-flash"},"temperature":{"type":"number","default":0.7},"max_tokens":{"type":"integer","default":4096}},"required":["messages"]}),
                output_schema: json!({"type":"object","properties":{"content":{"type":"string"},"usage":{"type":"object"}}}),
                compiled_template: r#"{"messages":"___","model":"deepseek-v4-flash"}"#.into(),
                signature: None,
                signature_algorithm: None,
                input_mapping: vec![
                    ("messages", "iri://schema/llm/messages"),
                    ("model", "iri://schema/llm/model"),
                    ("temperature", "iri://schema/llm/temperature"),
                    ("max_tokens", "iri://schema/llm/max-tokens"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                output_mapping: vec![
                    ("content", "iri://schema/llm/response-content"),
                    ("usage", "iri://schema/llm/usage"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                skill_types: vec![
                    "iri://skill-types/AIOperation".to_string(),
                    "iri://skill-types/LLMOperation".to_string(),
                    "iri://skill-types/ChatOperation".to_string(),
                ],
            },
            SkillMeta {
                skill_iri: "iri://skills/code_execute".to_string(),
                name: "code_execute".to_string(),
                description: "Execute code in a sandboxed environment".to_string(),
                version: "1.0.0".to_string(),
                category: "execution".to_string(),
                security_level: "critical".to_string(),
                allowed_roles: vec!["DA"].into_iter().map(String::from).collect(),
                input_schema: json!({"type":"object","properties":{"code":{"type":"string"},"language":{"type":"string","enum":["python","javascript","bash"],"default":"python"},"timeout":{"type":"integer","default":60}},"required":["code"]}),
                output_schema: json!({"type":"object","properties":{"stdout":{"type":"string"},"stderr":{"type":"string"},"exit_code":{"type":"integer"}}}),
                compiled_template: r#"{"code":"___","language":"python","timeout":60}"#.into(),
                signature: None,
                signature_algorithm: None,
                input_mapping: vec![
                    ("code", "iri://schema/code/source-code"),
                    ("language", "iri://schema/code/language"),
                    ("timeout", "iri://schema/code/timeout"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                output_mapping: vec![
                    ("stdout", "iri://schema/code/stdout"),
                    ("stderr", "iri://schema/code/stderr"),
                    ("exit_code", "iri://schema/code/exit-code"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                skill_types: vec![
                    "iri://skill-types/ExecutionOperation".to_string(),
                    "iri://skill-types/CodeExecution".to_string(),
                    "iri://skill-types/SandboxedOperation".to_string(),
                ],
            },
            SkillMeta {
                skill_iri: "iri://skills/jsonld_validate".to_string(),
                name: "jsonld_validate".to_string(),
                description: "Validate a JSON-LD document".to_string(),
                version: "1.0.0".to_string(),
                category: "validation".to_string(),
                security_level: "low".to_string(),
                allowed_roles: vec!["CA","SA"].into_iter().map(String::from).collect(),
                input_schema: json!({"type":"object","properties":{"document":{"type":"object"},"schema_iri":{"type":"string"}},"required":["document"]}),
                output_schema: json!({"type":"object","properties":{"valid":{"type":"boolean"},"errors":{"type":"array"}}}),
                compiled_template: r#"{"document":"___","schema_iri":null}"#.into(),
                signature: None,
                signature_algorithm: None,
                input_mapping: vec![
                    ("document", "iri://schema/jsonld/document"),
                    ("schema_iri", "iri://schema/jsonld/schema-iri"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                output_mapping: vec![
                    ("valid", "iri://schema/validation/valid"),
                    ("errors", "iri://schema/validation/errors"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                skill_types: vec![
                    "iri://skill-types/ValidationOperation".to_string(),
                    "iri://skill-types/JSONLDOperation".to_string(),
                    "iri://skill-types/SchemaValidation".to_string(),
                ],
            },
            SkillMeta {
                skill_iri: "iri://skills/create_skill".to_string(),
                name: "create_skill".to_string(),
                description: "Automatically create new Skill definitions from natural language descriptions, using LLM to generate JSON-LD Skill format".to_string(),
                version: "1.0.0".to_string(),
                category: "meta".to_string(),
                security_level: "high".to_string(),
                allowed_roles: vec!["DA".to_string()],
                input_schema: json!({"type":"object","properties":{"description":{"type":"string","description":"Natural language description of Skill functionality"},"skill_name_hint":{"type":"string","description":"Suggested Skill name (optional)"},"category_hint":{"type":"string","description":"Suggested category (optional): file|network|ai|execution|validation|data|meta|system"},"security_level_override":{"type":"string","description":"Security level override (optional): low|normal|high|critical"}},"required":["description"]}),
                output_schema: json!({"type":"object","properties":{"skill_iri":{"type":"string"},"name":{"type":"string"},"json_ld":{"type":"object"},"registered":{"type":"boolean"}}}),
                compiled_template: r#"{"description":"___"}"#.into(),
                signature: None,
                signature_algorithm: None,
                input_mapping: vec![
                    ("description", "iri://schema/skill-creator/description"),
                    ("skill_name_hint", "iri://schema/skill-creator/name-hint"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                output_mapping: vec![
                    ("skill_iri", "iri://schema/skill-creator/output/skill-iri"),
                    ("name", "iri://schema/skill-creator/output/name"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                skill_types: vec![
                    "iri://skill-types/MetaOperation".to_string(),
                    "iri://skill-types/SkillCreation".to_string(),
                ],
            },
            SkillMeta {
                skill_iri: "iri://skills/convert_skill".to_string(),
                name: "convert_skill".to_string(),
                description: "Automatically convert Markdown Skill descriptions to JSON-LD Skill definitions".to_string(),
                version: "1.0.0".to_string(),
                category: "meta".to_string(),
                security_level: "normal".to_string(),
                allowed_roles: vec!["DA".to_string(), "CA".to_string()],
                input_schema: json!({"type":"object","properties":{"markdown_content":{"type":"string","description":"Markdown formatted Skill description content"},"source_path":{"type":"string","description":"Source file path (optional)"}},"required":["markdown_content"]}),
                output_schema: json!({"type":"object","properties":{"skill_iri":{"type":"string"},"name":{"type":"string"},"json_ld":{"type":"object"},"registered":{"type":"boolean"}}}),
                compiled_template: r#"{"markdown_content":"___"}"#.into(),
                signature: None,
                signature_algorithm: None,
                input_mapping: vec![
                    ("markdown_content", "iri://schema/skill-converter/markdown"),
                    ("source_path", "iri://schema/skill-converter/source-path"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                output_mapping: vec![
                    ("skill_iri", "iri://schema/skill-converter/output/skill-iri"),
                    ("name", "iri://schema/skill-converter/output/name"),
                ].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                skill_types: vec![
                    "iri://skill-types/MetaOperation".to_string(),
                    "iri://skill-types/SkillConversion".to_string(),
                ],
            },
        ]
    }

    pub fn load_from_jsonld<P: AsRef<Path>>(&self, path: P) -> Result<usize, CoreError> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading skills from JSON-LD");

        let content = std::fs::read_to_string(path).map_err(|e| CoreError::Internal {
            message: format!("Failed to read JSON-LD file: {}", e),
        })?;

        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| CoreError::InvalidJsonLd {
                message: format!("Invalid JSON: {}", e),
            })?;

        let skills = json
            .get("skills")
            .and_then(|s| s.as_array())
            .ok_or_else(|| CoreError::InvalidJsonLd {
                message: "No skills array found".to_string(),
            })?;

        let mut loaded = 0;
        for skill_json in skills {
            if let Ok(skill) = self.parse_skill_from_jsonld(skill_json) {
                self.register_skill(skill);
                loaded += 1;
            }
        }

        info!(count = loaded, "Skills loaded from JSON-LD");
        Ok(loaded)
    }

    /// Load skills from a `skills/*/skill.jsonld` directory layout: each
    /// immediate subdirectory may contain a `skill.jsonld` file whose
    /// `skills` array is merged (existing IRIs replaced). Subdirectories
    /// without a `skill.jsonld` are skipped.
    pub fn load_from_jsonld_dir<P: AsRef<Path>>(&self, dir: P) -> Result<usize, CoreError> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err(CoreError::Internal {
                message: format!("Skill directory not found: {}", dir.display()),
            });
        }
        let mut loaded = 0;
        for entry in std::fs::read_dir(dir).map_err(|e| CoreError::Internal {
            message: format!("Failed to read skill dir {}: {}", dir.display(), e),
        })? {
            let entry = entry.map_err(|e| CoreError::Internal {
                message: format!("read_dir entry: {}", e),
            })?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let def = path.join("skill.jsonld");
            if def.is_file() {
                loaded += self.load_from_jsonld(&def)?;
            }
        }
        info!(count = loaded, path = %dir.display(), "Skills loaded from JSON-LD directory");
        Ok(loaded)
    }

    fn parse_skill_from_jsonld(&self, json: &serde_json::Value) -> Result<SkillMeta, CoreError> {
        let skill_iri = json
            .get("@id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CoreError::InvalidJsonLd {
                message: "Missing @id".to_string(),
            })?
            .to_string();

        let name = json
            .get("skill:name")
            .or_else(|| json.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let description = json
            .get("skill:description")
            .or_else(|| json.get("description"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let version = json
            .get("skill:version")
            .or_else(|| json.get("version"))
            .and_then(|v| v.as_str())
            .unwrap_or("1.0.0")
            .to_string();

        let category = json
            .get("skill:category")
            .or_else(|| json.get("category"))
            .and_then(|v| v.as_str())
            .unwrap_or("general")
            .to_string();

        let security_level = json
            .get("skill:securityLevel")
            .or_else(|| json.get("security_level"))
            .and_then(|v| v.as_str())
            .unwrap_or("normal")
            .to_string();

        let allowed_roles = json
            .get("skill:allowedRoles")
            .or_else(|| json.get("allowed_roles"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| r.as_str().map(String::from))
                    .collect()
            })
            .or_else(|| {
                json.get("skill:who")
                    .and_then(|who| {
                        who.get("skill:roleName")
                            .or_else(|| who.get("skill:requiredRole"))
                    })
                    .and_then(|v| v.as_str())
                    .map(|r| vec![r.to_string()])
            })
            .unwrap_or_default();

        let input_schema_raw = json
            .get("skill:inputSchema")
            .or_else(|| json.get("input_schema"))
            .cloned()
            .unwrap_or(serde_json::json!({"type": "object"}));

        let output_schema_raw = json
            .get("skill:outputSchema")
            .or_else(|| json.get("output_schema"))
            .cloned()
            .unwrap_or(serde_json::json!({"type": "object"}));

        let input_schema = Self::convert_jsonld_schema_to_jsonschema(&input_schema_raw);
        let output_schema = Self::convert_jsonld_schema_to_jsonschema(&output_schema_raw);

        let compiled_template = json
            .get("skill:compiledTemplate")
            .or_else(|| json.get("compiled_template"))
            .and_then(|v| serde_json::to_string(v).ok())
            .unwrap_or_else(|| "{}".to_string());

        let signature = json
            .get("skill:signature")
            .or_else(|| json.get("signature"))
            .and_then(|v| v.as_str().map(String::from));

        let signature_algorithm = json
            .get("skill:signatureAlgorithm")
            .or_else(|| json.get("signature_algorithm"))
            .and_then(|v| v.as_str().map(String::from));

        let input_mapping = json
            .get("skill:inputMapping")
            .or_else(|| json.get("input_mapping"))
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        v.get("@id")
                            .and_then(|id| id.as_str())
                            .map(|id| (k.clone(), id.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let output_mapping = json
            .get("skill:outputMapping")
            .or_else(|| json.get("output_mapping"))
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        v.get("@id")
                            .and_then(|id| id.as_str())
                            .map(|id| (k.clone(), id.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let skill_types = json
            .get("@type")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str().map(String::from))
                    .filter(|t| t != "skill:Skill")
                    .collect()
            })
            .unwrap_or_default();

        Ok(SkillMeta {
            skill_iri,
            name,
            description,
            version,
            category,
            security_level,
            allowed_roles,
            input_schema,
            output_schema,
            compiled_template,
            signature,
            signature_algorithm,
            input_mapping,
            output_mapping,
            skill_types,
        })
    }

    /// Convert JSON-LD skill schema to standard JSON Schema format
    fn convert_jsonld_schema_to_jsonschema(schema: &serde_json::Value) -> serde_json::Value {
        if !schema.is_object() {
            return schema.clone();
        }

        let obj = schema.as_object().expect("checked is_object above");

        if obj.contains_key("type") && obj.get("type").and_then(|t| t.as_str()).is_some() {
            return schema.clone();
        }

        let mut result = serde_json::Map::new();
        result.insert("type".to_string(), serde_json::json!("object"));

        if let Some(props) = obj
            .get("skill:properties")
            .or_else(|| obj.get("properties"))
        {
            let converted_props = Self::convert_properties(props);
            result.insert("properties".to_string(), converted_props);
        }

        if let Some(required) = obj.get("skill:required").or_else(|| obj.get("required")) {
            result.insert("required".to_string(), required.clone());
        }

        serde_json::Value::Object(result)
    }

    /// Convert JSON-LD properties to JSON Schema properties
    fn convert_properties(props: &serde_json::Value) -> serde_json::Value {
        if !props.is_object() {
            return props.clone();
        }

        let obj = props.as_object().expect("checked is_object above");
        let mut result = serde_json::Map::new();

        for (key, value) in obj {
            let converted = Self::convert_property(value);
            result.insert(key.clone(), converted);
        }

        serde_json::Value::Object(result)
    }

    /// Convert a single JSON-LD property to JSON Schema property
    fn convert_property(prop: &serde_json::Value) -> serde_json::Value {
        if !prop.is_object() {
            return prop.clone();
        }

        let obj = prop.as_object().expect("checked is_object above");
        let mut result = serde_json::Map::new();

        if let Some(type_val) = obj.get("skill:type").or_else(|| obj.get("type")) {
            result.insert("type".to_string(), type_val.clone());
        }

        if let Some(desc) = obj
            .get("skill:description")
            .or_else(|| obj.get("description"))
        {
            result.insert("description".to_string(), desc.clone());
        }

        if let Some(default) = obj.get("skill:default").or_else(|| obj.get("default")) {
            result.insert("default".to_string(), default.clone());
        }

        if let Some(enum_val) = obj.get("skill:enum").or_else(|| obj.get("enum")) {
            result.insert("enum".to_string(), enum_val.clone());
        }

        if let Some(items) = obj.get("skill:items").or_else(|| obj.get("items")) {
            result.insert("items".to_string(), items.clone());
        }

        if let Some(min_val) = obj.get("skill:minimum").or_else(|| obj.get("minimum")) {
            result.insert("minimum".to_string(), min_val.clone());
        }

        if let Some(max_val) = obj.get("skill:maximum").or_else(|| obj.get("maximum")) {
            result.insert("maximum".to_string(), max_val.clone());
        }

        if let Some(props) = obj
            .get("skill:properties")
            .or_else(|| obj.get("properties"))
        {
            result.insert("properties".to_string(), Self::convert_properties(props));
        }

        if result.is_empty() {
            return prop.clone();
        }

        serde_json::Value::Object(result)
    }

    pub fn register_skill(&self, skill: SkillMeta) {
        let cached = CachedSkill {
            basic: SkillBasic {
                skill_iri: skill.skill_iri.clone(),
                name: skill.name.clone(),
                description: skill.description.clone(),
                version: skill.version.clone(),
                category: skill.category.clone(),
                security_level: skill.security_level.clone(),
                allowed_roles: skill.allowed_roles.clone(),
            },
            schema: Some(SkillSchema {
                input_schema: skill.input_schema.clone(),
                output_schema: skill.output_schema.clone(),
            }),
            full: Some(SkillFull {
                compiled_template: skill.compiled_template.clone(),
                execution_handler: None,
                timeout_seconds: 60,
                retry_count: 0,
                metadata: serde_json::Map::new(),
            }),
            loaded_layer: DisclosureLayer::Full,
            source_path: None,
            input_mapping: skill.input_mapping.clone(),
            output_mapping: skill.output_mapping.clone(),
            skill_types: skill.skill_types.clone(),
        };

        let iri = skill.skill_iri.clone();
        // Never hold the primary map while acquiring secondary-index locks:
        // role readers acquire in the opposite order (index → primary map).
        // A brief replacement window can omit this IRI from a concurrent read,
        // but cannot deadlock the agent runtime.
        if self.skills.write().remove(&iri).is_some() {
            self.remove_from_secondary_indexes(&iri);
        }
        {
            let mut by_role = self.skills_by_role.write();
            for role in &skill.allowed_roles {
                by_role.entry(role.clone()).or_default().push(iri.clone());
            }
        }
        {
            let mut by_cat = self.skills_by_category.write();
            by_cat
                .entry(skill.category.clone())
                .or_default()
                .push(iri.clone());
        }
        self.skills.write().insert(iri, cached);

        debug!(skill_iri = %skill.skill_iri, "Skill registered");
    }

    /// Remove a skill and every role/category secondary-index reference.
    /// This is used by compensating workflows such as dynamic-skill creation
    /// when a later required store write fails.
    pub fn unregister_skill(&self, skill_iri: &str) -> bool {
        let removed = self.skills.write().remove(skill_iri).is_some();
        if removed {
            self.remove_from_secondary_indexes(skill_iri);
            debug!(skill_iri, "Skill unregistered");
        }
        removed
    }

    fn remove_from_secondary_indexes(&self, skill_iri: &str) {
        let mut by_role = self.skills_by_role.write();
        by_role.retain(|_, iris| {
            iris.retain(|iri| iri != skill_iri);
            !iris.is_empty()
        });
        drop(by_role);
        let mut by_category = self.skills_by_category.write();
        by_category.retain(|_, iris| {
            iris.retain(|iri| iri != skill_iri);
            !iris.is_empty()
        });
    }

    pub fn get_skill(&self, skill_iri: &str) -> Option<SkillMeta> {
        let skills = self.skills.read();
        let cached = skills.get(skill_iri)?;

        Some(SkillMeta {
            skill_iri: cached.basic.skill_iri.clone(),
            name: cached.basic.name.clone(),
            description: cached.basic.description.clone(),
            version: cached.basic.version.clone(),
            category: cached.basic.category.clone(),
            security_level: cached.basic.security_level.clone(),
            allowed_roles: cached.basic.allowed_roles.clone(),
            input_schema: cached
                .schema
                .as_ref()
                .map(|s| s.input_schema.clone())
                .unwrap_or(serde_json::json!({"type": "object"})),
            output_schema: cached
                .schema
                .as_ref()
                .map(|s| s.output_schema.clone())
                .unwrap_or(serde_json::json!({"type": "object"})),
            compiled_template: cached
                .full
                .as_ref()
                .map(|f| f.compiled_template.clone())
                .unwrap_or_default(),
            signature: None,
            signature_algorithm: None,
            input_mapping: cached.input_mapping.clone(),
            output_mapping: cached.output_mapping.clone(),
            skill_types: cached.skill_types.clone(),
        })
    }

    /// Resolve a runtime tool name to the canonical skill IRI.
    ///
    /// Tool names are part of the execution protocol while IRIs are part of
    /// the graph/evolution protocol. Keeping this lookup in the registry
    /// prevents callers from manufacturing a second IRI namespace.
    pub fn skill_iri_for_tool_name(&self, tool_name: &str) -> Option<String> {
        self.skills
            .read()
            .values()
            .find(|cached| cached.basic.name == tool_name)
            .map(|cached| cached.basic.skill_iri.clone())
    }

    pub fn get_skill_basic(&self, skill_iri: &str) -> Option<SkillBasic> {
        let skills = self.skills.read();
        skills.get(skill_iri).map(|c| c.basic.clone())
    }

    pub fn get_skill_schema(&self, skill_iri: &str) -> Option<SkillSchema> {
        let skills = self.skills.read();
        skills.get(skill_iri).and_then(|c| c.schema.clone())
    }

    pub fn get_skill_full(&self, skill_iri: &str) -> Option<(SkillBasic, SkillSchema, SkillFull)> {
        let skills = self.skills.read();
        let cached = skills.get(skill_iri)?;

        Some((
            cached.basic.clone(),
            cached.schema.clone()?,
            cached.full.clone()?,
        ))
    }

    pub fn list_skills_for_role(&self, role: &str) -> Vec<SkillMeta> {
        let by_role = self.skills_by_role.read();
        let _skills = self.skills.read();

        by_role
            .get(role)
            .map(|iris| {
                iris.iter()
                    .filter_map(|iri| self.get_skill(iri))
                    .filter(|skill| {
                        !skill
                            .skill_types
                            .iter()
                            .any(|kind| kind == "definition_only")
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn list_skills_for_category(&self, category: &str) -> Vec<SkillMeta> {
        let by_cat = self.skills_by_category.read();

        by_cat
            .get(category)
            .map(|iris| iris.iter().filter_map(|iri| self.get_skill(iri)).collect())
            .unwrap_or_default()
    }

    pub fn list_all_skills(&self) -> Vec<SkillMeta> {
        let skills = self.skills.read();
        skills
            .keys()
            .filter_map(|iri| self.get_skill(iri))
            .collect()
    }

    pub fn list_skills_basic(&self) -> Vec<SkillBasic> {
        let skills = self.skills.read();
        skills.values().map(|c| c.basic.clone()).collect()
    }

    pub fn validate_input(
        &self,
        skill_iri: &str,
        input: &str,
    ) -> Result<serde_json::Value, CoreError> {
        let schema = self
            .get_skill_schema(skill_iri)
            .ok_or_else(|| CoreError::SkillNotFound {
                iri: skill_iri.to_string(),
            })?;

        let input_value: serde_json::Value =
            serde_json::from_str(input).map_err(|e| CoreError::InvalidJsonLd {
                message: format!("Invalid JSON input: {}", e),
            })?;

        let compiled = jsonschema::JSONSchema::options()
            .compile(&schema.input_schema)
            .map_err(|e| CoreError::Internal {
                message: e.to_string(),
            })?;

        if let Err(errors) = compiled.validate(&input_value) {
            let error_messages: Vec<String> = errors
                .map(|e| format!("{}: {}", e.instance_path, e))
                .collect();

            return Err(CoreError::ValidationFailed {
                message: error_messages.join("; "),
            });
        }

        Ok(input_value)
    }

    pub fn validate_output(
        &self,
        skill_iri: &str,
        output: &str,
    ) -> Result<serde_json::Value, CoreError> {
        let schema = self
            .get_skill_schema(skill_iri)
            .ok_or_else(|| CoreError::SkillNotFound {
                iri: skill_iri.to_string(),
            })?;

        let output_value: serde_json::Value =
            serde_json::from_str(output).map_err(|e| CoreError::InvalidJsonLd {
                message: format!("Invalid JSON output: {}", e),
            })?;

        let compiled = jsonschema::JSONSchema::options()
            .compile(&schema.output_schema)
            .map_err(|e| CoreError::Internal {
                message: e.to_string(),
            })?;

        if let Err(errors) = compiled.validate(&output_value) {
            let error_messages: Vec<String> = errors
                .map(|e| format!("{}: {}", e.instance_path, e))
                .collect();

            return Err(CoreError::ValidationFailed {
                message: error_messages.join("; "),
            });
        }

        Ok(output_value)
    }

    pub fn get_compiled_template(&self, skill_iri: &str) -> Option<String> {
        let skills = self.skills.read();
        skills
            .get(skill_iri)
            .and_then(|c| c.full.as_ref().map(|f| f.compiled_template.clone()))
    }

    pub fn check_signature(&self, skill_iri: &str) -> bool {
        let skills = self.skills.read();
        skills
            .get(skill_iri)
            .and_then(|c| c.full.as_ref())
            .map(|_| true)
            .unwrap_or(false)
    }

    /// Sync all registered skills to the `system:skills` named graph in oxigraph.
    /// This enables SPARQL-based tool discovery.
    pub fn sync_to_oxigraph(&self, blackboard: &Blackboard) -> Result<usize, CoreError> {
        use std::fmt::Write;
        let skills = self.list_all_skills();
        let mut sparql = String::from("PREFIX skill: <https://agent-os.org/ontology/skill#>\n");
        for s in &skills {
            let iri = &s.skill_iri;
            let name = s.name.replace('\\', "\\\\").replace('\'', "\\'");
            let desc = s.description.replace('\\', "\\\\").replace('\'', "\\'");
            write!(
                sparql,
                "INSERT DATA {{ GRAPH <system:skills> {{ <{iri}> a skill:RegisteredSkill ; skill:name '{name}' ; skill:description '{desc}' ; skill:category '{cat}' ; skill:securityLevel '{sec}' . }} }}\n",
                iri = iri,
                name = name,
                desc = desc,
                cat = s.category,
                sec = s.security_level,
            ).ok();
        }
        if !skills.is_empty() {
            blackboard.sparql_update(&sparql)?;
        }
        Ok(skills.len())
    }

    /// SPARQL-based tool discovery against `system:skills` named graph.
    /// Falls back to in-memory search if blackboard is None.
    pub fn sparql_search_skills(
        &self,
        blackboard: &Blackboard,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SkillMeta>, CoreError> {
        let safe_query = query.replace('\'', "\\'");
        let sparql = format!(
            "PREFIX skill: <https://agent-os.org/ontology/skill#>
             SELECT ?iri ?name ?desc ?cat ?sec WHERE {{
               GRAPH <system:skills> {{
                 ?iri a skill:RegisteredSkill ;
                      skill:name ?name ;
                      skill:description ?desc ;
                      skill:category ?cat ;
                      skill:securityLevel ?sec .
                 FILTER(CONTAINS(LCASE(?name), '{q}') || CONTAINS(LCASE(?desc), '{q}') || CONTAINS(LCASE(?cat), '{q}'))
               }}
             }} LIMIT {limit}",
            q = safe_query.to_lowercase(),
            limit = limit
        );
        let results = blackboard.query(&sparql)?;
        let mut metas = Vec::new();
        for row in &results {
            if let (Some(iri), Some(_name)) = (
                row.get("iri").and_then(|v| v.as_str()),
                row.get("name").and_then(|v| v.as_str()),
            ) {
                // Get from cache for complete metadata
                if let Some(skill) = self.get_skill(iri) {
                    metas.push(skill);
                }
            }
        }
        Ok(metas)
    }

    /// Search skills by keyword across name, description, category, IRI
    pub fn search_skills(&self, query: &str, limit: usize) -> Vec<SkillMeta> {
        let q = query.to_lowercase();
        let mut results: Vec<SkillMeta> = self
            .list_all_skills()
            .into_iter()
            .filter(|s| {
                s.name.to_lowercase().contains(&q)
                    || s.description.to_lowercase().contains(&q)
                    || s.category.to_lowercase().contains(&q)
                    || s.skill_iri.to_lowercase().contains(&q)
            })
            .collect();
        results.truncate(limit);
        results
    }

    pub fn skill_count(&self) -> usize {
        self.skills.read().len()
    }

    /// Discover skills by semantic capability
    /// Find matching skills based on skill_types IRIs
    pub fn find_skills_by_semantic_capability(&self, capability_iri: &str) -> Vec<SkillMeta> {
        let skills = self.skills.read();
        skills
            .values()
            .filter(|cached| cached.skill_types.iter().any(|t| t == capability_iri))
            .filter_map(|cached| self.get_skill(&cached.basic.skill_iri))
            .collect()
    }

    /// Discover skills by input type
    /// Find matching skills based on input_mapping values
    pub fn find_skills_by_input_type(&self, type_iri: &str) -> Vec<SkillMeta> {
        let skills = self.skills.read();
        skills
            .values()
            .filter(|cached| cached.input_mapping.values().any(|v| v == type_iri))
            .filter_map(|cached| self.get_skill(&cached.basic.skill_iri))
            .collect()
    }

    /// Map local parameter names to unified IRIs
    /// Returns a mapped parameter dictionary with IRI keys and original parameter values
    pub fn map_input_params(
        &self,
        skill_iri: &str,
        params: &HashMap<String, serde_json::Value>,
    ) -> HashMap<String, serde_json::Value> {
        let skills = self.skills.read();
        if let Some(cached) = skills.get(skill_iri) {
            params
                .iter()
                .filter_map(|(key, value)| {
                    cached
                        .input_mapping
                        .get(key)
                        .map(|iri| (iri.clone(), value.clone()))
                })
                .collect()
        } else {
            HashMap::new()
        }
    }

    /// Convert SkillMeta to full JSON-LD format
    pub fn to_json_ld(&self, skill_iri: &str) -> Option<serde_json::Value> {
        let skill = self.get_skill(skill_iri)?;

        let mut context_map = JsonLdContext::context_value()
            .as_object()
            .cloned()
            .unwrap_or_default();
        context_map.insert(
            "schema".to_string(),
            serde_json::json!("https://agent-os.org/ontology/schema#"),
        );

        let mut input_props = serde_json::Map::new();
        for (local_name, iri) in &skill.input_mapping {
            input_props.insert(
                local_name.clone(),
                serde_json::json!({
                    "@id": iri,
                    "@type": "schema:Parameter"
                }),
            );
        }

        let mut output_props = serde_json::Map::new();
        for (local_name, iri) in &skill.output_mapping {
            output_props.insert(
                local_name.clone(),
                serde_json::json!({
                    "@id": iri,
                    "@type": "schema:OutputField"
                }),
            );
        }

        let mut skill_types_json = vec![serde_json::json!("skill:Skill")];
        for skill_type in &skill.skill_types {
            skill_types_json.push(serde_json::Value::String(skill_type.clone()));
        }

        let mut result = serde_json::Map::new();
        result.insert(
            "@context".to_string(),
            serde_json::Value::Object(context_map),
        );
        result.insert(
            "@id".to_string(),
            serde_json::Value::String(skill.skill_iri.clone()),
        );
        result.insert(
            "@type".to_string(),
            serde_json::Value::Array(skill_types_json),
        );
        result.insert(
            "skill:name".to_string(),
            serde_json::Value::String(skill.name),
        );
        result.insert(
            "skill:description".to_string(),
            serde_json::Value::String(skill.description),
        );
        result.insert(
            "skill:version".to_string(),
            serde_json::Value::String(skill.version),
        );
        result.insert(
            "skill:category".to_string(),
            serde_json::Value::String(skill.category),
        );
        result.insert(
            "skill:securityLevel".to_string(),
            serde_json::Value::String(skill.security_level),
        );
        result.insert(
            "skill:allowedRoles".to_string(),
            serde_json::json!(skill.allowed_roles),
        );
        result.insert("skill:inputSchema".to_string(), skill.input_schema);
        result.insert("skill:outputSchema".to_string(), skill.output_schema);
        result.insert(
            "skill:inputMapping".to_string(),
            serde_json::Value::Object(input_props),
        );
        result.insert(
            "skill:outputMapping".to_string(),
            serde_json::Value::Object(output_props),
        );

        Some(serde_json::Value::Object(result))
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_registry() {
        let registry = SkillRegistry::new();

        let skill = registry.get_skill("iri://skills/file_read");
        assert!(skill.is_some());

        let da_skills = registry.list_skills_for_role("DA");
        assert!(!da_skills.is_empty());
    }

    #[test]
    fn definition_only_skill_is_retained_but_not_offered_as_executable_role_skill() {
        let registry = SkillRegistry::new();
        let mut definition = registry.get_skill("iri://skills/file_read").unwrap();
        definition.skill_iri = "iri://skills/generated_definition".to_string();
        definition.name = "generated_definition".to_string();
        definition.skill_types = vec!["definition_only".to_string()];
        registry.register_skill(definition);

        assert!(registry
            .get_skill("iri://skills/generated_definition")
            .is_some());
        assert!(registry
            .list_all_skills()
            .iter()
            .any(|skill| skill.skill_iri == "iri://skills/generated_definition"));
        assert!(!registry
            .list_skills_for_role("DA")
            .iter()
            .any(|skill| skill.skill_iri == "iri://skills/generated_definition"));
    }

    #[test]
    fn replacing_or_unregistering_skill_cleans_secondary_indexes() {
        let registry = SkillRegistry::new();
        let mut replacement = registry.get_skill("iri://skills/file_read").unwrap();
        replacement.allowed_roles = vec!["AA".to_string()];
        replacement.category = "replacement-category".to_string();
        registry.register_skill(replacement);

        assert!(!registry
            .list_skills_for_role("DA")
            .iter()
            .any(|skill| skill.skill_iri == "iri://skills/file_read"));
        assert_eq!(
            registry
                .list_skills_for_role("AA")
                .iter()
                .filter(|skill| skill.skill_iri == "iri://skills/file_read")
                .count(),
            1
        );
        assert!(registry.unregister_skill("iri://skills/file_read"));
        assert!(registry.get_skill("iri://skills/file_read").is_none());
        assert!(!registry
            .list_skills_for_role("AA")
            .iter()
            .any(|skill| skill.skill_iri == "iri://skills/file_read"));
    }

    #[test]
    fn test_resolve_canonical_skill_iri_from_tool_name() {
        let registry = SkillRegistry::new();

        assert_eq!(
            registry.skill_iri_for_tool_name("file_read").as_deref(),
            Some("iri://skills/file_read")
        );
        assert_eq!(registry.skill_iri_for_tool_name("unknown_tool"), None);
    }

    #[test]
    fn test_validate_input() {
        let registry = SkillRegistry::new();

        let valid_input = r#"{"path": "/tmp/test.txt"}"#;
        let result = registry.validate_input("iri://skills/file_read", valid_input);
        assert!(result.is_ok());

        let invalid_input = r#"{}"#;
        let result = registry.validate_input("iri://skills/file_read", invalid_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_progressive_disclosure() {
        let registry = SkillRegistry::new();

        let basic = registry.get_skill_basic("iri://skills/file_read");
        assert!(basic.is_some());

        let schema = registry.get_skill_schema("iri://skills/file_read");
        assert!(schema.is_some());

        let full = registry.get_skill_full("iri://skills/file_read");
        assert!(full.is_some());
    }

    #[test]
    fn test_list_basic() {
        let registry = SkillRegistry::new();

        let basics = registry.list_skills_basic();
        assert!(!basics.is_empty());

        for basic in basics {
            assert!(!basic.skill_iri.is_empty());
            assert!(!basic.name.is_empty());
        }
    }

    #[test]
    fn test_jsonld_schema_conversion() {
        let jsonld_schema = serde_json::json!({
            "@type": "skill:InputSchema",
            "skill:properties": {
                "path": {
                    "@id": "skill:path",
                    "@type": "skill:Property",
                    "skill:type": "string",
                    "skill:description": "File path to read",
                    "skill:required": true
                },
                "encoding": {
                    "@id": "skill:encoding",
                    "@type": "skill:Property",
                    "skill:type": "string",
                    "skill:description": "File encoding",
                    "skill:default": "utf-8"
                }
            },
            "skill:required": ["path"]
        });

        let converted = SkillRegistry::convert_jsonld_schema_to_jsonschema(&jsonld_schema);

        assert_eq!(
            converted.get("type").and_then(|t| t.as_str()),
            Some("object")
        );
        assert!(converted.get("properties").is_some());
        assert!(converted.get("required").is_some());

        let props = converted.get("properties").unwrap().as_object().unwrap();
        assert!(props.contains_key("path"));
        assert!(props.contains_key("encoding"));

        let path_prop = props.get("path").unwrap().as_object().unwrap();
        assert_eq!(
            path_prop.get("type").and_then(|t| t.as_str()),
            Some("string")
        );
        assert_eq!(
            path_prop.get("description").and_then(|d| d.as_str()),
            Some("File path to read")
        );

        let encoding_prop = props.get("encoding").unwrap().as_object().unwrap();
        assert_eq!(
            encoding_prop.get("default").and_then(|d| d.as_str()),
            Some("utf-8")
        );
    }

    #[test]
    fn test_semantic_capability_discovery() {
        let registry = SkillRegistry::new();

        let file_ops =
            registry.find_skills_by_semantic_capability("iri://skill-types/FileOperation");
        assert_eq!(file_ops.len(), 2);
        assert!(file_ops.iter().any(|s| s.name == "file_read"));
        assert!(file_ops.iter().any(|s| s.name == "file_write"));

        let ai_ops = registry.find_skills_by_semantic_capability("iri://skill-types/AIOperation");
        assert_eq!(ai_ops.len(), 1);
        assert_eq!(ai_ops[0].name, "llm_chat");

        let non_existent =
            registry.find_skills_by_semantic_capability("iri://skill-types/NonExistent");
        assert!(non_existent.is_empty());
    }

    #[test]
    fn test_input_type_discovery() {
        let registry = SkillRegistry::new();

        let path_input_skills = registry.find_skills_by_input_type("iri://schema/file/path");
        assert_eq!(path_input_skills.len(), 2);
        assert!(path_input_skills.iter().any(|s| s.name == "file_read"));
        assert!(path_input_skills.iter().any(|s| s.name == "file_write"));

        let url_input_skills = registry.find_skills_by_input_type("iri://schema/http/url");
        assert_eq!(url_input_skills.len(), 1);
        assert_eq!(url_input_skills[0].name, "http_request");

        let code_input_skills = registry.find_skills_by_input_type("iri://schema/code/source-code");
        assert_eq!(code_input_skills.len(), 1);
        assert_eq!(code_input_skills[0].name, "code_execute");
    }

    #[test]
    fn test_map_input_params() {
        let registry = SkillRegistry::new();

        let mut params = HashMap::new();
        params.insert("path".to_string(), serde_json::json!("/tmp/test.txt"));
        params.insert("encoding".to_string(), serde_json::json!("utf-8"));
        params.insert("unknown_param".to_string(), serde_json::json!("value"));

        let mapped = registry.map_input_params("iri://skills/file_read", &params);

        assert_eq!(mapped.len(), 2);
        assert_eq!(
            mapped.get("iri://schema/file/path"),
            Some(&serde_json::json!("/tmp/test.txt"))
        );
        assert_eq!(
            mapped.get("iri://schema/file/encoding"),
            Some(&serde_json::json!("utf-8"))
        );
        assert!(!mapped.contains_key("unknown_param"));

        let empty_mapped = registry.map_input_params("iri://skills/non_existent", &params);
        assert!(empty_mapped.is_empty());
    }

    #[test]
    fn test_to_json_ld() {
        let registry = SkillRegistry::new();

        let jsonld = registry.to_json_ld("iri://skills/file_read");
        assert!(jsonld.is_some());

        let jsonld = jsonld.unwrap();
        assert_eq!(
            jsonld.get("@id").and_then(|v| v.as_str()),
            Some("iri://skills/file_read")
        );
        assert_eq!(
            jsonld.get("skill:name").and_then(|v| v.as_str()),
            Some("file_read")
        );

        let types = jsonld.get("@type").and_then(|v| v.as_array()).unwrap();
        assert!(types.contains(&serde_json::json!("skill:Skill")));
        assert!(types.contains(&serde_json::json!("iri://skill-types/FileOperation")));
        assert!(types.contains(&serde_json::json!("iri://skill-types/ReadOperation")));

        let input_mapping = jsonld
            .get("skill:inputMapping")
            .and_then(|v| v.as_object())
            .unwrap();
        assert!(input_mapping.contains_key("path"));
        assert!(input_mapping.contains_key("encoding"));

        let path_mapping = input_mapping
            .get("path")
            .and_then(|v| v.as_object())
            .unwrap();
        assert_eq!(
            path_mapping.get("@id").and_then(|v| v.as_str()),
            Some("iri://schema/file/path")
        );

        let none_jsonld = registry.to_json_ld("iri://skills/non_existent");
        assert!(none_jsonld.is_none());
    }

    #[test]
    fn test_skill_meta_fields() {
        let registry = SkillRegistry::new();

        let skill = registry.get_skill("iri://skills/file_read");
        assert!(skill.is_some());

        let skill = skill.unwrap();
        assert!(!skill.input_mapping.is_empty());
        assert!(!skill.output_mapping.is_empty());
        assert!(!skill.skill_types.is_empty());

        assert!(skill.input_mapping.contains_key("path"));
        assert!(skill.input_mapping.contains_key("encoding"));
        assert_eq!(
            skill.input_mapping.get("path"),
            Some(&"iri://schema/file/path".to_string())
        );

        assert!(skill.output_mapping.contains_key("content"));
        assert!(skill.output_mapping.contains_key("size"));

        assert!(skill
            .skill_types
            .contains(&"iri://skill-types/FileOperation".to_string()));
        assert!(skill
            .skill_types
            .contains(&"iri://skill-types/ReadOperation".to_string()));
    }

    #[test]
    fn test_parse_skill_from_jsonld_with_mappings() {
        let registry = SkillRegistry::new();

        let jsonld = serde_json::json!({
            "@id": "iri://skills/test_skill",
            "@type": ["skill:Skill", "iri://skill-types/TestOperation"],
            "skill:name": "test_skill",
            "skill:description": "A test skill",
            "skill:version": "1.0.0",
            "skill:category": "test",
            "skill:securityLevel": "normal",
            "skill:allowedRoles": ["DA"],
            "skill:inputSchema": {"type": "object"},
            "skill:outputSchema": {"type": "object"},
            "skill:inputMapping": {
                "param1": {
                    "@id": "iri://schema/test/param1",
                    "@type": "schema:Parameter"
                }
            },
            "skill:outputMapping": {
                "result": {
                    "@id": "iri://schema/test/result",
                    "@type": "schema:OutputField"
                }
            }
        });

        let skill = registry.parse_skill_from_jsonld(&jsonld);
        assert!(skill.is_ok());

        let skill = skill.unwrap();
        assert_eq!(skill.skill_iri, "iri://skills/test_skill");
        assert_eq!(skill.name, "test_skill");
        assert_eq!(skill.skill_types, vec!["iri://skill-types/TestOperation"]);
        assert_eq!(
            skill.input_mapping.get("param1"),
            Some(&"iri://schema/test/param1".to_string())
        );
        assert_eq!(
            skill.output_mapping.get("result"),
            Some(&"iri://schema/test/result".to_string())
        );
    }

    #[test]
    fn load_from_jsonld_dir_merges_skill_jsonld_subdirectories() {
        let dir = tempfile::tempdir().unwrap();
        let skills_root = dir.path().join("skills");
        let skill_dir = skills_root.join("dir_test");
        std::fs::create_dir_all(&skill_dir).unwrap();
        let file = skill_dir.join("skill.jsonld");
        std::fs::write(
            &file,
            r#"{
                "skills": [{
                    "@id": "iri://skills/dir_test",
                    "skill:name": "dir_test",
                    "skill:description": "Loaded from directory",
                    "skill:version": "1.0.0",
                    "skill:category": "external",
                    "skill:allowedRoles": ["DA"],
                    "skill:inputSchema": {"type": "object"},
                    "skill:outputSchema": {"type": "object"}
                }]
            }"#,
        )
        .unwrap();

        let registry = SkillRegistry::new();
        let loaded = registry.load_from_jsonld_dir(&skills_root).unwrap();
        assert_eq!(loaded, 1);
        let skill = registry.get_skill("iri://skills/dir_test").unwrap();
        assert_eq!(skill.name, "dir_test");
        assert_eq!(skill.category, "external");
        assert!(registry
            .list_all_skills()
            .iter()
            .any(|s| s.skill_iri == "iri://skills/dir_test"));
    }

    #[test]
    fn load_from_jsonld_dir_skips_leaf_files_and_missing_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        let skills_root = dir.path().join("skills");
        std::fs::create_dir_all(&skills_root).unwrap();
        std::fs::write(skills_root.join("stray.jsonld"), "{}").unwrap();
        let sub = skills_root.join("no_def");
        std::fs::create_dir_all(&sub).unwrap();

        let registry = SkillRegistry::new();
        let loaded = registry.load_from_jsonld_dir(&skills_root).unwrap();
        assert_eq!(loaded, 0);

        assert!(registry
            .load_from_jsonld_dir(dir.path().join("missing"))
            .is_err());
    }
}
