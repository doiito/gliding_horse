use std::collections::HashMap;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::CoreError;

#[derive(Debug, Clone)]
pub struct Template {
    pub name: String,
    pub content: String,
    pub source_path: Option<PathBuf>,
    pub role: String,
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub name: String,
    pub schema: Value,
    pub source_path: Option<PathBuf>,
}

pub struct TemplateManager {
    templates: RwLock<HashMap<String, Template>>,
    schemas: RwLock<HashMap<String, Schema>>,
    template_dir: PathBuf,
}

impl TemplateManager {
    pub fn new(template_dir: &Path) -> Result<Self, CoreError> {
        let manager = Self {
            templates: RwLock::new(HashMap::new()),
            schemas: RwLock::new(HashMap::new()),
            template_dir: template_dir.to_path_buf(),
        };

        manager.load_all()?;
        info!(
            dir = %template_dir.display(),
            "TemplateManager initialized"
        );
        Ok(manager)
    }

    pub fn load_all(&self) -> Result<usize, CoreError> {
        let mut count = 0;

        if !self.template_dir.exists() {
            debug!(dir = %self.template_dir.display(), "Template directory does not exist");
            return Ok(0);
        }

        if let Ok(entries) = std::fs::read_dir(&self.template_dir) {
            for entry in entries.flatten() {
                // file_type() does not follow symlinks, so self-referential
                // links in the template dir cannot cause unbounded recursion
                let Ok(ft) = entry.file_type() else { continue };
                if !ft.is_dir() {
                    continue;
                }
                let path = entry.path();
                let dir_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");

                match dir_name {
                    "prompts" => {
                        count += self.load_prompts_from_dir(&path)?;
                    }
                    "schemas" => {
                        count += self.load_schemas_from_dir(&path)?;
                    }
                    _ => {
                        count += self.load_prompts_from_dir(&path)?;
                    }
                }
            }
        }

        info!(count = count, dir = %self.template_dir.display(), "Templates and schemas loaded");
        Ok(count)
    }

    fn load_prompts_from_dir(&self, dir: &Path) -> Result<usize, CoreError> {
        let mut count = 0;

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let Ok(ft) = entry.file_type() else { continue };
                let path = entry.path();
                if ft.is_dir() {
                    count += self.load_prompts_from_dir(&path)?;
                } else if ft.is_file() && path.extension().is_some_and(|e| e == "md") {
                    let relative = path.strip_prefix(&self.template_dir).unwrap_or(&path);
                    let template_name = relative
                        .to_string_lossy()
                        .replace('\\', "/")
                        .trim_end_matches(".md")
                        .to_string();

                    let content =
                        std::fs::read_to_string(&path).map_err(|e| CoreError::Internal {
                            message: format!("Failed to read template {}: {}", path.display(), e),
                        })?;

                    let role = relative
                        .iter()
                        .nth(1)
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    self.add_template_internal(Template {
                        name: template_name.clone(),
                        content,
                        source_path: Some(path),
                        role,
                    });
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    fn load_schemas_from_dir(&self, dir: &Path) -> Result<usize, CoreError> {
        let mut count = 0;

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let Ok(ft) = entry.file_type() else { continue };
                let path = entry.path();
                if ft.is_dir() {
                    count += self.load_schemas_from_dir(&path)?;
                } else if ft.is_file() && path.extension().is_some_and(|e| e == "json") {
                    let relative = path.strip_prefix(&self.template_dir).unwrap_or(&path);
                    let schema_name = relative
                        .to_string_lossy()
                        .replace('\\', "/")
                        .trim_end_matches(".json")
                        .to_string();

                    let content =
                        std::fs::read_to_string(&path).map_err(|e| CoreError::Internal {
                            message: format!("Failed to read schema {}: {}", path.display(), e),
                        })?;

                    let schema: Value =
                        serde_json::from_str(&content).map_err(|e| CoreError::InvalidJsonLd {
                            message: format!("Invalid JSON schema {}: {}", path.display(), e),
                        })?;

                    self.add_schema_internal(Schema {
                        name: schema_name.clone(),
                        schema,
                        source_path: Some(path),
                    });
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    fn add_template_internal(&self, template: Template) {
        let mut templates = self.templates.write();
        templates.insert(template.name.clone(), template);
    }

    fn add_schema_internal(&self, schema: Schema) {
        let mut schemas = self.schemas.write();
        schemas.insert(schema.name.clone(), schema);
    }

    pub fn add_template(&self, name: &str, content: &str, role: &str) {
        let template = Template {
            name: name.to_string(),
            content: content.to_string(),
            source_path: None,
            role: role.to_string(),
        };
        self.add_template_internal(template);
        debug!(template = %name, "Template registered");
    }

    pub fn add_schema(&self, name: &str, schema: Value) {
        let schema = Schema {
            name: name.to_string(),
            schema,
            source_path: None,
        };
        self.add_schema_internal(schema);
        debug!(schema = %name, "Schema registered");
    }

    pub fn get_template(&self, name: &str) -> Option<Template> {
        self.templates.read().get(name).cloned()
    }

    pub fn get_schema(&self, name: &str) -> Option<Schema> {
        self.schemas.read().get(name).cloned()
    }

    pub fn load_prompt_template(
        &self,
        agent_type: &str,
        template_name: &str,
    ) -> Result<String, CoreError> {
        let key = if agent_type == "sa" {
            format!("prompts/sa/{}", template_name)
        } else {
            format!("prompts/workers/{}/{}", agent_type, template_name)
        };

        self.templates
            .read()
            .get(&key)
            .map(|t| t.content.clone())
            .or_else(|| {
                let alt_key = format!("{}/{}", agent_type, template_name);
                self.templates
                    .read()
                    .get(&alt_key)
                    .map(|t| t.content.clone())
            })
            .ok_or_else(|| CoreError::Internal {
                message: format!("Template not found: {}/{}", agent_type, template_name),
            })
    }

    pub fn load_schema(&self, agent_type: &str, schema_name: &str) -> Result<Value, CoreError> {
        let key = if agent_type == "sa" {
            format!("schemas/sa/{}", schema_name)
        } else if agent_type == "workers" {
            format!("schemas/workers/{}", schema_name)
        } else {
            format!("schemas/{}/{}", agent_type, schema_name)
        };

        self.schemas
            .read()
            .get(&key)
            .map(|s| s.schema.clone())
            .or_else(|| {
                let alt_key = format!("{}/{}", agent_type, schema_name);
                self.schemas.read().get(&alt_key).map(|s| s.schema.clone())
            })
            .or_else(|| {
                let direct_key = format!("schemas/{}", schema_name);
                self.schemas
                    .read()
                    .get(&direct_key)
                    .map(|s| s.schema.clone())
            })
            .ok_or_else(|| CoreError::Internal {
                message: format!("Schema not found: {}/{}", agent_type, schema_name),
            })
    }

    pub fn load_base_schema(&self) -> Result<Value, CoreError> {
        self.load_schema("base", "base")
            .or_else(|_| self.load_schema("", "base"))
            .or_else(|_| {
                self.schemas
                    .read()
                    .get("schemas/base")
                    .map(|s| s.schema.clone())
                    .ok_or_else(|| CoreError::Internal {
                        message: "Base schema not found".to_string(),
                    })
            })
    }

    pub fn render_prompt(
        &self,
        agent_type: &str,
        template_name: &str,
        context: &HashMap<String, Value>,
        include_schema: bool,
        schema_name: Option<&str>,
    ) -> Result<String, CoreError> {
        let template = self.load_prompt_template(agent_type, template_name)?;

        let mut context = context.clone();

        if include_schema {
            let schema_key = schema_name.unwrap_or(template_name);
            if let Ok(schema) = self.load_schema(agent_type, schema_key) {
                context.insert(
                    "output_schema".to_string(),
                    serde_json::to_string_pretty(&schema)
                        .unwrap_or_default()
                        .into(),
                );
            } else {
                warn!(agent_type = %agent_type, schema = %schema_key, "Schema not found, skipping schema inclusion");
            }
        }

        Ok(Self::render_string(&template, &context))
    }

    pub fn render_string(template: &str, variables: &HashMap<String, Value>) -> String {
        let mut result = template.to_string();
        for (key, value) in variables {
            let placeholder = format!("{{{}}}", key);
            let replacement = match value {
                Value::String(s) => s.clone(),
                Value::Object(_) | Value::Array(_) => {
                    serde_json::to_string_pretty(value).unwrap_or_default()
                }
                Value::Null => String::new(),
                _ => value.to_string(),
            };
            result = result.replace(&placeholder, &replacement);
        }
        result
    }

    pub fn validate_output(
        &self,
        agent_type: &str,
        schema_name: &str,
        output: &Value,
    ) -> Result<bool, CoreError> {
        let schema = self.load_schema(agent_type, schema_name)?;

        let compiled = jsonschema::JSONSchema::options()
            .compile(&schema)
            .map_err(|e| CoreError::Internal {
                message: format!("Schema compilation error: {}", e),
            })?;

        let result = compiled.validate(output);
        match result {
            Ok(_) => Ok(true),
            Err(errors) => {
                let error_messages: Vec<String> = errors
                    .map(|e| format!("{}: {}", e.instance_path, e))
                    .collect();
                warn!(errors = ?error_messages, "Output validation failed");
                Ok(false)
            }
        }
    }

    pub fn list_available_templates(&self, agent_type: &str) -> Vec<String> {
        let prefix = if agent_type == "sa" {
            "prompts/sa/"
        } else {
            &format!("prompts/workers/{}/", agent_type)
        };

        self.templates
            .read()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .map(|k| k.trim_start_matches(prefix).to_string())
            .collect()
    }

    pub fn list_available_schemas(&self, agent_type: &str) -> Vec<String> {
        let prefix = if agent_type == "sa" {
            "schemas/sa/"
        } else if agent_type == "workers" {
            "schemas/workers/"
        } else {
            &format!("schemas/{}/", agent_type)
        };

        self.schemas
            .read()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .map(|k| k.trim_start_matches(prefix).to_string())
            .collect()
    }

    pub fn reload_templates(&self) -> Result<usize, CoreError> {
        self.templates.write().clear();
        self.schemas.write().clear();
        self.load_all()
    }

    pub fn get_template_path(&self, agent_type: &str, template_name: &str) -> PathBuf {
        if agent_type == "sa" {
            self.template_dir
                .join("prompts")
                .join("sa")
                .join(format!("{}.md", template_name))
        } else {
            self.template_dir
                .join("prompts")
                .join("workers")
                .join(agent_type)
                .join(format!("{}.md", template_name))
        }
    }

    pub fn get_schema_path(&self, agent_type: &str, schema_name: &str) -> PathBuf {
        if agent_type == "sa" {
            self.template_dir
                .join("schemas")
                .join("sa")
                .join(format!("{}.json", schema_name))
        } else if agent_type == "workers" {
            self.template_dir
                .join("schemas")
                .join("workers")
                .join(format!("{}.json", schema_name))
        } else {
            self.template_dir
                .join("schemas")
                .join(agent_type)
                .join(format!("{}.json", schema_name))
        }
    }

    pub fn template_count(&self) -> usize {
        self.templates.read().len()
    }

    pub fn schema_count(&self) -> usize {
        self.schemas.read().len()
    }
}

pub type TemplateEngine = TemplateManager;

pub fn build_system_prompt(
    engine: &TemplateEngine,
    role: &str,
    task_description: &str,
    available_skills: &[String],
    context_summary: &str,
    additional_constraints: &HashMap<String, String>,
) -> Result<String, CoreError> {
    let mut vars = HashMap::new();
    vars.insert(
        "task_description".to_string(),
        Value::String(task_description.to_string()),
    );
    vars.insert(
        "available_skills".to_string(),
        Value::String(available_skills.join(", ")),
    );
    vars.insert(
        "context_summary".to_string(),
        Value::String(context_summary.to_string()),
    );
    for (k, v) in additional_constraints {
        vars.insert(k.clone(), Value::String(v.clone()));
    }

    let template_name = if role.to_lowercase() == "sa" {
        "system"
    } else {
        "system"
    };

    engine.render_prompt(role, template_name, &vars, false, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_render_string() {
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), Value::String("World".to_string()));
        vars.insert("greeting".to_string(), Value::String("Hello".to_string()));

        let result = TemplateManager::render_string("{greeting}, {name}!", &vars);
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_render_with_json() {
        let mut vars = HashMap::new();
        vars.insert(
            "config".to_string(),
            Value::Object(
                vec![("key".to_string(), Value::String("value".to_string()))]
                    .into_iter()
                    .collect(),
            ),
        );

        let result = TemplateManager::render_string("Config: {config}", &vars);
        assert!(result.contains("key"));
        assert!(result.contains("value"));
    }

    #[test]
    fn test_add_and_get() {
        let engine = TemplateManager::new(Path::new("/nonexistent")).unwrap();
        engine.add_template(
            "da/system",
            "You are a DA agent. Task: {task_description}",
            "da",
        );

        let template = engine.get_template("da/system");
        assert!(template.is_some());
        assert_eq!(template.unwrap().name, "da/system");
    }

    #[test]
    fn test_add_and_get_schema() {
        let engine = TemplateManager::new(Path::new("/nonexistent")).unwrap();
        engine.add_schema(
            "test/output",
            json!({
                "type": "object",
                "properties": {
                    "status": {"type": "string"}
                }
            }),
        );

        let schema = engine.get_schema("test/output");
        assert!(schema.is_some());
    }

    #[test]
    fn test_validate_output() {
        let engine = TemplateManager::new(Path::new("/nonexistent")).unwrap();
        engine.add_schema(
            "test/output",
            json!({
                "type": "object",
                "properties": {
                    "status": {"type": "string"}
                },
                "required": ["status"]
            }),
        );

        let valid = json!({"status": "success"});
        let result = engine.validate_output("test", "output", &valid).unwrap();
        assert!(result);

        let invalid = json!({"other": "value"});
        let result = engine.validate_output("test", "output", &invalid).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_render_template() {
        let engine = TemplateManager::new(Path::new("/nonexistent")).unwrap();
        engine.add_template(
            "sa/system",
            "Role: SA\nTask: {task_description}\nSkills: {available_skills}",
            "sa",
        );

        let mut vars = HashMap::new();
        vars.insert(
            "task_description".to_string(),
            Value::String("Build a web app".to_string()),
        );
        vars.insert(
            "available_skills".to_string(),
            Value::String("file_read, http_request".to_string()),
        );

        let result = engine
            .render_prompt("sa", "system", &vars, false, None)
            .unwrap();
        assert!(result.contains("Build a web app"));
        assert!(result.contains("file_read, http_request"));
    }

    #[test]
    fn test_new_skips_symlink_loop() {
        #[cfg(unix)]
        {
            let dir = tempfile::TempDir::new().unwrap();
            std::os::unix::fs::symlink(dir.path(), dir.path().join("self")).unwrap();
            let engine = TemplateManager::new(dir.path()).unwrap();
            assert_eq!(engine.template_count(), 0);
        }
        #[cfg(not(unix))]
        {
            let engine = TemplateManager::new(Path::new("/nonexistent")).unwrap();
            assert_eq!(engine.template_count(), 0);
        }
    }
}
