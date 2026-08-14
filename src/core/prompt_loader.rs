use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use serde_json::Value;
use tracing::debug;

use crate::templates::template_engine::TemplateEngine;

fn builtin_fallback(role: &str) -> &'static str {
    match role {
        "pa" => "You are the Planning Agent (PA). Analyze the task and create an execution plan. Output JSON-formatted results when done.",
        "da" => "You are the Doing Agent (DA). Execute the task and create artifacts. Prioritize using web_search for current information. Output JSON-formatted results when done.",
        "ca" => "You are the Checking Agent (CA). Verify whether execution results meet requirements. Output JSON-formatted results when done.",
        "aa" => "You are the Acting Agent (AA). Make final decisions and summaries based on audit results. Output JSON-formatted results when done.",
        _ => "",
    }
}

#[derive(Debug, Clone)]
pub struct PromptConfig {
    pub user_prefix: Option<String>,
    pub role_overrides: HashMap<String, String>,
    pub env_refs: Vec<String>,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            user_prefix: None,
            role_overrides: HashMap::new(),
            env_refs: Vec::new(),
        }
    }
}

pub struct PromptLoader {
    config: PromptConfig,
    engine: Arc<TemplateEngine>,
    /// (content, mtime) cache, indexed by file path
    file_cache: std::sync::Mutex<HashMap<PathBuf, (String, SystemTime)>>,
}

impl PromptLoader {
    pub fn new(config: PromptConfig, engine: Arc<TemplateEngine>) -> Self {
        Self {
            config,
            engine,
            file_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// File read cache with mtime check. Skips IO when file hasn't changed.
    fn read_cached(&self, path: &PathBuf) -> Option<String> {
        let metadata = std::fs::metadata(path).ok()?;
        let modified = metadata.modified().ok()?;

        if let Ok(cache) = self.file_cache.lock() {
            if let Some((cached_content, cached_mtime)) = cache.get(path) {
                if *cached_mtime == modified {
                    return Some(cached_content.clone());
                }
            }
        }

        let content = std::fs::read_to_string(path).ok()?;
        if let Ok(mut cache) = self.file_cache.lock() {
            cache.insert(path.clone(), (content.clone(), modified));
        }
        Some(content)
    }

    pub fn load(&self, role: &str, template: &str, vars: &HashMap<String, Value>) -> String {
        let fname = format!("{}/{}.md", role, template);

        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_default();
        if !home.is_empty() {
            let path = PathBuf::from(&home)
                .join(".gliding_horse")
                .join("prompts")
                .join(&fname);
            if let Some(content) = self.read_cached(&path) {
                debug!(role = %role, template = %template, source = "user", "prompt template hit");
                return self.post_process(role, &Self::render_string(&content, vars));
            }
        }

        let proj_path = PathBuf::from(".gliding_horse").join("prompts").join(&fname);
        if let Some(content) = self.read_cached(&proj_path) {
            debug!(role = %role, template = %template, source = "project", "prompt template hit");
            return self.post_process(role, &Self::render_string(&content, vars));
        }

        if let Ok(content) = self.engine.render_prompt(role, template, vars, false, None) {
            debug!(role = %role, template = %template, source = "engine", "prompt template hit");
            return self.post_process(role, &content);
        }

        debug!(
            role = %role,
            template = %template,
            "no prompt template found for {}/{}.md, using builtin fallback",
            role, template
        );
        let fallback = builtin_fallback(role);
        self.post_process(role, &Self::render_string(fallback, vars))
    }

    fn post_process(&self, role: &str, content: &str) -> String {
        let mut result = content.to_string();

        for ref_name in &self.config.env_refs {
            if let Ok(val) = std::env::var(ref_name) {
                result = result.replace(&format!("${{{}}}", ref_name), &val);
            }
        }

        if let Some(ref prefix) = self.config.user_prefix {
            result = format!("{}\n\n{}", prefix, result);
        }

        if let Some(ref override_content) = self.config.role_overrides.get(role) {
            result = format!(
                "{}\n\n---\n\n## User Additional Constraints\n{}",
                result, override_content
            );
        }

        result
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
}
