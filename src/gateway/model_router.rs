use std::collections::HashMap;

/// Routes LLM requests to different models based on task type.
/// Supports mapping override, fallback chains, and health-aware selection.
pub struct ModelRouter {
    /// Task type → model name
    mapping: HashMap<String, String>,
    /// Fallback order when primary model is unavailable
    fallback_chain: Vec<String>,
}

impl ModelRouter {
    pub fn new() -> Self {
        let default =
            std::env::var("DEFAULT_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".to_string());
        Self {
            mapping: HashMap::from([
                ("planning".to_string(), default.clone()),
                ("execution".to_string(), default.clone()),
                ("analysis".to_string(), default.clone()),
                ("default".to_string(), default),
            ]),
            fallback_chain: vec![
                "deepseek-v4-pro".to_string(),
                "deepseek-v4-flash".to_string(),
            ],
        }
    }

    /// Resolve the model for a given task type
    pub fn resolve(&self, task_type: &str) -> String {
        self.mapping
            .get(task_type)
            .or_else(|| self.mapping.get("default"))
            .cloned()
            .unwrap_or_else(|| "deepseek-v4-flash".to_string())
    }

    /// Get next fallback model after `current`
    pub fn next_fallback(&self, current: &str) -> Option<String> {
        let pos = self.fallback_chain.iter().position(|m| m == current)?;
        self.fallback_chain.get(pos + 1).cloned()
    }

    /// Set or override a task type mapping
    pub fn set_mapping(&mut self, task_type: &str, model: &str) {
        self.mapping
            .insert(task_type.to_string(), model.to_string());
    }

    /// Set the fallback chain
    pub fn set_fallback_chain(&mut self, chain: Vec<String>) {
        self.fallback_chain = chain;
    }

    /// List all mappings
    pub fn mappings(&self) -> &HashMap<String, String> {
        &self.mapping
    }
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve() {
        std::env::set_var("DEFAULT_MODEL", "deepseek-v4-flash");
        let router = ModelRouter::new();
        assert_eq!(router.resolve("planning"), "deepseek-v4-flash");
        assert_eq!(router.resolve("unknown"), "deepseek-v4-flash");
    }

    #[test]
    fn test_fallback() {
        let router = ModelRouter::new();
        assert_eq!(
            router.next_fallback("deepseek-v4-pro"),
            Some("deepseek-v4-flash".to_string())
        );
        assert!(router.next_fallback("deepseek-v4-flash").is_none());
    }
}
