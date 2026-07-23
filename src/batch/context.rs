use std::sync::Arc;

use serde_json::Value;
use tracing::{debug, warn};

use crate::batch::error::BatchError;
use crate::batch::types::{BatchAgentConfig, EmphasisItem, PromptContext};
use crate::batch::window::SlidingWindow;
use crate::knowledge_graph::store::KnowledgeGraphStore;
use crate::memory::l0_store::L0Store;
use crate::memory::l3_projection::ProjectionEngine;

pub struct ContextCollector {
    l0_store: Option<Arc<L0Store>>,
    projection: Option<Arc<ProjectionEngine>>,
    kg_store: Option<Arc<KnowledgeGraphStore>>,
}

impl ContextCollector {
    pub fn new(
        l0_store: Option<Arc<L0Store>>,
        projection: Option<Arc<ProjectionEngine>>,
        kg_store: Option<Arc<KnowledgeGraphStore>>,
    ) -> Self {
        Self {
            l0_store,
            projection,
            kg_store,
        }
    }

    pub async fn collect(
        &self,
        config: &BatchAgentConfig,
        window: &SlidingWindow,
    ) -> PromptContext {
        let mut context = PromptContext::default();

        // 1. User reminders from L0 emphasis
        if config.inject_user_reminders {
            match self.load_user_reminders(10).await {
                Ok(items) => {
                    context.user_reminders = items;
                    debug!(
                        agent = %config.name,
                        count = %context.user_reminders.len(),
                        "User reminders loaded"
                    );
                }
                Err(e) => {
                    warn!(error = %e, "Failed to load user reminders");
                }
            }
        }

        // 2. Conversation context from L3 projection
        if config.inject_context_summary {
            let task_iri = format!("batch://{}", config.name);
            match self.load_context_summary(&task_iri).await {
                Ok(Some(summary)) => {
                    context.context_summary = Some(summary);
                    debug!(
                        agent = %config.name,
                        "Context summary loaded from projection"
                    );
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "Failed to load context summary");
                }
            }
        }

        // 3. Related entities from knowledge graph
        if config.inject_related_entities {
            let keywords = self.extract_window_keywords(window);
            match self.load_related_entities(&keywords, 5).await {
                Ok(entities) => {
                    context.related_entities = entities;
                    debug!(
                        agent = %config.name,
                        count = %context.related_entities.len(),
                        "Related entities loaded from KG"
                    );
                }
                Err(e) => {
                    warn!(error = %e, "Failed to load related entities");
                }
            }
        }

        // 4. Window summary (if maintained)
        context.window_summary = window.get_running_summary().map(|s| s.to_string());

        context
    }

    async fn load_user_reminders(&self, limit: usize) -> Result<Vec<EmphasisItem>, BatchError> {
        let l0 = match self.l0_store.as_ref() {
            Some(s) => s,
            None => return Ok(vec![]),
        };

        let entries = l0.search_by_tags(&["emphasis".to_string()]).map_err(|e| {
            BatchError::MemoryOperationFailed {
                message: format!("L0 search_by_tags failed: {}", e),
            }
        })?;

        let mut items: Vec<EmphasisItem> = entries
            .into_iter()
            .filter_map(|entry| {
                let parsed: Value = serde_json::from_str(&entry.content).ok()?;
                let text = parsed
                    .get("emphasis_text")
                    .or_else(|| parsed.get("text"))
                    .and_then(|v| v.as_str())?;
                let importance = parsed
                    .get("importance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5);
                let created_at = entry.last_accessed.max(entry.created_at);
                Some(EmphasisItem {
                    text: text.to_string(),
                    importance,
                    created_at,
                })
            })
            .collect();

        items.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(limit);

        Ok(items)
    }

    async fn load_context_summary(&self, task_iri: &str) -> Result<Option<String>, BatchError> {
        let projection = match self.projection.as_ref() {
            Some(p) => p,
            None => return Ok(None),
        };

        let params = std::collections::HashMap::new();
        match projection.project(task_iri, "summary_only", params).await {
            Ok(json_str) => {
                // Try to extract a meaningful summary from the projection JSON
                let parsed: Value =
                    serde_json::from_str(&json_str).unwrap_or(Value::String(json_str.clone()));
                let summary = parsed
                    .get("summary")
                    .or_else(|| parsed.get("content"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        // Fall back to the entire projection string if it's short enough
                        if json_str.len() < 500 {
                            Some(json_str)
                        } else {
                            None
                        }
                    });
                Ok(summary)
            }
            Err(e) => {
                debug!(
                    task_iri = %task_iri,
                    error = %e,
                    "Projection returned error (may not exist yet)"
                );
                Ok(None)
            }
        }
    }

    async fn load_related_entities(
        &self,
        keywords: &[String],
        limit: usize,
    ) -> Result<Vec<String>, BatchError> {
        let kg = match self.kg_store.as_ref() {
            Some(s) => s,
            None => return Ok(vec![]),
        };

        let mut related = Vec::new();
        for keyword in keywords.iter().take(5) {
            match kg.search_entities(keyword, None) {
                Ok(results) => {
                    for result in results {
                        if let Some(label) = result
                            .get("?label")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                        {
                            if !related.contains(&label) {
                                related.push(label);
                                if related.len() >= limit {
                                    return Ok(related);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    debug!(keyword = %keyword, error = %e, "KG search failed");
                }
            }
        }

        Ok(related)
    }

    fn extract_window_keywords(&self, window: &SlidingWindow) -> Vec<String> {
        let mut word_freq: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for entry in window.entries() {
            for word in entry.content.split_whitespace() {
                let cleaned: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                if cleaned.len() > 3 {
                    *word_freq.entry(cleaned.to_lowercase()).or_insert(0) += 1;
                }
            }
        }

        let mut words: Vec<(String, usize)> = word_freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1));
        words.into_iter().take(5).map(|(w, _)| w).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::types::WindowEntry;
    use crate::batch::window::{SlidingWindow, WindowConfig};
    use chrono::Utc;

    fn make_entry(content: &str) -> WindowEntry {
        WindowEntry {
            message_id: "test_msg".into(),
            role: "user".into(),
            content: content.to_string(),
            timestamp: Utc::now(),
            estimated_intent: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_extract_window_keywords() {
        let collector = ContextCollector::new(None, None, None);
        let mut window = SlidingWindow::new(WindowConfig {
            max_entries: 10,
            min_entries: 1,
            time_window_secs: 600,
            intent_shift_threshold: 0.3,
        });

        window
            .push(make_entry("Rust backend framework performance"))
            .unwrap();
        window
            .push(make_entry("PostgreSQL database schema design"))
            .unwrap();
        window
            .push(make_entry("Rust async web framework comparison"))
            .unwrap();

        let keywords = collector.extract_window_keywords(&window);
        assert!(!keywords.is_empty(), "Should extract keywords");
        assert!(
            keywords.contains(&"framework".to_string()),
            "framework should be a top keyword"
        );
    }

    #[tokio::test]
    async fn test_collect_with_no_stores() {
        let collector = ContextCollector::new(None, None, None);
        let config = BatchAgentConfig::default();
        let window = SlidingWindow::new(WindowConfig {
            max_entries: 10,
            min_entries: 1,
            time_window_secs: 600,
            intent_shift_threshold: 0.3,
        });

        let ctx = collector.collect(&config, &window).await;
        assert!(ctx.user_reminders.is_empty());
        assert!(ctx.context_summary.is_none());
        assert!(ctx.related_entities.is_empty());
    }
}
