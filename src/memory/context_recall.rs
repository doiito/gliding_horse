//! Typed, bounded recall input.
//!
//! Task identity is useful for scoping and audit, but a generated IRI carries
//! no semantic signal. This value object keeps those roles separate so callers
//! cannot accidentally embed `iri://task/<uuid>` as the retrieval query.

use serde::{Deserialize, Serialize};

pub const CONTEXT_RECALL_QUERY_VERSION: u32 = 1;
const MAX_TOTAL_CHARS: usize = 6_000;
const MAX_FIELD_CHARS: usize = 1_500;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextRecallQuery {
    pub query_version: u32,
    pub task_iri: String,
    /// Bounded, normalized semantic text suitable for embedding and lexical
    /// recall. It intentionally excludes arbitrary constraints, prior agent
    /// output, tool arguments and raw interaction content.
    pub semantic_text: String,
    /// Stable names of input fields that actually contributed text. This is
    /// audit metadata, not a second copy of the original values.
    pub field_sources: Vec<String>,
}

impl ContextRecallQuery {
    pub fn from_fields<I, S>(task_iri: &str, fields: I) -> Self
    where
        I: IntoIterator<Item = (&'static str, S)>,
        S: AsRef<str>,
    {
        let mut text = String::new();
        let mut field_sources = Vec::new();
        for (name, raw) in fields {
            let normalized = normalize_and_bound(raw.as_ref(), MAX_FIELD_CHARS);
            if normalized.is_empty() || field_sources.iter().any(|existing| existing == name) {
                continue;
            }
            let prefix = if text.is_empty() { "" } else { "\n" };
            let remaining = MAX_TOTAL_CHARS.saturating_sub(text.chars().count());
            if remaining <= prefix.chars().count() {
                break;
            }
            let body_limit = remaining.saturating_sub(prefix.chars().count());
            text.push_str(prefix);
            text.push_str(&truncate_chars(&normalized, body_limit));
            field_sources.push(name.to_string());
        }
        Self {
            query_version: CONTEXT_RECALL_QUERY_VERSION,
            task_iri: task_iri.to_string(),
            semantic_text: text,
            field_sources,
        }
    }

    /// Legacy compatibility only. New task execution paths must construct a
    /// query from task semantics, not from the opaque IRI.
    pub fn legacy(task_iri: &str) -> Self {
        Self {
            query_version: CONTEXT_RECALL_QUERY_VERSION,
            task_iri: task_iri.to_string(),
            semantic_text: String::new(),
            field_sources: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.semantic_text.trim().is_empty()
    }
}

fn normalize_and_bound(value: &str, max_chars: usize) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_chars(&normalized, max_chars)
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_task_identity_is_not_used_as_semantic_query_text() {
        let query = ContextRecallQuery::from_fields(
            "iri://task/4d826d77-76c8-4e8e-a6c3-8eafbb836a6a",
            [("objective", "修复 parseHTTPResponse 的 E0425 错误")],
        );
        assert!(query.semantic_text.contains("parseHTTPResponse"));
        assert!(!query.semantic_text.contains("iri://task/"));
        assert_eq!(query.field_sources, vec!["objective"]);
    }

    #[test]
    fn recall_query_is_bounded_and_normalizes_whitespace() {
        let source = format!("  alpha\n\t{}  ", "x".repeat(MAX_FIELD_CHARS + 200));
        let query = ContextRecallQuery::from_fields("iri://task/t", [("objective", source)]);
        assert!(query.semantic_text.starts_with("alpha x"));
        assert!(query.semantic_text.chars().count() <= MAX_FIELD_CHARS);
    }
}
