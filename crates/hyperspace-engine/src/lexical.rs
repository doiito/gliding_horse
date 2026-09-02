//! Deterministic, in-process lexical index derived from persisted JSON-LD payloads.
//!
//! The index is deliberately a cache rather than a second source of truth:
//! only payloads that explicitly set `_gh_lexical_recall: true` participate,
//! and the index is rebuilt from durable engine metadata on open. This keeps
//! recovery simple and prevents ordinary/raw vector records from silently
//! becoming keyword-searchable.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use roaring::RoaringBitmap;
use serde_json::Value;

pub const LEXICAL_RECALL_FLAG: &str = "_gh_lexical_recall";
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;
const MAX_TERMS_PER_DOCUMENT: usize = 8_192;
const MAX_QUERY_TERMS: usize = 128;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct LexicalDocument {
    term_frequencies: BTreeMap<String, u16>,
    term_count: u32,
}

/// Local BM25-style inverted index. It stores normalized tokens only; source
/// text remains in the engine payload governed by the caller's allow-list.
#[derive(Debug, Default)]
pub struct LexicalIndex {
    documents: HashMap<u32, LexicalDocument>,
    postings: HashMap<String, BTreeMap<u32, u16>>,
    total_terms: u64,
}

impl LexicalIndex {
    pub fn rebuild<'a>(&mut self, payloads: impl IntoIterator<Item = (u32, &'a Value)>) {
        self.documents.clear();
        self.postings.clear();
        self.total_terms = 0;
        for (id, payload) in payloads {
            self.upsert_payload(id, payload);
        }
    }

    pub fn upsert_payload(&mut self, id: u32, payload: &Value) {
        self.remove(id);
        if payload.get(LEXICAL_RECALL_FLAG).and_then(Value::as_bool) != Some(true) {
            return;
        }
        let Some(text) = payload.get("text").and_then(Value::as_str) else {
            return;
        };
        let terms = normalized_terms(text, MAX_TERMS_PER_DOCUMENT);
        if terms.is_empty() {
            return;
        }

        let mut frequencies = BTreeMap::<String, u16>::new();
        for term in terms {
            let frequency = frequencies.entry(term).or_insert(0);
            *frequency = frequency.saturating_add(1);
        }
        let term_count = frequencies.values().map(|value| *value as u32).sum();
        for (term, frequency) in &frequencies {
            self.postings
                .entry(term.clone())
                .or_default()
                .insert(id, *frequency);
        }
        self.total_terms = self.total_terms.saturating_add(term_count as u64);
        self.documents.insert(
            id,
            LexicalDocument {
                term_frequencies: frequencies,
                term_count,
            },
        );
    }

    pub fn remove(&mut self, id: u32) {
        let Some(existing) = self.documents.remove(&id) else {
            return;
        };
        self.total_terms = self.total_terms.saturating_sub(existing.term_count as u64);
        for term in existing.term_frequencies.keys() {
            let remove_posting = if let Some(posting) = self.postings.get_mut(term) {
                posting.remove(&id);
                posting.is_empty()
            } else {
                false
            };
            if remove_posting {
                self.postings.remove(term);
            }
        }
    }

    pub fn search(
        &self,
        query: &str,
        limit: usize,
        allowed: Option<&RoaringBitmap>,
    ) -> Vec<(u32, f32)> {
        if limit == 0 || self.documents.is_empty() {
            return Vec::new();
        }
        let terms = normalized_terms(query, MAX_QUERY_TERMS);
        if terms.is_empty() {
            return Vec::new();
        }
        let unique_terms = terms.into_iter().collect::<BTreeSet<_>>();
        let eligible_count = match allowed {
            Some(bitmap) => self
                .documents
                .keys()
                .filter(|id| bitmap.contains(**id))
                .count(),
            None => self.documents.len(),
        };
        if eligible_count == 0 {
            return Vec::new();
        }
        let total_terms = match allowed {
            Some(bitmap) => self
                .documents
                .iter()
                .filter(|(id, _)| bitmap.contains(**id))
                .map(|(_, doc)| doc.term_count as u64)
                .sum::<u64>(),
            None => self.total_terms,
        };
        let average_length = (total_terms as f64 / eligible_count as f64).max(1.0);
        let mut scores = HashMap::<u32, f64>::new();

        for term in unique_terms {
            let Some(posting) = self.postings.get(&term) else {
                continue;
            };
            let document_frequency = posting
                .keys()
                .filter(|id| match allowed {
                    Some(bitmap) => bitmap.contains(**id),
                    None => true,
                })
                .count();
            if document_frequency == 0 {
                continue;
            }
            let idf = ((eligible_count as f64 - document_frequency as f64 + 0.5)
                / (document_frequency as f64 + 0.5)
                + 1.0)
                .ln();
            for (id, frequency) in posting {
                if allowed.is_some_and(|bitmap| !bitmap.contains(*id)) {
                    continue;
                }
                let Some(document) = self.documents.get(id) else {
                    continue;
                };
                let tf = *frequency as f64;
                let normalization = tf
                    + BM25_K1
                        * (1.0 - BM25_B + BM25_B * document.term_count as f64 / average_length);
                let contribution = idf * (tf * (BM25_K1 + 1.0)) / normalization.max(f64::EPSILON);
                *scores.entry(*id).or_insert(0.0) += contribution;
            }
        }

        let mut ranked = scores.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(limit);
        ranked
            .into_iter()
            .map(|(id, score)| (id, score as f32))
            .collect()
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }
}

/// Unicode-aware enough for identifiers and CJK without making whitespace a
/// correctness dependency. It intentionally preserves contiguous CJK phrases
/// and also indexes individual CJK characters for partial query recall.
pub fn normalized_terms(input: &str, max_terms: usize) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut previous_was_lower_or_digit = false;

    let flush = |current: &mut String, terms: &mut Vec<String>| {
        if !current.is_empty() && current.chars().count() >= 2 {
            terms.push(std::mem::take(current));
        } else {
            current.clear();
        }
    };

    for character in input.chars() {
        if terms.len() >= max_terms {
            break;
        }
        if is_cjk(character) {
            flush(&mut current, &mut terms);
            terms.push(character.to_string());
            previous_was_lower_or_digit = false;
            continue;
        }
        if character.is_alphanumeric() {
            let starts_upper = character.is_uppercase();
            if starts_upper && previous_was_lower_or_digit && !current.is_empty() {
                flush(&mut current, &mut terms);
            }
            for lowered in character.to_lowercase() {
                current.push(lowered);
            }
            previous_was_lower_or_digit = character.is_lowercase() || character.is_ascii_digit();
        } else {
            flush(&mut current, &mut terms);
            previous_was_lower_or_digit = false;
        }
    }
    flush(&mut current, &mut terms);
    terms.truncate(max_terms);
    terms
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2EBEF
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tokenizer_splits_code_identifiers_and_cjk_without_whitespace_dependency() {
        let terms = normalized_terms("parseHTTPResponse src/foo-bar.rs 修复检索", 64);
        assert!(terms.contains(&"parse".to_string()));
        assert!(terms.contains(&"httpresponse".to_string()));
        assert!(terms.contains(&"src".to_string()));
        assert!(terms.contains(&"foo".to_string()));
        assert!(terms.contains(&"检".to_string()));
        assert!(terms.contains(&"索".to_string()));
    }

    #[test]
    fn upsert_and_delete_do_not_leave_stale_postings() {
        let mut index = LexicalIndex::default();
        index.upsert_payload(
            1,
            &json!({LEXICAL_RECALL_FLAG: true, "text": "firstAlbatross"}),
        );
        assert_eq!(index.search("firstAlbatross", 10, None)[0].0, 1);
        index.upsert_payload(
            1,
            &json!({LEXICAL_RECALL_FLAG: true, "text": "secondOtter"}),
        );
        assert!(index.search("firstAlbatross", 10, None).is_empty());
        assert_eq!(index.search("secondOtter", 10, None)[0].0, 1);
        index.remove(1);
        assert!(index.search("secondOtter", 10, None).is_empty());
    }

    #[test]
    fn lexical_search_honors_hard_filter_membership() {
        let mut index = LexicalIndex::default();
        index.upsert_payload(
            1,
            &json!({LEXICAL_RECALL_FLAG: true, "text": "rust error E0425"}),
        );
        index.upsert_payload(
            2,
            &json!({LEXICAL_RECALL_FLAG: true, "text": "rust error E0425"}),
        );
        let allowed = RoaringBitmap::from_iter([2]);
        let results = index.search("E0425", 10, Some(&allowed));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }
}
