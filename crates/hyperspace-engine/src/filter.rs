use roaring::RoaringBitmap;

use crate::jsonld_meta::JsonLdMetadataIndex;

/// JSON-LD aware filter expressions for search.
#[derive(Debug, Clone)]
pub enum JsonLdFilter {
    Type(String),
    NamedGraph(String),
    Context(String),
    Tag {
        key: String,
        value: String,
    },
    Range {
        key: String,
        gte: Option<f64>,
        lte: Option<f64>,
    },
    Must(Vec<JsonLdFilter>),
    Should(Vec<JsonLdFilter>),
    MustNot(Vec<JsonLdFilter>),
}

impl JsonLdFilter {
    pub fn tag(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Tag {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// Compiled filter for fast membership testing against bitmaps.
#[derive(Debug, Clone)]
pub enum CompiledFilter {
    MatchTag(String, String),
    MatchType(String),
    MatchGraph(String),
    MatchContext(String),
    Range {
        key: String,
        gte: Option<f64>,
        lte: Option<f64>,
    },
    And(Vec<CompiledFilter>),
    Or(Vec<CompiledFilter>),
    Not(Box<CompiledFilter>),
}

/// Result of evaluating the search filter set.
///
/// `Unrestricted` and `Empty` must remain distinct: treating an empty match
/// as no filter silently exposes unrelated vectors.
#[derive(Debug, Clone)]
pub enum FilterEvaluation {
    Unrestricted,
    Matched(RoaringBitmap),
    Empty,
}

impl CompiledFilter {
    /// Evaluate this filter against the metadata index, returning a RoaringBitmap
    /// of matching IDs. Returns None if the filter would match nothing.
    pub fn evaluate(&self, meta: &JsonLdMetadataIndex) -> Option<RoaringBitmap> {
        let result = self.evaluate_bitmap(meta);
        (!result.is_empty()).then_some(result)
    }

    /// Evaluate to a bitmap without collapsing the empty set into `None`.
    fn evaluate_bitmap(&self, meta: &JsonLdMetadataIndex) -> RoaringBitmap {
        match self {
            Self::MatchTag(key, val) => meta.ids_for_tag(key, val).unwrap_or_default(),
            Self::MatchType(typ) => meta.ids_for_type(typ).unwrap_or_default(),
            Self::MatchGraph(g) => meta.ids_for_graph(g).unwrap_or_default(),
            Self::MatchContext(ctx) => meta.ids_for_context(ctx).unwrap_or_default(),
            Self::Range { key, gte, lte } => meta
                .ids_for_numeric_range(key, *gte, *lte)
                .unwrap_or_default(),
            Self::And(conds) => {
                let mut result: RoaringBitmap = meta.all_ids().into_iter().collect();
                for cond in conds {
                    result &= cond.evaluate_bitmap(meta);
                    if result.is_empty() {
                        break;
                    }
                }
                result
            }
            Self::Or(conds) => {
                let mut result = RoaringBitmap::new();
                for c in conds {
                    result |= c.evaluate_bitmap(meta);
                }
                result
            }
            Self::Not(cond) => {
                // Not is tricky: we need the universe of live IDs minus matching ones
                let all: RoaringBitmap = meta.all_ids().iter().collect();
                all - cond.evaluate_bitmap(meta)
            }
        }
    }

    /// Check if a set of tags/values satisfies this filter (legacy function-based path).
    /// `has_tag` returns true if the key:value pair exists.
    /// `get_numeric` returns numeric value for a key.
    pub fn check(
        &self,
        has_tag: &impl Fn(&str, &str) -> bool,
        get_numeric: &impl Fn(&str) -> Option<f64>,
        has_type: &impl Fn(&str) -> bool,
        has_graph: &impl Fn(&str) -> bool,
    ) -> bool {
        match self {
            Self::MatchTag(key, val) => has_tag(key, val),
            Self::MatchType(typ) => has_type(typ),
            Self::MatchGraph(g) => has_graph(g),
            Self::MatchContext(_) => true,
            Self::Range { key, gte, lte } => {
                if let Some(val) = get_numeric(key) {
                    let ok_ge = gte.is_none_or(|g| val >= g);
                    let ok_le = lte.is_none_or(|l| val <= l);
                    ok_ge && ok_le
                } else {
                    false
                }
            }
            Self::And(conds) => conds
                .iter()
                .all(|c| c.check(has_tag, get_numeric, has_type, has_graph)),
            Self::Or(conds) => conds
                .iter()
                .any(|c| c.check(has_tag, get_numeric, has_type, has_graph)),
            Self::Not(cond) => !cond.check(has_tag, get_numeric, has_type, has_graph),
        }
    }
}

/// Convert JsonLdFilter to CompiledFilter.
pub fn compile_filter(f: &JsonLdFilter) -> CompiledFilter {
    match f {
        JsonLdFilter::Tag { key, value } => CompiledFilter::MatchTag(key.clone(), value.clone()),
        JsonLdFilter::Type(t) => CompiledFilter::MatchType(t.clone()),
        JsonLdFilter::NamedGraph(g) => CompiledFilter::MatchGraph(g.clone()),
        JsonLdFilter::Context(c) => CompiledFilter::MatchContext(c.clone()),
        JsonLdFilter::Range { key, gte, lte } => CompiledFilter::Range {
            key: key.clone(),
            gte: *gte,
            lte: *lte,
        },
        JsonLdFilter::Must(children) => {
            CompiledFilter::And(children.iter().map(compile_filter).collect())
        }
        JsonLdFilter::Should(children) => {
            CompiledFilter::Or(children.iter().map(compile_filter).collect())
        }
        JsonLdFilter::MustNot(children) => {
            let inner: Vec<CompiledFilter> = children.iter().map(compile_filter).collect();
            // Exclude the union of all forbidden conditions. `NOT(AND(...))`
            // would incorrectly keep entries matching one forbidden condition.
            CompiledFilter::Not(Box::new(CompiledFilter::Or(inner)))
        }
    }
}

/// Evaluate a set of JSON-LD filters against the metadata index and return
/// a RoaringBitmap of matching IDs. This is the main entry point for
/// filter-integrated search.
pub fn evaluate_filters(meta: &JsonLdMetadataIndex, filters: &[JsonLdFilter]) -> FilterEvaluation {
    if filters.is_empty() {
        return FilterEvaluation::Unrestricted;
    }
    let compiled: Vec<CompiledFilter> = filters.iter().map(compile_filter).collect();
    let filter = CompiledFilter::And(compiled);
    let bitmap = filter.evaluate_bitmap(meta);
    if bitmap.is_empty() {
        FilterEvaluation::Empty
    } else {
        FilterEvaluation::Matched(bitmap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_filter_match() {
        let filter = compile_filter(&JsonLdFilter::tag("tags", "important"));
        let result = filter.check(
            &|k, v| k == "tags" && v == "important",
            &|_| None,
            &|_| false,
            &|_| false,
        );
        assert!(result);
    }

    #[test]
    fn test_tag_filter_no_match() {
        let filter = compile_filter(&JsonLdFilter::tag("tags", "important"));
        let result = filter.check(
            &|k, v| k == "tags" && v == "unimportant",
            &|_| None,
            &|_| false,
            &|_| false,
        );
        assert!(!result);
    }

    #[test]
    fn test_must_all_pass() {
        let must = JsonLdFilter::Must(vec![
            JsonLdFilter::tag("a", "1"),
            JsonLdFilter::tag("b", "2"),
        ]);
        let filter = compile_filter(&must);
        let result = filter.check(
            &|k, v| matches!((k, v), ("a", "1") | ("b", "2")),
            &|_| None,
            &|_| false,
            &|_| false,
        );
        assert!(result);
    }

    #[test]
    fn test_must_fail_if_one_missing() {
        let must = JsonLdFilter::Must(vec![
            JsonLdFilter::tag("a", "1"),
            JsonLdFilter::tag("b", "2"),
        ]);
        let filter = compile_filter(&must);
        let result = filter.check(
            &|k, v| k == "a" && v == "1", // missing b:2
            &|_| None,
            &|_| false,
            &|_| false,
        );
        assert!(!result);
    }

    #[test]
    fn test_should_any_pass() {
        let should = JsonLdFilter::Should(vec![
            JsonLdFilter::tag("x", "1"),
            JsonLdFilter::tag("y", "2"),
        ]);
        let filter = compile_filter(&should);
        let result = filter.check(&|k, v| k == "y" && v == "2", &|_| None, &|_| false, &|_| {
            false
        });
        assert!(result);
    }

    #[test]
    fn test_must_not_excludes() {
        let must_not = JsonLdFilter::MustNot(vec![JsonLdFilter::tag("bad", "true")]);
        let filter = compile_filter(&must_not);
        let result = filter.check(
            &|k, v| k == "bad" && v == "true",
            &|_| None,
            &|_| false,
            &|_| false,
        );
        assert!(!result);
    }

    #[test]
    fn test_range_filter() {
        let range = JsonLdFilter::Range {
            key: "importance".into(),
            gte: Some(0.5),
            lte: None,
        };
        let filter = compile_filter(&range);
        let result = filter.check(
            &|_, _| false,
            &|k| {
                if k == "importance" {
                    Some(0.8)
                } else {
                    None
                }
            },
            &|_| false,
            &|_| false,
        );
        assert!(result);

        let result2 = filter.check(
            &|_, _| false,
            &|k| {
                if k == "importance" {
                    Some(0.2)
                } else {
                    None
                }
            },
            &|_| false,
            &|_| false,
        );
        assert!(!result2);
    }

    // ── Bitmap evaluation tests (require JsonLdMetadataIndex) ──

    use crate::jsonld_meta::JsonLdMetadataIndex;
    use serde_json::json;

    fn setup_test_index() -> JsonLdMetadataIndex {
        let meta = JsonLdMetadataIndex::new();
        meta.index(1, &json!({"@type": ["Document"], "tags": ["important"], "importance": 0.9, "named_graph": "g1"}));
        meta.index(2, &json!({"@type": ["Document", "Report"], "tags": ["normal"], "importance": 0.5, "named_graph": "g1"}));
        meta.index(3, &json!({"@type": ["Note"], "tags": ["important"], "importance": 0.3, "named_graph": "g2"}));
        meta
    }

    #[test]
    fn test_evaluate_tag() {
        let meta = setup_test_index();
        let cf = compile_filter(&JsonLdFilter::tag("tags", "important"));
        let bm = cf.evaluate(&meta).unwrap();
        assert!(bm.contains(1));
        assert!(!bm.contains(2));
        assert!(bm.contains(3));
    }

    #[test]
    fn test_evaluate_type() {
        let meta = setup_test_index();
        let cf = compile_filter(&JsonLdFilter::Type("Report".into()));
        let bm = cf.evaluate(&meta).unwrap();
        assert!(!bm.contains(1));
        assert!(bm.contains(2));
        assert!(!bm.contains(3));
    }

    #[test]
    fn test_evaluate_range() {
        let meta = setup_test_index();
        let cf = compile_filter(&JsonLdFilter::Range {
            key: "importance".into(),
            gte: Some(0.6),
            lte: None,
        });
        let bm = cf.evaluate(&meta).unwrap();
        assert!(bm.contains(1));
        assert!(!bm.contains(2));
        assert!(!bm.contains(3));
    }

    #[test]
    fn test_evaluate_must() {
        let meta = setup_test_index();
        let filter = JsonLdFilter::Must(vec![
            JsonLdFilter::tag("tags", "important"),
            JsonLdFilter::Type("Document".into()),
        ]);
        let cf = compile_filter(&filter);
        let bm = cf.evaluate(&meta).unwrap();
        assert!(bm.contains(1));
        assert!(!bm.contains(2)); // tags=normal
        assert!(!bm.contains(3)); // type=Note
    }

    #[test]
    fn test_evaluate_should() {
        let meta = setup_test_index();
        let filter = JsonLdFilter::Should(vec![
            JsonLdFilter::Type("Note".into()),
            JsonLdFilter::tag("tags", "normal"),
        ]);
        let cf = compile_filter(&filter);
        let bm = cf.evaluate(&meta).unwrap();
        assert!(!bm.contains(1));
        assert!(bm.contains(2));
        assert!(bm.contains(3));
    }

    #[test]
    fn test_evaluate_must_not() {
        let meta = setup_test_index();
        let filter = JsonLdFilter::MustNot(vec![JsonLdFilter::tag("tags", "important")]);
        let cf = compile_filter(&filter);
        let bm = cf.evaluate(&meta).unwrap();
        assert!(!bm.contains(1));
        assert!(bm.contains(2));
        assert!(!bm.contains(3));
    }

    #[test]
    fn test_evaluate_multiple_must_not_excludes_union() {
        let meta = setup_test_index();
        let filter = JsonLdFilter::MustNot(vec![
            JsonLdFilter::Type("Document".into()),
            JsonLdFilter::tag("tags", "normal"),
        ]);
        let bm = compile_filter(&filter).evaluate(&meta).unwrap();
        assert!(!bm.contains(1));
        assert!(!bm.contains(2));
        assert!(bm.contains(3));
    }

    #[test]
    fn test_evaluate_filters_keeps_empty_match_distinct_from_no_filter() {
        let meta = setup_test_index();
        assert!(matches!(
            evaluate_filters(&meta, &[]),
            FilterEvaluation::Unrestricted
        ));
        assert!(matches!(
            evaluate_filters(&meta, &[JsonLdFilter::Type("Missing".into())]),
            FilterEvaluation::Empty
        ));
    }
}
