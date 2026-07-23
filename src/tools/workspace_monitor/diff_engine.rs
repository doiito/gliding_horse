use similar::{ChangeTag, DiffOp, TextDiff};

pub struct DiffEngine;

impl DiffEngine {
    pub fn unified_diff(
        old_lines: &[String],
        new_lines: &[String],
        file_path: &str,
        old_version: u64,
        new_version: u64,
    ) -> String {
        let old_text = old_lines.join("\n") + "\n";
        let new_text = new_lines.join("\n") + "\n";
        let diff = TextDiff::from_lines(&old_text, &new_text);

        let mut out = String::new();
        out.push_str(&format!(
            "--- {} (v{})\n+++ {} (v{})\n",
            file_path, old_version, file_path, new_version
        ));

        for group in diff.grouped_ops(3) {
            let (start_a, end_a, start_b, end_b) = group_span(&group);
            out.push_str(&format!(
                "@@ -{},{} +{},{} @@\n",
                start_a + 1,
                end_a - start_a,
                start_b + 1,
                end_b - start_b,
            ));

            for op in &group {
                for change in diff.iter_changes(op) {
                    let sigil = match change.tag() {
                        ChangeTag::Delete => "-",
                        ChangeTag::Insert => "+",
                        ChangeTag::Equal => " ",
                    };
                    let value: String = change.value().to_string();
                    let trimmed = value.trim_end_matches('\n');
                    for line in trimmed.split('\n') {
                        out.push_str(&format!("{}{}\n", sigil, line));
                    }
                }
            }
        }

        out
    }

    pub fn changed_ranges(old_lines: &[String], new_lines: &[String]) -> Vec<(usize, usize)> {
        let old_text = old_lines.join("\n") + "\n";
        let new_text = new_lines.join("\n") + "\n";
        let diff = TextDiff::from_lines(&old_text, &new_text);

        let new_len = new_lines.len();
        if new_len == 0 {
            return Vec::new();
        }
        let mut changed = vec![false; new_len];

        for op in diff.ops() {
            match op {
                DiffOp::Insert { .. } | DiffOp::Replace { .. } => {
                    let new_range = op.new_range();
                    for idx in new_range.start..new_range.end {
                        if idx < new_len {
                            changed[idx] = true;
                        }
                    }
                }
                _ => {}
            }
        }

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < changed.len() {
            if changed[i] {
                let start = i;
                while i < changed.len() && changed[i] {
                    i += 1;
                }
                ranges.push((start, i));
            } else {
                i += 1;
            }
        }

        ranges
    }

    pub fn has_changes(old_content: &str, new_content: &str) -> bool {
        old_content != new_content
    }
}

fn group_span(ops: &[similar::DiffOp]) -> (usize, usize, usize, usize) {
    if ops.is_empty() {
        return (0, 0, 0, 0);
    }
    let range_a = ops.first().map(|o| o.old_range()).unwrap_or(0..0);
    let range_b = ops.last().map(|o| o.new_range()).unwrap_or(0..0);
    (range_a.start, range_a.end, range_b.start, range_b.end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_diff() {
        let old = vec!["line1".into(), "line2".into(), "line3".into()];
        let new = vec!["line1".into(), "line2_modified".into(), "line3".into()];

        let diff = DiffEngine::unified_diff(&old, &new, "test.txt", 1, 2);
        assert!(diff.contains("--- test.txt (v1)"));
        assert!(diff.contains("+++ test.txt (v2)"));
        assert!(diff.contains("-line2"));
        assert!(diff.contains("+line2_modified"));
    }

    #[test]
    fn test_changed_ranges() {
        let old = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let new = vec!["a".into(), "b".into(), "x".into(), "d".into()];

        let ranges = DiffEngine::changed_ranges(&old, &new);
        assert_eq!(ranges, vec![(2, 3)]);
    }

    #[test]
    fn test_has_changes() {
        assert!(DiffEngine::has_changes("hello", "world"));
        assert!(!DiffEngine::has_changes("same", "same"));
    }

    #[test]
    fn test_empty_diff() {
        let lines = vec!["a".into(), "b".into()];
        let diff = DiffEngine::unified_diff(&lines, &lines, "f", 1, 2);
        assert!(diff.contains("(v1)"));
        assert!(diff.contains("(v2)"));
    }

    #[test]
    fn test_changed_ranges_insert() {
        let old = vec!["a".into()];
        let new = vec!["a".into(), "b".into(), "c".into()];

        let ranges = DiffEngine::changed_ranges(&old, &new);
        assert_eq!(ranges, vec![(1, 3)]);
    }

    #[test]
    fn test_changed_ranges_delete() {
        let old = vec!["a".into(), "b".into(), "c".into()];
        let new = vec!["a".into()];

        let ranges = DiffEngine::changed_ranges(&old, &new);
        assert_eq!(ranges, vec![]); // deletion means nothing new to read
    }
}
