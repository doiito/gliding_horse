use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Return the display width of a string (fullwidth CJK = 2, ASCII = 1).
#[inline]
pub fn display_width(s: &str) -> usize {
    UnicodeWidthStr::width(s)
}

/// Truncate to at most `max_bytes` bytes, always at a UTF-8 char boundary.
/// This never panics and never splits a multi-byte character.
pub fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Truncate to at most `max_width` display columns, keeping whole grapheme
/// clusters so that combining marks, flags, and multi-byte CJK are never
/// split visually.
///
/// Uses `unicode-width` for width calculation and `unicode-segmentation`
/// for grapheme-aware iteration.
pub fn truncate_to_width(s: &str, max_width: usize) -> String {
    if UnicodeWidthStr::width(s) <= max_width {
        return s.to_string();
    }

    let mut result = String::new();
    let mut current_width = 0;

    for grapheme in s.graphemes(true) {
        let g_width = UnicodeWidthStr::width(grapheme);
        if current_width + g_width > max_width {
            break;
        }
        result.push_str(grapheme);
        current_width += g_width;
    }

    result
}

/// Count the number of grapheme clusters (user-perceived characters).
/// Prefer this over `s.chars().count()` when the result is used for
/// display purposes, because combining sequences and emoji ZWJ sequences
/// consist of multiple `char` values but should be treated as one
/// visual unit.
pub fn grapheme_count(s: &str) -> usize {
    s.graphemes(true).count()
}

/// Truncate text for a preview, appending a truncation notice.
/// `max_width` limits the display columns of the preview body (not
/// including the notice).
pub fn truncate_preview(text: &str, max_width: usize) -> String {
    let width = UnicodeWidthStr::width(text);
    if width <= max_width {
        return text.to_string();
    }

    let truncated = truncate_to_width(text, max_width.saturating_sub(3));
    format!("{}...", truncated)
}

/// Smart truncate a block of text: keeps whole lines up to roughly
/// `max_bytes` (at a char boundary), then appends a summary line
/// showing the total size.  Uses byte-safe truncation rather than
/// display-width truncation because `\n` has display width 0 in the
/// `unicode-width` crate, which would cause over-inclusion when the
/// text contains many newlines.
pub fn smart_truncate_text(text: &str, max_bytes: usize) -> String {
    let text_width = UnicodeWidthStr::width(text);
    if text_width <= max_bytes {
        return text.to_string();
    }

    let truncated = safe_truncate(text, max_bytes);

    if let Some(last_newline) = truncated.rfind('\n') {
        let result = truncated[..last_newline].to_string();
        let total_lines = text.lines().count();
        let kept_lines = result.lines().count();
        format!(
            "{}\n\n[truncated: {} lines total, {} lines kept | {} original chars]",
            result, total_lines, kept_lines, text_width
        )
    } else {
        format!(
            "{}...\n\n[truncated: {} original chars, {} chars kept]",
            truncated,
            text_width,
            truncated.width()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_width_ascii() {
        assert_eq!(display_width("hello"), 5);
    }

    #[test]
    fn test_display_width_cjk() {
        assert_eq!(display_width("你好"), 4);
    }

    #[test]
    fn test_display_width_mixed() {
        assert_eq!(display_width("你好Rust"), 8);
    }

    #[test]
    fn test_safe_truncate_ascii() {
        assert_eq!(safe_truncate("hello world", 5), "hello");
    }

    #[test]
    fn test_safe_truncate_cjk_boundary() {
        let s = "你好世界";
        // "你好世" is 9 bytes (3*3), "你好世界" is 12 bytes
        assert_eq!(safe_truncate(s, 9), "你好世");
        // At byte 8, "你" is 3 so 8%3 != 0 → walk back to 6
        assert_eq!(safe_truncate(s, 8), "你好");
        assert!(safe_truncate(s, 8).is_char_boundary(safe_truncate(s, 8).len()));
    }

    #[test]
    fn test_safe_truncate_noop() {
        let s = "short";
        assert_eq!(safe_truncate(s, 100), s);
    }

    #[test]
    fn test_truncate_to_width_cjk() {
        let s = "你好世界";
        // Each CJK char is width 2. "你好世" = 6, "你好世界" = 8
        assert_eq!(truncate_to_width(s, 6), "你好世");
        assert_eq!(truncate_to_width(s, 4), "你好");
    }

    #[test]
    fn test_truncate_to_width_mixed() {
        let s = "你好Rust";
        // 你(2) + 好(2) + R(1) + u(1) + s(1) + t(1) = 8
        assert_eq!(truncate_to_width(s, 5), "你好R");
        assert_eq!(truncate_to_width(s, 4), "你好");
    }

    #[test]
    fn test_truncate_to_width_noop() {
        let s = "hello";
        assert_eq!(truncate_to_width(s, 10), s);
    }

    #[test]
    fn test_grapheme_count() {
        assert_eq!(grapheme_count("hello"), 5);
        assert_eq!(grapheme_count("你好"), 2);
        assert_eq!(grapheme_count("e\u{301}"), 1); // é as combining
    }

    #[test]
    fn test_truncate_preview() {
        assert_eq!(truncate_preview("hello", 10), "hello");
        assert_eq!(truncate_preview("hello world", 8), "hello...");
    }

    #[test]
    fn test_truncate_preview_cjk() {
        let s = "你好世界Rust";
        // 你(2) + 好(2) = 4, plus "..."(3) = 7 ≤ max_width=7
        assert_eq!(truncate_preview(s, 7), "你好...");
    }

    #[test]
    fn test_smart_truncate_text() {
        let text = "line1\nline2\nline3\nline4\nline5";
        let result = smart_truncate_text(text, 15);
        assert!(result.contains("line1"));
        assert!(result.contains("truncated"));
    }

    #[test]
    fn test_smart_truncate_text_utf8() {
        let text = "你好世界\n".repeat(500);
        let result = smart_truncate_text(&text, 100);
        assert!(result.contains("truncated"));
        assert!(result.is_char_boundary(result.len()));
    }
}
