//! Shared inbound API authentication helpers.

/// Compare secrets without returning early on the first differing byte.
///
/// This is intentionally small and dependency-free. It also incorporates the
/// length difference into the result so differently sized values never match.
pub fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    let max_len = left.len().max(right.len());
    for index in 0..max_len {
        let left_byte = left.get(index).copied().unwrap_or(0);
        let right_byte = right.get(index).copied().unwrap_or(0);
        difference |= usize::from(left_byte ^ right_byte);
    }
    difference == 0
}

pub fn valid_bearer_header(header: Option<&str>, expected_token: &str) -> bool {
    let Some(candidate) = header.and_then(|value| value.strip_prefix("Bearer ")) else {
        return false;
    };
    constant_time_eq(candidate.as_bytes(), expected_token.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_header_requires_exact_scheme_and_token() {
        assert!(valid_bearer_header(Some("Bearer secret"), "secret"));
        assert!(!valid_bearer_header(Some("bearer secret"), "secret"));
        assert!(!valid_bearer_header(Some("Bearer secrets"), "secret"));
        assert!(!valid_bearer_header(None, "secret"));
    }
}
