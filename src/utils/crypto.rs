use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct CryptoUtils;

impl CryptoUtils {
    pub fn sha256_hex(data: &str) -> String {
        use base64::Engine;
        let digest = ring::digest::digest(&ring::digest::SHA256, data.as_bytes());
        base64::engine::general_purpose::STANDARD.encode(digest.as_ref())
    }

    pub fn fast_hash(data: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256() {
        let h = CryptoUtils::sha256_hex("hello");
        assert!(!h.is_empty());
    }

    #[test]
    fn test_deterministic() {
        assert_eq!(
            CryptoUtils::sha256_hex("abc"),
            CryptoUtils::sha256_hex("abc")
        );
        assert_ne!(
            CryptoUtils::sha256_hex("abc"),
            CryptoUtils::sha256_hex("abd")
        );
    }
}
