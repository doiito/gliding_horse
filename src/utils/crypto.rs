use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct CryptoUtils;

impl CryptoUtils {
    pub fn sha256_hex(data: &str) -> String {
        Self::sha256_hex_bytes(data.as_bytes())
    }

    pub fn sha256_hex_bytes(data: &[u8]) -> String {
        let digest = ring::digest::digest(&ring::digest::SHA256, data);
        hex::encode(digest.as_ref())
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
        assert_eq!(
            h,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        assert_eq!(h.len(), 64);
        assert!(h.bytes().all(|byte| byte.is_ascii_hexdigit()));
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

    #[test]
    fn byte_hash_supports_non_utf8_content() {
        assert_eq!(
            CryptoUtils::sha256_hex_bytes(&[0xff, 0x00, 0x80]),
            "ef192b7af54e943f206ab27075ec1805384c972c9959fc5820f1fa7d5268fcef"
        );
    }
}
