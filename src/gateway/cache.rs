use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde_json::Value;
use tracing::debug;

const DEFAULT_CAPACITY: usize = 256;
const DEFAULT_TTL_SECS: u64 = 3600;

struct CacheEntry {
    key: String,
    value: Value,
    created: Instant,
    access_count: u64,
}

/// LRU response cache with TTL expiration
pub struct ResponseCache {
    entries: RwLock<VecDeque<CacheEntry>>,
    capacity: usize,
    ttl: Duration,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ResponseCache {
    pub fn new(capacity: usize, ttl_secs: u64) -> Self {
        Self {
            entries: RwLock::new(VecDeque::with_capacity(capacity)),
            capacity,
            ttl: Duration::from_secs(ttl_secs),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Build cache key from (model, message payload hash)
    pub fn build_key(model: &str, messages: &[Value]) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        model.hash(&mut hasher);
        for msg in messages {
            if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                role.hash(&mut hasher);
            }
            if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                content.hash(&mut hasher);
            }
            if let Some(reasoning) = msg.get("reasoning_content").and_then(|r| r.as_str()) {
                reasoning.hash(&mut hasher);
            }
            if let Some(tool_call_id) = msg.get("tool_call_id").and_then(|t| t.as_str()) {
                tool_call_id.hash(&mut hasher);
            }
            if let Some(tool_calls) = msg.get("tool_calls") {
                tool_calls.to_string().hash(&mut hasher);
            }
        }
        format!("{}:{:x}", model, hasher.finish())
    }

    /// Look up a cache entry
    pub fn get(&self, key: &str) -> Option<Value> {
        let mut entries = self.entries.write();
        if let Some(pos) = entries.iter().position(|e| e.key == key) {
            let mut entry = entries
                .remove(pos)
                .expect("pos confirmed by position() above");
            // Check TTL
            if entry.created.elapsed() > self.ttl {
                self.misses.fetch_add(1, Ordering::Relaxed);
                debug!(key = %key, "Cache entry expired");
                return None;
            }
            entry.access_count += 1;
            entry.created = Instant::now(); // refresh on access
            entries.push_back(entry);
            self.hits.fetch_add(1, Ordering::Relaxed);
            debug!(key = %key, "Cache hit");
            // Return the last entry's value
            entries.back().map(|e| e.value.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert a response into cache
    pub fn set(&self, key: &str, value: Value) {
        let mut entries = self.entries.write();
        if entries.len() >= self.capacity {
            entries.pop_front();
        }
        entries.push_back(CacheEntry {
            key: key.to_string(),
            value,
            created: Instant::now(),
            access_count: 0,
        });
        debug!(key = %key, "Cached");
    }

    /// Invalidate a specific key
    pub fn invalidate(&self, key: &str) {
        let mut entries = self.entries.write();
        entries.retain(|e| e.key != key);
    }

    /// Clear all entries
    pub fn clear(&self) {
        self.entries.write().clear();
        debug!("Cache cleared");
    }

    /// Hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Current size
    pub fn size(&self) -> usize {
        self.entries.read().len()
    }
}

impl Default for ResponseCache {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY, DEFAULT_TTL_SECS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_cache_set_get() {
        let cache = ResponseCache::new(10, 60);
        let key = "test_key";
        cache.set(key, json!({"result": "hello"}));
        let val = cache.get(key);
        assert!(val.is_some());
        assert_eq!(val.unwrap()["result"], "hello");
    }

    #[test]
    fn test_cache_miss() {
        let cache = ResponseCache::new(10, 60);
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let cache = ResponseCache::new(2, 60);
        cache.set("a", json!(1));
        cache.set("b", json!(2));
        cache.set("c", json!(3)); // evicts "a"
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
    }

    #[test]
    fn test_build_key() {
        let msgs = vec![json!({"role": "user", "content": "hello"})];
        let key = ResponseCache::build_key("deepseek-v4-pro", &msgs);
        assert!(key.starts_with("deepseek-v4-pro:"));
    }
}
