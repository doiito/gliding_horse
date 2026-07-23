use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tracing::debug;

/// Token bucket rate limiter, per-model
pub struct RateLimiter {
    buckets: RwLock<HashMap<String, TokenBucket>>,
    default_rate: u64,  // tokens per second
    default_burst: u64, // max burst size
}

struct TokenBucket {
    tokens: f64,
    capacity: f64,
    rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(rate: u64, burst: u64) -> Self {
        Self {
            tokens: burst as f64,
            capacity: burst as f64,
            rate: rate as f64,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rate).min(self.capacity);
        self.last_refill = Instant::now();
    }

    fn try_consume(&mut self, count: u64) -> bool {
        self.refill();
        let needed = count as f64;
        if self.tokens >= needed {
            self.tokens -= needed;
            true
        } else {
            false
        }
    }

    fn wait_time(&self) -> Option<Duration> {
        if self.tokens > 0.0 {
            return None;
        }
        Some(Duration::from_secs_f64(1.0 / self.rate.max(0.001)))
    }
}

impl RateLimiter {
    pub fn new(rate: u64, burst: u64) -> Self {
        Self {
            buckets: RwLock::new(HashMap::new()),
            default_rate: rate,
            default_burst: burst,
        }
    }

    /// Check if a request is allowed, consuming `count` tokens
    pub fn check(&self, model: &str, count: u64) -> bool {
        let mut buckets = self.buckets.write();
        let bucket = buckets
            .entry(model.to_string())
            .or_insert_with(|| TokenBucket::new(self.default_rate, self.default_burst));
        bucket.try_consume(count)
    }

    /// Wait until a request can be made (blocking approximate)
    pub fn wait_if_needed(&self, model: &str, count: u64) -> Option<Duration> {
        let wait = {
            let buckets = self.buckets.read();
            buckets.get(model).and_then(|b| b.wait_time())
        };
        if let Some(dur) = wait {
            let h = tokio::runtime::Handle::try_current();
            if h.is_ok() {
                tracing::warn!(
                    "RateLimiter::wait_if_needed called in async context; returning wait duration instead of blocking. \
                     Caller should use wait_if_needed_async()."
                );
            } else {
                std::thread::sleep(dur);
            }
        }
        let allowed = self.check(model, count);
        if allowed {
            None
        } else {
            Some(Duration::from_millis(100))
        }
    }

    /// Async version: wait until a request can be made without blocking the runtime
    pub async fn wait_if_needed_async(&self, model: &str, count: u64) -> Option<Duration> {
        let wait = {
            let buckets = self.buckets.read();
            buckets.get(model).and_then(|b| b.wait_time())
        };
        if let Some(dur) = wait {
            tokio::time::sleep(dur).await;
        }
        let allowed = self.check(model, count);
        if allowed {
            None
        } else {
            Some(Duration::from_millis(100))
        }
    }

    /// Configure rate for a specific model
    pub fn configure(&self, model: &str, rate: u64, burst: u64) {
        let mut buckets = self.buckets.write();
        buckets.insert(model.to_string(), TokenBucket::new(rate, burst));
        debug!(model = %model, rate = rate, burst = burst, "Rate limit configured");
    }

    /// Total configured models
    pub fn model_count(&self) -> usize {
        self.buckets.read().len()
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(10, 20) // 10 req/s, burst 20
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_rate_limit() {
        let limiter = RateLimiter::new(100, 200);
        assert!(limiter.check("deepseek-v4-pro", 1));
        // Burst should allow many at once
        for _ in 0..150 {
            limiter.check("deepseek-v4-pro", 1);
        }
    }

    #[test]
    fn test_configure_model() {
        let limiter = RateLimiter::new(10, 10);
        limiter.configure("deepseek", 5, 5);
        assert_eq!(limiter.model_count(), 1);
    }
}
