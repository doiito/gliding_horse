use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use tracing::{info, warn};

const DEFAULT_VEC_SIZE: usize = 128;

/// Trait for embedding text into vectors.
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, String>;
    fn dimension(&self) -> usize;
}

pub struct OneApiEmbeddingService {
    client: reqwest::Client,
    api_url: String,
    api_key: String,
    model: String,
    dimension: usize,
    failure_warned: AtomicBool,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl OneApiEmbeddingService {
    pub fn new(api_url: &str, api_key: &str, model: &str, dimension: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_url: api_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            dimension,
            failure_warned: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl EmbeddingService for OneApiEmbeddingService {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let url = format!("{}/v1/embeddings", self.api_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.model,
            "input": text
        });
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                warn_once_on_first_failure(&self.failure_warned, &format!("{}: {}", self.model, e));
                format!("Embedding request failed: {}", e)
            })?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let err = format!(
                "Embedding API returned error: {} - {}",
                status, body
            );
            warn_once_on_first_failure(&self.failure_warned, &err);
            return Err(err);
        }
        let result: EmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| format!("Embedding response parse failed: {}", e))?;
        result
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| "No data in Embedding response".to_string())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

pub struct FallbackEmbeddingService {
    dimension: usize,
}

impl FallbackEmbeddingService {
    pub fn new() -> Self {
        Self {
            dimension: DEFAULT_VEC_SIZE,
        }
    }

    pub fn with_dimension(dimension: usize) -> Self {
        Self { dimension }
    }

    pub fn embed_fallback(&self, text: &str) -> Vec<f32> {
        let dim = self.dimension;
        let mut v = vec![0.0f32; dim];
        for word in text.split_whitespace() {
            v[(fnv_hash(word) % dim as u64) as usize] += 1.0;
        }
        let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 0.0 {
            for x in &mut v {
                *x /= mag;
            }
        }
        v
    }
}

impl Default for FallbackEmbeddingService {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EmbeddingService for FallbackEmbeddingService {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        Ok(self.embed_fallback(text))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

pub struct OllamaEmbeddingService {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dimension: usize,
    failure_warned: AtomicBool,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaEmbeddingService {
    pub fn new(base_url: &str, model: &str, dimension: usize, timeout_secs: u64) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .build()
                .unwrap_or_default(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            dimension,
            failure_warned: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl EmbeddingService for OllamaEmbeddingService {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let url = format!("{}/api/embed", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": text
        });
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                warn_once_on_first_failure(&self.failure_warned, &format!("{}: {}", self.model, e));
                format!("Ollama Embedding request failed: {}", e)
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let err = format!(
                "Ollama Embedding API returned error: {} - {}",
                status, body
            );
            warn_once_on_first_failure(&self.failure_warned, &err);
            return Err(err);
        }

        let result: OllamaEmbedResponse = resp
            .json()
            .await
            .map_err(|e| format!("Ollama Embedding response parse failed: {}", e))?;

        result
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| "No embeddings data in Ollama Embedding response".to_string())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

pub fn create_embedding_service_from_config(
    config: &crate::config::settings::EmbeddingSettings,
    timeout_secs: u64,
) -> Arc<dyn EmbeddingService> {
    if !config.enabled {
        info!("Embedding disabled, using Fallback service");
        return Arc::new(FallbackEmbeddingService::with_dimension(
            config.fallback.dimension,
        ));
    }

    match config.provider.as_str() {
        "ollama" => {
            info!(
                url = %config.ollama.base_url,
                model = %config.ollama.model,
                dim = config.ollama.dimension,
                "Using Ollama Embedding service"
            );
            Arc::new(OllamaEmbeddingService::new(
                &config.ollama.base_url,
                &config.ollama.model,
                config.ollama.dimension,
                timeout_secs,
            ))
        }
        "oneapi" => {
            if config.oneapi.base_url.is_empty() || config.oneapi.api_key.is_empty() {
                warn!("OneAPI Embedding config incomplete, falling back to Fallback");
                return Arc::new(FallbackEmbeddingService::with_dimension(
                    config.fallback.dimension,
                ));
            }
            info!(
                url = %config.oneapi.base_url,
                model = %config.oneapi.model,
                dim = config.oneapi.dimension,
                "Using OneAPI Embedding service"
            );
            Arc::new(OneApiEmbeddingService::new(
                &config.oneapi.base_url,
                &config.oneapi.api_key,
                &config.oneapi.model,
                config.oneapi.dimension,
            ))
        }
        "fallback" | "" => {
            info!("Using Fallback Embedding service");
            Arc::new(FallbackEmbeddingService::with_dimension(
                config.fallback.dimension,
            ))
        }
        other => {
            warn!(
                provider = other,
                "Unknown Embedding provider, falling back to Fallback"
            );
            Arc::new(FallbackEmbeddingService::with_dimension(
                config.fallback.dimension,
            ))
        }
    }
}

fn fnv_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Emit an embedding-service failure warning exactly once per process.
///
/// Returns `true` when this call is the first failure (and the warning was
/// emitted), `false` for every subsequent failure. Callers (Ollama/OneAPI
/// `embed`) invoke this on the error path so a chronically unreachable
/// embedding backend cannot spam the log on every search/upsert.
fn warn_once_on_first_failure(failure_warned: &AtomicBool, detail: &str) -> bool {
    if failure_warned
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        warn!("Embedding service unreachable — semantic index/search degraded: {}", detail);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_embed() {
        let svc = FallbackEmbeddingService::new();
        let v = svc.embed_fallback("hello world");
        assert_eq!(v.len(), DEFAULT_VEC_SIZE);
        assert!(v.iter().any(|x| *x > 0.0));
    }

    #[test]
    fn test_fallback_embed_custom_dimension() {
        let svc = FallbackEmbeddingService::with_dimension(256);
        let v = svc.embed_fallback("hello world");
        assert_eq!(v.len(), 256);
        assert!(v.iter().any(|x| *x > 0.0));
    }

    #[tokio::test]
    async fn test_fallback_embedding_service_trait() {
        let svc = FallbackEmbeddingService::new();
        let v = svc.embed("hello world").await.unwrap();
        assert_eq!(v.len(), DEFAULT_VEC_SIZE);
        assert!(v.iter().any(|x| *x > 0.0));
    }

#[test]
    fn test_fnv() {
        assert_eq!(fnv_hash("a"), fnv_hash("a"));
    }

    #[test]
    fn warn_once_emits_only_on_first_failure() {
        let flag = std::sync::atomic::AtomicBool::new(false);
        assert!(warn_once_on_first_failure(&flag, "ollama down"));
        assert!(!warn_once_on_first_failure(&flag, "ollama down"));
        assert!(!warn_once_on_first_failure(&flag, "still down"));
    }
}
