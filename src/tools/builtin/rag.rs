use std::path::PathBuf;

use serde::Deserialize;
use serde_json::{json, Value};

fn gliding_data_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(&home).join(".gliding_horse").join("data")
}

fn rag_index_dir() -> PathBuf {
    gliding_data_dir().join("rag_index")
}

#[derive(Debug, Deserialize)]
struct RagSearchInput {
    query: String,
    limit: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RagIndexInput {
    content: String,
    iri: Option<String>,
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct RagChunkInput {
    content: String,
    chunk_size: Option<usize>,
    overlap: Option<usize>,
}

pub fn execute_rag_search(input: &Value) -> Result<Value, String> {
    let params: RagSearchInput =
        serde_json::from_value(input.clone()).map_err(|e| format!("Invalid input: {}", e))?;
    let query = &params.query;
    let limit = params.limit.unwrap_or(5);

    let query_lower = query.to_lowercase();
    let keywords: Vec<&str> = query_lower.split_whitespace().collect();

    let index_dir = rag_index_dir();
    if !index_dir.exists() {
        return Ok(json!({
            "query": query,
            "results": [],
            "count": 0,
            "message": "RAG index directory does not exist. Please use rag_index to index documents first"
        }));
    }

    let mut results = Vec::new();
    let entries = std::fs::read_dir(index_dir)
        .map_err(|e| format!("Failed to read index directory: {}", e))?;

    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().ends_with(".json") {
            continue;
        }
        let content = std::fs::read_to_string(entry.path())
            .map_err(|e| format!("Failed to read index file: {}", e))?;
        let doc: Value = serde_json::from_str(&content).unwrap_or_default();

        let text = doc["content"].as_str().unwrap_or("").to_lowercase();
        let score = keywords.iter().filter(|kw| text.contains(*kw)).count() as f64
            / keywords.len().max(1) as f64;

        if score > 0.0 {
            results.push(json!({
                "iri": doc["iri"].as_str().unwrap_or(""),
                "content": doc["content"].as_str().unwrap_or(""),
                "tags": doc["tags"].as_array().unwrap_or(&vec![]),
                "score": score,
            }));
        }
    }

    results.sort_by(|a, b| {
        b["score"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&a["score"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit as usize);

    Ok(json!({
        "query": query,
        "results": results,
        "count": results.len(),
    }))
}

pub fn execute_rag_index(input: &Value) -> Result<Value, String> {
    let params: RagIndexInput =
        serde_json::from_value(input.clone()).map_err(|e| format!("Invalid input: {}", e))?;

    let index_dir = rag_index_dir();
    std::fs::create_dir_all(&index_dir)
        .map_err(|e| format!("Failed to create index directory: {}", e))?;

    let iri = params
        .iri
        .unwrap_or_else(|| format!("iri://rag/{}", uuid::Uuid::new_v4()));

    let doc = json!({
        "iri": iri,
        "content": params.content,
        "tags": params.tags.unwrap_or_default(),
        "indexed_at": chrono::Utc::now().to_rfc3339(),
    });

    let file_name = iri.replace([':', '/', '#'], "_");
    let file_path = index_dir.join(format!("{}.json", file_name));
    let json_str = serde_json::to_string_pretty(&doc)
        .map_err(|e| format!("Failed to serialize index document: {}", e))?;
    std::fs::write(&file_path, json_str)
        .map_err(|e| format!("Failed to write index file: {}", e))?;

    Ok(json!({
        "iri": iri,
        "indexed": true,
        "path": file_path.to_string_lossy(),
    }))
}

pub fn execute_rag_chunk(input: &Value) -> Result<Value, String> {
    let params: RagChunkInput =
        serde_json::from_value(input.clone()).map_err(|e| format!("Invalid input: {}", e))?;

    let chunk_size = params.chunk_size.unwrap_or(500);
    let overlap = params.overlap.unwrap_or(50);
    let content = &params.content;

    if content.len() <= chunk_size {
        return Ok(json!({
            "chunks": [content],
            "count": 1,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }));
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    let content_chars: Vec<char> = content.chars().collect();
    let total_len = content_chars.len();

    while start < total_len {
        let end = (start + chunk_size).min(total_len);
        let chunk: String = content_chars[start..end].iter().collect();
        chunks.push(chunk);

        let next_start = end.saturating_sub(overlap);
        if next_start <= start || end >= total_len {
            break;
        }
        start = next_start;
    }

    Ok(json!({
        "chunks": chunks,
        "count": chunks.len(),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_chunk() {
        let input = json!({
            "content": "This is a test text for verifying document chunking. Chunking is a core feature of the RAG system.",
            "chunk_size": 20,
            "overlap": 5
        });
        let result = execute_rag_chunk(&input).unwrap();
        assert!(result["count"].as_u64().unwrap() > 1);
        assert!(result["chunks"].as_array().unwrap().len() > 1);
    }

    #[test]
    fn test_rag_chunk_short() {
        let input = json!({
            "content": "Short text"
        });
        let result = execute_rag_chunk(&input).unwrap();
        assert_eq!(result["count"].as_u64().unwrap(), 1);
    }

    #[test]
    fn test_rag_search_no_index() {
        let dir = tempfile::tempdir().unwrap();
        let orig_home = std::env::var("HOME").ok();
        std::env::set_var("HOME", dir.path());

        let input = json!({
            "query": "Test query"
        });
        let result = execute_rag_search(&input).unwrap();
        assert_eq!(result["count"].as_u64().unwrap(), 0);

        match orig_home {
            Some(h) => std::env::set_var("HOME", h),
            None => std::env::remove_var("HOME"),
        }
    }
}
