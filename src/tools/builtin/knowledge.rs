use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

fn rag_index_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(&home)
        .join(".gliding_horse")
        .join("data")
        .join("rag_index")
}

#[derive(Debug, Deserialize)]
pub struct KnowledgeImportFileInput {
    pub path: String,
    pub tags: Option<Vec<String>>,
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
    pub auto_detect_title: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct KnowledgeImportUrlInput {
    pub url: String,
    pub tags: Option<Vec<String>>,
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
    pub selector: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct KnowledgeImportDirectoryInput {
    pub path: String,
    pub pattern: Option<String>,
    pub tags: Option<Vec<String>>,
    pub recursive: Option<bool>,
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct KnowledgeListInput {
    pub tags: Option<Vec<String>>,
    pub source_type: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct KnowledgeDeleteInput {
    pub iri: Option<String>,
    pub tags: Option<Vec<String>>,
    pub all: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct KnowledgeSearchInput {
    pub query: String,
    pub tags: Option<Vec<String>>,
    pub limit: Option<usize>,
    pub min_score: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct KnowledgeUpdateInput {
    pub iri: String,
    pub content: Option<String>,
    pub tags: Option<Vec<String>>,
    pub append_tags: Option<bool>,
}

fn ensure_index_dir() -> Result<(), String> {
    let dir = rag_index_dir();
    if !dir.exists() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("Failed to create index directory: {}", e))?;
    }
    Ok(())
}

fn generate_iri(source: &str, path: &str) -> String {
    let hash = format!("{:x}", md5::compute(format!("{}:{}", source, path)));
    format!("iri://knowledge/{}/{}", source, &hash[..12])
}

fn extract_title(content: &str, file_ext: &str) -> Option<String> {
    match file_ext {
        "md" | "markdown" => {
            for line in content.lines().take(10) {
                let trimmed = line.trim();
                if trimmed.starts_with("# ") {
                    return Some(trimmed[2..].to_string());
                }
            }
            None
        }
        "html" | "htm" => {
            let re = regex::Regex::new(r"<title[^>]*>([^<]+)</title>").ok()?;
            re.captures(content)
                .and_then(|c| c.get(1).map(|m| m.as_str().to_string()))
        }
        _ => None,
    }
}

fn html_to_text(html: &str) -> String {
    let mut text = html.to_string();

    let script_re =
        regex::Regex::new(r"<script[^>]*>[\s\S]*?</script>").expect("valid regex literal");
    let style_re = regex::Regex::new(r"<style[^>]*>[\s\S]*?</style>").expect("valid regex literal");
    let tag_re = regex::Regex::new(r"<[^>]+>").expect("valid regex literal");
    let ws_re = regex::Regex::new(r"\s+").expect("valid regex literal");

    text = script_re.replace_all(&text, "").to_string();
    text = style_re.replace_all(&text, "").to_string();
    text = tag_re.replace_all(&text, " ").to_string();
    text = ws_re.replace_all(&text, " ").to_string();

    text.trim().to_string()
}

fn smart_chunk(content: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if content.len() <= chunk_size {
        return vec![content.to_string()];
    }

    let paragraphs: Vec<&str> = content.split("\n\n").collect();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for para in paragraphs {
        if current_chunk.len() + para.len() + 2 > chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            if overlap > 0 && current_chunk.len() > overlap {
                let chars: Vec<char> = current_chunk.chars().collect();
                let start = chars.len().saturating_sub(overlap);
                let overlap_text: String = chars[start..].iter().collect();
                current_chunk = overlap_text;
            } else {
                current_chunk = String::new();
            }
        }
        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(para);
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

fn index_chunk(
    content: &str,
    iri: &str,
    chunk_index: usize,
    tags: &[String],
    source: &str,
    source_path: &str,
    title: Option<&str>,
) -> Result<String, String> {
    ensure_index_dir()?;

    let chunk_iri = format!("{}#chunk{}", iri, chunk_index);
    let now: DateTime<Utc> = Utc::now();

    let doc = json!({
        "iri": chunk_iri,
        "parent_iri": iri,
        "content": content,
        "chunk_index": chunk_index,
        "tags": tags,
        "source": {
            "type": source,
            "path": source_path,
        },
        "title": title,
        "indexed_at": now.to_rfc3339(),
        "char_count": content.len(),
    });

    let file_name = chunk_iri.replace([':', '/', '#'], "_");
    let file_path = rag_index_dir().join(format!("{}.json", file_name));
    let json_str = serde_json::to_string_pretty(&doc)
        .map_err(|e| format!("Failed to serialize chunk document: {}", e))?;
    std::fs::write(&file_path, json_str)
        .map_err(|e| format!("Failed to write index file: {}", e))?;

    Ok(chunk_iri)
}

pub async fn execute_knowledge_import_file(input: Value) -> Result<Value, String> {
    let params: KnowledgeImportFileInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    let path = Path::new(&params.path);
    if !path.exists() {
        return Err(format!("File does not exist: {}", params.path));
    }

    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let processed_content = match ext.as_str() {
        "html" | "htm" => html_to_text(&content),
        "md" | "markdown" | "txt" | "json" | "yaml" | "yml" | "toml" | "rs" | "py" | "js"
        | "ts" => content,
        _ => content,
    };

    let title = if params.auto_detect_title.unwrap_or(true) {
        extract_title(&processed_content, &ext)
    } else {
        None
    };

    let chunk_size = params.chunk_size.unwrap_or(1000);
    let overlap = params.overlap.unwrap_or(100);
    let chunks = smart_chunk(&processed_content, chunk_size, overlap);

    let iri = generate_iri("file", &params.path);
    let tags = params.tags.unwrap_or_default();

    let mut indexed_chunks = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_iri = index_chunk(
            chunk,
            &iri,
            i,
            &tags,
            "file",
            &params.path,
            title.as_deref(),
        )?;
        indexed_chunks.push(chunk_iri);
    }

    Ok(json!({
        "iri": iri,
        "source": "file",
        "path": params.path,
        "title": title,
        "total_chunks": chunks.len(),
        "chunk_iris": indexed_chunks,
        "char_count": processed_content.len(),
        "tags": tags,
        "success": true,
    }))
}

pub async fn execute_knowledge_import_url(input: Value) -> Result<Value, String> {
    let params: KnowledgeImportUrlInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
        .build()
        .map_err(|e| format!("HTTP client: {}", e))?;
    let resp = client
        .get(&params.url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;
    let html = resp
        .text()
        .await
        .map_err(|e| format!("Read failed: {}", e))?;

    let content = if let Some(ref selector) = params.selector {
        let re = regex::Regex::new(selector).map_err(|e| format!("Invalid selector: {}", e))?;
        re.find(&html)
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| html_to_text(&html))
    } else {
        html_to_text(&html)
    };

    let title = extract_title(&html, "html");

    let chunk_size = params.chunk_size.unwrap_or(1000);
    let overlap = params.overlap.unwrap_or(100);
    let chunks = smart_chunk(&content, chunk_size, overlap);

    let iri = generate_iri("url", &params.url);
    let tags = params.tags.unwrap_or_default();

    let mut indexed_chunks = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_iri = index_chunk(chunk, &iri, i, &tags, "url", &params.url, title.as_deref())?;
        indexed_chunks.push(chunk_iri);
    }

    Ok(json!({
        "iri": iri,
        "source": "url",
        "url": params.url,
        "title": title,
        "total_chunks": chunks.len(),
        "chunk_iris": indexed_chunks,
        "char_count": content.len(),
        "tags": tags,
        "success": true,
    }))
}

pub async fn execute_knowledge_import_directory(input: Value) -> Result<Value, String> {
    let params: KnowledgeImportDirectoryInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    let path = Path::new(&params.path);
    if !path.exists() {
        return Err(format!("Directory does not exist: {}", params.path));
    }

    let pattern = params
        .pattern
        .clone()
        .unwrap_or_else(|| "*.md,*.txt,*.html,*.json".to_string());
    let extensions: Vec<&str> = pattern
        .split(',')
        .map(|s| s.trim().trim_start_matches('*').trim_start_matches('.'))
        .collect();

    let recursive = params.recursive.unwrap_or(true);
    let chunk_size = params.chunk_size.unwrap_or(1000);
    let overlap = params.overlap.unwrap_or(100);
    let tags = params.tags.unwrap_or_default();

    let mut results = Vec::new();
    let mut total_chunks = 0;
    let mut total_chars = 0;
    let mut errors = Vec::new();

    let walker = if recursive {
        walkdir::WalkDir::new(path).into_iter()
    } else {
        walkdir::WalkDir::new(path).max_depth(1).into_iter()
    };

    for entry in walker.filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }

        let ext = entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !extensions.contains(&ext.as_str()) {
            continue;
        }

        let file_path = entry.path().to_string_lossy().to_string();

        match std::fs::read_to_string(entry.path()) {
            Ok(content) => {
                let processed_content = match ext.as_str() {
                    "html" | "htm" => html_to_text(&content),
                    _ => content,
                };

                let title = extract_title(&processed_content, &ext);
                let chunks = smart_chunk(&processed_content, chunk_size, overlap);
                let iri = generate_iri("file", &file_path);

                let mut indexed_chunks = Vec::new();
                for (i, chunk) in chunks.iter().enumerate() {
                    match index_chunk(chunk, &iri, i, &tags, "file", &file_path, title.as_deref()) {
                        Ok(chunk_iri) => indexed_chunks.push(chunk_iri),
                        Err(e) => errors.push(format!("{}: {}", file_path, e)),
                    }
                }

                total_chunks += chunks.len();
                total_chars += processed_content.len();

                results.push(json!({
                    "path": file_path,
                    "iri": iri,
                    "chunks": chunks.len(),
                    "title": title,
                }));
            }
            Err(e) => errors.push(format!("{}: {}", file_path, e)),
        }
    }

    Ok(json!({
        "source": "directory",
        "path": params.path,
        "files_processed": results.len(),
        "total_chunks": total_chunks,
        "total_chars": total_chars,
        "results": results,
        "errors": errors,
        "tags": tags,
        "success": errors.is_empty(),
    }))
}

pub async fn execute_knowledge_list(input: Value) -> Result<Value, String> {
    let params: KnowledgeListInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    ensure_index_dir()?;

    let index_dir = rag_index_dir();
    let entries = std::fs::read_dir(index_dir)
        .map_err(|e| format!("Failed to read index directory: {}", e))?;

    let mut results = Vec::new();
    let limit = params.limit.unwrap_or(100);
    let offset = params.offset.unwrap_or(0);

    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().ends_with(".json") {
            continue;
        }

        let content = std::fs::read_to_string(entry.path())
            .map_err(|e| format!("Failed to read index file: {}", e))?;
        let doc: Value = serde_json::from_str(&content).unwrap_or_default();

        if let Some(ref filter_tags) = params.tags {
            let empty_vec = vec![];
            let doc_tags = doc["tags"].as_array().unwrap_or(&empty_vec);
            let has_tag = filter_tags
                .iter()
                .any(|t| doc_tags.iter().any(|dt| dt.as_str() == Some(t.as_str())));
            if !has_tag {
                continue;
            }
        }

        if let Some(ref source_type) = params.source_type {
            if doc["source"]["type"].as_str() != Some(source_type.as_str()) {
                continue;
            }
        }

        results.push(json!({
            "iri": doc["iri"].as_str().unwrap_or(""),
            "parent_iri": doc["parent_iri"].as_str().unwrap_or(""),
            "title": doc["title"].as_str().unwrap_or(""),
            "tags": doc["tags"].as_array().unwrap_or(&vec![]),
            "source": doc["source"].clone(),
            "indexed_at": doc["indexed_at"].as_str().unwrap_or(""),
            "char_count": doc["char_count"].as_u64().unwrap_or(0),
            "chunk_index": doc["chunk_index"].as_u64().unwrap_or(0),
        }));
    }

    let total = results.len();
    results.sort_by(|a, b| {
        b["indexed_at"]
            .as_str()
            .unwrap_or("")
            .cmp(&a["indexed_at"].as_str().unwrap_or(""))
    });

    let paginated: Vec<_> = results.into_iter().skip(offset).take(limit).collect();

    Ok(json!({
        "results": paginated,
        "total": total,
        "limit": limit,
        "offset": offset,
    }))
}

pub async fn execute_knowledge_delete(input: Value) -> Result<Value, String> {
    let params: KnowledgeDeleteInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    ensure_index_dir()?;

    let index_dir = rag_index_dir();
    let entries = std::fs::read_dir(index_dir)
        .map_err(|e| format!("Failed to read index directory: {}", e))?;

    let mut deleted = Vec::new();
    let mut deleted_count = 0;

    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().ends_with(".json") {
            continue;
        }

        let content = std::fs::read_to_string(entry.path())
            .map_err(|e| format!("Failed to read index file: {}", e))?;
        let doc: Value = serde_json::from_str(&content).unwrap_or_default();

        let should_delete = if params.all.unwrap_or(false) {
            true
        } else if let Some(ref iri) = params.iri {
            doc["iri"].as_str() == Some(iri.as_str())
                || doc["parent_iri"].as_str() == Some(iri.as_str())
        } else if let Some(ref filter_tags) = params.tags {
            let empty = vec![];
            let doc_tags = doc["tags"].as_array().unwrap_or(&empty);
            filter_tags
                .iter()
                .any(|t| doc_tags.iter().any(|dt| dt.as_str() == Some(t.as_str())))
        } else {
            false
        };

        if should_delete {
            let iri = doc["iri"].as_str().unwrap_or("").to_string();
            std::fs::remove_file(entry.path()).ok();
            deleted.push(iri);
            deleted_count += 1;
        }
    }

    Ok(json!({
        "deleted_count": deleted_count,
        "deleted_iris": deleted,
        "success": true,
    }))
}

pub async fn execute_knowledge_search(input: Value) -> Result<Value, String> {
    let params: KnowledgeSearchInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    ensure_index_dir()?;

    let query = params.query.to_lowercase();
    let keywords: Vec<&str> = query.split_whitespace().collect();
    let limit = params.limit.unwrap_or(10);
    let min_score = params.min_score.unwrap_or(0.1);

    let index_dir = rag_index_dir();
    let entries = std::fs::read_dir(index_dir)
        .map_err(|e| format!("Failed to read index directory: {}", e))?;

    let mut results = Vec::new();

    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().ends_with(".json") {
            continue;
        }

        let content = std::fs::read_to_string(entry.path())
            .map_err(|e| format!("Failed to read index file: {}", e))?;
        let doc: Value = serde_json::from_str(&content).unwrap_or_default();

        if let Some(ref filter_tags) = params.tags {
            let empty = vec![];
            let doc_tags = doc["tags"].as_array().unwrap_or(&empty);
            let has_tag = filter_tags
                .iter()
                .any(|t| doc_tags.iter().any(|dt| dt.as_str() == Some(t.as_str())));
            if !has_tag {
                continue;
            }
        }

        let text = doc["content"].as_str().unwrap_or("").to_lowercase();
        let title = doc["title"].as_str().unwrap_or("").to_lowercase();

        let content_score = keywords.iter().filter(|kw| text.contains(*kw)).count() as f64
            / keywords.len().max(1) as f64;

        let title_score = keywords.iter().filter(|kw| title.contains(*kw)).count() as f64
            / keywords.len().max(1) as f64;

        let score = content_score * 0.7 + title_score * 0.3;

        if score >= min_score {
            results.push(json!({
                "iri": doc["iri"].as_str().unwrap_or(""),
                "parent_iri": doc["parent_iri"].as_str().unwrap_or(""),
                "title": doc["title"].as_str().unwrap_or(""),
                "content": doc["content"].as_str().unwrap_or(""),
                "tags": doc["tags"].as_array().unwrap_or(&vec![]),
                "source": doc["source"].clone(),
                "score": score,
                "chunk_index": doc["chunk_index"].as_u64().unwrap_or(0),
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

    results.truncate(limit);

    Ok(json!({
        "query": params.query,
        "results": results,
        "count": results.len(),
        "keywords": keywords,
    }))
}

pub async fn execute_knowledge_update(input: Value) -> Result<Value, String> {
    let params: KnowledgeUpdateInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    ensure_index_dir()?;

    let index_dir = rag_index_dir();
    let entries = std::fs::read_dir(index_dir)
        .map_err(|e| format!("Failed to read index directory: {}", e))?;

    let mut updated_count = 0;

    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().ends_with(".json") {
            continue;
        }

        let content = std::fs::read_to_string(entry.path())
            .map_err(|e| format!("Failed to read index file: {}", e))?;
        let mut doc: Value = serde_json::from_str(&content).unwrap_or_default();

        if doc["iri"].as_str() == Some(params.iri.as_str())
            || doc["parent_iri"].as_str() == Some(params.iri.as_str())
        {
            if let Some(ref new_content) = params.content {
                doc["content"] = json!(new_content);
                doc["char_count"] = json!(new_content.len());
            }

            if let Some(ref new_tags) = params.tags {
                if params.append_tags.unwrap_or(false) {
                    let mut existing_tags: Vec<String> = doc["tags"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .filter_map(|t| t.as_str().map(|s| s.to_string()))
                        .collect();
                    for tag in new_tags {
                        if !existing_tags.contains(tag) {
                            existing_tags.push(tag.clone());
                        }
                    }
                    doc["tags"] = json!(existing_tags);
                } else {
                    doc["tags"] = json!(new_tags);
                }
            }

            doc["updated_at"] = json!(Utc::now().to_rfc3339());

            let json_str = serde_json::to_string_pretty(&doc)
                .map_err(|e| format!("Failed to serialize updated document: {}", e))?;
            std::fs::write(entry.path(), json_str)
                .map_err(|e| format!("Failed to write index file: {}", e))?;

            updated_count += 1;
        }
    }

    Ok(json!({
        "iri": params.iri,
        "updated_count": updated_count,
        "success": updated_count > 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_chunk() {
        let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = smart_chunk(content, 20, 5);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_html_to_text() {
        let html =
            "<html><head><title>Test</title></head><body><p>Hello <b>World</b></p></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Hello"));
        assert!(!text.contains("<"));
    }

    #[test]
    fn test_extract_title_markdown() {
        let content = "# My Title\n\nSome content";
        let title = extract_title(content, "md");
        assert_eq!(title, Some("My Title".to_string()));
    }
}
