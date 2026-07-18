use std::collections::HashMap;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::knowledge_graph::code_ast::CodeAstExtractor;
use crate::knowledge_graph::extractor::KnowledgeExtractor;
use crate::knowledge_graph::ontology::OntologyManager;
use crate::knowledge_graph::rdf_mapper::RdfMapper;
use crate::knowledge_graph::bridge::KnowledgeBridge;
use crate::knowledge_graph::store::KnowledgeGraphStore;
use crate::skill_graph::graph_store::SkillGraphStore;
use crate::knowledge_graph::types::{BridgeRelationType, NodeDef, EdgeDef, RdfQuad, RdfValue};
use crate::utils::text::safe_truncate;

use super::{GlobSearchInput, GrepSearchInput, ToolSearchInput, WebFetchInput, WebSearchInput};

// ========== Tool implementations ==========

pub(super) async fn execute_glob_search(input: Value) -> Result<Value, String> {
    let params: GlobSearchInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let root = params.path.as_deref().unwrap_or(".");

    // check if search path is within workspace
    if root != "." {
        if let Err(msg) = check_path_in_workspace(root) {
            return Err(format!(
                "{}\nPlease focus on the current workspace, search within the working directory.",
                msg
            ));
        }
    }

    let mut files = Vec::new();
    let glob_pattern = if root != "." {
        format!("{}/{}", root.trim_end_matches('/'), &params.pattern)
    } else {
        params.pattern.clone()
    };
    match glob::glob(&glob_pattern) {
        Ok(entries) => {
            for entry in entries.flatten() {
                if let Some(p) = entry.to_str() {
                    files.push(p.to_string());
                }
            }
        }
        Err(e) => return Err(format!("Glob error: {}", e)),
    }
    files.sort();
    files.dedup();
    Ok(json!({ "files": files, "count": files.len(), "pattern": params.pattern }))
}

pub(super) async fn execute_grep_search(input: Value) -> Result<Value, String> {
    let params: GrepSearchInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    let root = params.path.as_deref().unwrap_or(".");
    let mode = params.output_mode.as_deref().unwrap_or("files_with_matches");
    if mode != "files_with_matches" && mode != "content" && mode != "count" {
        return Err(format!("Invalid output_mode: {}. Must be files_with_matches, content, or count", mode));
    }

    let ci = params.case_insensitive.unwrap_or(false);
    let ml = params.multiline.unwrap_or(false);
    let re = regex::RegexBuilder::new(&params.pattern)
        .case_insensitive(ci)
        .multi_line(ml)
        .dot_matches_new_line(ml)
        .build()
        .map_err(|e| format!("Invalid regex: {}", e))?;

    let before = params.before.or(params.context).unwrap_or(0);
    let after = params.after.or(params.context).unwrap_or(0);
    let show_line_numbers = params.line_numbers.unwrap_or(true);
    let limit = params.head_limit.unwrap_or(250);
    let offset = params.offset.unwrap_or(0);

    let file_glob = resolve_file_glob(params.glob.as_deref(), params.file_type.as_deref());

    let mut filenames: Vec<String> = Vec::new();
    let mut total_matches: usize = 0;
    let mut files_with_matches: usize = 0;
    let mut all_matches: Vec<(String, usize, String)> = Vec::new();
    let mut file_contents: HashMap<String, String> = HashMap::new();

    for entry in walkdir::WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let path_str = path.to_string_lossy().to_string();

        if !match_glob(&path_str, &file_glob) {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut file_had_match = false;
        for (line_idx, line) in content.lines().enumerate() {
            if !re.is_match(line) {
                continue;
            }
            total_matches += 1;
            if !file_had_match {
                files_with_matches += 1;
                filenames.push(path_str.clone());
                file_had_match = true;
            }
            all_matches.push((path_str.clone(), line_idx, line.to_string()));
        }

        if file_had_match {
            file_contents.insert(path_str.clone(), content);
        }
    }

    let skipped = std::cmp::min(offset, all_matches.len());
    let selected: Vec<_> = all_matches.iter().skip(skipped).take(limit).collect();
    let applied_limit = limit;
    let applied_offset = offset;

    match mode {
        "files_with_matches" => {
            let limited_files: Vec<String> = filenames.iter().skip(skipped).take(limit).cloned().collect();
            Ok(json!({
                "mode": "files_with_matches",
                "num_files": files_with_matches,
                "filenames": limited_files,
                "applied_limit": applied_limit,
                "applied_offset": applied_offset,
            }))
        }
        "content" => {
            let mut output_parts: Vec<String> = Vec::new();
            for (file, line_idx, _line) in &selected {
                let content = match file_contents.get(file) {
                    Some(c) => c,
                    None => continue,
                };
                let lines_vec: Vec<&str> = content.lines().collect();
                let start = line_idx.saturating_sub(before);
                let end = std::cmp::min(line_idx + after + 1, lines_vec.len());
                for i in start..end {
                    let prefix = if show_line_numbers {
                        format!("{}:{}:", file, i + 1)
                    } else {
                        format!("{}:", file)
                    };
                    output_parts.push(format!("{}{}", prefix, lines_vec[i]));
                }
            }
            Ok(json!({
                "mode": "content",
                "num_files": files_with_matches,
                "filenames": filenames,
                "content": output_parts.join("\n"),
                "num_matches": total_matches,
                "applied_limit": applied_limit,
                "applied_offset": applied_offset,
            }))
        }
        "count" => {
            let mut file_counts: Vec<Value> = Vec::new();
            for fname in &filenames {
                let content = match file_contents.get(fname) {
                    Some(c) => c,
                    None => continue,
                };
                let cnt = content.lines().filter(|l| re.is_match(l)).count();
                file_counts.push(json!({"file": fname, "count": cnt}));
            }
            let limited_counts: Vec<Value> = file_counts.into_iter().skip(skipped).take(limit).collect();
            Ok(json!({
                "mode": "count",
                "num_files": files_with_matches,
                "num_matches": total_matches,
                "counts": limited_counts,
                "applied_limit": applied_limit,
                "applied_offset": applied_offset,
            }))
        }
        _ => Err("Unreachable".to_string()),
    }
}

fn resolve_file_glob(glob: Option<&str>, file_type: Option<&str>) -> String {
    if let Some(ft) = file_type {
        let ext = match ft {
            "rust" => "*.rs",
            "python" => "*.py",
            "javascript" => "*.js",
            "typescript" => "*.ts",
            "java" => "*.java",
            "c" => "*.c",
            "cpp" => "*.cpp",
            "go" => "*.go",
            "ruby" => "*.rb",
            "swift" => "*.swift",
            "kotlin" => "*.kt",
            "scala" => "*.scala",
            "haskell" => "*.hs",
            "lua" => "*.lua",
            "perl" => "*.pl",
            "php" => "*.php",
            "shell" => "*.sh",
            "sql" => "*.sql",
            "html" => "*.html",
            "css" => "*.css",
            "json" => "*.json",
            "yaml" => "*.yml",
            "toml" => "*.toml",
            "xml" => "*.xml",
            "markdown" => "*.md",
            "dockerfile" => "Dockerfile",
            _ => ft,
        };
        return ext.to_string();
    }
    glob.unwrap_or("*").to_string()
}

fn match_glob(path: &str, pattern: &str) -> bool {
    if pattern == "*" || pattern == "**" {
        return true;
    }
    let file_name = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if let Ok(glob_matcher) = glob::Pattern::new(pattern) {
        return glob_matcher.matches(file_name);
    }
    file_name.contains(pattern.trim_start_matches('*'))
}

pub(super) async fn execute_web_fetch(input: Value) -> Result<Value, String> {
    let params: WebFetchInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let started = Instant::now();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("HTTP client: {}", e))?;

    let mut last_err = String::new();
    let mut resp = None;
    for attempt in 0..3 {
        match client.get(&params.url).send().await {
            Ok(r) => {
                resp = Some(r);
                break;
            }
            Err(e) => {
                last_err = format!("Request (attempt {}/3): {}", attempt + 1, e);
                tracing::warn!("Web fetch attempt {}/3 failed: {}", attempt + 1, e);
                if attempt < 2 {
                    tokio::time::sleep(std::time::Duration::from_secs(1 << attempt)).await;
                }
            }
        }
    }
    let resp = resp.ok_or_else(|| last_err)?;

    let code = resp.status().as_u16();
    if code >= 400 {
        return Err(format!(
            "HTTP {} {} — target URL returned an error. Please verify the URL or use web_search to find an accessible link.",
            code,
            resp.status().canonical_reason().unwrap_or("Unknown")
        ));
    }

    let ct = resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let content_length = resp.content_length().unwrap_or(0);
    if content_length > 10_000_000 {
        return Err(format!(
            "Content too large ({} bytes, max 10MB). Use a more specific URL or bash curl for chunked download.",
            content_length
        ));
    }

    let body = match resp.bytes().await {
        Ok(b) => {
            if b.len() > 10_000_000 {
                return Err(format!(
            "Content too large ({} bytes, max 10MB). Use a more specific URL or bash curl for chunked download.",
                    b.len()
                ));
            }
            String::from_utf8_lossy(&b).to_string()
        }
        Err(e) => return Err(format!("Failed to read response body: {}", e)),
    };

    let content = if ct.contains("html") { html_to_text(&body) } else { safe_truncate(&body, 8000).to_string() };

    Ok(json!({
        "url": params.url, "status_code": code,
        "content": content, "content_type": ct,
        "duration_ms": started.elapsed().as_millis(),
    }))
}

/// Execute search using Exa API (preferred, requires EXA_API_KEY env var).
/// Return format compatible with execute_web_search.
pub(super) async fn execute_exa_search(query: &str, started: Instant) -> Result<Value, String> {
    let api_key = std::env::var("EXA_API_KEY")
        .map_err(|_| "EXA_API_KEY not set".to_string())?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("HTTP client: {}", e))?;

    let resp = client.post("https://api.exa.ai/search")
        .header("x-api-key", &api_key)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "query": query,
            "type": "auto",
            "numResults": 8,
            "highlights": {"maxCharacters": 2000}
        }))
        .send()
        .await
        .map_err(|e| format!("Exa search request failed: {}", e))?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().await
        .map_err(|e| format!("Exa response parse failed: {}", e))?;

    if !status.is_success() {
        let error_msg = body.get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown error");
        return Ok(json!({
            "query": query,
            "duration_seconds": started.elapsed().as_secs_f64(),
            "results": [],
            "error": format!("Exa API returned {}: {}", status.as_u16(), error_msg),
            "suggestion": "Exa search unavailable, please check API Key or network connection"
        }));
    }

    let results: Vec<Value> = body.get("results")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter().map(|r| {
                json!({
                    "title": r.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                    "url": r.get("url").and_then(|v| v.as_str()).unwrap_or(""),
                    "snippet": r.get("highlights")
                        .and_then(|v| v.as_array())
                        .and_then(|h| h.first())
                        .and_then(|v| v.as_str())
                        .unwrap_or_else(|| {
                            r.get("text").and_then(|v| v.as_str()).unwrap_or("")
                        }),
                })
            }).collect()
        })
        .unwrap_or_default();

    Ok(json!({
        "query": query,
        "duration_seconds": started.elapsed().as_secs_f64(),
        "results": results,
    }))
}

pub(super) async fn execute_web_search(input: Value) -> Result<Value, String> {
    let params: WebSearchInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let started = Instant::now();

    if std::env::var("EXA_API_KEY").is_ok() {
        let result = execute_exa_search(&params.query, started).await;
        match result {
            Ok(v) => {
                if v.get("error").is_none() {
                    return Ok(v);
                }
                tracing::warn!(
                    "Exa search failed ({}), falling back to DuckDuckGo",
                    v.get("error").and_then(|e| e.as_str()).unwrap_or("Unknown error")
                );
            }
            Err(e) => {
                tracing::warn!("Exa search error: {}, falling back to DuckDuckGo", e);
            }
        };
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()
        .map_err(|e| format!("HTTP client: {}", e))?;

    // all search fallbacks inside one async block; any network error is caught
    // by the match Err(e) => Ok(json!({error, suggestion})) guard below,
    // preventing Plan from exiting on network failure.
    let html_result: Result<(reqwest::StatusCode, String, String), String> = (async {
        // prefer DuckDuckGo Lite
        let lite_url = format!("https://lite.duckduckgo.com/lite/?q={}", urlencode(&params.query));
        let resp = client.get(&lite_url)
            .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            .send().await.map_err(|e| format!("Search: {}", e))?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| format!("Read: {}", e))?;

        if status.as_u16() == 200 && (body.contains("result-link") || body.contains("result__a")) {
            return Ok((status, body, "lite".to_string()));
        }

        // fallback: DuckDuckGo HTML
        let html_url = format!("https://html.duckduckgo.com/html/?q={}", urlencode(&params.query));
        let resp = client.get(&html_url)
            .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            .send().await.map_err(|e| format!("Search: {}", e))?;
        let status2 = resp.status();
        let body2 = resp.text().await.map_err(|e| format!("Read: {}", e))?;

        if status2.as_u16() == 200 && body2.contains("result__a") {
            return Ok((status2, body2, "html".to_string()));
        }

        // fallback: DuckDuckGo Instant Answer API
        let api_url = format!("https://api.duckduckgo.com/?q={}&format=json&no_html=1", urlencode(&params.query));
        let resp = client.get(&api_url).send().await.map_err(|e| format!("API: {}", e))?;
        let api_body = resp.text().await.map_err(|e| format!("Read: {}", e))?;
        Ok((status, api_body, "api".to_string()))
    }).await;

    match html_result {
        Ok((status, body, source)) => {
            let mut results = Vec::new();

            if source == "lite" {
                results = extract_ddg_lite_results(&body);
                if results.is_empty() {
                    results = extract_ddg_results(&body);
                }
            } else if source == "html" {
                results = extract_ddg_results(&body);
            } else if source == "api" {
                results = extract_ddg_api_results(&body);
            }

            if results.is_empty() && status.as_u16() != 200 && source != "api" {
                return Ok(json!({
                    "query": params.query,
                    "duration_seconds": started.elapsed().as_secs_f64(),
                    "results": [],
                    "error": format!("Search engine returned non-200 status code: {}", status),
                    "suggestion": "Web search unavailable, please answer based on your own knowledge"
                }));
            }

            if let Some(ref allowed) = params.allowed_domains {
                results.retain(|r| {
                    r["url"].as_str().map_or(false, |url| {
                        allowed.iter().any(|d| url.contains(d.as_str()))
                    })
                });
            }
            if let Some(ref blocked) = params.blocked_domains {
                results.retain(|r| {
                    r["url"].as_str().map_or(true, |url| {
                        !blocked.iter().any(|d| url.contains(d.as_str()))
                    })
                });
            }
            results.truncate(8);
            Ok(json!({
                "query": params.query,
                "duration_seconds": started.elapsed().as_secs_f64(),
                "results": results,
            }))
        }
        Err(e) => {
            Ok(json!({
                "query": params.query,
                "duration_seconds": started.elapsed().as_secs_f64(),
                "results": [],
                "error": format!("Search request failed: {}", e),
                "suggestion": "Web search unavailable, please answer based on your own knowledge"
            }))
        }
    }
}

pub(super) async fn execute_tool_search(input: Value) -> Result<Value, String> {
    let params: ToolSearchInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let q = params.query.to_lowercase();
    let max = params.max_results.unwrap_or(10);
    let all = vec![
        ("glob_search", "Find files by glob pattern. Supports **, *, ? wildcards. Part of system:skills namespace."),
        ("grep_search", "Search file contents with a regex pattern. Part of system:skills namespace."),
        ("web_fetch", "Fetch a URL and convert it into readable text. Network tool."),
        ("web_search", "Search the web for current information. Network tool."),
        ("tool_search", "Search available tools by name or keyword. System tool."),
    ];
    let matches: Vec<Value> = all.iter()
        .filter(|(n, d)| n.to_lowercase().contains(&q) || d.to_lowercase().contains(&q))
        .take(max)
        .map(|(n, d)| json!({"name": n, "description": d, "source": "system:skills"}))
        .collect();
    Ok(json!({"matches": matches, "count": matches.len(), "query": params.query}))
}

// ===== File and Bash tool inputs =====

#[derive(Debug, Deserialize)]
struct FileReadInput {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct FileWriteInput {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct FileListInput {
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BashInput {
    command: String,
    #[allow(dead_code)]
    description: Option<String>,
    timeout: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct FileEditInput {
    path: String,
    old_string: String,
    new_string: String,
    replace_all: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PowerShellInput {
    command: String,
    timeout: Option<u64>,
    description: Option<String>,
    run_in_background: Option<bool>,
}

pub(super) async fn execute_file_read(input: Value) -> Result<Value, String> {
    let params: FileReadInput = serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let path = &params.path;
    let path_obj = std::path::Path::new(path);

    // directory not readable → guide LLM to use file_list to view contents
    if path_obj.is_dir() {
        return Err(format!(
            "Read error: \"{}\" is a directory and cannot be read directly. Use file_list(\"{}\") to view files in this directory, then retry with the correct filename.",
            path, path
        ));
    }

    // check if file is within workspace, provide helpful message
    if let Err(scope_msg) = check_path_in_workspace(path) {
        return Err(scope_msg);
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            let hint = if !path_obj.exists() {
                // file not found → auto-list parent directory contents to help LLM find correct filename
                let parent = path_obj.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| std::path::PathBuf::from("."));
                let parent_display = parent.display().to_string();
                let listing;
                if let Ok(entries) = std::fs::read_dir(&parent) {
                    let files: Vec<String> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| {
                            let kind = if e.file_type().map(|t| t.is_dir()).unwrap_or(false) { "[dir]" } else { "[file]" };
                            format!("  {} {}", kind, e.file_name().to_string_lossy())
                        })
                        .collect();
                    if files.is_empty() {
                        listing = format!("\nDirectory {} is empty.", parent_display);
                    } else {
                        listing = format!("\nFiles in directory {}:\n{}", parent_display, files.join("\n"));
                    }
                } else {
                    listing = format!("\nDirectory {} does not exist either. Use file_list(\".\") to see workspace root structure.", parent_display);
                }
                format!(
                    "Read error: {}.\nFile \"{}\" does not exist. {}\nPlease verify the filename and path, then retry.",
                    e, path, listing
                )
            } else if e.kind() == std::io::ErrorKind::InvalidData {
                // binary/non-UTF-8 file → guide LLM to use bash tool instead
                format!(
                    "Read error: file \"{}\" contains binary/non-text content and cannot be read directly.\n\
                     To check file type: bash(\"file '{}'\")\n\
                     To check file size: bash(\"ls -lh '{}'\")\n\
                     To view beginning (text embedded in binary): bash(\"head -c 200 '{}' | strings\")\n\
                     Please focus on workspace files relevant to the current task.",
                    path, path, path, path
                )
            } else {
                format!("Read error: {}", e)
            };
            return Err(hint);
        }
    };
    let lines: Vec<&str> = content.lines().collect();
    let start = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(usize::MAX);
    let selected: Vec<String> = lines.iter().skip(start).take(limit).map(|l| l.to_string()).collect();
    let total = lines.len();
    Ok(json!({
        "path": params.path, "total_lines": total,
        "offset": start, "lines": selected, "returned": selected.len(),
    }))
}

pub(super) async fn execute_file_write(input: Value) -> Result<Value, String> {
    let params: FileWriteInput = serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    check_path_in_workspace(&params.path)?;
    if let Some(parent) = std::path::Path::new(&params.path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Mkdir error: {}", e))?;
    }
    std::fs::write(&params.path, &params.content).map_err(|e| format!("Write error: {}", e))?;
    Ok(json!({"path": params.path, "bytes_written": params.content.len(), "success": true}))
}

pub(super) async fn execute_file_list(input: Value) -> Result<Value, String> {
    let params: FileListInput = serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let dir = params.path.as_deref().unwrap_or(".");

    // check if within workspace
    if dir != "." {
        if let Err(msg) = check_path_in_workspace(dir) {
            return Err(format!(
                "{}\nPlease focus on the current workspace ({}), list files under the working directory.",
                msg,
                std::env::current_dir().map(|d| d.display().to_string()).unwrap_or_else(|_| ".".to_string())
            ));
        }
    }

    let mut entries = Vec::new();
    let read_dir = std::fs::read_dir(dir).map_err(|e| format!("List error: {}", e))?;
    for entry in read_dir.flatten() {
        let ft = entry.file_type().ok();
        let kind = if ft.map_or(false, |t| t.is_dir()) { "dir" } else { "file" };
        if let Ok(name) = entry.file_name().into_string() {
            entries.push(json!({"name": name, "type": kind}));
        }
    }
    Ok(json!({"path": dir, "entries": entries, "count": entries.len()}))
}

pub(super) async fn execute_bash(input: Value) -> Result<Value, String> {
    // Windows: delegate to execute_powershell, LLM doesn't need to know
    #[cfg(windows)]
    {
        let params: BashInput = serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
        let ps_input = serde_json::to_value(PowerShellInput {
            command: params.command,
            timeout: params.timeout,
            description: params.description,
            run_in_background: None,
        }).map_err(|e| format!("Serialize error: {}", e))?;
        return execute_powershell(ps_input).await;
    }

    // Unix: execute via sh -c
    #[cfg(not(windows))]
    {
        let params: BashInput = serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
        use std::process::Command;
        use std::sync::{Arc, Mutex};
        use std::thread;
        let timeout_ms = params.timeout.unwrap_or(60_000);

        let spawn_child = |cmd: &str| -> Result<std::process::Child, String> {
            let mut c = Command::new("sh");
            c.arg("-c")
                .arg(cmd)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            #[cfg(unix)]
            {
                c.process_group(0);
            }
            c.spawn().map_err(|e| format!("Spawn error: {}", e))
        };

        let spawn_readers = |child: &mut std::process::Child|
            -> (Arc<Mutex<String>>, Arc<Mutex<String>>)
        {
            let out_buf = Arc::new(Mutex::new(String::new()));
            let err_buf = Arc::new(Mutex::new(String::new()));
            if let Some(stdout) = child.stdout.take() {
                let buf = out_buf.clone();
                thread::spawn(move || {
                    if let Ok(output) = std::io::read_to_string(stdout) {
                        *buf.lock().expect("out_buf Mutex poisoned") = output;
                    }
                });
            }
            if let Some(stderr) = child.stderr.take() {
                let buf = err_buf.clone();
                thread::spawn(move || {
                    if let Ok(output) = std::io::read_to_string(stderr) {
                        *buf.lock().expect("err_buf Mutex poisoned") = output;
                    }
                });
            }
            (out_buf, err_buf)
        };

        let mut child = spawn_child(&params.command)?;
        let (mut stdout_buf, mut stderr_buf) = spawn_readers(&mut child);

        let mut start = std::time::Instant::now();
        let mut max_dur = std::time::Duration::from_millis(timeout_ms);
        let mut attempts = 0u32;
        let result = loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    // brief pause for reader threads to finish
                    thread::sleep(std::time::Duration::from_millis(50));
                    let stdout = stdout_buf.lock().expect("stdout_buf Mutex poisoned").clone();
                    let stderr = stderr_buf.lock().expect("stderr_buf Mutex poisoned").clone();
                    let code = status.code().unwrap_or(-1);
                    break json!({
                        "command": params.command, "exit_code": code,
                        "stdout": stdout, "stderr": stderr,
                        "duration_ms": start.elapsed().as_millis() as u64,
                    });
                }
                Ok(None) => {
                    if start.elapsed() > max_dur {
                        if attempts == 0 {
                            tracing::warn!(
                                "[execute_bash] command timed out after {}ms, retrying with {}ms timeout",
                                timeout_ms, timeout_ms * 2,
                            );
                            let _ = child.kill();
                            kill_process_group(&child);
                            child = spawn_child(&params.command)?;
                            let (o, e) = spawn_readers(&mut child);
                            stdout_buf = o;
                            stderr_buf = e;
                            max_dur = std::time::Duration::from_millis(timeout_ms * 2);
                            start = std::time::Instant::now();
                            attempts = 1;
                            continue;
                        }
                        let _ = child.kill();
                        kill_process_group(&child);
                        let stdout = stdout_buf.lock().expect("stdout_buf Mutex poisoned").clone();
                        let stderr = stderr_buf.lock().expect("stderr_buf Mutex poisoned").clone();
                        break json!({
                            "command": params.command, "timed_out": true,
                            "stdout": stdout, "stderr": stderr,
                            "error": format!("Timeout after {}ms", timeout_ms * 2),
                        });
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => break json!({"command": params.command, "error": e.to_string()}),
            }
        };
        Ok(result)
    }
}

#[cfg(unix)]
fn kill_process_group(child: &std::process::Child) {
    let pgid = child.id();
    let _ = std::process::Command::new("kill")
        .arg(format!("-{}", pgid))
        .output();
}

#[cfg(not(unix))]
fn kill_process_group(_child: &std::process::Child) {}

/// Check if a path is within the current working directory (workspace).
/// Returns an error if the path is outside the workspace.
/// For non-existent paths (e.g. file_write creating new files), checks the parent directory.
fn check_path_in_workspace(path: &str) -> Result<(), String> {
    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cwd_canonical = match cwd.canonicalize() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    let requested = std::path::Path::new(path);
    // relative path: join with cwd then resolve; absolute path: resolve directly
    let requested_abs = if requested.is_relative() {
        cwd.join(requested)
    } else {
        requested.to_path_buf()
    };
    // try to canonicalize path; if file doesn't exist, check parent directory is within workspace
    let check_path = match requested_abs.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            // file doesn't exist, check parent
            match requested_abs.parent() {
                Some(parent) => match parent.canonicalize() {
                    Ok(p) => p,
                    Err(_) => return Ok(()), // parent also doesn't exist, don't block
                },
                None => return Ok(()), // no parent (e.g. root path), don't block
            }
        }
    };

    if !check_path.starts_with(&cwd_canonical) {
        return Err(format!(
            "Path is not within the workspace: {}. The current task should only access files in the workspace.",
            check_path.display(),
        ));
    }
    Ok(())
}

pub(super) async fn execute_file_edit(input: Value) -> Result<Value, String> {
    let params: FileEditInput = serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    check_path_in_workspace(&params.path)?;

    let content = std::fs::read_to_string(&params.path).map_err(|e| format!("Read error: {}", e))?;

    let count = content.matches(&params.old_string).count();
    if count == 0 {
        return Err(format!("old_string not found in {}", params.path));
    }
    if count > 1 && !params.replace_all.unwrap_or(false) {
        return Err(format!(
            "old_string found {} times in {}. Set replace_all=true to replace all occurrences.",
            count, params.path
        ));
    }

    let old_lines: Vec<&str> = params.old_string.lines().collect();
    let new_lines: Vec<&str> = params.new_string.lines().collect();

    let diff = generate_diff(&params.path, &old_lines, &new_lines);

    let new_content = if params.replace_all.unwrap_or(false) {
        content.replace(&params.old_string, &params.new_string)
    } else {
        content.replacen(&params.old_string, &params.new_string, 1)
    };

    let replacements = if params.replace_all.unwrap_or(false) { count } else { 1 };

    std::fs::write(&params.path, &new_content).map_err(|e| format!("Write error: {}", e))?;

    Ok(json!({
        "path": params.path,
        "success": true,
        "replacements": replacements,
        "diff": diff,
    }))
}

fn generate_diff(path: &str, old_lines: &[&str], new_lines: &[&str]) -> String {
    let file_name = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
    let mut diff = String::new();
    diff.push_str(&format!("--- a/{}\n", file_name));
    diff.push_str(&format!("+++ b/{}\n", file_name));

    let old_count = old_lines.len();
    let new_count = new_lines.len();
    diff.push_str(&format!("@@ -1,{} +1,{} @@\n", old_count, new_count));

    for line in old_lines {
        diff.push_str(&format!("-{}\n", line));
    }
    for line in new_lines {
        diff.push_str(&format!("+{}\n", line));
    }

    diff
}

pub(super) async fn execute_powershell(input: Value) -> Result<Value, String> {
    let params: PowerShellInput = serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    let exe = if cfg!(target_os = "windows") { "powershell" } else { "pwsh" };

    let exe_path = match which_powershell(exe) {
        Some(p) => p,
        None => return Err(format!("{} not found on this system", exe)),
    };

    let timeout_ms = params.timeout.unwrap_or(60_000);

    let mut child = std::process::Command::new(&exe_path)
        .args(["-NoProfile", "-NonInteractive", "-Command", &params.command])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Spawn error: {}", e))?;

    let start = std::time::Instant::now();
    let max_dur = std::time::Duration::from_millis(timeout_ms);

    let result = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = child.stdout.take()
                    .map(|out| std::io::read_to_string(out).unwrap_or_default())
                    .unwrap_or_default();
                let stderr = child.stderr.take()
                    .map(|err| std::io::read_to_string(err).unwrap_or_default())
                    .unwrap_or_default();
                let code = status.code().unwrap_or(-1);
                break json!({
                    "command": params.command, "exit_code": code,
                    "stdout": stdout, "stderr": stderr,
                    "duration_ms": start.elapsed().as_millis() as u64,
                    "shell": exe,
                });
            }
            Ok(None) => {
                if start.elapsed() > max_dur {
                    let _ = child.kill();
                    break json!({
                        "command": params.command, "timed_out": true,
                        "error": format!("Timeout after {}ms", timeout_ms),
                        "shell": exe,
                    });
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => break json!({"command": params.command, "error": e.to_string(), "shell": exe}),
        }
    };
    Ok(result)
}

fn which_powershell(exe: &str) -> Option<String> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(exe);
        if candidate.exists() {
            return candidate.to_str().map(|s| s.to_string());
        }
        let with_ext = dir.join(format!("{}.exe", exe));
        if with_ext.exists() {
            return with_ext.to_str().map(|s| s.to_string());
        }
    }
    None
}
fn html_to_text(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut prev_space = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => { in_tag = false; if !text.is_empty() && !text.ends_with(' ') { text.push(' '); prev_space = true; } }
            _ if in_tag => {}
            ch if ch.is_whitespace() => { if !prev_space { text.push(' '); prev_space = true; } }
            _ => { text.push(ch); prev_space = false; }
        }
    }
    let decoded = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&quot;", "\"").replace("&nbsp;", " ");
    let mut result = String::with_capacity(decoded.len());
    let mut ps = false;
    for ch in decoded.chars() {
        if ch.is_whitespace() { if !ps { result.push(' '); ps = true; } }
        else { result.push(ch); ps = false; }
    }
    let len = safe_truncate(&result, 8000).len();
    result.truncate(len);
    result
}

fn extract_ddg_results(html: &str) -> Vec<Value> {
    let mut results = Vec::new();
    let mut rem = html;
    while let Some(s) = rem.find("result__a") {
        let after = &rem[s..];
        let Some(hp) = after.find("href=") else { rem = &after[1..]; continue; };
        let hs = &after[hp + 5..];
        let Some((url, rest)) = extract_quoted_str(hs) else { rem = &after[1..]; continue; };
        let Some(ct) = rest.find('>') else { rem = &after[1..]; continue; };
        let at = &rest[ct + 1..];
        let Some(ea) = at.find("</a>") else { rem = &after[1..]; continue; };
        let title = html_to_text(&at[..ea]);
        rem = &at[ea + 4..];
        let snippet = if let Some(sp) = rem.find("result__snippet") {
            let sa = &rem[sp..];
            if let Some(tc) = sa.find('>') {
                let sc = &sa[tc + 1..];
                if let Some(es) = sc.find("</") { html_to_text(&sc[..es]) } else { String::new() }
            } else { String::new() }
        } else { String::new() };
        if !title.trim().is_empty() && (url.starts_with("http://") || url.starts_with("https://")) {
            results.push(json!({"title": title.trim(), "url": url, "snippet": snippet.trim()}));
        }
    }
    results
}

fn extract_ddg_api_results(body: &str) -> Vec<Value> {
    let mut results = Vec::new();
    let Ok(data) = serde_json::from_str::<Value>(body) else { return results; };

    // Abstract result
    if let Some(abstract_text) = data["AbstractText"].as_str() {
        if !abstract_text.is_empty() {
            results.push(json!({
                "title": data["AbstractSource"].as_str().unwrap_or(""),
                "url": data["AbstractURL"].as_str().unwrap_or(""),
                "snippet": abstract_text,
            }));
        }
    }

    // Related topics
    if let Some(topics) = data["RelatedTopics"].as_array() {
        for topic in topics {
            if let Some(text) = topic["Text"].as_str() {
                if !text.is_empty() {
                    results.push(json!({
                        "title": text.split_whitespace().take(5).collect::<Vec<_>>().join(" "),
                        "url": topic["FirstURL"].as_str().unwrap_or(""),
                        "snippet": text,
                    }));
                }
            } else if let Some(sub_topics) = topic["Topics"].as_array() {
                for sub in sub_topics {
                    if let Some(text) = sub["Text"].as_str() {
                        if !text.is_empty() {
                            results.push(json!({
                                "title": text.split_whitespace().take(5).collect::<Vec<_>>().join(" "),
                                "url": sub["FirstURL"].as_str().unwrap_or(""),
                                "snippet": text,
                            }));
    }
}

// ---- L0 store read functions removed: agents should not access L0 directly, use L3 projection instead ----
                }
            }
        }
    }

    results
}

fn extract_ddg_lite_results(html: &str) -> Vec<Value> {
    let mut results = Vec::new();
    let mut rem = html;
    loop {
        let Some(link_start) = rem.find("class=\"result-link\"") else { break; };
        let link_section = &rem[link_start..];

        let Some(href_pos) = link_section.find("href=") else { rem = &link_section[1..]; continue; };
        let href_str = &link_section[href_pos + 5..];
        let Some((url, after_url)) = extract_quoted_str(href_str) else { rem = &link_section[1..]; continue; };

        let Some(gt_pos) = after_url.find('>') else { rem = &link_section[1..]; continue; };
        let title_start = &after_url[gt_pos + 1..];
        let Some(title_end) = title_start.find("</a>") else { rem = &link_section[1..]; continue; };
        let title = html_to_text(&title_start[..title_end]);

        let snippet = if let Some(snip_pos) = after_url.find("class=\"result-snippet\"") {
            let snip_section = &after_url[snip_pos..];
            if let Some(gt) = snip_section.find('>') {
                let snip_text = &snip_section[gt + 1..];
                if let Some(end_tag) = snip_text.find("</td>") {
                    html_to_text(&snip_text[..end_tag])
                } else if let Some(end_tag) = snip_text.find("</") {
                    html_to_text(&snip_text[..end_tag])
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if !title.trim().is_empty() && (url.starts_with("http://") || url.starts_with("https://")) {
            results.push(json!({"title": title.trim(), "url": url, "snippet": snippet.trim()}));
        }

        rem = &title_start[title_end + 4..];
    }
    results
}

fn extract_quoted_str(input: &str) -> Option<(String, &str)> {
    let q = input.chars().next()?;
    if q != '"' && q != '\'' { return None; }
    let rest = &input[q.len_utf8()..];
    let end = rest.find(q)?;
    Some((rest[..end].to_string(), &rest[end + q.len_utf8()..]))
}

fn urlencode(s: &str) -> String {
    s.chars().map(|c| match c {
        'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
        ' ' => "+".to_string(),
        c => format!("%{:02X}", c as u8),
    }).collect()
}

static SKILL_CREATOR_GATEWAY: OnceCell<std::sync::Arc<crate::gateway::unified_gateway::UnifiedGateway>> = OnceCell::new();

#[allow(dead_code)]
pub(crate) fn init_skill_creator_gateway(gateway: std::sync::Arc<crate::gateway::unified_gateway::UnifiedGateway>) {
    let _ = SKILL_CREATOR_GATEWAY.set(gateway);
}

pub(super) async fn execute_create_skill(input: Value, shared_graph: Option<Arc<SkillGraphStore>>) -> Result<Value, String> {
    let description = input["description"].as_str().unwrap_or("").to_string();
    if description.is_empty() {
        return Err("description is required".to_string());
    }

    let skill_name_hint = input["skill_name_hint"].as_str().map(String::from);
    let category_hint = input["category_hint"].as_str().map(String::from);
    let security_level_override = input["security_level_override"].as_str().map(String::from);

    if let Some(gateway) = SKILL_CREATOR_GATEWAY.get() {
        let graph_store = shared_graph.unwrap_or_else(|| Arc::new(crate::skill_graph::SkillGraphStore::new()));
        let registry = std::sync::Arc::new(crate::tools::SkillRegistry::new());
        let config = crate::skill_graph::SkillCreatorConfig::default();
        let creator = crate::skill_graph::SkillCreator::new(
            gateway.clone(), graph_store, registry, config,
        );

        let request = crate::skill_graph::CreateSkillRequest {
            description,
            skill_name_hint,
            category_hint,
            security_level_override,
        };

        let result = creator.create_from_description(request).await
            .map_err(|e| format!("Create Skill failed: {:?}", e))?;

        Ok(json!({
            "skill_iri": result.skill_iri,
            "name": result.name,
            "registered": true,
            "json_ld": result.json_ld,
        }))
    } else {
        let name = skill_name_hint.unwrap_or_else(|| {
            description.split_whitespace().take(2).collect::<Vec<_>>().join("_").to_lowercase()
        });
        let category = category_hint.unwrap_or_else(|| "system".to_string());

        Ok(json!({
            "skill_iri": format!("iri://skills/{}", name),
            "name": name,
            "description": description,
            "category": category,
            "registered": false,
            "note": "Gateway not initialized, returning template only. Use SkillCreator API to create the full Skill."
        }))
    }
}

pub(super) async fn execute_convert_skill(input: Value, shared_graph: Option<Arc<SkillGraphStore>>) -> Result<Value, String> {
    let markdown_content = input["markdown_content"].as_str().unwrap_or("").to_string();
    if markdown_content.is_empty() {
        return Err("markdown_content is required".to_string());
    }
    let source_path = input["source_path"].as_str().map(String::from);

    if let Some(gateway) = SKILL_CREATOR_GATEWAY.get() {
        let graph_store = shared_graph.unwrap_or_else(|| Arc::new(crate::skill_graph::SkillGraphStore::new()));
        let registry = std::sync::Arc::new(crate::tools::SkillRegistry::new());
        let config = crate::skill_graph::SkillCreatorConfig::default();
        let creator = crate::skill_graph::SkillCreator::new(
            gateway.clone(), graph_store, registry, config,
        );

        let request = crate::skill_graph::ConvertMarkdownRequest {
            markdown_content,
            source_path,
        };

        let result = creator.convert_from_markdown(request).await
            .map_err(|e| format!("Convert Skill failed: {:?}", e))?;

        Ok(json!({
            "skill_iri": result.skill_iri,
            "name": result.name,
            "registered": true,
            "json_ld": result.json_ld,
        }))
    } else {
        let def = crate::skill_graph::SkillCreator::convert_markdown_static(&markdown_content)
            .map_err(|e| format!("Static parse failed: {:?}", e))?;

        Ok(json!({
            "skill_iri": format!("iri://skills/{}", def.name),
            "name": def.name,
            "description": def.description,
            "steps": def.steps.len(),
            "tags": def.tags,
            "registered": false,
            "note": "Gateway not initialized, using static parse. Full conversion requires SkillCreator API."
        }))
    }
}

// ========== Knowledge graph tool implementation ==========

pub(super) async fn execute_knowledge_extract(input: Value, kg_store: Arc<RwLock<KnowledgeGraphStore>>) -> Result<Value, String> {
    let text = input["text"].as_str().unwrap_or("").to_string();
    if text.is_empty() {
        return Err("text parameter cannot be empty".to_string());
    }
    let domain = input["domain"].as_str().map(String::from);

    let api_url = std::env::var("ONE_API_URL")
        .or_else(|_| std::env::var("OPENAI_API_BASE"))
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = std::env::var("ONE_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .map_err(|_| "API key not configured: set ONE_API_KEY or OPENAI_API_KEY environment variable".to_string())?;
    let model = std::env::var("KG_EXTRACT_MODEL")
        .unwrap_or_else(|_| "deepseek-v4-flash".to_string());

    let ontology = OntologyManager::new();
    let temp_store = KnowledgeGraphStore::new()
        .map_err(|e| format!("Create temporary store failed: {}", e))?;
    let extractor = KnowledgeExtractor::new(
        ontology,
        temp_store,
        api_url,
        api_key,
        model,
    );

    let result = extractor.extract(&text, domain.as_deref())?;

    let store = kg_store.write().map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
    let graph = store.default_graph();
    store.write_quads(&result.quads, graph)?;

    Ok(json!({
        "success": true,
        "entity_count": result.entity_count,
        "relation_count": result.relation_count,
        "quad_count": result.quads.len(),
        "graph": graph,
    }))
}

pub(super) async fn execute_knowledge_query(input: Value, kg_store: Arc<RwLock<KnowledgeGraphStore>>) -> Result<Value, String> {
    let sparql = input["sparql"].as_str().unwrap_or("").to_string();
    if sparql.is_empty() {
        return Err("sparql parameter cannot be empty".to_string());
    }
    let named_graph = input["named_graph"].as_str().map(String::from);

    let store = kg_store.read().map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
    let results = store.query_sparql(&sparql, named_graph.as_deref())?;

    Ok(json!({
        "success": true,
        "results": results,
        "count": results.len(),
    }))
}

pub(super) async fn execute_knowledge_search(input: Value, kg_store: Arc<RwLock<KnowledgeGraphStore>>) -> Result<Value, String> {
    let keyword = input["keyword"].as_str().unwrap_or("").to_string();
    if keyword.is_empty() {
        return Err("keyword parameter cannot be empty".to_string());
    }
    let entity_type = input["entity_type"].as_str().map(String::from);

    let store = kg_store.read().map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
    let results = store.search_entities(&keyword, entity_type.as_deref())?;

    Ok(json!({
        "success": true,
        "results": results,
        "count": results.len(),
        "keyword": keyword,
    }))
}

pub(super) async fn execute_knowledge_neighbors(input: Value, kg_store: Arc<RwLock<KnowledgeGraphStore>>) -> Result<Value, String> {
    let entity_id = input["entity_id"].as_str().unwrap_or("").to_string();
    if entity_id.is_empty() {
        return Err("entity_id parameter cannot be empty".to_string());
    }
    let depth = input["depth"].as_u64().unwrap_or(1).min(3) as usize;

    let store = kg_store.read().map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
    let result = store.get_neighbors(&entity_id, depth)?;

    Ok(json!({
        "success": true,
        "result": result,
    }))
}

pub(super) async fn execute_knowledge_import_json(input: Value, kg_store: Arc<RwLock<KnowledgeGraphStore>>) -> Result<Value, String> {
    let json_data_str = input["json_data"].as_str().unwrap_or("");
    if json_data_str.is_empty() {
        return Err("json_data parameter cannot be empty".to_string());
    }
    let mapping_config_str = input["mapping_config"].as_str().unwrap_or("");
    if mapping_config_str.is_empty() {
        return Err("mapping_config parameter cannot be empty".to_string());
    }

    let json_data: Value = serde_json::from_str(json_data_str)
        .map_err(|e| format!("json_data JSON parse failed: {}", e))?;
    let mapping: Value = serde_json::from_str(mapping_config_str)
        .map_err(|e| format!("mapping_config JSON parse failed: {}", e))?;

    let id_field = mapping["id_field"].as_str().unwrap_or("id");
    let type_field = mapping["type_field"].as_str().unwrap_or("type");
    let label_field = mapping["label_field"].as_str().unwrap_or("label");
    let desc_field = mapping["description_field"].as_str();

    let items = match json_data {
        Value::Array(arr) => arr,
        Value::Object(_) => vec![json_data],
        _ => return Err("json_data must be a JSON object or array".to_string()),
    };

    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    for item in &items {
        let id = item[id_field].as_str().unwrap_or("").to_string();
        if id.is_empty() {
            continue;
        }
        let node_type = item[type_field].as_str().unwrap_or("Concept").to_string();
        let label = item[label_field].as_str().unwrap_or(&id).to_string();
        let description = desc_field.and_then(|f| item[f].as_str()).map(String::from);

        let mut properties = HashMap::new();
        if let Some(obj) = item.as_object() {
            for (key, value) in obj {
                if key != id_field && key != type_field && key != label_field {
                    if desc_field == Some(key.as_str()) { continue; }
                    properties.insert(key.clone(), value.clone());
                }
            }
        }

        nodes.push(NodeDef {
            id: id.clone(),
            node_type,
            label,
            description,
            properties,
        });

        if let Some(relations) = mapping["relations"].as_array() {
            for rel in relations {
                let field = rel["field"].as_str().unwrap_or("");
                let relation = rel["relation"].as_str().unwrap_or("relatedTo");
                let target_prefix = rel["target_prefix"].as_str().unwrap_or("");

                if let Some(target_val) = item[field].as_str() {
                    let target_id = if target_prefix.is_empty() {
                        target_val.to_string()
                    } else {
                        format!("{}{}", target_prefix.trim_end_matches('/'), target_val.trim_start_matches('/'))
                    };
                    if !target_id.is_empty() {
                        edges.push(EdgeDef {
                            source: id.clone(),
                            target: target_id,
                            relation: relation.to_string(),
                            properties: HashMap::new(),
                        });
                    }
                }
            }
        }
    }

    if nodes.is_empty() {
        return Ok(json!({
            "success": true,
            "entity_count": 0,
            "relation_count": 0,
            "message": "No importable entities found",
        }));
    }

    let graph = {
        let store = kg_store.read().map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
        store.default_graph().to_string()
    };

    let extraction = crate::knowledge_graph::types::LLMExtractionOutput {
        nodes: nodes.clone(),
        edges: edges.clone(),
    };
    let result = RdfMapper::map_extraction(&extraction, &graph);

    {
        let store = kg_store.write().map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
        store.write_quads(&result.quads, &graph)?;
    }

    Ok(json!({
        "success": true,
        "entity_count": result.entity_count,
        "relation_count": result.relation_count,
        "quad_count": result.quads.len(),
        "graph": graph,
    }))
}

pub(super) async fn execute_ontology_register(input: Value, kg_store: Arc<RwLock<KnowledgeGraphStore>>) -> Result<Value, String> {
    let terms = input["terms"].as_array().ok_or("terms parameter must be an array")?;
    if terms.is_empty() {
        return Err("terms array cannot be empty".to_string());
    }

    let graph = "graph:ontology";
    let mut quads = Vec::new();

    for term in terms {
        let iri = term["iri"].as_str().unwrap_or("").to_string();
        let label = term["label"].as_str().unwrap_or("").to_string();
        let description = term["description"].as_str().unwrap_or("").to_string();
        let term_type = term["term_type"].as_str().unwrap_or("").to_string();

        if iri.is_empty() || label.is_empty() {
            continue;
        }

        let type_iri = match term_type.as_str() {
            "Class" => "http://www.w3.org/2000/01/rdf-schema#Class",
            "Property" => "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
            "Relation" => "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
            _ => "http://www.w3.org/2000/01/rdf-schema#Resource",
        };

        quads.push(RdfQuad {
            subject: iri.clone(),
            predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            object: RdfValue::Iri(type_iri.to_string()),
            graph: Some(graph.to_string()),
        });
        quads.push(RdfQuad {
            subject: iri.clone(),
            predicate: "http://www.w3.org/2000/01/rdf-schema#label".to_string(),
            object: RdfValue::Literal(label),
            graph: Some(graph.to_string()),
        });
        if !description.is_empty() {
            quads.push(RdfQuad {
                subject: iri.clone(),
                predicate: "http://www.w3.org/2000/01/rdf-schema#comment".to_string(),
                object: RdfValue::Literal(description),
                graph: Some(graph.to_string()),
            });
        }
        quads.push(RdfQuad {
            subject: iri,
            predicate: "https://agentos.ontology/meta/termType".to_string(),
            object: RdfValue::Literal(term_type),
            graph: Some(graph.to_string()),
        });
    }

    let registered = quads.len() / 3;
    {
        let store = kg_store.write().map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
        store.write_quads(&quads, graph)?;
    }

    Ok(json!({
        "success": true,
        "registered_terms": registered,
        "graph": graph,
    }))
}

pub(super) async fn execute_knowledge_bridge_with_store(input: Value, kg_store: Arc<RwLock<KnowledgeGraphStore>>) -> Result<Value, String> {
    let entity_id = input["entity_id"].as_str().unwrap_or("").to_string();
    if entity_id.is_empty() {
        return Err("entity_id parameter cannot be empty".to_string());
    }
    let skill_iri = input["skill_iri"].as_str().unwrap_or("").to_string();
    if skill_iri.is_empty() {
        return Err("skill_iri parameter cannot be empty".to_string());
    }
    let relation_type_str = input["relation_type"].as_str().unwrap_or("HasSkill");

    let relation = match relation_type_str {
        "HasSkill" => BridgeRelationType::HasSkill,
        "ApplicableIn" => BridgeRelationType::ApplicableIn,
        "RelatedTo" => BridgeRelationType::RelatedTo,
        _ => return Err(format!("Unsupported relation type: {}, options: HasSkill, ApplicableIn, RelatedTo", relation_type_str)),
    };

    // Create a bridge that shares the same underlying Oxigraph store.
    // Both `kg_store` and the bridge operate on the same triples.
    let guard = kg_store.read().map_err(|e| format!("Failed to acquire kg_store lock: {}", e))?;
    let bridge = KnowledgeBridge::from_kg_store(&guard)
        .map_err(|e| format!("Failed to create KnowledgeBridge: {}", e))?;
    drop(guard); // release the read lock before write operations

    bridge.create_bridge(&entity_id, &skill_iri, relation)?;

    Ok(json!({
        "success": true,
        "entity_id": entity_id,
        "skill_iri": skill_iri,
        "relation_type": relation_type_str,
    }))
}

pub(super) async fn execute_knowledge_extract_code(input: Value, kg_store: Arc<RwLock<KnowledgeGraphStore>>) -> Result<Value, String> {
    let file_path = input["file_path"].as_str().unwrap_or("").to_string();
    if file_path.is_empty() {
        return Err("file_path parameter cannot be empty".to_string());
    }
    let graph = input["named_graph"].as_str().unwrap_or("graph:code").to_string();
    let force = input["force"].as_bool().unwrap_or(false);

    let store = kg_store.write().map_err(|e| format!("Failed to acquire storage lock: {}", e))?;

    if force {
        let result = CodeAstExtractor::extract_from_file(&file_path, &graph)?;
        store.delete_quads_by_subject_prefix(&format!("iri://entity/file:{}", file_path), &graph)?;
        store.write_quads(&result.quads, &graph)?;
        Ok(json!({
            "success": true,
            "file_path": file_path,
            "mode": "force",
            "entity_count": result.entity_count,
            "relation_count": result.relation_count,
            "quad_count": result.quads.len(),
            "graph": graph,
        }))
    } else {
        use crate::knowledge_graph::code_ast::IncrementalResult;
        let result = CodeAstExtractor::extract_incremental(&file_path, &graph, &store)?;
        match result {
            IncrementalResult::Unchanged => Ok(json!({
                "success": true,
                "file_path": file_path,
                "mode": "incremental",
                "status": "unchanged",
                "message": "File content unchanged, skipping AST extraction",
            })),
            IncrementalResult::Created { entity_count, relation_count, quad_count } => Ok(json!({
                "success": true,
                "file_path": file_path,
                "mode": "incremental",
                "status": "created",
                "entity_count": entity_count,
                "relation_count": relation_count,
                "quad_count": quad_count,
                "graph": graph,
            })),
            IncrementalResult::Updated { entity_count, relation_count, quad_count, deleted_quads } => Ok(json!({
                "success": true,
                "file_path": file_path,
                "mode": "incremental",
                "status": "updated",
                "entity_count": entity_count,
                "relation_count": relation_count,
                "quad_count": quad_count,
                "deleted_quads": deleted_quads,
                "graph": graph,
            })),
        }
    }
}
