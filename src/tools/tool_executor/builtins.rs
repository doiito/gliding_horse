use std::collections::HashMap;
#[cfg(unix)]
use std::ffi::{CStr, CString, OsStr};
use std::net::{IpAddr, SocketAddr};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
#[cfg(unix)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use futures::StreamExt;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::knowledge_graph::bridge::KnowledgeBridge;
use crate::knowledge_graph::code_ast::CodeAstExtractor;
use crate::knowledge_graph::extractor::KnowledgeExtractor;
use crate::knowledge_graph::ontology::OntologyManager;
use crate::knowledge_graph::rdf_mapper::RdfMapper;
use crate::knowledge_graph::store::KnowledgeGraphStore;
use crate::knowledge_graph::types::{BridgeRelationType, EdgeDef, NodeDef, RdfQuad, RdfValue};
use crate::skill_graph::graph_store::SkillGraphStore;
use crate::tools::builtin::sandbox::{
    build_linux_sandbox_command, resolve_sandbox_status_for_request, FilesystemIsolationMode,
    SandboxConfig, SandboxStatus,
};
use crate::utils::text::safe_truncate;
use crate::utils::CryptoUtils;

use super::{
    GlobSearchInput, GrepSearchInput, ToolExecutionProfile, ToolSearchInput, WebFetchInput,
    WebSearchInput,
};

// ========== Tool implementations ==========

pub(super) async fn execute_glob_search(input: Value) -> Result<Value, String> {
    let params: GlobSearchInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let root = params.path.as_deref().unwrap_or(".");

    validate_workspace_glob_pattern(&params.pattern)?;

    // check if search path is within workspace
    if let Err(msg) = check_path_in_workspace(root) {
        return Err(format!(
            "{}\nPlease focus on the current workspace, search within the working directory.",
            msg
        ));
    }

    let mut files = Vec::new();
    let glob_pattern = if root != "." {
        format!("{}/{}", root.trim_end_matches('/'), &params.pattern)
    } else {
        params.pattern.clone()
    };
    match glob::glob(&glob_pattern) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry.map_err(|error| format!("Glob traversal error: {error}"))?;
                // A safe root does not make every expansion safe: a matching
                // entry can traverse a symlink below that root. Validate every
                // concrete result before exposing any of the collected names.
                resolve_path_in_workspace(
                    entry
                        .to_str()
                        .ok_or_else(|| "Glob result path is not valid UTF-8".to_string())?,
                )?;
                if crate::tools::workspace_monitor::inventory::is_workspace_runtime_path(&entry) {
                    continue;
                }
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

fn validate_workspace_glob_pattern(pattern: &str) -> Result<(), String> {
    let path = Path::new(pattern);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(format!(
            "Glob pattern must be relative to its workspace search path and must not contain '..': {pattern}"
        ));
    }
    Ok(())
}

pub(super) async fn execute_grep_search(input: Value) -> Result<Value, String> {
    let params: GrepSearchInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    let root = params.path.as_deref().unwrap_or(".");
    if let Err(message) = check_path_in_workspace(root) {
        return Err(format!(
            "{message}\nPlease focus on the current workspace, search within the working directory."
        ));
    }
    let mode = params
        .output_mode
        .as_deref()
        .unwrap_or("files_with_matches");
    if mode != "files_with_matches" && mode != "content" && mode != "count" {
        return Err(format!(
            "Invalid output_mode: {}. Must be files_with_matches, content, or count",
            mode
        ));
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

    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| entry.depth() == 0 || !is_build_or_vendored_dir(entry.path()))
        .filter_map(|e| e.ok())
    {
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
            let limited_files: Vec<String> = filenames
                .iter()
                .skip(skipped)
                .take(limit)
                .cloned()
                .collect();
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
            let limited_counts: Vec<Value> =
                file_counts.into_iter().skip(skipped).take(limit).collect();
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

/// Directories excluded from recursive file scans so a repository-scale
/// search does not traverse build artifacts and vendored dependencies.
fn is_build_or_vendored_dir(path: &std::path::Path) -> bool {
    const EXCLUDED: &[&str] = &[
        "target",
        "node_modules",
        ".git",
        "dist",
        "build",
        "vendor",
        ".venv",
        "__pycache__",
        ".next",
        ".gliding_horse",
    ];
    let name = path.file_name().and_then(|n| n.to_str());
    matches!(name, Some(name) if EXCLUDED.contains(&name))
}

const MAX_WEB_FETCH_BYTES: usize = 10_000_000;
const MAX_WEB_FETCH_REDIRECTS: usize = 10;

pub(super) fn is_public_web_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !(ip.is_unspecified()
                || ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_multicast()
                || ip.is_broadcast()
                || ip.is_documentation()
                || a == 0
                || (a == 100 && (64..=127).contains(&b))
                || (a == 192 && b == 0 && c == 0)
                || (a == 198 && (b == 18 || b == 19))
                || a >= 240)
        }
        IpAddr::V6(ip) => {
            if let Some(v4) = ip.to_ipv4() {
                return is_public_web_ip(IpAddr::V4(v4));
            }
            let segments = ip.segments();
            !(ip.is_unspecified()
                || ip.is_loopback()
                || ip.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
                || (segments[0] == 0x2001 && segments[1] == 0x0db8))
        }
    }
}

pub(super) async fn validated_web_target(url: &reqwest::Url) -> Result<Vec<SocketAddr>, String> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err("Only http:// and https:// URLs are allowed".to_string());
    }
    let host = url
        .host_str()
        .ok_or_else(|| "URL must contain a hostname".to_string())?;
    if host.eq_ignore_ascii_case("localhost") || host.ends_with(".localhost") {
        return Err("Localhost targets are blocked by web_fetch".to_string());
    }
    let port = url
        .port_or_known_default()
        .ok_or_else(|| "URL uses an unsupported port".to_string())?;
    let addresses: Vec<SocketAddr> = if let Ok(ip) = host.parse::<IpAddr>() {
        vec![SocketAddr::new(ip, port)]
    } else {
        tokio::net::lookup_host((host, port))
            .await
            .map_err(|error| format!("DNS resolution failed for {host}: {error}"))?
            .collect()
    };
    if addresses.is_empty() {
        return Err(format!("DNS resolution returned no addresses for {host}"));
    }
    if let Some(blocked) = addresses
        .iter()
        .find(|address| !is_public_web_ip(address.ip()))
    {
        return Err(format!(
            "Private, local, or reserved network target is blocked: {}",
            blocked.ip()
        ));
    }
    Ok(addresses)
}

async fn send_validated_web_request(url: &reqwest::Url) -> Result<reqwest::Response, String> {
    let addresses = validated_web_target(url).await?;
    let host = url
        .host_str()
        .ok_or_else(|| "URL must contain a hostname".to_string())?;
    let mut builder = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy();
    if host.parse::<IpAddr>().is_err() {
        builder = builder.resolve_to_addrs(host, &addresses);
    }
    let client = builder
        .build()
        .map_err(|error| format!("HTTP client: {error}"))?;

    let mut last_error = None;
    for attempt in 0..3 {
        match client.get(url.clone()).send().await {
            Ok(response) => return Ok(response),
            Err(error) => {
                last_error = Some(error.to_string());
                if attempt < 2 {
                    tokio::time::sleep(std::time::Duration::from_secs(1 << attempt)).await;
                }
            }
        }
    }
    Err(format!(
        "Request failed after 3 attempts: {}",
        last_error.unwrap_or_else(|| "unknown error".to_string())
    ))
}

pub(super) async fn read_limited_web_body(response: reqwest::Response) -> Result<Vec<u8>, String> {
    if response.content_length().unwrap_or(0) > MAX_WEB_FETCH_BYTES as u64 {
        return Err(format!(
            "Content too large (declared size exceeds {} bytes)",
            MAX_WEB_FETCH_BYTES
        ));
    }
    let mut body = Vec::with_capacity(
        response
            .content_length()
            .unwrap_or(0)
            .min(MAX_WEB_FETCH_BYTES as u64) as usize,
    );
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| format!("Failed to read response body: {error}"))?;
        let new_size = body
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| "Content size overflow".to_string())?;
        if new_size > MAX_WEB_FETCH_BYTES {
            return Err(format!(
                "Content too large (stream exceeded {} bytes)",
                MAX_WEB_FETCH_BYTES
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

pub(super) async fn execute_web_fetch(input: Value) -> Result<Value, String> {
    let params: WebFetchInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let started = Instant::now();

    let mut current_url = reqwest::Url::parse(&params.url)
        .map_err(|error| format!("Invalid URL '{}': {error}", params.url))?;
    let mut redirect_count = 0usize;
    let resp = loop {
        let response = send_validated_web_request(&current_url).await?;
        if response.status().is_redirection() {
            if redirect_count >= MAX_WEB_FETCH_REDIRECTS {
                return Err(format!(
                    "Too many redirects (maximum {})",
                    MAX_WEB_FETCH_REDIRECTS
                ));
            }
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .ok_or_else(|| "Redirect response is missing Location header".to_string())?
                .to_str()
                .map_err(|error| format!("Invalid redirect Location header: {error}"))?;
            current_url = current_url
                .join(location)
                .map_err(|error| format!("Invalid redirect target: {error}"))?;
            redirect_count += 1;
            continue;
        }
        break response;
    };

    let code = resp.status().as_u16();
    if code >= 400 {
        return Err(format!(
            "HTTP {} {} — target URL returned an error. Please verify the URL or use web_search to find an accessible link.",
            code,
            resp.status().canonical_reason().unwrap_or("Unknown")
        ));
    }

    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let body_bytes = read_limited_web_body(resp).await?;
    let body = String::from_utf8_lossy(&body_bytes).to_string();

    let content = if ct.contains("html") {
        html_to_text(&body)
    } else {
        safe_truncate(&body, 8000).to_string()
    };

    Ok(json!({
        "url": current_url.as_str(), "status_code": code,
        "content": content, "content_type": ct,
        "duration_ms": started.elapsed().as_millis(),
    }))
}

/// Execute search using Exa API (preferred, requires EXA_API_KEY env var).
/// Return format compatible with execute_web_search.
pub(super) async fn execute_exa_search(query: &str, started: Instant) -> Result<Value, String> {
    let api_key = std::env::var("EXA_API_KEY").map_err(|_| "EXA_API_KEY not set".to_string())?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("HTTP client: {}", e))?;

    let resp = client
        .post("https://api.exa.ai/search")
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
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Exa response parse failed: {}", e))?;

    if !status.is_success() {
        let error_msg = body
            .get("error")
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

    let results: Vec<Value> = body
        .get("results")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|r| {
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
                })
                .collect()
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
            Ok(mut v) => {
                if v.get("error").is_none() {
                    filter_search_response(&mut v, &params);
                    return Ok(v);
                }
                tracing::warn!(
                    "Exa search failed ({}), falling back to DuckDuckGo",
                    v.get("error")
                        .and_then(|e| e.as_str())
                        .unwrap_or("Unknown error")
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
        let lite_url = format!(
            "https://lite.duckduckgo.com/lite/?q={}",
            urlencode(&params.query)
        );
        let resp = client
            .get(&lite_url)
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .send()
            .await
            .map_err(|e| format!("Search: {}", e))?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| format!("Read: {}", e))?;

        if status.as_u16() == 200 && (body.contains("result-link") || body.contains("result__a")) {
            return Ok((status, body, "lite".to_string()));
        }

        // fallback: DuckDuckGo HTML
        let html_url = format!(
            "https://html.duckduckgo.com/html/?q={}",
            urlencode(&params.query)
        );
        let resp = client
            .get(&html_url)
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .send()
            .await
            .map_err(|e| format!("Search: {}", e))?;
        let status2 = resp.status();
        let body2 = resp.text().await.map_err(|e| format!("Read: {}", e))?;

        if status2.as_u16() == 200 && body2.contains("result__a") {
            return Ok((status2, body2, "html".to_string()));
        }

        // fallback: DuckDuckGo Instant Answer API
        let api_url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1",
            urlencode(&params.query)
        );
        let resp = client
            .get(&api_url)
            .send()
            .await
            .map_err(|e| format!("API: {}", e))?;
        let api_body = resp.text().await.map_err(|e| format!("Read: {}", e))?;
        Ok((status, api_body, "api".to_string()))
    })
    .await;

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

            filter_search_results(&mut results, &params);
            results.truncate(8);
            Ok(json!({
                "query": params.query,
                "duration_seconds": started.elapsed().as_secs_f64(),
                "results": results,
            }))
        }
        Err(e) => Ok(json!({
            "query": params.query,
            "duration_seconds": started.elapsed().as_secs_f64(),
            "results": [],
            "error": format!("Search request failed: {}", e),
            "suggestion": "Web search unavailable, please answer based on your own knowledge"
        })),
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
    let matches: Vec<Value> = all
        .iter()
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
    /// Kernel-issued optimistic-concurrency precondition. Model/tool JSON is
    /// rejected if it contains any `__gh_*` field; only ToolExecutor may add
    /// this value after the final Hook and workspace-lease checks.
    #[serde(rename = "__gh_expected_current_sha256")]
    expected_current_sha256: Option<String>,
    /// Keep the low-level helper usable in focused filesystem tests while the
    /// production ToolExecutor path fails closed for changed existing files.
    #[serde(rename = "__gh_require_overwrite_baseline", default)]
    require_overwrite_baseline: bool,
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
    #[serde(rename = "run_in_background")]
    run_in_background: Option<bool>,
    #[serde(rename = "dangerouslyDisableSandbox")]
    dangerously_disable_sandbox: Option<bool>,
    #[serde(rename = "namespaceRestrictions")]
    namespace_restrictions: Option<bool>,
    #[serde(rename = "isolateNetwork")]
    isolate_network: Option<bool>,
    #[serde(rename = "filesystemMode")]
    filesystem_mode: Option<FilesystemIsolationMode>,
    #[serde(rename = "allowedMounts")]
    allowed_mounts: Option<Vec<String>>,
    /// Injected by ToolExecutor after every model/Hook-visible trust boundary.
    /// The public bash schema never advertises this reserved field.
    #[serde(rename = "__gh_execution_profile", default)]
    execution_profile: ToolExecutionProfile,
}

#[derive(Debug, Deserialize)]
struct FileEditInput {
    path: String,
    old_string: String,
    new_string: String,
    replace_all: Option<bool>,
    #[serde(rename = "__gh_expected_current_sha256")]
    expected_current_sha256: Option<String>,
    #[serde(rename = "__gh_require_overwrite_baseline", default)]
    require_overwrite_baseline: bool,
}

fn enforce_kernel_overwrite_baseline(
    existing: &ExistingWorkspaceFile,
    expected_current_sha256: Option<&str>,
    required: bool,
) -> Result<(), String> {
    if !required {
        return Ok(());
    }
    let expected = expected_current_sha256.ok_or_else(|| {
        "overwrite_baseline_required: changing an existing file requires a successful whole-file read in the active Agent/L1 context"
            .to_string()
    })?;
    let actual = CryptoUtils::sha256_hex_bytes(&existing.content);
    if expected != actual {
        return Err(format!(
            "overwrite_baseline_stale: the file changed after the active Agent/L1 read it (expected {expected}, current {actual}); read the complete current file and retry"
        ));
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct PowerShellInput {
    command: String,
    timeout: Option<u64>,
    description: Option<String>,
    run_in_background: Option<bool>,
    #[serde(rename = "__gh_execution_profile", default)]
    execution_profile: ToolExecutionProfile,
}

pub(super) async fn execute_file_read(input: Value) -> Result<Value, String> {
    let params: FileReadInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
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
                let parent = path_obj
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| std::path::PathBuf::from("."));
                let parent_display = parent.display().to_string();
                let listing;
                if let Ok(entries) = std::fs::read_dir(&parent) {
                    let files: Vec<String> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| {
                            let kind = if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                                "[dir]"
                            } else {
                                "[file]"
                            };
                            format!("  {} {}", kind, e.file_name().to_string_lossy())
                        })
                        .collect();
                    if files.is_empty() {
                        listing = format!("\nDirectory {} is empty.", parent_display);
                    } else {
                        listing = format!(
                            "\nFiles in directory {}:\n{}",
                            parent_display,
                            files.join("\n")
                        );
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
    let selected: Vec<String> = lines
        .iter()
        .skip(start)
        .take(limit)
        .map(|l| l.to_string())
        .collect();
    let total = lines.len();
    Ok(json!({
        "path": params.path, "total_lines": total,
        "offset": start, "lines": selected, "returned": selected.len(),
        "content_sha256": CryptoUtils::sha256_hex(&content),
    }))
}

pub(super) async fn execute_file_write(input: Value) -> Result<Value, String> {
    let params: FileWriteInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let target = SecureWorkspaceTarget::open(&params.path, true)?;
    let existing = target.read_existing()?;
    let existed = existing.is_some();
    let changed = existing
        .as_ref()
        .map(|existing| existing.content != params.content.as_bytes())
        .unwrap_or(true);
    let content_sha256 = if changed {
        if let Some(existing) = existing.as_ref() {
            enforce_kernel_overwrite_baseline(
                existing,
                params.expected_current_sha256.as_deref(),
                params.require_overwrite_baseline,
            )?;
        }
        target.commit(params.content.as_bytes(), existing.as_ref())?;
        CryptoUtils::sha256_hex(&params.content)
    } else {
        let existing = existing
            .as_ref()
            .ok_or_else(|| "Workspace target disappeared before no-op verification".to_string())?;
        let verified = target.verify_unchanged(existing, params.content.as_bytes())?;
        let verified = std::str::from_utf8(&verified).map_err(|error| {
            format!("Workspace target stopped being valid UTF-8 during final verification: {error}")
        })?;
        CryptoUtils::sha256_hex(verified)
    };
    Ok(json!({
        "path": params.path,
        "bytes_written": if changed { params.content.len() } else { 0 },
        // A changed write is atomically committed from this payload. A no-op
        // hash is instead computed from bytes re-opened and revalidated at the
        // final secure target, after the initial equality decision.
        "content_sha256": content_sha256,
        "success": true,
        "changed": changed,
        "created": changed && !existed,
    }))
}

pub(super) async fn execute_file_list(input: Value) -> Result<Value, String> {
    let params: FileListInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let dir = params.path.as_deref().unwrap_or(".");

    if crate::tools::workspace_monitor::inventory::is_workspace_runtime_path(Path::new(dir)) {
        return Err("Workspace runtime state is not part of the project inventory".to_string());
    }

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
        if crate::tools::workspace_monitor::inventory::is_workspace_runtime_path(&entry.path()) {
            continue;
        }
        let ft = entry.file_type().ok();
        let kind = if ft.map_or(false, |t| t.is_dir()) {
            "dir"
        } else {
            "file"
        };
        if let Ok(name) = entry.file_name().into_string() {
            entries.push(json!({"name": name, "type": kind}));
        }
    }
    Ok(json!({"path": dir, "entries": entries, "count": entries.len()}))
}

#[derive(Debug, Clone)]
struct BackgroundProcessRecord {
    command: String,
    status: String,
    finished_at: Option<Instant>,
}

static BACKGROUND_PROCESSES: Lazy<std::sync::Mutex<HashMap<u32, BackgroundProcessRecord>>> =
    Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

#[derive(Default)]
struct OutputCapture {
    text: String,
    total_bytes: usize,
    truncated: bool,
}

async fn read_bounded_output<R>(mut reader: R) -> OutputCapture
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let mut captured = Vec::with_capacity(MAX_OUTPUT_BYTES);
    let mut total_bytes = 0usize;
    let mut buffer = [0u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                total_bytes = total_bytes.saturating_add(read);
                let remaining = MAX_OUTPUT_BYTES.saturating_sub(captured.len());
                captured.extend_from_slice(&buffer[..read.min(remaining)]);
            }
        }
    }
    let truncated = total_bytes > captured.len();
    let mut text = String::from_utf8_lossy(&captured).into_owned();
    if truncated {
        text.push_str(&format!(
            "\n...[output truncated: {} bytes total]",
            total_bytes
        ));
    }
    OutputCapture {
        text,
        total_bytes,
        truncated,
    }
}

async fn join_output_capture(
    task: Option<tokio::task::JoinHandle<OutputCapture>>,
) -> OutputCapture {
    match task {
        Some(task) => task.await.unwrap_or_default(),
        None => OutputCapture::default(),
    }
}

fn register_background_process(task_id: u32, command: String, mut child: tokio::process::Child) {
    {
        let mut registry = BACKGROUND_PROCESSES
            .lock()
            .expect("background process registry poisoned");
        registry.retain(|_, record| {
            record
                .finished_at
                .is_none_or(|finished| finished.elapsed() < std::time::Duration::from_secs(300))
        });
        registry.insert(
            task_id,
            BackgroundProcessRecord {
                command,
                status: "running".to_string(),
                finished_at: None,
            },
        );
    }
    tokio::spawn(async move {
        let status = child.wait().await;
        if let Ok(mut registry) = BACKGROUND_PROCESSES.lock() {
            if let Some(record) = registry.get_mut(&task_id) {
                record.status = match status {
                    Ok(status) => format!("exited:{}", status.code().unwrap_or(-1)),
                    Err(error) => format!("wait_error:{error}"),
                };
                record.finished_at = Some(Instant::now());
                tracing::debug!(background_task_id = task_id, command = %record.command,
                    status = %record.status, "Background process reaped");
            }
        }
    });
}

#[cfg(test)]
pub(super) fn background_process_status(task_id: u32) -> Option<String> {
    BACKGROUND_PROCESSES
        .lock()
        .ok()
        .and_then(|registry| registry.get(&task_id).map(|record| record.status.clone()))
}

pub(super) async fn execute_bash(input: Value) -> Result<Value, String> {
    #[cfg(windows)]
    {
        let params: BashInput =
            serde_json::from_value(input).map_err(|e| format!("Invalid input: {e}"))?;
        let ps_input = serde_json::to_value(PowerShellInput {
            command: params.command,
            timeout: params.timeout,
            description: params.description,
            run_in_background: params.run_in_background,
            execution_profile: params.execution_profile,
        })
        .map_err(|e| format!("Serialize error: {e}"))?;
        return execute_powershell(ps_input).await;
    }

    #[cfg(not(windows))]
    {
        let params: BashInput =
            serde_json::from_value(input).map_err(|e| format!("Invalid input: {e}"))?;
        let timeout_ms = params.timeout.unwrap_or(60_000);
        let cwd = std::env::current_dir().map_err(|e| format!("Current dir error: {e}"))?;
        let sandbox_status = sandbox_status_for_input(&params, &cwd);
        let sandbox_status_json = serde_json::to_value(&sandbox_status)
            .map_err(|e| format!("Sandbox status serialize error: {e}"))?;
        if params.execution_profile.is_clean_verification()
            && params.run_in_background.unwrap_or(false)
        {
            return Err(
                "Clean verification profile requires a foreground verifier so its isolated cache lifetime can be bounded"
                    .to_string(),
            );
        }
        let verification_cache = verification_cache_isolation(params.execution_profile)?;

        if params.run_in_background.unwrap_or(false) {
            let mut command = prepare_bash_spawn(
                &params.command,
                &cwd,
                &sandbox_status,
                params.execution_profile,
                verification_cache.as_ref().map(tempfile::TempDir::path),
            )?;
            command
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            let mut command = tokio::process::Command::from(command);
            command.kill_on_drop(true);
            let child = command.spawn().map_err(|e| format!("Spawn error: {e}"))?;
            let task_id = child
                .id()
                .ok_or_else(|| "Spawned background process has no PID".to_string())?;
            register_background_process(task_id, params.command.clone(), child);
            return Ok(json!({
                "command": params.command,
                "background_task_id": task_id.to_string(),
                "background_status": "running",
                "sandbox_status": sandbox_status_json,
            }));
        }

        let guarded_command = self_protect_bash_command(&params.command);
        let mut command = tokio::process::Command::from(prepare_bash_spawn(
            &guarded_command,
            &cwd,
            &sandbox_status,
            params.execution_profile,
            verification_cache.as_ref().map(tempfile::TempDir::path),
        )?);
        command.kill_on_drop(true);
        let mut child = command.spawn().map_err(|e| format!("Spawn error: {e}"))?;
        let pid = child.id();
        let stdout_task = child
            .stdout
            .take()
            .map(|out| tokio::spawn(read_bounded_output(out)));
        let stderr_task = child
            .stderr
            .take()
            .map(|err| tokio::spawn(read_bounded_output(err)));
        let started = Instant::now();
        let wait_result =
            tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), child.wait()).await;
        let (status, timed_out, process_group_cleanup) = match wait_result {
            Ok(result) => {
                let status = result.map_err(|e| format!("Wait error: {e}"))?;
                let cleanup = match pid {
                    Some(pid) => settle_foreground_process_group(pid, None).await,
                    None => Ok(ProcessGroupCleanup::default()),
                };
                (Some(status), false, cleanup)
            }
            Err(_) => {
                let cleanup = match pid {
                    Some(pid) => settle_foreground_process_group(pid, Some(&mut child)).await,
                    None => Ok(ProcessGroupCleanup::default()),
                };
                // `settle_foreground_process_group` normally reaps the group
                // leader while confirming that the PGID disappeared. Keep a
                // direct-child fallback for platforms/errors where no usable
                // process-group identity was available.
                if child.try_wait().ok().flatten().is_none() {
                    let _ = child.kill().await;
                }
                let _ = child.wait().await;
                (None, true, cleanup)
            }
        };
        let stdout = join_output_capture(stdout_task).await;
        let stderr = join_output_capture(stderr_task).await;
        let original_size = stdout.total_bytes.saturating_add(stderr.total_bytes);
        if timed_out {
            let mut response = json!({
                "command": params.command, "timed_out": true,
                "stdout": stdout.text, "stderr": stderr.text,
                "truncated": stdout.truncated || stderr.truncated,
                "original_size": original_size,
                "duration_ms": started.elapsed().as_millis() as u64,
                "error": format!("Timeout after {}ms", timeout_ms),
                "sandbox_status": sandbox_status_json,
            });
            attach_process_group_cleanup(&mut response, process_group_cleanup);
            attach_verification_profile(&mut response, params.execution_profile);
            return Ok(response);
        }
        let mut response = json!({
            "command": params.command,
            "exit_code": status.and_then(|status| status.code()).unwrap_or(-1),
            "stdout": stdout.text, "stderr": stderr.text,
            "duration_ms": started.elapsed().as_millis() as u64,
            "truncated": stdout.truncated || stderr.truncated,
            "original_size": original_size,
            "sandbox_status": sandbox_status_json,
        });
        attach_process_group_cleanup(&mut response, process_group_cleanup);
        attach_verification_profile(&mut response, params.execution_profile);
        Ok(response)
    }
}

/// Resolve the effective sandbox status from the per-command overrides.
/// Sandboxing is opt-in: without an explicit `dangerouslyDisableSandbox`
/// value the sandbox stays disabled, preserving existing behaviour (and
/// keeping pkill/pgrep able to manage host processes across the namespace
/// boundary, which a default-enabled PID namespace would break).
fn sandbox_status_for_input(input: &BashInput, cwd: &std::path::Path) -> SandboxStatus {
    let enabled = input
        .dangerously_disable_sandbox
        .map(|disabled| !disabled)
        .unwrap_or(false);
    let request = SandboxConfig::default().resolve_request(
        Some(enabled),
        input.namespace_restrictions,
        input.isolate_network,
        input.filesystem_mode,
        input.allowed_mounts.clone(),
    );
    resolve_sandbox_status_for_request(&request, cwd)
}

/// Prepare a bash spawn: unshare launcher when the sandbox is active,
/// otherwise a plain `sh -lc` (with sandbox HOME/TMPDIR when filesystem
/// isolation is requested).
fn prepare_bash_spawn(
    command: &str,
    cwd: &std::path::Path,
    sandbox_status: &SandboxStatus,
    execution_profile: ToolExecutionProfile,
    verification_cache: Option<&Path>,
) -> Result<std::process::Command, String> {
    use std::process::Command;
    let command = profiled_shell_command(command, execution_profile, verification_cache)?;
    if sandbox_status.filesystem_active {
        let _ = crate::tools::builtin::sandbox::ensure_sandbox_dirs(cwd);
    }
    if let Some(launcher) = build_linux_sandbox_command(&command, cwd, sandbox_status) {
        let mut c = Command::new(launcher.program);
        c.args(launcher.args);
        c.current_dir(cwd);
        c.env_clear();
        c.envs(crate::tools::process_env::sanitized_child_environment(true));
        // Agent shell probes should not leave interpreter cache artifacts in
        // the user's deliverable. An explicit command-local assignment can
        // still opt back in when bytecode-cache behavior is itself under test.
        c.env("PYTHONDONTWRITEBYTECODE", "1");
        c.envs(launcher.env);
        apply_clean_verification_environment(&mut c, execution_profile, verification_cache)?;
        c.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        #[cfg(unix)]
        {
            c.process_group(0);
        }
        return Ok(c);
    }
    let mut c = Command::new("sh");
    c.arg("-lc").arg(&command).current_dir(cwd);
    c.env_clear();
    c.envs(crate::tools::process_env::sanitized_child_environment(true));
    c.env("PYTHONDONTWRITEBYTECODE", "1");
    if sandbox_status.filesystem_active {
        c.env("HOME", cwd.join(".sandbox-home"));
        c.env("TMPDIR", cwd.join(".sandbox-tmp"));
    }
    apply_clean_verification_environment(&mut c, execution_profile, verification_cache)?;
    c.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    {
        c.process_group(0);
    }
    Ok(c)
}

fn profiled_shell_command(
    command: &str,
    execution_profile: ToolExecutionProfile,
    verification_cache: Option<&Path>,
) -> Result<String, String> {
    if execution_profile != ToolExecutionProfile::CleanMermaidVerification {
        return Ok(command.to_string());
    }
    let cache = verification_cache.ok_or_else(|| {
        "Clean Mermaid verification profile is missing its kernel-owned directory".to_string()
    })?;
    let config = cache.join("puppeteer.json");
    let escaped_config = config.to_string_lossy().replace('\'', "'\"'\"'");
    Ok(format!(
        "mmdc() {{ command mmdc -p '{escaped_config}' \"$@\"; }}\n{command}"
    ))
}

fn verification_cache_isolation(
    execution_profile: ToolExecutionProfile,
) -> Result<Option<tempfile::TempDir>, String> {
    if !execution_profile.is_clean_verification() {
        return Ok(None);
    }
    tempfile::Builder::new()
        .prefix("glidinghorse-verification-")
        .tempdir()
        .map(Some)
        .map_err(|error| format!("Cannot create isolated verification cache: {error}"))
}

fn apply_clean_verification_environment(
    command: &mut std::process::Command,
    execution_profile: ToolExecutionProfile,
    verification_cache: Option<&Path>,
) -> Result<(), String> {
    if !execution_profile.is_clean_verification() {
        return Ok(());
    }
    let cache = verification_cache.ok_or_else(|| {
        "Clean verification profile is missing its kernel-owned cache directory".to_string()
    })?;
    match execution_profile {
        ToolExecutionProfile::CleanPythonVerification => {
            command.env("PYTHONPYCACHEPREFIX", cache.join("pycache"));
        }
        ToolExecutionProfile::CleanPytestVerification => {
            command.env("PYTHONPYCACHEPREFIX", cache.join("pycache"));
            command.env(
                "PYTEST_ADDOPTS",
                format!("-o cache_dir={}", cache.join("pytest-cache").display()),
            );
        }
        ToolExecutionProfile::CleanMermaidVerification => {
            std::fs::write(
                cache.join("puppeteer.json"),
                r#"{"args":["--no-sandbox","--disable-setuid-sandbox"]}
"#,
            )
            .map_err(|error| {
                format!("Cannot create kernel-owned Mermaid verifier config: {error}")
            })?;
        }
        ToolExecutionProfile::Standard => {}
    }
    Ok(())
}

fn attach_verification_profile(response: &mut Value, execution_profile: ToolExecutionProfile) {
    if !execution_profile.is_clean_verification() {
        return;
    }
    if let Some(object) = response.as_object_mut() {
        object.insert(
            "execution_profile".to_string(),
            serde_json::to_value(execution_profile)
                .expect("ToolExecutionProfile is always JSON serializable"),
        );
        object.insert(
            "isolated_environment".to_string(),
            json!({
                "PYTHONPYCACHEPREFIX": execution_profile.is_python_verification(),
                "PYTEST_ADDOPTS": execution_profile == ToolExecutionProfile::CleanPytestVerification,
                "PUPPETEER_CONFIG": execution_profile == ToolExecutionProfile::CleanMermaidVerification,
            }),
        );
    }
}

const MAX_OUTPUT_BYTES: usize = 16_384;

/// Truncate output to `MAX_OUTPUT_BYTES`, appending a marker when trimmed.
#[cfg(test)]
pub(super) fn truncate_output(s: &str) -> (String, bool) {
    if s.len() <= MAX_OUTPUT_BYTES {
        return (s.to_string(), false);
    }
    let mut end = MAX_OUTPUT_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut truncated = s[..end].to_string();
    truncated.push_str("\n\n[output truncated — exceeded 16384 bytes]");
    (truncated, true)
}

/// Guards a bash command against self-kill. The DA frequently cleans up
/// spawned processes with `pkill -f <name>` / `killall <name>`; the agent's
/// own process command line embeds the task prompt, which can contain the
/// same `<name>` (e.g. a file the DA created and then pkill -f's). A plain
/// full-command-line match then kills the agent OS process itself.
///
/// When the command mentions pkill/killall, we prepend shell function
/// overrides that resolve targets via `pgrep` and exclude both the agent's
/// own PID and the wrapper shell's PID before signaling.
#[cfg(unix)]
fn self_protect_bash_command(command: &str) -> String {
    if !command.contains("pkill") && !command.contains("killall") {
        return command.to_string();
    }
    let self_pid = std::process::id();
    // pkill/killall drop-in: keep flags (signal, -f) but route matching
    // through pgrep so we can filter out our own PID. `command pgrep`
    // bypasses any function alias; `$$` is the wrapper shell PID.
    format!(
        r#"_agent_self_pid={self_pid}
pkill() {{
  local sig="TERM" f=""
  for a in "$@"; do
    case "$a" in
      -[0-9]*|-SIG*) sig="${{a#-}}"; ;;
      -f) f="-f"; ;;
      -*) ;;
      *) pat="$a"; ;;
    esac
  done
  [ -z "${{pat:-}}" ] && return 1
  local pids
  pids="$(command pgrep $f -- "$pat" 2>/dev/null | grep -vw "$_agent_self_pid" | grep -vw "$$" || true)"
  [ -z "$pids" ] && return 1
  command kill "-$sig" $pids 2>/dev/null
}}
killall() {{
  local sig="TERM"
  for a in "$@"; do
    case "$a" in
      -[0-9]*|-SIG*) sig="${{a#-}}"; ;;
      -*) ;;
      *) pat="$a"; ;;
    esac
  done
  [ -z "${{pat:-}}" ] && return 1
  local pids
  pids="$(command pgrep -- "$pat" 2>/dev/null | grep -vw "$_agent_self_pid" | grep -vw "$$" || true)"
  [ -z "$pids" ] && return 1
  command kill "-$sig" $pids 2>/dev/null
}}
{command}
"#
    )
}

#[derive(Debug, Clone, Copy, Default)]
struct ProcessGroupCleanup {
    residual_processes_detected: bool,
    forced_kill: bool,
}

fn attach_process_group_cleanup(
    response: &mut Value,
    cleanup: Result<ProcessGroupCleanup, String>,
) {
    let Some(object) = response.as_object_mut() else {
        return;
    };
    match cleanup {
        Ok(cleanup) if cleanup.residual_processes_detected => {
            object.insert(
                "process_group_cleanup".to_string(),
                json!({
                    "residual_processes_detected": true,
                    "forced_kill": cleanup.forced_kill,
                    "confirmed_gone": true,
                }),
            );
        }
        Ok(_) => {}
        Err(error) => {
            object.insert(
                "process_group_cleanup".to_string(),
                json!({
                    "residual_processes_detected": true,
                    "confirmed_gone": false,
                    "error": error,
                }),
            );
            object.entry("error".to_string()).or_insert_with(|| {
                Value::String(
                    "Foreground command finished, but its process group could not be settled"
                        .to_string(),
                )
            });
        }
    }
}

#[cfg(unix)]
fn checked_process_group_id(process_group_id: u32) -> Result<libc::pid_t, String> {
    let process_group_id = libc::pid_t::try_from(process_group_id)
        .map_err(|_| "spawned process-group id exceeds the platform PID range".to_string())?;
    let current_pid = unsafe { libc::getpid() };
    let current_group = unsafe { libc::getpgrp() };
    if process_group_id <= 1 || process_group_id == current_pid || process_group_id == current_group
    {
        return Err(format!(
            "refusing unsafe foreground process-group target {process_group_id}"
        ));
    }
    Ok(process_group_id)
}

#[cfg(unix)]
fn process_group_exists(process_group_id: libc::pid_t) -> Result<bool, String> {
    if unsafe { libc::kill(-process_group_id, 0) } == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        // The group exists even though the current process cannot signal it.
        Some(libc::EPERM) => Ok(true),
        _ => Err(format!(
            "cannot inspect foreground process group {process_group_id}: {error}"
        )),
    }
}

#[cfg(unix)]
fn signal_process_group(process_group_id: libc::pid_t, signal: libc::c_int) -> Result<(), String> {
    if unsafe { libc::kill(-process_group_id, signal) } == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        return Ok(());
    }
    Err(format!(
        "cannot signal foreground process group {process_group_id} with {signal}: {error}"
    ))
}

#[cfg(unix)]
fn reap_group_leader_if_exited(
    group_leader: &mut Option<&mut tokio::process::Child>,
) -> Result<(), String> {
    let exited = match group_leader.as_deref_mut() {
        Some(child) => child
            .try_wait()
            .map_err(|error| format!("cannot reap foreground process-group leader: {error}"))?
            .is_some(),
        None => false,
    };
    if exited {
        *group_leader = None;
    }
    Ok(())
}

#[cfg(unix)]
async fn wait_for_process_group_exit(
    process_group_id: libc::pid_t,
    group_leader: &mut Option<&mut tokio::process::Child>,
    timeout: std::time::Duration,
) -> Result<bool, String> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        reap_group_leader_if_exited(group_leader)?;
        if !process_group_exists(process_group_id)? {
            return Ok(true);
        }
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Ok(false);
        }
        tokio::time::sleep(
            std::time::Duration::from_millis(10).min(deadline.saturating_duration_since(now)),
        )
        .await;
    }
}

/// Settle every process which remains in the foreground shell's dedicated
/// process group. A model can spell `nohup ... &` inside a nominally
/// foreground command without setting the structured `run_in_background`
/// flag. Waiting only for the group leader would then let a descendant write
/// after AgentRunner's post-execution workspace snapshot.
///
/// This function uses kernel signals directly, never a PATH-resolved `kill`
/// utility. The PGID is the child PID assigned by `CommandExt::process_group`
/// and is rejected if it could identify PID 1, this process, or the host
/// process group. TERM offers a short graceful window; KILL and a second
/// bounded poll make successful return an explicit no-live-group receipt.
#[cfg(unix)]
async fn settle_foreground_process_group(
    process_group_id: u32,
    group_leader: Option<&mut tokio::process::Child>,
) -> Result<ProcessGroupCleanup, String> {
    const TERM_GRACE: std::time::Duration = std::time::Duration::from_millis(200);
    const KILL_GRACE: std::time::Duration = std::time::Duration::from_millis(800);

    let process_group_id = checked_process_group_id(process_group_id)?;
    let mut group_leader = group_leader;
    reap_group_leader_if_exited(&mut group_leader)?;
    if !process_group_exists(process_group_id)? {
        return Ok(ProcessGroupCleanup::default());
    }

    signal_process_group(process_group_id, libc::SIGTERM)?;
    if wait_for_process_group_exit(process_group_id, &mut group_leader, TERM_GRACE).await? {
        return Ok(ProcessGroupCleanup {
            residual_processes_detected: true,
            forced_kill: false,
        });
    }

    signal_process_group(process_group_id, libc::SIGKILL)?;
    if wait_for_process_group_exit(process_group_id, &mut group_leader, KILL_GRACE).await? {
        return Ok(ProcessGroupCleanup {
            residual_processes_detected: true,
            forced_kill: true,
        });
    }
    Err(format!(
        "foreground process group {process_group_id} still exists after TERM and KILL"
    ))
}

#[cfg(not(unix))]
async fn settle_foreground_process_group(
    _process_group_id: u32,
    _group_leader: Option<&mut tokio::process::Child>,
) -> Result<ProcessGroupCleanup, String> {
    Ok(ProcessGroupCleanup::default())
}

/// Resolve a path against the current workspace using the nearest existing
/// ancestor. Unlike `Path::exists`, `symlink_metadata` treats dangling links as
/// existing, so an unresolved link is rejected instead of being mistaken for
/// a harmless missing directory.
pub(super) fn resolve_path_in_workspace(path: &str) -> Result<(PathBuf, PathBuf), String> {
    let cwd = std::env::current_dir()
        .map_err(|error| format!("Cannot determine the current workspace: {error}"))?;
    let workspace = std::fs::canonicalize(&cwd)
        .map_err(|error| format!("Cannot resolve the current workspace: {error}"))?;
    let requested = Path::new(path);
    let requested = if requested.is_absolute() {
        normalize_absolute_path(requested)?
    } else {
        normalize_absolute_path(&cwd.join(requested))?
    };

    let mut ancestor = requested.clone();
    let mut missing = Vec::new();
    loop {
        match std::fs::symlink_metadata(&ancestor) {
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let name = ancestor
                    .file_name()
                    .ok_or_else(|| format!("Path has no resolvable workspace ancestor: {path}"))?;
                missing.push(name.to_os_string());
                if !ancestor.pop() {
                    return Err(format!("Path has no resolvable workspace ancestor: {path}"));
                }
            }
            Err(error) => {
                return Err(format!("Cannot inspect requested path ancestor: {error}"));
            }
        }
    }

    let mut resolved = std::fs::canonicalize(&ancestor)
        .map_err(|error| format!("Cannot resolve requested path ancestor: {error}"))?;
    if !resolved.starts_with(&workspace) {
        return Err(format!(
            "Path is not within the workspace: {}. The current task should only access files in the workspace.",
            resolved.display(),
        ));
    }
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    if !resolved.starts_with(&workspace) {
        return Err(format!(
            "Path is not within the workspace: {}. The current task should only access files in the workspace.",
            resolved.display(),
        ));
    }
    Ok((workspace, resolved))
}

fn normalize_absolute_path(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("Workspace path normalization requires an absolute path".to_string());
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(format!(
                        "Path escapes the filesystem root during normalization: {}",
                        path.display()
                    ));
                }
            }
        }
    }
    if !normalized.is_absolute() {
        return Err(format!(
            "Path is not absolute after normalization: {}",
            path.display()
        ));
    }
    Ok(normalized)
}

/// Check whether a path resolves inside the workspace. Mutation handlers do
/// not rely on this check alone: they use `SecureWorkspaceTarget`, which binds
/// traversal and commit to directory file descriptors to close the TOCTOU
/// window between validation and I/O.
fn check_path_in_workspace(path: &str) -> Result<(), String> {
    resolve_path_in_workspace(path).map(|_| ())
}

struct ExistingWorkspaceFile {
    content: Vec<u8>,
    #[cfg(unix)]
    mode: libc::mode_t,
    #[cfg(unix)]
    device: libc::dev_t,
    #[cfg(unix)]
    inode: libc::ino_t,
}

#[cfg(unix)]
struct SecureWorkspaceTarget {
    root: OwnedFd,
    parent: OwnedFd,
    workspace: PathBuf,
    parent_components: Vec<CString>,
    file_name: CString,
}

#[cfg(unix)]
static SECURE_WRITE_NONCE: AtomicU64 = AtomicU64::new(1);

#[cfg(unix)]
impl SecureWorkspaceTarget {
    fn open(path: &str, create_parents: bool) -> Result<Self, String> {
        // Open `.` before resolving its printable path. This descriptor is the
        // capability boundary used by every later filesystem operation.
        let root = open_directory(CStr::from_bytes_with_nul(b".\0").expect("static C string"))
            .map_err(|error| format!("Cannot open current workspace: {error}"))?;
        let (workspace, resolved) = resolve_path_in_workspace(path)?;
        verify_workspace_identity(&root, &workspace)?;
        let relative = resolved.strip_prefix(&workspace).map_err(|_| {
            format!(
                "Resolved path is outside the workspace: {}",
                resolved.display()
            )
        })?;
        let mut components = relative.components().peekable();
        let mut parent_components = Vec::new();
        let mut file_name = None;
        while let Some(component) = components.next() {
            let Component::Normal(name) = component else {
                return Err(format!(
                    "Workspace path is not normalized: {}",
                    resolved.display()
                ));
            };
            if components.peek().is_none() {
                file_name = Some(os_str_to_cstring(name)?);
                break;
            }
            parent_components.push(os_str_to_cstring(name)?);
        }
        let file_name =
            file_name.ok_or_else(|| "Workspace root is not a file target".to_string())?;
        let parent = open_parent_from_root(&root, &parent_components, create_parents)
            .map_err(|error| format!("Cannot securely traverse workspace path: {error}"))?;
        Ok(Self {
            root,
            parent,
            workspace,
            parent_components,
            file_name,
        })
    }

    fn read_existing(&self) -> Result<Option<ExistingWorkspaceFile>, String> {
        let flags = libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK;
        let raw = unsafe { libc::openat(self.parent.as_raw_fd(), self.file_name.as_ptr(), flags) };
        if raw < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::NotFound {
                return Ok(None);
            }
            return Err(format!("Cannot securely open workspace file: {error}"));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let metadata = metadata_for_fd(fd.as_raw_fd())
            .map_err(|error| format!("Cannot inspect workspace file: {error}"))?;
        if metadata.st_mode & libc::S_IFMT != libc::S_IFREG {
            return Err("Workspace mutation target must be a regular file".to_string());
        }
        let mode = metadata.st_mode;
        let device = metadata.st_dev;
        let inode = metadata.st_ino;
        let mut file: std::fs::File = fd.into();
        let mut content = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut content)
            .map_err(|error| format!("Read error: {error}"))?;
        Ok(Some(ExistingWorkspaceFile {
            content,
            mode,
            device,
            inode,
        }))
    }

    /// Re-establish the full descriptor-relative path and final-name binding
    /// before certifying a no-op write. The initial equality read is not
    /// enough: another process could replace the parent, swap in a symlink, or
    /// modify/replace the file before `execute_file_write` returns its digest.
    fn verify_unchanged(
        &self,
        existing: &ExistingWorkspaceFile,
        expected_content: &[u8],
    ) -> Result<Vec<u8>, String> {
        if existing.content != expected_content {
            return Err(
                "Workspace target did not match the requested no-op content initially".to_string(),
            );
        }

        let parent = self.revalidate_parent().map_err(|error| {
            format!("Workspace target changed during no-op verification: {error}")
        })?;
        let flags = libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK;
        let raw = unsafe { libc::openat(parent.as_raw_fd(), self.file_name.as_ptr(), flags) };
        if raw < 0 {
            return Err(format!(
                "Workspace target is no longer safe during no-op verification: {}",
                std::io::Error::last_os_error()
            ));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let metadata = metadata_for_fd(fd.as_raw_fd()).map_err(|error| {
            format!("Cannot inspect workspace target during no-op verification: {error}")
        })?;
        if metadata.st_mode & libc::S_IFMT != libc::S_IFREG {
            return Err(
                "Workspace no-op target must remain a regular file during verification".to_string(),
            );
        }
        if metadata.st_dev != existing.device || metadata.st_ino != existing.inode {
            return Err(
                "Workspace target identity changed concurrently during no-op verification"
                    .to_string(),
            );
        }

        let mut file: std::fs::File = fd.into();
        let mut observed = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut observed).map_err(|error| {
            format!("Cannot read workspace target during no-op verification: {error}")
        })?;
        if observed != existing.content || observed != expected_content {
            return Err(
                "Workspace target bytes changed concurrently during no-op verification".to_string(),
            );
        }

        self.ensure_same_commit_parent(&parent).map_err(|error| {
            format!("Workspace target changed during no-op verification: {error}")
        })?;
        self.verify_named_target_identity(&parent, existing)?;

        // Re-read after the pathname checks so the returned digest is derived
        // from the last fully observed target bytes, not the earlier equality
        // read. A final name/parent check then catches a replacement during
        // this read.
        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(0)).map_err(|error| {
            format!("Cannot rewind workspace target during no-op verification: {error}")
        })?;
        let mut final_content = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut final_content).map_err(|error| {
            format!("Cannot re-read workspace target during no-op verification: {error}")
        })?;
        if final_content != observed || final_content != expected_content {
            return Err(
                "Workspace target bytes changed concurrently during final no-op verification"
                    .to_string(),
            );
        }
        self.ensure_same_commit_parent(&parent).map_err(|error| {
            format!("Workspace target changed during final no-op verification: {error}")
        })?;
        self.verify_named_target_identity(&parent, existing)?;
        Ok(final_content)
    }

    fn verify_named_target_identity(
        &self,
        parent: &OwnedFd,
        existing: &ExistingWorkspaceFile,
    ) -> Result<(), String> {
        let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe {
            libc::fstatat(
                parent.as_raw_fd(),
                self.file_name.as_ptr(),
                metadata.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } != 0
        {
            return Err(format!(
                "Cannot inspect final workspace name during no-op verification: {}",
                std::io::Error::last_os_error()
            ));
        }
        let metadata = unsafe { metadata.assume_init() };
        if metadata.st_mode & libc::S_IFMT != libc::S_IFREG {
            return Err(
                "Workspace final name stopped identifying a regular file during no-op verification"
                    .to_string(),
            );
        }
        if metadata.st_dev != existing.device || metadata.st_ino != existing.inode {
            return Err(
                "Workspace final name identity changed concurrently during no-op verification"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn commit(
        &self,
        content: &[u8],
        existing: Option<&ExistingWorkspaceFile>,
    ) -> Result<(), String> {
        let parent = self.revalidate_parent()?;
        let (temp_name, raw) = self.create_temp_file(parent.as_raw_fd())?;
        let mut file = unsafe { std::fs::File::from_raw_fd(raw) };
        let result = (|| -> Result<(), String> {
            if let Some(existing) = existing {
                let permissions = existing.mode & 0o7777;
                if unsafe { libc::fchmod(file.as_raw_fd(), permissions) } != 0 {
                    return Err(format!(
                        "Cannot preserve workspace file permissions: {}",
                        std::io::Error::last_os_error()
                    ));
                }
            }
            std::io::Write::write_all(&mut file, content)
                .map_err(|error| format!("Write error: {error}"))?;
            file.sync_all()
                .map_err(|error| format!("Cannot sync workspace file: {error}"))?;
            drop(file);
            self.ensure_same_commit_parent(&parent)?;
            self.verify_final_target(&parent, existing)?;
            if unsafe {
                libc::renameat(
                    parent.as_raw_fd(),
                    temp_name.as_ptr(),
                    parent.as_raw_fd(),
                    self.file_name.as_ptr(),
                )
            } != 0
            {
                return Err(format!(
                    "Cannot atomically commit workspace file: {}",
                    std::io::Error::last_os_error()
                ));
            }
            Ok(())
        })();
        if result.is_err() {
            unsafe {
                libc::unlinkat(parent.as_raw_fd(), temp_name.as_ptr(), 0);
            }
        }
        result
    }

    fn verify_final_target(
        &self,
        parent: &OwnedFd,
        existing: Option<&ExistingWorkspaceFile>,
    ) -> Result<(), String> {
        let Some(existing) = existing else {
            let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
            let result = unsafe {
                libc::fstatat(
                    parent.as_raw_fd(),
                    self.file_name.as_ptr(),
                    metadata.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            };
            if result == 0 {
                return Err("Workspace target appeared concurrently before commit".to_string());
            }
            let error = std::io::Error::last_os_error();
            return if error.kind() == std::io::ErrorKind::NotFound {
                Ok(())
            } else {
                Err(format!("Cannot verify new workspace target: {error}"))
            };
        };

        // Re-open the final name with write permission immediately before the
        // rename. Besides preserving historical read-only-file behavior, the
        // byte comparison prevents silently overwriting an in-place edit that
        // happened after our initial read.
        let flags = libc::O_RDWR | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK;
        let raw = unsafe { libc::openat(parent.as_raw_fd(), self.file_name.as_ptr(), flags) };
        if raw < 0 {
            return Err(format!(
                "Workspace target is no longer safely writable: {}",
                std::io::Error::last_os_error()
            ));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let metadata = metadata_for_fd(fd.as_raw_fd())
            .map_err(|error| format!("Cannot inspect workspace target before commit: {error}"))?;
        if metadata.st_mode & libc::S_IFMT != libc::S_IFREG {
            return Err("Workspace mutation target must remain a regular file".to_string());
        }
        if metadata.st_dev != existing.device || metadata.st_ino != existing.inode {
            return Err("Workspace target identity changed concurrently before commit".to_string());
        }
        let mut file: std::fs::File = fd.into();
        let mut current = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut current)
            .map_err(|error| format!("Cannot re-read workspace target before commit: {error}"))?;
        if current != existing.content {
            return Err("Workspace target changed concurrently before commit".to_string());
        }
        Ok(())
    }

    fn revalidate_parent(&self) -> Result<OwnedFd, String> {
        verify_workspace_identity(&self.root, &self.workspace)?;
        let current = open_parent_from_root(&self.root, &self.parent_components, false)
            .map_err(|error| format!("Workspace target changed before commit: {error}"))?;
        if !same_open_file(&self.parent, &current)? {
            return Err("Workspace target directory identity changed before commit".to_string());
        }
        Ok(current)
    }

    fn ensure_same_commit_parent(&self, parent: &OwnedFd) -> Result<(), String> {
        verify_workspace_identity(&self.root, &self.workspace)?;
        let current = open_parent_from_root(&self.root, &self.parent_components, false)
            .map_err(|error| format!("Workspace target changed during commit: {error}"))?;
        if !same_open_file(parent, &current)? {
            return Err("Workspace target directory identity changed during commit".to_string());
        }
        Ok(())
    }

    fn create_temp_file(&self, parent: RawFd) -> Result<(CString, RawFd), String> {
        for _ in 0..128 {
            let nonce = SECURE_WRITE_NONCE.fetch_add(1, Ordering::Relaxed);
            let name = CString::new(format!(
                ".glidinghorse-write-{}-{nonce}.tmp",
                std::process::id()
            ))
            .expect("generated temporary name contains no NUL");
            let flags =
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW;
            let raw = unsafe { libc::openat(parent, name.as_ptr(), flags, 0o666) };
            if raw >= 0 {
                return Ok((name, raw));
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::AlreadyExists {
                return Err(format!("Cannot create workspace temporary file: {error}"));
            }
        }
        Err("Cannot allocate a unique workspace temporary file".to_string())
    }
}

#[cfg(unix)]
fn open_parent_from_root(
    root: &OwnedFd,
    components: &[CString],
    create: bool,
) -> std::io::Result<OwnedFd> {
    let raw = unsafe { libc::fcntl(root.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
    if raw < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut parent = unsafe { OwnedFd::from_raw_fd(raw) };
    for component in components {
        parent = open_or_create_directory_at(parent.as_raw_fd(), component, create)?;
    }
    Ok(parent)
}

#[cfg(unix)]
fn same_open_file(left: &OwnedFd, right: &OwnedFd) -> Result<bool, String> {
    let left = metadata_for_fd(left.as_raw_fd())
        .map_err(|error| format!("Cannot inspect original workspace directory: {error}"))?;
    let right = metadata_for_fd(right.as_raw_fd())
        .map_err(|error| format!("Cannot inspect current workspace directory: {error}"))?;
    Ok(left.st_dev == right.st_dev && left.st_ino == right.st_ino)
}

#[cfg(unix)]
fn open_directory(path: &CStr) -> std::io::Result<OwnedFd> {
    let raw = unsafe {
        libc::open(
            path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if raw < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(unsafe { OwnedFd::from_raw_fd(raw) })
    }
}

#[cfg(unix)]
fn open_or_create_directory_at(
    parent: RawFd,
    name: &CStr,
    create: bool,
) -> std::io::Result<OwnedFd> {
    let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW;
    let mut raw = unsafe { libc::openat(parent, name.as_ptr(), flags) };
    if raw < 0 && create {
        let open_error = std::io::Error::last_os_error();
        if open_error.kind() == std::io::ErrorKind::NotFound {
            if unsafe { libc::mkdirat(parent, name.as_ptr(), 0o777) } != 0 {
                let mkdir_error = std::io::Error::last_os_error();
                if mkdir_error.kind() != std::io::ErrorKind::AlreadyExists {
                    return Err(mkdir_error);
                }
            }
            raw = unsafe { libc::openat(parent, name.as_ptr(), flags) };
        }
    }
    if raw < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(unsafe { OwnedFd::from_raw_fd(raw) })
    }
}

#[cfg(unix)]
fn verify_workspace_identity(root: &OwnedFd, workspace: &Path) -> Result<(), String> {
    let descriptor = metadata_for_fd(root.as_raw_fd())
        .map_err(|error| format!("Cannot inspect workspace descriptor: {error}"))?;
    let path_metadata = std::fs::metadata(workspace)
        .map_err(|error| format!("Cannot inspect resolved workspace: {error}"))?;
    if descriptor.st_dev != path_metadata.dev() || descriptor.st_ino != path_metadata.ino() {
        return Err("Current workspace identity changed during path resolution".to_string());
    }
    Ok(())
}

#[cfg(unix)]
fn metadata_for_fd(fd: RawFd) -> std::io::Result<libc::stat> {
    let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
    if unsafe { libc::fstat(fd, metadata.as_mut_ptr()) } != 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(unsafe { metadata.assume_init() })
    }
}

#[cfg(unix)]
fn os_str_to_cstring(value: &OsStr) -> Result<CString, String> {
    CString::new(value.as_bytes())
        .map_err(|_| "Workspace paths must not contain NUL bytes".to_string())
}

// Unix provides the descriptor-relative primitives needed for strict
// race-resistant traversal. Other targets retain fail-closed containment
// checks; platform-specific handle-relative APIs can replace this fallback
// without changing the mutation handlers.
#[cfg(not(unix))]
struct SecureWorkspaceTarget {
    path: PathBuf,
}

#[cfg(not(unix))]
impl SecureWorkspaceTarget {
    fn open(path: &str, create_parents: bool) -> Result<Self, String> {
        let (_, resolved) = resolve_path_in_workspace(path)?;
        let parent = resolved
            .parent()
            .ok_or_else(|| "Workspace root is not a file target".to_string())?;
        if create_parents {
            std::fs::create_dir_all(parent).map_err(|error| format!("Mkdir error: {error}"))?;
        }
        let canonical_parent = std::fs::canonicalize(parent)
            .map_err(|error| format!("Cannot resolve workspace parent: {error}"))?;
        let workspace = std::fs::canonicalize(
            std::env::current_dir()
                .map_err(|error| format!("Cannot determine the current workspace: {error}"))?,
        )
        .map_err(|error| format!("Cannot resolve the current workspace: {error}"))?;
        if !canonical_parent.starts_with(&workspace) {
            return Err("Workspace parent escaped during path preparation".to_string());
        }
        let file_name = resolved
            .file_name()
            .ok_or_else(|| "Workspace root is not a file target".to_string())?;
        Ok(Self {
            path: canonical_parent.join(file_name),
        })
    }

    fn read_existing(&self) -> Result<Option<ExistingWorkspaceFile>, String> {
        match std::fs::read(&self.path) {
            Ok(content) => Ok(Some(ExistingWorkspaceFile { content })),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("Read error: {error}")),
        }
    }

    fn verify_unchanged(
        &self,
        existing: &ExistingWorkspaceFile,
        expected_content: &[u8],
    ) -> Result<Vec<u8>, String> {
        if existing.content != expected_content {
            return Err(
                "Workspace target did not match the requested no-op content initially".to_string(),
            );
        }
        let metadata = std::fs::symlink_metadata(&self.path)
            .map_err(|error| format!("Cannot inspect no-op workspace target: {error}"))?;
        if !metadata.file_type().is_file() {
            return Err("Workspace no-op target must remain a regular file".to_string());
        }
        let canonical = std::fs::canonicalize(&self.path)
            .map_err(|error| format!("Cannot resolve no-op workspace target: {error}"))?;
        let workspace = std::fs::canonicalize(
            std::env::current_dir()
                .map_err(|error| format!("Cannot determine the current workspace: {error}"))?,
        )
        .map_err(|error| format!("Cannot resolve the current workspace: {error}"))?;
        if !canonical.starts_with(&workspace) {
            return Err("Workspace target escaped during no-op verification".to_string());
        }
        let observed = std::fs::read(&self.path)
            .map_err(|error| format!("Cannot read no-op workspace target: {error}"))?;
        if observed != existing.content || observed != expected_content {
            return Err(
                "Workspace target bytes changed concurrently during no-op verification".to_string(),
            );
        }
        let final_metadata = std::fs::symlink_metadata(&self.path)
            .map_err(|error| format!("Cannot re-inspect no-op workspace target: {error}"))?;
        let final_content = std::fs::read(&self.path)
            .map_err(|error| format!("Cannot re-read no-op workspace target: {error}"))?;
        if !final_metadata.file_type().is_file()
            || final_content != observed
            || final_content != expected_content
        {
            return Err("Workspace target changed during final no-op verification".to_string());
        }
        Ok(final_content)
    }

    fn commit(
        &self,
        content: &[u8],
        _existing: Option<&ExistingWorkspaceFile>,
    ) -> Result<(), String> {
        std::fs::write(&self.path, content).map_err(|error| format!("Write error: {error}"))
    }
}

pub(super) async fn execute_file_edit(input: Value) -> Result<Value, String> {
    let params: FileEditInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;
    let target = SecureWorkspaceTarget::open(&params.path, false)?;
    let existing = target
        .read_existing()?
        .ok_or_else(|| format!("Read error: file not found: {}", params.path))?;
    let content = String::from_utf8(existing.content.clone())
        .map_err(|error| format!("Read error: workspace file is not UTF-8 text: {error}"))?;

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

    let replacements = if params.replace_all.unwrap_or(false) {
        count
    } else {
        1
    };

    let changed = new_content != content;
    if changed {
        enforce_kernel_overwrite_baseline(
            &existing,
            params.expected_current_sha256.as_deref(),
            params.require_overwrite_baseline,
        )?;
        target.commit(new_content.as_bytes(), Some(&existing))?;
    }

    Ok(json!({
        "path": params.path,
        "success": true,
        "changed": changed,
        "replacements": if changed { replacements } else { 0 },
        "content_sha256": CryptoUtils::sha256_hex(&new_content),
        "size_bytes": new_content.len(),
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
    let params: PowerShellInput =
        serde_json::from_value(input).map_err(|e| format!("Invalid input: {}", e))?;

    if params.execution_profile.is_clean_verification() && params.run_in_background.unwrap_or(false)
    {
        return Err(
            "Clean verification profile requires a foreground verifier so its isolated cache lifetime can be bounded"
                .to_string(),
        );
    }
    let verification_cache = verification_cache_isolation(params.execution_profile)?;

    let exe = if cfg!(target_os = "windows") {
        "powershell"
    } else {
        "pwsh"
    };

    let exe_path = match which_powershell(exe) {
        Some(p) => p,
        None => return Err(format!("{} not found on this system", exe)),
    };

    let timeout_ms = params.timeout.unwrap_or(60_000);
    let profiled_command = profiled_powershell_command(
        &params.command,
        params.execution_profile,
        verification_cache.as_ref().map(tempfile::TempDir::path),
    )?;

    let mut command = tokio::process::Command::new(&exe_path);
    command
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &profiled_command,
        ])
        .env_clear()
        .envs(crate::tools::process_env::sanitized_child_environment(true))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    apply_clean_verification_environment_tokio(
        &mut command,
        params.execution_profile,
        verification_cache.as_ref().map(tempfile::TempDir::path),
    )?;
    let mut child = command.spawn().map_err(|e| format!("Spawn error: {}", e))?;

    let stdout_task = child
        .stdout
        .take()
        .map(|stdout| tokio::spawn(read_bounded_output(stdout)));
    let stderr_task = child
        .stderr
        .take()
        .map(|stderr| tokio::spawn(read_bounded_output(stderr)));
    let started = Instant::now();
    let wait_result =
        tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), child.wait()).await;
    let (status, timed_out) = match wait_result {
        Ok(result) => (
            Some(result.map_err(|error| format!("Wait error: {error}"))?),
            false,
        ),
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            (None, true)
        }
    };
    let stdout = join_output_capture(stdout_task).await;
    let stderr = join_output_capture(stderr_task).await;
    let original_size = stdout.total_bytes.saturating_add(stderr.total_bytes);

    if timed_out {
        let mut response = json!({
            "command": params.command,
            "timed_out": true,
            "stdout": stdout.text,
            "stderr": stderr.text,
            "truncated": stdout.truncated || stderr.truncated,
            "original_size": original_size,
            "duration_ms": started.elapsed().as_millis() as u64,
            "error": format!("Timeout after {}ms", timeout_ms),
            "shell": exe,
        });
        attach_verification_profile(&mut response, params.execution_profile);
        return Ok(response);
    }

    let mut response = json!({
        "command": params.command,
        "exit_code": status.and_then(|status| status.code()).unwrap_or(-1),
        "stdout": stdout.text,
        "stderr": stderr.text,
        "truncated": stdout.truncated || stderr.truncated,
        "original_size": original_size,
        "duration_ms": started.elapsed().as_millis() as u64,
        "shell": exe,
    });
    attach_verification_profile(&mut response, params.execution_profile);
    Ok(response)
}

fn apply_clean_verification_environment_tokio(
    command: &mut tokio::process::Command,
    execution_profile: ToolExecutionProfile,
    verification_cache: Option<&Path>,
) -> Result<(), String> {
    if !execution_profile.is_clean_verification() {
        return Ok(());
    }
    let cache = verification_cache.ok_or_else(|| {
        "Clean verification profile is missing its kernel-owned cache directory".to_string()
    })?;
    match execution_profile {
        ToolExecutionProfile::CleanPythonVerification => {
            command.env("PYTHONPYCACHEPREFIX", cache.join("pycache"));
        }
        ToolExecutionProfile::CleanPytestVerification => {
            command.env("PYTHONPYCACHEPREFIX", cache.join("pycache"));
            command.env(
                "PYTEST_ADDOPTS",
                format!("-o cache_dir={}", cache.join("pytest-cache").display()),
            );
        }
        ToolExecutionProfile::CleanMermaidVerification => {
            std::fs::write(
                cache.join("puppeteer.json"),
                r#"{"args":["--no-sandbox","--disable-setuid-sandbox"]}
"#,
            )
            .map_err(|error| {
                format!("Cannot create kernel-owned Mermaid verifier config: {error}")
            })?;
        }
        ToolExecutionProfile::Standard => {}
    }
    Ok(())
}

fn profiled_powershell_command(
    command: &str,
    execution_profile: ToolExecutionProfile,
    verification_cache: Option<&Path>,
) -> Result<String, String> {
    if execution_profile != ToolExecutionProfile::CleanMermaidVerification {
        return Ok(command.to_string());
    }
    let cache = verification_cache.ok_or_else(|| {
        "Clean Mermaid verification profile is missing its kernel-owned directory".to_string()
    })?;
    let config = cache.join("puppeteer.json");
    let escaped_config = config.to_string_lossy().replace('\'', "''");
    Ok(format!(
        "$ghMmdc = (Get-Command mmdc -CommandType Application).Source; function mmdc {{ & $ghMmdc -p '{escaped_config}' @args }}; {command}"
    ))
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
            '>' => {
                in_tag = false;
                if !text.is_empty() && !text.ends_with(' ') {
                    text.push(' ');
                    prev_space = true;
                }
            }
            _ if in_tag => {}
            ch if ch.is_whitespace() => {
                if !prev_space {
                    text.push(' ');
                    prev_space = true;
                }
            }
            _ => {
                text.push(ch);
                prev_space = false;
            }
        }
    }
    let decoded = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&nbsp;", " ");
    let mut result = String::with_capacity(decoded.len());
    let mut ps = false;
    for ch in decoded.chars() {
        if ch.is_whitespace() {
            if !ps {
                result.push(' ');
                ps = true;
            }
        } else {
            result.push(ch);
            ps = false;
        }
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
        let Some(hp) = after.find("href=") else {
            rem = &after[1..];
            continue;
        };
        let hs = &after[hp + 5..];
        let Some((url, rest)) = extract_quoted_str(hs) else {
            rem = &after[1..];
            continue;
        };
        let Some(ct) = rest.find('>') else {
            rem = &after[1..];
            continue;
        };
        let at = &rest[ct + 1..];
        let Some(ea) = at.find("</a>") else {
            rem = &after[1..];
            continue;
        };
        let title = html_to_text(&at[..ea]);
        rem = &at[ea + 4..];
        let snippet = if let Some(sp) = rem.find("result__snippet") {
            let sa = &rem[sp..];
            if let Some(tc) = sa.find('>') {
                let sc = &sa[tc + 1..];
                if let Some(es) = sc.find("</") {
                    html_to_text(&sc[..es])
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
    }
    results
}

fn extract_ddg_api_results(body: &str) -> Vec<Value> {
    let mut results = Vec::new();
    let Ok(data) = serde_json::from_str::<Value>(body) else {
        return results;
    };

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
        let Some(link_start) = rem.find("class=\"result-link\"") else {
            break;
        };
        let link_section = &rem[link_start..];

        let Some(href_pos) = link_section.find("href=") else {
            rem = &link_section[1..];
            continue;
        };
        let href_str = &link_section[href_pos + 5..];
        let Some((url, after_url)) = extract_quoted_str(href_str) else {
            rem = &link_section[1..];
            continue;
        };

        let Some(gt_pos) = after_url.find('>') else {
            rem = &link_section[1..];
            continue;
        };
        let title_start = &after_url[gt_pos + 1..];
        let Some(title_end) = title_start.find("</a>") else {
            rem = &link_section[1..];
            continue;
        };
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
    if q != '"' && q != '\'' {
        return None;
    }
    let rest = &input[q.len_utf8()..];
    let end = rest.find(q)?;
    Some((rest[..end].to_string(), &rest[end + q.len_utf8()..]))
}

fn urlencode(s: &str) -> String {
    let mut encoded = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(char::from(byte));
            }
            b' ' => encoded.push('+'),
            byte => {
                use std::fmt::Write as _;
                let _ = write!(encoded, "%{byte:02X}");
            }
        }
    }
    encoded
}

fn normalized_domain(pattern: &str) -> Option<String> {
    let trimmed = pattern.trim().trim_end_matches('.');
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(url) = reqwest::Url::parse(trimmed) {
        return url.host_str().map(|host| host.to_ascii_lowercase());
    }
    let domain = trimmed.trim_start_matches('.').to_ascii_lowercase();
    if domain.contains('/') || domain.contains(':') || domain.contains(char::is_whitespace) {
        None
    } else {
        Some(domain)
    }
}

fn domain_matches(url: &str, pattern: &str) -> bool {
    let Some(host) = reqwest::Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
    else {
        return false;
    };
    let Some(domain) = normalized_domain(pattern) else {
        return false;
    };
    host == domain || host.ends_with(&format!(".{domain}"))
}

fn filter_search_results(results: &mut Vec<Value>, params: &WebSearchInput) {
    if let Some(allowed) = &params.allowed_domains {
        results.retain(|result| {
            result["url"]
                .as_str()
                .is_some_and(|url| allowed.iter().any(|domain| domain_matches(url, domain)))
        });
    }
    if let Some(blocked) = &params.blocked_domains {
        results.retain(|result| {
            result["url"]
                .as_str()
                .is_none_or(|url| !blocked.iter().any(|domain| domain_matches(url, domain)))
        });
    }
}

fn filter_search_response(response: &mut Value, params: &WebSearchInput) {
    if let Some(results) = response.get_mut("results").and_then(Value::as_array_mut) {
        filter_search_results(results, params);
        results.truncate(8);
    }
}

#[cfg(test)]
mod web_search_filter_tests {
    use super::*;

    #[test]
    fn form_encoding_uses_utf8_bytes() {
        assert_eq!(
            urlencode("滑翔 🐎 rust"),
            "%E6%BB%91%E7%BF%94+%F0%9F%90%8E+rust"
        );
    }

    #[test]
    fn domain_filter_matches_only_host_or_subdomain() {
        assert!(domain_matches(
            "https://docs.example.com/path?next=evil.test",
            "example.com"
        ));
        assert!(domain_matches(
            "https://example.com/path",
            "https://example.com"
        ));
        assert!(!domain_matches(
            "https://example.com.evil.test/example.com",
            "example.com"
        ));
        assert!(!domain_matches(
            "https://evil.test/?target=example.com",
            "example.com"
        ));
    }

    #[test]
    fn allowed_and_blocked_domains_share_the_same_filter() {
        let params = WebSearchInput {
            query: "test".to_string(),
            allowed_domains: Some(vec!["example.com".to_string()]),
            blocked_domains: Some(vec!["private.example.com".to_string()]),
        };
        let mut results = vec![
            json!({"url": "https://www.example.com/public"}),
            json!({"url": "https://private.example.com/secret"}),
            json!({"url": "https://example.com.evil.test/"}),
        ];
        filter_search_results(&mut results, &params);
        assert_eq!(
            results,
            vec![json!({"url": "https://www.example.com/public"})]
        );
    }
}

pub(super) async fn execute_create_skill(
    input: Value,
    interactions: Option<Arc<crate::llm::LlmInteractionService>>,
    gateway: Option<Arc<crate::gateway::unified_gateway::UnifiedGateway>>,
    shared_graph: Option<Arc<SkillGraphStore>>,
    shared_registry: Option<Arc<crate::tools::SkillRegistry>>,
    vector_store: Option<Arc<crate::memory::hyperspace_store::HyperspaceStore>>,
    interaction_scope: Option<crate::llm::LlmInteractionScope>,
) -> Result<Value, String> {
    let description = input["description"].as_str().unwrap_or("").to_string();
    if description.is_empty() {
        return Err("description is required".to_string());
    }

    let skill_name_hint = input["skill_name_hint"].as_str().map(String::from);
    let category_hint = input["category_hint"].as_str().map(String::from);
    let security_level_override = input["security_level_override"].as_str().map(String::from);

    if interactions.is_some() || gateway.is_some() {
        let graph_store =
            shared_graph.unwrap_or_else(|| Arc::new(crate::skill_graph::SkillGraphStore::new()));
        let registry = shared_registry
            .unwrap_or_else(|| std::sync::Arc::new(crate::tools::SkillRegistry::new()));
        let config = crate::skill_graph::SkillCreatorConfig::default();
        let creator = if let Some(interactions) = interactions {
            crate::skill_graph::SkillCreator::new_with_interactions(
                interactions,
                graph_store,
                registry,
                config,
            )
        } else {
            crate::skill_graph::SkillCreator::new(
                gateway.expect("gateway checked above"),
                graph_store,
                registry,
                config,
            )
        };
        let creator = match vector_store {
            Some(store) => creator.with_vector_store(store),
            None => creator,
        };

        let request = crate::skill_graph::CreateSkillRequest {
            description,
            skill_name_hint,
            category_hint,
            security_level_override,
        };

        let result = match interaction_scope {
            Some(scope) => {
                creator
                    .create_from_description_with_scope(request, scope)
                    .await
            }
            None => creator.create_from_description(request).await,
        }
        .map_err(|e| format!("Create Skill failed: {:?}", e))?;

        Ok(json!({
            "skill_iri": result.skill_iri,
            "name": result.name,
            "registered": true,
            "executable": false,
            "activation_status": "definition_only",
            "note": "The definition is registered for review and discovery. No ToolExecutor handler is generated automatically.",
            "json_ld": result.json_ld,
        }))
    } else {
        let name = skill_name_hint.unwrap_or_else(|| {
            description
                .split_whitespace()
                .take(2)
                .collect::<Vec<_>>()
                .join("_")
                .to_lowercase()
        });
        let category = category_hint.unwrap_or_else(|| "system".to_string());

        Ok(json!({
            "skill_iri": format!("iri://skills/{}", name),
            "name": name,
            "description": description,
            "category": category,
            "registered": false,
            "executable": false,
            "activation_status": "template_only",
            "note": "Gateway not initialized, returning template only. Use SkillCreator API to create the full Skill."
        }))
    }
}

pub(super) async fn execute_convert_skill(
    input: Value,
    interactions: Option<Arc<crate::llm::LlmInteractionService>>,
    gateway: Option<Arc<crate::gateway::unified_gateway::UnifiedGateway>>,
    shared_graph: Option<Arc<SkillGraphStore>>,
    shared_registry: Option<Arc<crate::tools::SkillRegistry>>,
    vector_store: Option<Arc<crate::memory::hyperspace_store::HyperspaceStore>>,
    interaction_scope: Option<crate::llm::LlmInteractionScope>,
) -> Result<Value, String> {
    let markdown_content = input["markdown_content"].as_str().unwrap_or("").to_string();
    if markdown_content.is_empty() {
        return Err("markdown_content is required".to_string());
    }
    let source_path = input["source_path"].as_str().map(String::from);

    if interactions.is_some() || gateway.is_some() {
        let graph_store =
            shared_graph.unwrap_or_else(|| Arc::new(crate::skill_graph::SkillGraphStore::new()));
        let registry = shared_registry
            .unwrap_or_else(|| std::sync::Arc::new(crate::tools::SkillRegistry::new()));
        let config = crate::skill_graph::SkillCreatorConfig::default();
        let creator = if let Some(interactions) = interactions {
            crate::skill_graph::SkillCreator::new_with_interactions(
                interactions,
                graph_store,
                registry,
                config,
            )
        } else {
            crate::skill_graph::SkillCreator::new(
                gateway.expect("gateway checked above"),
                graph_store,
                registry,
                config,
            )
        };
        let creator = match vector_store {
            Some(store) => creator.with_vector_store(store),
            None => creator,
        };

        let request = crate::skill_graph::ConvertMarkdownRequest {
            markdown_content,
            source_path,
        };

        let result = match interaction_scope {
            Some(scope) => {
                creator
                    .convert_from_markdown_with_scope(request, scope)
                    .await
            }
            None => creator.convert_from_markdown(request).await,
        }
        .map_err(|e| format!("Convert Skill failed: {:?}", e))?;

        Ok(json!({
            "skill_iri": result.skill_iri,
            "name": result.name,
            "registered": true,
            "executable": false,
            "activation_status": "definition_only",
            "note": "The definition is registered for review and discovery. No ToolExecutor handler is generated automatically.",
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
            "executable": false,
            "activation_status": "template_only",
            "note": "Gateway not initialized, using static parse. Full conversion requires SkillCreator API."
        }))
    }
}

// ========== Knowledge graph tool implementation ==========

pub(super) async fn execute_knowledge_extract(
    input: Value,
    kg_store: Arc<RwLock<KnowledgeGraphStore>>,
) -> Result<Value, String> {
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
        .map_err(|_| {
            "API key not configured: set ONE_API_KEY or OPENAI_API_KEY environment variable"
                .to_string()
        })?;
    let model =
        std::env::var("KG_EXTRACT_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".to_string());

    let ontology = OntologyManager::new();
    let temp_store =
        KnowledgeGraphStore::new().map_err(|e| format!("Create temporary store failed: {}", e))?;
    let extractor = KnowledgeExtractor::new(ontology, temp_store, api_url, api_key, model);

    let result = extractor.extract(&text, domain.as_deref()).await?;

    let store = kg_store
        .write()
        .map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
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

pub(super) async fn execute_knowledge_query(
    input: Value,
    kg_store: Arc<RwLock<KnowledgeGraphStore>>,
) -> Result<Value, String> {
    let sparql = input["sparql"].as_str().unwrap_or("").to_string();
    if sparql.is_empty() {
        return Err("sparql parameter cannot be empty".to_string());
    }
    let named_graph = input["named_graph"].as_str().map(String::from);

    let store = kg_store
        .read()
        .map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
    match store.query_sparql(&sparql, named_graph.as_deref()) {
        Ok(results) => Ok(json!({
            "success": true,
            "results": results,
            "count": results.len(),
        })),
        Err(e) => Err(diagnose_sparql_error(&sparql, &e)),
    }
}

/// Parse a SPARQL error and return a clean, actionable diagnostic for the LLM.
/// The raw Oxigraph error is cryptic; this extracts position context and suggests fixes.
fn diagnose_sparql_error(sparql: &str, raw_error: &str) -> String {
    let mut msg = String::from("SPARQL query syntax error. Fix the query and retry.\n\n");

    let error_body = raw_error
        .strip_prefix("SPARQL query failed: ")
        .unwrap_or(raw_error);
    let pos = parse_error_position(error_body);

    if let Some((line, col)) = pos {
        let lines: Vec<&str> = sparql.lines().collect();
        if line > 0 && line <= lines.len() {
            let problem_line = lines[line - 1];
            msg.push_str(&format!("┌─ Position: line {line}, column {col}\n"));
            msg.push_str("│\n");
            msg.push_str(&format!("│   {problem_line}\n"));
            let caret_pos = col.saturating_sub(1).min(problem_line.len());
            msg.push_str(&format!("│   {:indent$}↑ here\n", "", indent = caret_pos));
            msg.push_str("│\n");

            if let Some(detail_start) = error_body.find(": ").and_then(|p| {
                let rest = &error_body[p + 2..];
                rest.find(": ").map(|p2| &rest[p2 + 2..])
            }) {
                let detail = detail_start.trim();
                if !detail.is_empty() && detail.len() < 200 {
                    msg.push_str(&format!("└─ Parser: {detail}\n"));
                }
            }
        }
    } else {
        let first_line = raw_error.lines().next().unwrap_or(raw_error);
        msg.push_str(&format!("Error: {first_line}\n"));
    }

    msg.push_str("\nCommon fixes for LLM-generated SPARQL:\n");
    msg.push_str("  • Close all parentheses: FILTER(REGEX(...)) not FILTER(REGEX(...\n");
    msg.push_str("  • Add PREFIX declarations for every namespace used\n");
    msg.push_str("  • Use ex: prefix for ontology types: ?s a ex:Task\n");
    msg.push_str("  • Use single quotes for string literals: 'value' not \"value\"\n");
    msg.push_str("  • Check all curly braces { } are properly matched\n");
    msg.push_str("  • Try a simpler query without REGEX or nested FILTER\n");

    msg
}

/// Extract (line, col) from an Oxigraph error like "error at 3:15: expected ..."
fn parse_error_position(body: &str) -> Option<(usize, usize)> {
    let prefix = "error at ";
    let at_pos = body.find(prefix)?;
    let rest = &body[at_pos + prefix.len()..];
    let colon1 = rest.find(':')?;
    let line = rest[..colon1].parse::<usize>().ok()?;
    let rest2 = &rest[colon1 + 1..];
    let colon2 = rest2.find(':')?;
    let col = rest2[..colon2].parse::<usize>().ok()?;
    Some((line, col))
}

pub(super) async fn execute_knowledge_search(
    input: Value,
    kg_store: Arc<RwLock<KnowledgeGraphStore>>,
) -> Result<Value, String> {
    let keyword = input["keyword"].as_str().unwrap_or("").to_string();
    if keyword.is_empty() {
        return Err("keyword parameter cannot be empty".to_string());
    }
    let entity_type = input["entity_type"].as_str().map(String::from);

    let store = kg_store
        .read()
        .map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
    let results = store.search_entities(&keyword, entity_type.as_deref())?;

    Ok(json!({
        "success": true,
        "results": results,
        "count": results.len(),
        "keyword": keyword,
    }))
}

pub(super) async fn execute_knowledge_neighbors(
    input: Value,
    kg_store: Arc<RwLock<KnowledgeGraphStore>>,
) -> Result<Value, String> {
    let entity_id = input["entity_id"].as_str().unwrap_or("").to_string();
    if entity_id.is_empty() {
        return Err("entity_id parameter cannot be empty".to_string());
    }
    let depth = input["depth"].as_u64().unwrap_or(1).min(3) as usize;

    let store = kg_store
        .read()
        .map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
    let result = store.get_neighbors(&entity_id, depth)?;

    Ok(json!({
        "success": true,
        "result": result,
    }))
}

pub(super) async fn execute_knowledge_import_json(
    input: Value,
    kg_store: Arc<RwLock<KnowledgeGraphStore>>,
) -> Result<Value, String> {
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
                    if desc_field == Some(key.as_str()) {
                        continue;
                    }
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
                        format!(
                            "{}{}",
                            target_prefix.trim_end_matches('/'),
                            target_val.trim_start_matches('/')
                        )
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
        let store = kg_store
            .read()
            .map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
        store.default_graph().to_string()
    };

    let extraction = crate::knowledge_graph::types::LLMExtractionOutput {
        nodes: nodes.clone(),
        edges: edges.clone(),
    };
    let result = RdfMapper::map_extraction(&extraction, &graph);

    {
        let store = kg_store
            .write()
            .map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
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

pub(super) async fn execute_ontology_register(
    input: Value,
    kg_store: Arc<RwLock<KnowledgeGraphStore>>,
) -> Result<Value, String> {
    let terms = input["terms"]
        .as_array()
        .ok_or("terms parameter must be an array")?;
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
            predicate: "https://agent-os.org/ontology/meta/termType".to_string(),
            object: RdfValue::Literal(term_type),
            graph: Some(graph.to_string()),
        });
    }

    let registered = quads.len() / 3;
    {
        let store = kg_store
            .write()
            .map_err(|e| format!("Failed to acquire storage lock: {}", e))?;
        store.write_quads(&quads, graph)?;
    }

    Ok(json!({
        "success": true,
        "registered_terms": registered,
        "graph": graph,
    }))
}

pub(super) async fn execute_knowledge_bridge_with_store(
    input: Value,
    kg_store: Arc<RwLock<KnowledgeGraphStore>>,
) -> Result<Value, String> {
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
        _ => {
            return Err(format!(
                "Unsupported relation type: {}, options: HasSkill, ApplicableIn, RelatedTo",
                relation_type_str
            ))
        }
    };

    // Create a bridge that shares the same underlying Oxigraph store.
    // Both `kg_store` and the bridge operate on the same triples.
    let guard = kg_store
        .read()
        .map_err(|e| format!("Failed to acquire kg_store lock: {}", e))?;
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

pub(super) async fn execute_knowledge_extract_code(
    input: Value,
    kg_store: Arc<RwLock<KnowledgeGraphStore>>,
) -> Result<Value, String> {
    let file_path = input["file_path"].as_str().unwrap_or("").to_string();
    if file_path.is_empty() {
        return Err("file_path parameter cannot be empty".to_string());
    }
    let graph = input["named_graph"]
        .as_str()
        .unwrap_or("graph:code")
        .to_string();
    let force = input["force"].as_bool().unwrap_or(false);

    let store = kg_store
        .write()
        .map_err(|e| format!("Failed to acquire storage lock: {}", e))?;

    if force {
        let result = CodeAstExtractor::extract_from_file(&file_path, &graph)?;
        store
            .delete_quads_by_subject_prefix(&format!("iri://entity/file:{}", file_path), &graph)?;
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
            IncrementalResult::Created {
                entity_count,
                relation_count,
                quad_count,
            } => Ok(json!({
                "success": true,
                "file_path": file_path,
                "mode": "incremental",
                "status": "created",
                "entity_count": entity_count,
                "relation_count": relation_count,
                "quad_count": quad_count,
                "graph": graph,
            })),
            IncrementalResult::Updated {
                entity_count,
                relation_count,
                quad_count,
                deleted_quads,
            } => Ok(json!({
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_mermaid_profile_uses_a_kernel_owned_root_safe_browser_config() {
        let cache = tempfile::tempdir().unwrap();
        let mut process = std::process::Command::new("true");
        apply_clean_verification_environment(
            &mut process,
            ToolExecutionProfile::CleanMermaidVerification,
            Some(cache.path()),
        )
        .unwrap();
        let config = cache.path().join("puppeteer.json");
        let parsed: Value =
            serde_json::from_str(&std::fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(
            parsed,
            json!({"args": ["--no-sandbox", "--disable-setuid-sandbox"]})
        );

        let command = profiled_shell_command(
            "cd project && mmdc -i design.md -o /tmp/design.svg",
            ToolExecutionProfile::CleanMermaidVerification,
            Some(cache.path()),
        )
        .unwrap();
        assert!(command.starts_with("mmdc() { command mmdc -p '"));
        assert!(command.contains(&config.to_string_lossy().to_string()));
        assert!(command.ends_with("cd project && mmdc -i design.md -o /tmp/design.svg"));

        let powershell = profiled_powershell_command(
            "mmdc -i design.md -o $env:TEMP/design.svg",
            ToolExecutionProfile::CleanMermaidVerification,
            Some(cache.path()),
        )
        .unwrap();
        assert!(powershell.contains("Get-Command mmdc -CommandType Application"));
        assert!(powershell.contains(&config.to_string_lossy().to_string()));
        assert!(powershell.ends_with("mmdc -i design.md -o $env:TEMP/design.svg"));
    }

    #[cfg(unix)]
    #[test]
    fn clean_mermaid_profile_renders_when_mmdc_is_installed() {
        if std::process::Command::new("mmdc")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_err()
        {
            return;
        }
        let workspace = tempfile::tempdir().unwrap();
        let source = workspace.path().join("diagram.mmd");
        let output = workspace.path().join("diagram.svg");
        std::fs::write(&source, "flowchart LR\nA --> B\n").unwrap();
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(execute_bash(json!({
                "command": format!(
                    "mmdc -i '{}' -o '{}' -q",
                    source.display(),
                    output.display(),
                ),
                "__gh_execution_profile": "clean_mermaid_verification",
            })))
            .unwrap();
        assert_eq!(result["exit_code"], 0, "{result:?}");
        assert_eq!(result["execution_profile"], "clean_mermaid_verification");
        assert_eq!(result["isolated_environment"]["PUPPETEER_CONFIG"], true);
        assert!(output.is_file() && output.metadata().unwrap().len() > 0);
        assert!(
            !result.to_string().contains("glidinghorse-verification-"),
            "the ephemeral browser config path is kernel-private"
        );
    }

    #[cfg(unix)]
    fn workspace_tempdir(prefix: &str) -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(prefix)
            .tempdir_in(std::env::current_dir().expect("workspace cwd"))
            .expect("workspace temporary directory")
    }

    #[cfg(unix)]
    #[test]
    fn file_read_content_receipt_is_canonical_sha256_hex() {
        let container = workspace_tempdir(".workspace-file-read-hash-test-");
        let path = container.path().join("receipt.txt");
        std::fs::write(&path, "abc").unwrap();
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(execute_file_read(json!({"path": path})))
            .unwrap();

        assert_eq!(
            result["content_sha256"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[cfg(unix)]
    #[test]
    fn workspace_resolution_rejects_missing_path_below_outside_symlink() {
        use std::os::unix::fs::symlink;

        let container = workspace_tempdir(".workspace-ancestor-test-");
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), container.path().join("escape")).unwrap();
        let requested = container.path().join("escape/missing/deep/report.md");

        let error = check_path_in_workspace(requested.to_str().unwrap()).unwrap_err();
        assert!(error.contains("not within the workspace"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn workspace_resolution_fails_closed_for_dangling_symlink_ancestor() {
        use std::os::unix::fs::symlink;

        let container = workspace_tempdir(".workspace-dangling-test-");
        symlink(
            container.path().join("does-not-exist"),
            container.path().join("dangling"),
        )
        .unwrap();
        let requested = container.path().join("dangling/report.md");

        let error = check_path_in_workspace(requested.to_str().unwrap()).unwrap_err();
        assert!(
            error.contains("Cannot resolve requested path ancestor"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn grep_search_accepts_an_absolute_path_inside_the_workspace() {
        let container = workspace_tempdir(".workspace-grep-inside-test-");
        std::fs::write(container.path().join("sample.txt"), "alpha\nbeta\n").unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let result = runtime
            .block_on(execute_grep_search(json!({
                "pattern": "alpha",
                "path": container.path(),
                "output_mode": "content"
            })))
            .unwrap();

        assert_eq!(result["num_matches"], 1);
        assert!(result["content"]
            .as_str()
            .is_some_and(|content| content.contains("alpha")));
    }

    #[cfg(unix)]
    #[test]
    fn grep_search_rejects_an_absolute_path_outside_the_workspace() {
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "must not be read").unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let error = runtime
            .block_on(execute_grep_search(json!({
                "pattern": "secret",
                "path": outside.path()
            })))
            .unwrap_err();

        assert!(error.contains("not within the workspace"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn grep_search_rejects_a_workspace_symlink_escape() {
        use std::os::unix::fs::symlink;

        let container = workspace_tempdir(".workspace-grep-symlink-test-");
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "must not be read").unwrap();
        let escape = container.path().join("escape");
        symlink(outside.path(), &escape).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let error = runtime
            .block_on(execute_grep_search(json!({
                "pattern": "secret",
                "path": escape
            })))
            .unwrap_err();

        assert!(error.contains("not within the workspace"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn glob_search_accepts_an_absolute_root_inside_the_workspace() {
        let container = workspace_tempdir(".workspace-glob-inside-test-");
        std::fs::write(container.path().join("sample.txt"), "workspace data").unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let result = runtime
            .block_on(execute_glob_search(json!({
                "path": container.path(),
                "pattern": "*.txt"
            })))
            .unwrap();

        assert_eq!(result["count"], 1);
        assert!(result["files"].as_array().unwrap().iter().any(|path| {
            path.as_str()
                .is_some_and(|path| path.ends_with("sample.txt"))
        }));
    }

    #[cfg(unix)]
    #[test]
    fn glob_search_rejects_an_absolute_root_outside_the_workspace() {
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "must not be listed").unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let error = runtime
            .block_on(execute_glob_search(json!({
                "path": outside.path(),
                "pattern": "*.txt"
            })))
            .unwrap_err();

        assert!(error.contains("not within the workspace"), "{error}");
    }

    #[test]
    fn glob_search_rejects_absolute_and_parent_traversal_patterns() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        for pattern in ["../*", "nested/../../*", "/tmp/*"] {
            let error = runtime
                .block_on(execute_glob_search(json!({"pattern": pattern})))
                .unwrap_err();
            assert!(error.contains("Glob pattern must be relative"), "{error}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn glob_search_rejects_a_matching_symlink_escape() {
        use std::os::unix::fs::symlink;

        let container = workspace_tempdir(".workspace-glob-symlink-test-");
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "must not be listed").unwrap();
        symlink(outside.path(), container.path().join("escape")).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let error = runtime
            .block_on(execute_glob_search(json!({
                "path": container.path(),
                "pattern": "escape/*"
            })))
            .unwrap_err();

        assert!(error.contains("not within the workspace"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn discovery_tools_hide_workspace_runtime_state_before_monitor_scan() {
        let container = workspace_tempdir(".workspace-runtime-hide-test-");
        std::fs::write(container.path().join("visible.txt"), "project content").unwrap();
        std::fs::create_dir_all(container.path().join(".gliding_horse/ws_monitor")).unwrap();
        std::fs::write(
            container.path().join(".gliding_horse/ws_monitor/content"),
            "runtime state",
        )
        .unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let listed = runtime
            .block_on(execute_file_list(json!({"path": container.path()})))
            .unwrap();
        let names = listed["entries"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|entry| entry["name"].as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"visible.txt"));
        assert!(!names.contains(&".gliding_horse"));

        let globbed = runtime
            .block_on(execute_glob_search(json!({
                "path": container.path(),
                "pattern": "**/*"
            })))
            .unwrap();
        let files = globbed["files"].as_array().unwrap();
        assert!(files.iter().any(|path| {
            path.as_str()
                .is_some_and(|path| path.ends_with("visible.txt"))
        }));
        assert!(!files.iter().any(|path| {
            path.as_str()
                .is_some_and(|path| path.contains(".gliding_horse"))
        }));

        let error = runtime
            .block_on(execute_file_list(json!({
                "path": container.path().join(".gliding_horse")
            })))
            .unwrap_err();
        assert!(error.contains("runtime state"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn descriptor_bound_commit_cannot_follow_post_validation_symlink_swap() {
        use std::os::unix::fs::symlink;

        let container = workspace_tempdir(".workspace-toctou-test-");
        let outside = tempfile::tempdir().unwrap();
        let active = container.path().join("active");
        let parked = container.path().join("parked");
        std::fs::create_dir(&active).unwrap();
        let requested = active.join("report.md");

        // Prepare the capability while `active` is a real workspace directory,
        // then replace its pathname with a link to an outside directory before
        // committing. Revalidation from the root capability detects the
        // changed directory identity and fails closed.
        let target = SecureWorkspaceTarget::open(requested.to_str().unwrap(), true).unwrap();
        std::fs::rename(&active, &parked).unwrap();
        symlink(outside.path(), &active).unwrap();
        let error = target.commit(b"must not be written", None).unwrap_err();

        assert!(error.contains("changed before commit"), "{error}");
        assert!(!parked.join("report.md").exists());
        assert!(!outside.path().join("report.md").exists());
    }

    #[cfg(unix)]
    #[test]
    fn secure_commit_rejects_concurrent_final_file_changes() {
        let container = workspace_tempdir(".workspace-concurrent-write-test-");
        let path = container.path().join("report.md");
        std::fs::write(&path, "initial").unwrap();
        let target = SecureWorkspaceTarget::open(path.to_str().unwrap(), false).unwrap();
        let existing = target.read_existing().unwrap().unwrap();

        std::fs::write(&path, "concurrent update").unwrap();
        let error = target
            .commit(b"stale replacement", Some(&existing))
            .unwrap_err();

        assert!(error.contains("changed concurrently"), "{error}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "concurrent update");
    }

    #[cfg(unix)]
    #[test]
    fn secure_commit_does_not_overwrite_file_that_appeared_concurrently() {
        let container = workspace_tempdir(".workspace-concurrent-create-test-");
        let path = container.path().join("report.md");
        let target = SecureWorkspaceTarget::open(path.to_str().unwrap(), false).unwrap();
        assert!(target.read_existing().unwrap().is_none());

        std::fs::write(&path, "created by another actor").unwrap();
        let error = target.commit(b"must not overwrite", None).unwrap_err();

        assert!(error.contains("appeared concurrently"), "{error}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "created by another actor"
        );
    }

    #[cfg(unix)]
    #[test]
    fn secure_noop_revalidation_rejects_concurrent_byte_change() {
        let container = workspace_tempdir(".workspace-noop-byte-race-test-");
        let path = container.path().join("report.md");
        std::fs::write(&path, "expected").unwrap();
        let target = SecureWorkspaceTarget::open(path.to_str().unwrap(), false).unwrap();
        let existing = target.read_existing().unwrap().unwrap();

        std::fs::write(&path, "concurrent update").unwrap();
        let error = target.verify_unchanged(&existing, b"expected").unwrap_err();

        assert!(error.contains("changed concurrently"), "{error}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "concurrent update");
    }

    #[cfg(unix)]
    #[test]
    fn secure_noop_revalidation_rejects_final_symlink_swap() {
        use std::os::unix::fs::symlink;

        let container = workspace_tempdir(".workspace-noop-file-swap-test-");
        let outside = tempfile::tempdir().unwrap();
        let path = container.path().join("report.md");
        let parked = container.path().join("parked.md");
        let outside_path = outside.path().join("report.md");
        std::fs::write(&path, "same bytes").unwrap();
        std::fs::write(&outside_path, "same bytes").unwrap();
        let target = SecureWorkspaceTarget::open(path.to_str().unwrap(), false).unwrap();
        let existing = target.read_existing().unwrap().unwrap();

        std::fs::rename(&path, &parked).unwrap();
        symlink(&outside_path, &path).unwrap();
        let error = target
            .verify_unchanged(&existing, b"same bytes")
            .unwrap_err();

        assert!(error.contains("no longer safe"), "{error}");
        assert_eq!(
            std::fs::read_to_string(&outside_path).unwrap(),
            "same bytes"
        );
    }

    #[cfg(unix)]
    #[test]
    fn secure_noop_revalidation_rejects_identical_regular_file_replacement() {
        let container = workspace_tempdir(".workspace-noop-identity-swap-test-");
        let path = container.path().join("report.md");
        let parked = container.path().join("parked.md");
        std::fs::write(&path, "same bytes").unwrap();
        let target = SecureWorkspaceTarget::open(path.to_str().unwrap(), false).unwrap();
        let existing = target.read_existing().unwrap().unwrap();

        std::fs::rename(&path, &parked).unwrap();
        std::fs::write(&path, "same bytes").unwrap();
        let error = target
            .verify_unchanged(&existing, b"same bytes")
            .unwrap_err();

        assert!(error.contains("identity changed"), "{error}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "same bytes");
    }

    #[cfg(unix)]
    #[test]
    fn secure_noop_revalidation_rejects_parent_symlink_swap() {
        use std::os::unix::fs::symlink;

        let container = workspace_tempdir(".workspace-noop-parent-swap-test-");
        let outside = tempfile::tempdir().unwrap();
        let active = container.path().join("active");
        let parked = container.path().join("parked");
        std::fs::create_dir(&active).unwrap();
        std::fs::write(active.join("report.md"), "same bytes").unwrap();
        std::fs::write(outside.path().join("report.md"), "same bytes").unwrap();
        let target =
            SecureWorkspaceTarget::open(active.join("report.md").to_str().unwrap(), false).unwrap();
        let existing = target.read_existing().unwrap().unwrap();

        std::fs::rename(&active, &parked).unwrap();
        symlink(outside.path(), &active).unwrap();
        let error = target
            .verify_unchanged(&existing, b"same bytes")
            .unwrap_err();

        assert!(
            error.contains("changed during no-op verification"),
            "{error}"
        );
        assert_eq!(
            std::fs::read_to_string(outside.path().join("report.md")).unwrap(),
            "same bytes"
        );
    }

    #[cfg(unix)]
    #[test]
    fn file_write_noop_returns_hash_of_revalidated_final_bytes() {
        let container = workspace_tempdir(".workspace-secure-noop-test-");
        let path = container.path().join("report.md");
        std::fs::write(&path, "stable content").unwrap();
        let before = std::fs::metadata(&path).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let result = runtime
            .block_on(execute_file_write(json!({
                "path": path,
                "content": "stable content"
            })))
            .unwrap();

        let after = std::fs::metadata(&path).unwrap();
        assert_eq!(result["success"], true);
        assert_eq!(result["changed"], false);
        assert_eq!(result["created"], false);
        assert_eq!(result["bytes_written"], 0);
        assert_eq!(
            result["content_sha256"],
            CryptoUtils::sha256_hex(&std::fs::read_to_string(&path).unwrap())
        );
        assert_eq!(before.dev(), after.dev());
        assert_eq!(before.ino(), after.ino());
    }

    #[cfg(unix)]
    #[test]
    fn file_write_and_edit_use_secure_atomic_workspace_commit() {
        let container = workspace_tempdir(".workspace-secure-write-test-");
        let path = container.path().join("nested/report.md");
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let write = runtime
            .block_on(execute_file_write(json!({
                "path": path,
                "content": "alpha beta"
            })))
            .unwrap();
        assert_eq!(write["created"], true);
        let edit = runtime
            .block_on(execute_file_edit(json!({
                "path": path,
                "old_string": "beta",
                "new_string": "gamma"
            })))
            .unwrap();
        assert_eq!(edit["changed"], true);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "alpha gamma");
        let leftovers = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".glidinghorse-write-")
            })
            .count();
        assert_eq!(leftovers, 0);
    }

    #[test]
    fn test_is_build_or_vendored_dir() {
        let cases = [
            ("target", true),
            ("node_modules", true),
            (".git", true),
            ("dist", true),
            ("build", true),
            ("vendor", true),
            (".venv", true),
            ("__pycache__", true),
            (".next", true),
            ("src", false),
            ("crates", false),
            ("docs", false),
            ("target_arch", false),
            ("mybuild", false),
        ];
        for (name, expected) in cases {
            let path = std::path::Path::new(name);
            assert_eq!(is_build_or_vendored_dir(path), expected, "case: {name}");
        }
    }

    #[test]
    fn test_parse_error_position_standard() {
        let err = "error at 1:123: expected one of Prefix not found";
        assert_eq!(parse_error_position(err), Some((1, 123)));
    }

    #[test]
    fn test_parse_error_position_multi_line() {
        let err = "error at 3:15: expected '.', found 'x'";
        assert_eq!(parse_error_position(err), Some((3, 15)));
    }

    #[test]
    fn test_parse_error_position_no_match() {
        assert_eq!(parse_error_position("unexpected end of input"), None);
        assert_eq!(parse_error_position(""), None);
    }

    #[test]
    fn test_diagnose_sparql_error_includes_position() {
        let sparql = "SELECT ?s WHERE { ?s a ?type . FILTER(REGEX(?s, \"test\" )";
        let raw = "SPARQL query failed: error at 1:53: expected '}'";
        let msg = diagnose_sparql_error(sparql, raw);
        assert!(msg.contains("line 1"), "should mention the line number");
        assert!(msg.contains("column 53"), "should mention the column");
        assert!(msg.contains("↑"), "should have a caret marker");
        assert!(
            msg.contains("Common fixes"),
            "should include fix suggestions"
        );
        assert!(msg.contains("parentheses"), "should mention parentheses");
    }

    #[test]
    fn test_diagnose_sparql_error_missing_prefix() {
        let sparql = "SELECT ?x WHERE { ?x ex:name ?n }";
        let raw = "error at 1:22: expected one of Prefix not found";
        let msg = diagnose_sparql_error(sparql, raw);
        assert!(msg.contains("PREFIX"), "should suggest adding PREFIX");
        assert!(msg.contains("↑"), "should have caret");
    }

    #[test]
    fn test_diagnose_sparql_error_unparseable_fallback() {
        let sparql = "BAD SPARQL";
        let raw = "unknown parser error: catastrophic failure";
        let msg = diagnose_sparql_error(sparql, raw);
        assert!(msg.contains("unknown"), "should include the raw error text");
        assert!(msg.contains("Common fixes"), "should still suggest fixes");
    }

    #[test]
    fn test_diagnose_sparql_error_handles_empty_sparql() {
        let msg = diagnose_sparql_error("", "error at 1:1: unexpected end");
        assert!(msg.contains("Common fixes"), "should handle gracefully");
    }
}
