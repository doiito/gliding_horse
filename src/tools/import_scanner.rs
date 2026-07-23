use std::path::{Path, PathBuf};

use regex::Regex;
use tracing::debug;

/// A single discovered local file reference from scanning imports.
#[derive(Debug, Clone)]
pub struct DiscoveredImport {
    /// Absolute or workspace-relative path resolved from the import statement.
    pub resolved_path: String,
    /// The original import statement (e.g. "use crate::foo::bar").
    pub source: String,
    /// Language hint.
    pub language: &'static str,
}

/// Scan a file's content for local import/mod/use statements and resolve
/// them to file paths relative to the scanned file's directory.
///
/// Returns only references that appear to be local files (external crates
/// and packages are filtered out).
pub fn scan_imports(file_path: &str, content: &str) -> Vec<DiscoveredImport> {
    let path = Path::new(file_path);
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let lang = match ext {
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" => "javascript",
        "py" => "python",
        "go" => "golang",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "hpp" | "cc" | "cxx" => "cpp",
        _ => return vec![],
    };

    let scanners: Vec<&dyn Fn(&str, &Path) -> Vec<DiscoveredImport>> = match lang {
        "rust" => vec![&scan_rust_imports],
        "typescript" | "javascript" => vec![&scan_js_imports],
        "python" => vec![&scan_python_imports],
        "golang" => vec![&scan_golang_imports],
        "java" => vec![&scan_java_imports],
        "c" | "cpp" => vec![&scan_c_cpp_imports],
        _ => return vec![],
    };

    let mut results = Vec::new();
    for scanner in &scanners {
        results.extend(scanner(content, parent));
    }

    // Deduplicate by resolved_path
    results.sort_by(|a, b| a.resolved_path.cmp(&b.resolved_path));
    results.dedup_by(|a, b| a.resolved_path == b.resolved_path);

    if !results.is_empty() {
        debug!(
            file = %file_path,
            discovered = results.len(),
            "ImportScanner: discovered local file references"
        );
    }

    results
}

// ─── Rust ────────────────────────────────────────────────

fn scan_rust_imports(content: &str, parent: &Path) -> Vec<DiscoveredImport> {
    let mut results = Vec::new();

    // `mod foo;` / `pub mod foo;` / `pub(crate) mod foo;`
    let mod_re = Regex::new(r#"(?m)^\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(\w+)\s*;"#)
        .expect("valid regex literal");
    for cap in mod_re.captures_iter(content) {
        let name = cap[1].to_string();
        // Try `name.rs` first, then `name/mod.rs`
        for candidate in &[
            parent.join(format!("{}.rs", name)),
            parent.join(&name).join("mod.rs"),
        ] {
            if candidate.exists() {
                results.push(DiscoveredImport {
                    resolved_path: candidate.to_string_lossy().to_string(),
                    source: cap[0].to_string(),
                    language: "rust",
                });
                break;
            }
        }
    }

    // `use crate::<path>` → resolve relative to workspace src/
    let use_crate_re =
        Regex::new(r#"(?m)^\s*use\s+crate::(\w+(?:::\w+)*)\s*;"#).expect("valid regex literal");
    for cap in use_crate_re.captures_iter(content) {
        let path_str = cap[1].to_string();
        if let Some(p) = resolve_rust_module_path(&path_str, true) {
            if p.exists() {
                results.push(DiscoveredImport {
                    resolved_path: p.to_string_lossy().to_string(),
                    source: cap[0].to_string(),
                    language: "rust",
                });
            }
        }
    }

    // `use super::<path>` → resolve relative to parent
    let use_super_re =
        Regex::new(r#"(?m)^\s*use\s+(super::)+(\w+(?:::\w+)*)\s*;"#).expect("valid regex literal");
    for cap in use_super_re.captures_iter(content) {
        let supers = &cap[1]; // e.g. "super::" or "super::super::"
        let path_str = &cap[2];
        let levels = supers.matches("super").count() as u32;
        let mut base = parent.to_path_buf();
        for _ in 0..levels {
            base = base.parent().unwrap_or(&base).to_path_buf();
        }
        if let Some(p) = resolve_rust_module_path_rel(&base, path_str) {
            if p.exists() {
                results.push(DiscoveredImport {
                    resolved_path: p.to_string_lossy().to_string(),
                    source: cap[0].to_string(),
                    language: "rust",
                });
            }
        }
    }

    results
}

/// Resolve `foo::bar::baz` to a file path relative to workspace `src/`.
fn resolve_rust_module_path(path_str: &str, from_src: bool) -> Option<PathBuf> {
    let parts: Vec<&str> = path_str.split("::").collect();
    if parts.is_empty() {
        return None;
    }

    // Try to find workspace root by looking for Cargo.toml
    let workspace_root = find_workspace_root()?;
    let src_dir = workspace_root.join("src");

    let mut base = if from_src { src_dir } else { workspace_root };
    for (i, part) in parts.iter().enumerate() {
        if i < parts.len() - 1 {
            base = base.join(part);
        } else {
            // Last component: try `name.rs` and `name/mod.rs`
            let f1 = base.join(format!("{}.rs", part));
            if f1.exists() {
                return Some(f1);
            }
            let f2 = base.join(part).join("mod.rs");
            if f2.exists() {
                return Some(f2);
            }
        }
    }
    None
}

fn resolve_rust_module_path_rel(base: &Path, path_str: &str) -> Option<PathBuf> {
    let parts: Vec<&str> = path_str.split("::").collect();
    if parts.is_empty() {
        return None;
    }
    let mut base = base.to_path_buf();
    for (i, part) in parts.iter().enumerate() {
        if i < parts.len() - 1 {
            base = base.join(part);
        } else {
            let f1 = base.join(format!("{}.rs", part));
            if f1.exists() {
                return Some(f1);
            }
            let f2 = base.join(part).join("mod.rs");
            if f2.exists() {
                return Some(f2);
            }
        }
    }
    None
}

// ─── JavaScript / TypeScript ─────────────────────────────

fn scan_js_imports(content: &str, parent: &Path) -> Vec<DiscoveredImport> {
    let mut results = Vec::new();

    // import ... from '...'  or  import ... from "..."
    let import_re =
        Regex::new(r#"import\s+(?:\{[^}]*\}\s+from\s+|[\w*{},]+\s+from\s+)?['"](\.[^'"]+)['"]"#)
            .expect("valid regex literal");
    for cap in import_re.captures_iter(content) {
        let rel = &cap[1];
        if let Some(p) = resolve_js_path(parent, rel) {
            if p.exists() {
                results.push(DiscoveredImport {
                    resolved_path: p.to_string_lossy().to_string(),
                    source: cap[0].to_string(),
                    language: "typescript",
                });
            }
        }
    }

    // require('...') / require("...")
    let require_re =
        Regex::new(r#"(?m)require\s*\(\s*['"](\.[^'"]+)['"]\s*\)"#).expect("valid regex literal");
    for cap in require_re.captures_iter(content) {
        let rel = &cap[1];
        if let Some(p) = resolve_js_path(parent, rel) {
            if p.exists() {
                results.push(DiscoveredImport {
                    resolved_path: p.to_string_lossy().to_string(),
                    source: cap[0].to_string(),
                    language: "typescript",
                });
            }
        }
    }

    results
}

fn resolve_js_path(parent: &Path, rel: &str) -> Option<PathBuf> {
    let p = parent.join(rel);
    // If it's a directory, look for index files
    if p.is_dir() {
        for name in &[
            "index.ts",
            "index.tsx",
            "index.js",
            "index.jsx",
            "index.mjs",
        ] {
            let index = p.join(name);
            if index.exists() {
                return Some(index);
            }
        }
        return None;
    }
    // Try as-is
    if p.exists() {
        return Some(p);
    }
    // Try with extensions
    for ext in &[".ts", ".tsx", ".js", ".jsx", ".mjs", ".json"] {
        let with_ext = p.with_extension(ext.trim_start_matches('.'));
        if with_ext.exists() {
            return Some(with_ext);
        }
    }
    // Try parent/index files (for bare './routes' where routes is a directory)
    if p.is_dir() {
        for name in &["index.ts", "index.tsx", "index.js", "index.jsx"] {
            let index = p.join(name);
            if index.exists() {
                return Some(index);
            }
        }
    }
    None
}

// ─── Python ──────────────────────────────────────────────

fn scan_python_imports(content: &str, parent: &Path) -> Vec<DiscoveredImport> {
    let mut results = Vec::new();

    // `from .foo import bar` → relative
    let from_rel_re =
        Regex::new(r#"(?m)^\s*from\s+\.(\w+(?:\.\w+)*)\s+import\s+"#).expect("valid regex literal");
    for cap in from_rel_re.captures_iter(content) {
        let module = &cap[1];
        let candidate = parent.join(format!("{}.py", module.replace('.', "/")));
        if candidate.exists() {
            results.push(DiscoveredImport {
                resolved_path: candidate.to_string_lossy().to_string(),
                source: cap[0].to_string(),
                language: "python",
            });
        }
    }

    // `import foo` / `import foo.bar` (local only — check file exists)
    let import_re =
        Regex::new(r#"(?m)^\s*import\s+(\w+(?:\.\w+)*)\s*$"#).expect("valid regex literal");
    for cap in import_re.captures_iter(content) {
        let module = &cap[1];
        let candidate = parent.join(format!("{}.py", module.replace('.', "/")));
        if candidate.exists() {
            results.push(DiscoveredImport {
                resolved_path: candidate.to_string_lossy().to_string(),
                source: cap[0].to_string(),
                language: "python",
            });
        }
    }

    results
}

// ─── Go ──────────────────────────────────────────────────

fn scan_golang_imports(content: &str, parent: &Path) -> Vec<DiscoveredImport> {
    let results = Vec::new();

    // `import "path"` — can't easily distinguish local vs external for Go
    // without module context, so we skip Go for now.
    let _ = (content, parent);

    results
}

// ─── Java ────────────────────────────────────────────────

fn scan_java_imports(content: &str, parent: &Path) -> Vec<DiscoveredImport> {
    let _ = (content, parent);
    vec![]
}

// ─── C / C++ ─────────────────────────────────────────────

fn scan_c_cpp_imports(content: &str, parent: &Path) -> Vec<DiscoveredImport> {
    let mut results = Vec::new();

    // `#include "file.h"` — local include
    let include_local =
        Regex::new(r##"(?m)^\s*#include\s+"([^"]+)"##).expect("valid regex literal");
    for cap in include_local.captures_iter(content) {
        let inc = &cap[1];
        let candidate = parent.join(inc);
        if candidate.exists() {
            results.push(DiscoveredImport {
                resolved_path: candidate.to_string_lossy().to_string(),
                source: cap[0].to_string(),
                language: "c",
            });
        }
    }

    results
}

// ─── Helpers ─────────────────────────────────────────────

/// Find the workspace root by looking for Cargo.toml or package.json
/// walking up from the current directory.
fn find_workspace_root() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let mut current = Some(cwd.as_path());
    while let Some(dir) = current {
        if dir.join("Cargo.toml").exists() || dir.join("package.json").exists() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn tmp_dir() -> PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("import_scanner_test_{}", n));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_rust_mod_resolution() {
        let dir = tmp_dir();
        fs::write(dir.join("foo.rs"), "pub fn hello() {}").unwrap();
        fs::create_dir_all(dir.join("bar")).unwrap();
        fs::write(dir.join("bar").join("mod.rs"), "pub mod nested;").unwrap();
        fs::create_dir_all(dir.join("bar").join("nested")).unwrap();
        fs::write(dir.join("bar").join("nested").join("mod.rs"), "").unwrap();

        let content = r#"
mod foo;
pub mod bar;
"#;
        let results = scan_rust_imports(content, &dir);
        assert_eq!(results.len(), 2, "should find both existing modules");
        assert!(results.iter().any(|r| r.resolved_path.ends_with("foo.rs")));
        assert!(results
            .iter()
            .any(|r| r.resolved_path.ends_with("bar/mod.rs")));
    }

    #[test]
    fn test_rust_use_crate_resolution() {
        // Create src/lib.rs to simulate workspace
        let root = tmp_dir();
        let src = root.join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("lib.rs"), "").unwrap();
        fs::create_dir_all(src.join("utils")).unwrap();
        fs::write(src.join("utils").join("mod.rs"), "").unwrap();
        fs::write(root.join("Cargo.toml"), "[package]\nname = \"test\"\n").unwrap();

        // We need scan_imports to find workspace root via Cargo.toml
        let content = r#"
use crate::utils;
"#;
        // scan from a file inside src/
        let file_path = src.join("main.rs");
        let results = scan_imports(file_path.to_str().unwrap(), content);
        assert!(!results.is_empty(), "should resolve use crate::utils");
        assert!(results[0].resolved_path.ends_with("utils/mod.rs"));
    }

    #[test]
    fn test_rust_use_super_resolution() {
        let dir = tmp_dir();
        // Create parent/ subdir with a sibling module
        fs::create_dir_all(dir.join("parent")).unwrap();
        fs::write(dir.join("parent").join("mod.rs"), "pub mod child;").unwrap();
        fs::create_dir_all(dir.join("parent").join("sibling")).unwrap();
        fs::write(dir.join("parent").join("sibling").join("mod.rs"), "").unwrap();
        // Create child module
        let child = dir.join("parent").join("child");
        fs::create_dir_all(&child).unwrap();
        fs::write(child.join("mod.rs"), "use super::sibling;").unwrap();

        let content = "use super::sibling;";
        let results = scan_rust_imports(content, &child);
        assert!(!results.is_empty(), "should resolve use super::sibling");
        assert!(results[0].resolved_path.ends_with("parent/sibling/mod.rs"));
    }

    #[test]
    fn test_js_import_resolution() {
        let dir = tmp_dir();
        fs::write(dir.join("component.ts"), "export const x = 1;").unwrap();
        fs::write(dir.join("utils.ts"), "export const y = 2;").unwrap();

        let content = r#"
import { Component } from './component';
const utils = require('./utils');
import React from 'react';
"#;
        let results = scan_js_imports(content, &dir);
        assert_eq!(
            results.len(),
            2,
            "should find both local imports, skip react"
        );
        assert!(results
            .iter()
            .any(|r| r.resolved_path.ends_with("component.ts")));
        assert!(results
            .iter()
            .any(|r| r.resolved_path.ends_with("utils.ts")));
    }

    #[test]
    fn test_js_index_resolution() {
        let dir = tmp_dir();
        fs::create_dir_all(dir.join("routes")).unwrap();
        fs::write(dir.join("routes").join("index.ts"), "export const r = 1;").unwrap();

        let content = r#"import { r } from './routes';"#;
        let results = scan_js_imports(content, &dir);
        assert_eq!(results.len(), 1);
        assert!(results[0].resolved_path.ends_with("routes/index.ts"));
    }

    #[test]
    fn test_python_local_import() {
        let dir = tmp_dir();
        fs::write(dir.join("models.py"), "class User: pass").unwrap();
        fs::create_dir_all(dir.join("utils")).unwrap();
        fs::write(dir.join("utils").join("helpers.py"), "def parse(): pass").unwrap();

        let content = r#"
from .models import User
from .utils.helpers import parse
import os
import sys
"#;
        let results = scan_python_imports(content, &dir);
        assert_eq!(
            results.len(),
            2,
            "should find .models and .utils.helpers, skip stdlib"
        );
    }

    #[test]
    fn test_c_include_resolution() {
        let dir = tmp_dir();
        fs::write(dir.join("header.h"), "#define FOO 1").unwrap();

        let content = r##"
#include "header.h"
#include <stdio.h>
"##;
        let results = scan_c_cpp_imports(content, &dir);
        assert_eq!(
            results.len(),
            1,
            "should find local include, skip system include"
        );
        assert!(results[0].resolved_path.ends_with("header.h"));
    }

    #[test]
    fn test_no_results_for_missing_files() {
        let results = scan_imports("src/main.rs", "mod nonexistent;");
        assert!(
            results.is_empty(),
            "should return empty for non-existent files"
        );
    }

    #[test]
    fn test_dedup_same_path() {
        let dir = tmp_dir();
        fs::write(dir.join("foo.rs"), "").unwrap();

        let content = "mod foo;";
        let results1 = scan_rust_imports(content, &dir);
        // Same content scanned twice should produce same results
        let results2 = scan_rust_imports(content, &dir);
        assert_eq!(results1.len(), results2.len());
    }

    #[test]
    fn test_scan_imports_unknown_ext() {
        let results = scan_imports("data.bin", "some content");
        assert!(results.is_empty());
    }

    #[test]
    fn test_scan_imports_python_file() {
        let dir = tmp_dir();
        fs::write(dir.join("helper.py"), "def f(): pass").unwrap();

        let results = scan_imports(
            dir.join("main.py").to_str().unwrap(),
            "from .helper import f",
        );
        assert!(!results.is_empty());
        assert_eq!(results[0].language, "python");
    }
}
