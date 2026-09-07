//! Generic effect and completion contracts used by the orchestration kernel.
//!
//! The kernel deliberately does not classify software-engineering language.
//! Applications map their domain tasks onto these generic policies, while SA
//! and BizAgent use the same protocol for execution, evidence and recovery.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Component, Path, PathBuf};

pub const WORKSPACE_RESOURCE_LEASE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceLeaseAccess {
    Read,
    Write,
    Exclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceLeasePath {
    /// Normalized path relative to `workspace_root`; never absolute and never
    /// contains `.`/`..` components.
    pub relative_path: String,
    pub access: WorkspaceLeaseAccess,
}

/// An executable, fail-closed lease for a same-role child. The workspace root
/// is canonicalized when the parent creates the lease, and each concrete file
/// operation is resolved again immediately before ToolExecutor dispatch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceResourceLease {
    pub schema_version: u32,
    pub lease_id: String,
    pub workspace_root: PathBuf,
    pub paths: Vec<WorkspaceLeasePath>,
}

impl WorkspaceResourceLease {
    pub fn new(
        lease_id: impl Into<String>,
        workspace_root: &Path,
        paths: Vec<(String, WorkspaceLeaseAccess)>,
    ) -> Result<Self, String> {
        let lease_id = lease_id.into();
        if lease_id.trim().is_empty() {
            return Err("workspace lease id must not be empty".to_string());
        }
        let canonical_root = std::fs::canonicalize(workspace_root).map_err(|error| {
            format!(
                "workspace lease root '{}' cannot be canonicalized: {error}",
                workspace_root.display()
            )
        })?;
        if !canonical_root.is_dir() {
            return Err(format!(
                "workspace lease root '{}' is not a directory",
                canonical_root.display()
            ));
        }
        if paths.is_empty() {
            return Err("workspace lease must contain at least one exact path".to_string());
        }

        let mut normalized = Vec::<WorkspaceLeasePath>::with_capacity(paths.len());
        for (path, access) in paths {
            let relative = normalize_declared_workspace_path(&canonical_root, &path)?;
            let relative_path = relative
                .to_str()
                .ok_or_else(|| "workspace lease path must be valid UTF-8".to_string())?
                .replace(std::path::MAIN_SEPARATOR, "/");
            if let Some(existing) = normalized
                .iter_mut()
                .find(|entry| entry.relative_path == relative_path)
            {
                existing.access = strongest_lease_access(existing.access, access);
            } else {
                normalized.push(WorkspaceLeasePath {
                    relative_path,
                    access,
                });
            }
        }
        normalized.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        Ok(Self {
            schema_version: WORKSPACE_RESOURCE_LEASE_SCHEMA_VERSION,
            lease_id,
            workspace_root: canonical_root,
            paths: normalized,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != WORKSPACE_RESOURCE_LEASE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported workspace lease schema {}",
                self.schema_version
            ));
        }
        if self.lease_id.trim().is_empty() || self.paths.is_empty() {
            return Err("workspace lease is incomplete".to_string());
        }
        let current_root = std::fs::canonicalize(&self.workspace_root)
            .map_err(|error| format!("workspace lease root is unavailable: {error}"))?;
        if current_root != self.workspace_root {
            return Err("workspace lease root identity changed".to_string());
        }
        for path in &self.paths {
            normalize_relative_components(Path::new(&path.relative_path))?;
        }
        Ok(())
    }

    pub fn authorizes_file_tool(&self, tool_name: &str, input: &Value) -> Result<(), String> {
        self.resolve_authorized_file_tool_path(tool_name, input)
            .map(|_| ())
    }

    /// Validate one concrete file mutation and return the canonical execution
    /// target. Relative model arguments are rooted at the lease workspace,
    /// rather than at the process working directory, so authorization and the
    /// eventual handler operate on the same file.
    pub fn resolve_authorized_file_tool_path(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> Result<Option<PathBuf>, String> {
        if !matches!(tool_name, "file_write" | "file_edit") {
            return Ok(None);
        }
        self.validate()?;
        let requested = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{tool_name} requires a path for workspace lease validation"))?;
        let normalized = normalize_requested_workspace_path(&self.workspace_root, requested)?;
        let normalized = normalized
            .to_str()
            .ok_or_else(|| "requested workspace path must be valid UTF-8".to_string())?
            .replace(std::path::MAIN_SEPARATOR, "/");
        if self.paths.iter().any(|lease_path| {
            lease_path.relative_path == normalized
                && matches!(
                    lease_path.access,
                    WorkspaceLeaseAccess::Write | WorkspaceLeaseAccess::Exclusive
                )
        }) {
            Ok(Some(self.workspace_root.join(&normalized)))
        } else {
            Err(format!(
                "workspace lease '{}' does not authorize {tool_name} on exact path '{normalized}'",
                self.lease_id
            ))
        }
    }

    pub fn permits_writes(&self) -> bool {
        self.paths.iter().any(|path| {
            matches!(
                path.access,
                WorkspaceLeaseAccess::Write | WorkspaceLeaseAccess::Exclusive
            )
        })
    }

    pub fn conflicts_with(&self, other: &Self) -> bool {
        if self.workspace_root != other.workspace_root {
            return true;
        }
        self.paths.iter().any(|left| {
            other.paths.iter().any(|right| {
                workspace_paths_overlap(&left.relative_path, &right.relative_path)
                    && (left.access != WorkspaceLeaseAccess::Read
                        || right.access != WorkspaceLeaseAccess::Read)
            })
        })
    }
}

fn strongest_lease_access(
    left: WorkspaceLeaseAccess,
    right: WorkspaceLeaseAccess,
) -> WorkspaceLeaseAccess {
    use WorkspaceLeaseAccess::{Exclusive, Read, Write};
    match (left, right) {
        (Exclusive, _) | (_, Exclusive) => Exclusive,
        (Write, _) | (_, Write) => Write,
        (Read, Read) => Read,
    }
}

fn normalize_relative_components(path: &Path) -> Result<PathBuf, String> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err("workspace lease paths must be non-empty and relative".to_string());
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {
                return Err("workspace lease paths must not contain '.'".to_string());
            }
            Component::ParentDir => {
                return Err("workspace lease paths must not contain '..'".to_string());
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err("workspace lease paths must be relative".to_string());
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err("workspace lease path resolves to an empty path".to_string());
    }
    Ok(normalized)
}

fn resolve_candidate_under_root(root: &Path, candidate: &Path) -> Result<PathBuf, String> {
    let mut ancestor = candidate.to_path_buf();
    let mut missing = Vec::new();
    loop {
        match std::fs::symlink_metadata(&ancestor) {
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let name = ancestor.file_name().ok_or_else(|| {
                    format!(
                        "workspace path '{}' has no resolvable ancestor",
                        candidate.display()
                    )
                })?;
                missing.push(name.to_os_string());
                if !ancestor.pop() {
                    return Err(format!(
                        "workspace path '{}' has no resolvable ancestor",
                        candidate.display()
                    ));
                }
            }
            Err(error) => {
                return Err(format!(
                    "workspace path ancestor '{}' cannot be inspected: {error}",
                    ancestor.display()
                ));
            }
        }
    }
    let mut resolved = std::fs::canonicalize(&ancestor).map_err(|error| {
        format!(
            "workspace path ancestor '{}' cannot be canonicalized: {error}",
            ancestor.display()
        )
    })?;
    if !resolved.starts_with(root) {
        return Err(format!(
            "workspace path '{}' escapes lease root '{}'",
            candidate.display(),
            root.display()
        ));
    }
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    if !resolved.starts_with(root) {
        return Err(format!(
            "workspace path '{}' escapes lease root '{}'",
            candidate.display(),
            root.display()
        ));
    }
    Ok(resolved)
}

fn normalize_declared_workspace_path(root: &Path, path: &str) -> Result<PathBuf, String> {
    if path.contains(['*', '?', '[', ']', '{', '}']) || path.ends_with('/') || path.ends_with('\\')
    {
        return Err(format!(
            "workspace lease path '{path}' is not an exact file path"
        ));
    }
    let relative = normalize_relative_components(Path::new(path))?;
    let resolved = resolve_candidate_under_root(root, &root.join(&relative))?;
    resolved
        .strip_prefix(root)
        .map(Path::to_path_buf)
        .map_err(|_| format!("workspace lease path '{path}' escapes its root"))
}

fn normalize_requested_workspace_path(root: &Path, path: &str) -> Result<PathBuf, String> {
    let requested = Path::new(path);
    let candidate = if requested.is_absolute() {
        // Reject lexical traversal before resolving symlinks. Absolute paths
        // may still be used by callers, but only below the exact lease root.
        if requested
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
        {
            return Err(format!(
                "requested workspace path '{path}' is not normalized"
            ));
        }
        requested.to_path_buf()
    } else {
        root.join(normalize_relative_components(requested)?)
    };
    let resolved = resolve_candidate_under_root(root, &candidate)?;
    resolved
        .strip_prefix(root)
        .map(Path::to_path_buf)
        .map_err(|_| format!("requested workspace path '{path}' escapes its lease root"))
}

fn workspace_paths_overlap(left: &str, right: &str) -> bool {
    let left = Path::new(left);
    let right = Path::new(right);
    left == right || left.starts_with(right) || right.starts_with(left)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EffectKind {
    WorkspaceMutation,
    ExternalSideEffect,
    StateChange,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum EffectPolicy {
    None,
    Required {
        effect: EffectKind,
    },
    Conditional {
        effect: EffectKind,
        /// Older checkpoints and LLM-produced residual plans may omit the
        /// explanatory text.  The typed effect remains usable; orchestration
        /// supplies a generic revalidation condition at dispatch time.
        #[serde(default)]
        condition: String,
    },
    EvidenceOnly,
    DecisionOnly,
}

impl Default for EffectPolicy {
    fn default() -> Self {
        Self::None
    }
}

impl EffectPolicy {
    pub fn required_workspace_mutation() -> Self {
        Self::Required {
            effect: EffectKind::WorkspaceMutation,
        }
    }

    pub fn conditional_workspace_mutation(condition: impl Into<String>) -> Self {
        Self::Conditional {
            effect: EffectKind::WorkspaceMutation,
            condition: condition.into(),
        }
    }

    pub fn requires_workspace_mutation(&self) -> bool {
        matches!(
            self,
            Self::Required {
                effect: EffectKind::WorkspaceMutation
            }
        )
    }

    pub fn may_require_workspace_mutation(&self) -> bool {
        matches!(
            self,
            Self::Required {
                effect: EffectKind::WorkspaceMutation
            } | Self::Conditional {
                effect: EffectKind::WorkspaceMutation,
                ..
            }
        )
    }

    pub fn permits_mutation(&self) -> bool {
        !matches!(self, Self::EvidenceOnly | Self::DecisionOnly)
    }

    /// Compatibility bridge for checkpoints and external callers that still
    /// supply the former string constraint. New code should carry the typed
    /// policy directly in `TaskContext`/`PlanStep`.
    pub fn from_legacy_constraints(
        constraints: &std::collections::HashMap<String, String>,
    ) -> Self {
        match constraints.get("effect_policy").map(String::as_str) {
            Some("evidence_only") => Self::EvidenceOnly,
            Some("decision_only") => Self::DecisionOnly,
            Some("conditional_workspace_mutation") => {
                Self::conditional_workspace_mutation("condition declared by application")
            }
            Some("required_workspace_mutation") => Self::required_workspace_mutation(),
            _ if constraints
                .get("required_effect")
                .is_some_and(|value| value == "workspace_mutation") =>
            {
                Self::required_workspace_mutation()
            }
            _ => Self::None,
        }
    }
}

/// Conservative syntactic classification used at both the AgentRunner and
/// ToolExecutor boundaries. It is intentionally centralized so a hook cannot
/// mutate arguments between two different policy implementations.
pub fn is_substantive_workspace_effect(name: &str, args: &Value) -> bool {
    match name {
        "file_write" | "file_edit" => true,
        "bash" | "powershell" | "code_execute" => {
            let raw_command = args
                .get("command")
                .or_else(|| args.get("code"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let command = raw_command.to_lowercase();
            if command.is_empty() {
                return false;
            }
            let mutating_patterns = [
                "sed -i",
                "perl -pi",
                "git apply",
                "patch ",
                "mv ",
                "install ",
                "curl -o",
                "curl --output",
                "wget -o",
                "unzip ",
                "tar -x",
                "npm create",
                "npx create-",
                "cargo new",
                "cargo add",
                "django-admin startproject",
                "rails new",
                "write_text(",
                "write_bytes(",
            ];
            let python_execution = executes_python(name, args, raw_command);
            let python_open_effect = classify_python_open_effect(raw_command);
            let open_is_mutating = match python_open_effect {
                PythonOpenEffect::None => {
                    // Preserve the old fail-closed behavior for an occurrence
                    // which is not a Python `open` token (for example a
                    // language-specific `reopen(` API).
                    command.contains("open(")
                }
                PythonOpenEffect::ReadOnly => !python_execution,
                PythonOpenEffect::MutationOrUnknown => true,
            };
            mutating_patterns
                .iter()
                .any(|pattern| command.contains(pattern))
                || open_is_mutating
                || has_substantive_copy_effect(&command)
                || (matches!(name, "bash" | "powershell")
                    && has_substantive_output_redirection(&command))
        }
        _ => false,
    }
}

/// Broader pre-execution classifier: directory creation/deletion is a
/// mutation even though it does not count as substantive completion evidence.
pub fn is_workspace_mutation_candidate(name: &str, args: &Value) -> bool {
    if is_substantive_workspace_effect(name, args) {
        return true;
    }
    if !matches!(name, "bash" | "powershell" | "code_execute") {
        return false;
    }
    let command = args
        .get("command")
        .or_else(|| args.get("code"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_lowercase();
    if matches!(name, "bash" | "powershell") && has_workspace_output_redirection(&command) {
        return true;
    }
    [
        "mkdir ",
        "touch ",
        "rm ",
        "rmdir ",
        "del ",
        "remove-item",
        "new-item",
        "set-content",
        "add-content",
    ]
    .iter()
    .any(|pattern| command.contains(pattern))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PythonOpenEffect {
    None,
    ReadOnly,
    MutationOrUnknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PythonOpenSignature {
    /// `open(path, mode)` / `builtins.open(path, mode)` /
    /// `io.open(path, mode)`.
    FileThenMode,
    /// The type-identifiable `Path(path).open(mode)` form.
    ModeFirst,
    /// An arbitrary object's `.open()` method cannot safely inherit pathlib
    /// semantics: it may mutate even with no arguments.
    UnknownMethod,
}

/// Decide whether this tool invocation actually provides Python source.  A
/// read-only interpretation of `open()` is only safe inside Python; calls in
/// other languages retain the historical fail-closed classification.
fn executes_python(name: &str, args: &Value, command: &str) -> bool {
    if name == "code_execute" {
        return args
            .get("language")
            .and_then(Value::as_str)
            .map(|language| {
                matches!(
                    language.trim().to_ascii_lowercase().as_str(),
                    "python" | "python3" | "py"
                )
            })
            // `code_execute` defaults to Python when language is omitted.
            .unwrap_or(true);
    }
    if !matches!(name, "bash" | "powershell") {
        return false;
    }

    command
        .split(|character: char| {
            character.is_whitespace() || matches!(character, ';' | '&' | '|' | '(' | ')')
        })
        .map(|token| token.trim_matches(|character| matches!(character, '\'' | '"')))
        .filter(|token| !token.is_empty())
        .any(|token| {
            let file_name = token.rsplit(['/', '\\']).next().unwrap_or(token);
            let normalized_file_name = file_name.to_ascii_lowercase();
            let executable = normalized_file_name
                .strip_suffix(".exe")
                .unwrap_or(&normalized_file_name);
            executable == "py"
                || executable == "python"
                || executable == "pypy"
                || executable.strip_prefix("python").is_some_and(|suffix| {
                    !suffix.is_empty()
                        && suffix
                            .chars()
                            .all(|character| character.is_ascii_digit() || character == '.')
                })
                || executable.strip_prefix("pypy").is_some_and(|suffix| {
                    !suffix.is_empty()
                        && suffix
                            .chars()
                            .all(|character| character.is_ascii_digit() || character == '.')
                })
        })
}

/// Parse Python `open` calls just far enough to classify their mode. This is
/// deliberately not a general Python parser: calls with dynamic argument
/// expansion, non-literal modes, conflicting modes, or malformed delimiters
/// are classified as mutations. That fail-closed rule is what makes it safe
/// to exempt the common `open(path).read()` verification idiom.
fn classify_python_open_effect(source: &str) -> PythonOpenEffect {
    let bytes = source.as_bytes();
    let mut search_from = 0usize;
    let mut found_read = false;

    while search_from < bytes.len() {
        let Some(relative_start) = source[search_from..].find("open") else {
            break;
        };
        let start = search_from + relative_start;
        let after_name = start + "open".len();
        search_from = after_name;

        if start
            .checked_sub(1)
            .and_then(|index| bytes.get(index))
            .is_some_and(|byte| is_python_identifier_byte(*byte))
            || bytes
                .get(after_name)
                .is_some_and(|byte| is_python_identifier_byte(*byte))
        {
            continue;
        }

        let mut open_paren = after_name;
        while bytes.get(open_paren).is_some_and(u8::is_ascii_whitespace) {
            open_paren += 1;
        }
        if bytes.get(open_paren) != Some(&b'(') {
            continue;
        }

        let signature = python_open_signature(source, start);
        let Some((arguments, _call_end)) = parse_python_call_arguments(source, open_paren) else {
            return PythonOpenEffect::MutationOrUnknown;
        };
        match classify_python_open_arguments(&arguments, signature) {
            PythonOpenEffect::MutationOrUnknown => {
                return PythonOpenEffect::MutationOrUnknown;
            }
            PythonOpenEffect::ReadOnly => found_read = true,
            PythonOpenEffect::None => {}
        }
    }

    if found_read {
        PythonOpenEffect::ReadOnly
    } else {
        PythonOpenEffect::None
    }
}

fn is_python_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || !byte.is_ascii()
}

fn python_open_signature(source: &str, open_start: usize) -> PythonOpenSignature {
    let bytes = source.as_bytes();
    let mut cursor = open_start;
    while cursor > 0 && bytes[cursor - 1].is_ascii_whitespace() {
        cursor -= 1;
    }
    if cursor == 0 || bytes[cursor - 1] != b'.' {
        return PythonOpenSignature::FileThenMode;
    }

    cursor -= 1;
    while cursor > 0 && bytes[cursor - 1].is_ascii_whitespace() {
        cursor -= 1;
    }
    let receiver_end = cursor;
    while cursor > 0 && is_python_identifier_byte(bytes[cursor - 1]) {
        cursor -= 1;
    }
    let receiver = &source[cursor..receiver_end];
    if matches!(receiver, "io" | "builtins") {
        PythonOpenSignature::FileThenMode
    } else if is_path_constructor_receiver(source, receiver_end) {
        PythonOpenSignature::ModeFirst
    } else {
        PythonOpenSignature::UnknownMethod
    }
}

fn is_path_constructor_receiver(source: &str, receiver_end: usize) -> bool {
    let bytes = source.as_bytes();
    let mut search_from = 0usize;
    while search_from < receiver_end {
        let Some(relative_start) = source[search_from..receiver_end].find("Path") else {
            break;
        };
        let start = search_from + relative_start;
        let after_name = start + "Path".len();
        search_from = after_name;
        if start
            .checked_sub(1)
            .and_then(|index| bytes.get(index))
            .is_some_and(|byte| is_python_identifier_byte(*byte))
            || bytes
                .get(after_name)
                .is_some_and(|byte| is_python_identifier_byte(*byte))
        {
            continue;
        }

        let mut open_paren = after_name;
        while bytes.get(open_paren).is_some_and(u8::is_ascii_whitespace) {
            open_paren += 1;
        }
        if bytes.get(open_paren) != Some(&b'(') {
            continue;
        }
        let Some((_arguments, call_end)) = parse_python_call_arguments(source, open_paren) else {
            continue;
        };
        let mut normalized_end = call_end;
        while normalized_end < receiver_end
            && bytes
                .get(normalized_end)
                .is_some_and(u8::is_ascii_whitespace)
        {
            normalized_end += 1;
        }
        if normalized_end == receiver_end {
            return true;
        }
    }
    false
}

/// Return the top-level arguments and the byte immediately following the
/// closing parenthesis. Quotes and nested calls/containers are honored, so a
/// comma in a file name or nested expression cannot move the mode position.
fn parse_python_call_arguments(source: &str, open_paren: usize) -> Option<(Vec<&str>, usize)> {
    #[derive(Clone, Copy)]
    struct Quote {
        delimiter: u8,
        shell_escaped: bool,
    }

    let bytes = source.as_bytes();
    let mut delimiters = vec![b')'];
    let mut quote: Option<Quote> = None;
    let mut comment = false;
    let mut argument_start = open_paren + 1;
    let mut arguments = Vec::new();
    let mut cursor = argument_start;

    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if comment {
            if byte == b'\n' {
                comment = false;
            }
            cursor += 1;
            continue;
        }
        if let Some(active_quote) = quote {
            if active_quote.shell_escaped
                && byte == b'\\'
                && bytes.get(cursor + 1) == Some(&active_quote.delimiter)
            {
                quote = None;
                cursor += 2;
                continue;
            }
            if !active_quote.shell_escaped && byte == active_quote.delimiter {
                quote = None;
                cursor += 1;
                continue;
            }
            if byte == b'\\' {
                cursor = cursor.saturating_add(2);
            } else {
                cursor += 1;
            }
            continue;
        }

        if byte == b'\\'
            && bytes
                .get(cursor + 1)
                .is_some_and(|next| matches!(next, b'\'' | b'"'))
        {
            quote = Some(Quote {
                delimiter: bytes[cursor + 1],
                shell_escaped: true,
            });
            cursor += 2;
            continue;
        }
        if matches!(byte, b'\'' | b'"') {
            quote = Some(Quote {
                delimiter: byte,
                shell_escaped: false,
            });
            cursor += 1;
            continue;
        }
        if byte == b'#' {
            comment = true;
            cursor += 1;
            continue;
        }

        match byte {
            b'(' => delimiters.push(b')'),
            b'[' => delimiters.push(b']'),
            b'{' => delimiters.push(b'}'),
            b')' | b']' | b'}' => {
                if delimiters.pop() != Some(byte) {
                    return None;
                }
                if delimiters.is_empty() {
                    let argument = source[argument_start..cursor].trim();
                    if !argument.is_empty() {
                        arguments.push(argument);
                    }
                    return Some((arguments, cursor + 1));
                }
            }
            b',' if delimiters.len() == 1 => {
                let argument = source[argument_start..cursor].trim();
                if !argument.is_empty() {
                    arguments.push(argument);
                }
                argument_start = cursor + 1;
            }
            _ => {}
        }
        cursor += 1;
    }
    None
}

fn classify_python_open_arguments(
    arguments: &[&str],
    signature: PythonOpenSignature,
) -> PythonOpenEffect {
    let mode_index = match signature {
        PythonOpenSignature::FileThenMode => 1,
        PythonOpenSignature::ModeFirst => 0,
        PythonOpenSignature::UnknownMethod => {
            return PythonOpenEffect::MutationOrUnknown;
        }
    };
    let mut positional = Vec::new();
    let mut keyword_mode = None;
    let mut saw_keyword = false;

    for argument in arguments {
        let argument = argument.trim();
        if argument.starts_with('*') {
            // `*args` or `**kwargs` may supply a write mode at runtime.
            return PythonOpenEffect::MutationOrUnknown;
        }
        if let Some((keyword, value)) = python_keyword_argument(argument) {
            saw_keyword = true;
            if keyword == "mode" {
                if keyword_mode.replace(value).is_some() {
                    return PythonOpenEffect::MutationOrUnknown;
                }
            } else if keyword == "opener" && value != "None" {
                // A custom builtins.open opener executes application code and
                // may ignore the nominal read-only flags.
                return PythonOpenEffect::MutationOrUnknown;
            }
        } else {
            if saw_keyword {
                // Positional arguments after keyword arguments are malformed;
                // keep the pre-execution policy fail-closed.
                return PythonOpenEffect::MutationOrUnknown;
            }
            positional.push(argument);
        }
    }

    if let Some(mode) = keyword_mode {
        if positional.len() > mode_index {
            // A positional and keyword mode conflict. Python will reject it,
            // but policy parsing must not guess which value was intended.
            return PythonOpenEffect::MutationOrUnknown;
        }
        return classify_python_open_mode(mode);
    }
    positional
        .get(mode_index)
        .map_or(PythonOpenEffect::ReadOnly, |mode| {
            classify_python_open_mode(mode)
        })
}

fn python_keyword_argument(argument: &str) -> Option<(&str, &str)> {
    let identifier_end = argument
        .char_indices()
        .take_while(|(_, character)| character.is_ascii_alphanumeric() || *character == '_')
        .map(|(index, character)| index + character.len_utf8())
        .last()?;
    let keyword = &argument[..identifier_end];
    let remainder = argument[identifier_end..].trim_start();
    let value = remainder.strip_prefix('=')?;
    if value.starts_with('=') {
        return None;
    }
    Some((keyword, value.trim()))
}

fn classify_python_open_mode(expression: &str) -> PythonOpenEffect {
    let Some(mode) = parse_python_string_literal(expression) else {
        return PythonOpenEffect::MutationOrUnknown;
    };
    let mode = mode.to_ascii_lowercase();
    if !mode.is_empty()
        && mode.contains('r')
        && mode
            .chars()
            .all(|character| matches!(character, 'r' | 'b' | 't'))
    {
        PythonOpenEffect::ReadOnly
    } else {
        // Includes all valid write modes (`w`, `a`, `x`, and `+`) plus
        // invalid/unknown literals. Invalid input is deliberately fail-closed.
        PythonOpenEffect::MutationOrUnknown
    }
}

/// Parse a simple Python string literal used as an open mode. Prefixes which
/// can interpolate (`f`) and escaped contents remain unknown. `\"r\"` is
/// accepted because it is the representation produced inside a double-quoted
/// shell `python -c` payload.
fn parse_python_string_literal(expression: &str) -> Option<String> {
    let expression = expression.trim();
    let bytes = expression.as_bytes();
    let mut cursor = 0usize;
    while bytes
        .get(cursor)
        .is_some_and(|byte| byte.is_ascii_alphabetic())
    {
        cursor += 1;
    }
    let prefix = expression[..cursor].to_ascii_lowercase();
    if prefix
        .chars()
        .any(|character| !matches!(character, 'r' | 'u' | 'b'))
    {
        return None;
    }

    let (delimiter, shell_escaped, content_start) = match bytes.get(cursor) {
        Some(delimiter @ (b'\'' | b'"')) => (*delimiter, false, cursor + 1),
        Some(b'\\') if matches!(bytes.get(cursor + 1), Some(b'\'' | b'"')) => {
            (bytes[cursor + 1], true, cursor + 2)
        }
        _ => return None,
    };
    let mut end = content_start;
    while end < bytes.len() {
        if shell_escaped && bytes[end] == b'\\' && bytes.get(end + 1) == Some(&delimiter) {
            let trailing = expression[end + 2..].trim();
            return (trailing.is_empty() && !expression[content_start..end].contains('\\'))
                .then(|| expression[content_start..end].to_string());
        }
        if !shell_escaped && bytes[end] == delimiter {
            let trailing = expression[end + 1..].trim();
            return (trailing.is_empty() && !expression[content_start..end].contains('\\'))
                .then(|| expression[content_start..end].to_string());
        }
        end += 1;
    }
    None
}

fn has_substantive_copy_effect(command: &str) -> bool {
    let tokens = command.split_whitespace().collect::<Vec<_>>();
    let mut index = 0usize;
    while index < tokens.len() {
        let token = tokens[index]
            .trim_matches(|ch: char| matches!(ch, '\'' | '"' | '(' | ')' | ';' | '&' | '|'));
        if token.rsplit('/').next() != Some("cp") {
            index += 1;
            continue;
        }

        let mut operands = Vec::new();
        index += 1;
        while index < tokens.len() {
            let raw = tokens[index];
            if matches!(raw, "&&" | "||" | "|" | ";") {
                break;
            }
            let operand =
                raw.trim_matches(|ch: char| matches!(ch, '\'' | '"' | '(' | ')' | ';' | '&' | '|'));
            if !operand.is_empty() && !operand.starts_with('-') {
                operands.push(operand);
            }
            index += 1;
        }

        let Some(destination) = operands.last() else {
            continue;
        };
        let backup_only = destination.ends_with(".bak")
            || destination.ends_with(".backup")
            || destination.ends_with(".orig")
            || destination.ends_with('~');
        if !backup_only {
            return true;
        }
    }
    false
}

/// Detect shell output redirections which can create or replace project
/// content. This is deliberately only a pre-execution classifier: the runner
/// still requires a successful tool result and a changed semantic workspace
/// fingerprint before recording an effect.
///
/// A small quote-aware scanner is used instead of looking for the raw `>`
/// byte. That avoids treating comparisons or quoted documentation as writes,
/// handles both `>` and `>>`, and recognizes the heredoc form commonly used
/// by agents (`cat > docs/design.md <<'EOF'`). Descriptor duplication and
/// well-known transient/test outputs never qualify as substantive work.
fn has_substantive_output_redirection(command: &str) -> bool {
    has_output_redirection(command, false)
}

/// Detect non-transient shell output redirections for the pre-execution
/// mutation guard. Verification output is deliberately included here: a test
/// command writing `report.txt` still mutates the workspace even though that
/// output does not count as a substantive deliverable.
fn has_workspace_output_redirection(command: &str) -> bool {
    has_output_redirection(command, true)
}

fn has_output_redirection(command: &str, include_verification_output: bool) -> bool {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Quote {
        None,
        Single,
        Double,
    }

    let bytes = command.as_bytes();
    let mut quote = Quote::None;
    let mut segment_start = 0usize;
    let mut index = 0usize;

    while index < bytes.len() {
        match (quote, bytes[index]) {
            (Quote::Single, b'\'') => quote = Quote::None,
            (Quote::Double, b'"') => quote = Quote::None,
            (Quote::None, b'\'') => quote = Quote::Single,
            (Quote::None, b'"') => quote = Quote::Double,
            (Quote::None, b'\\') => {
                // The escaped byte cannot be a shell operator.
                index = index.saturating_add(1);
            }
            (Quote::None, b';' | b'\n' | b'|') => {
                segment_start = index.saturating_add(1);
                if bytes[index] == b'|' && bytes.get(index + 1) == Some(&b'|') {
                    index = index.saturating_add(1);
                    segment_start = index.saturating_add(1);
                }
            }
            (Quote::None, b'&') => {
                // `&>` redirects both streams; leave `>` for the next
                // iteration. Other ampersands terminate the command segment.
                if bytes.get(index + 1) != Some(&b'>') {
                    if bytes.get(index + 1) == Some(&b'&') {
                        index = index.saturating_add(1);
                    }
                    segment_start = index.saturating_add(1);
                }
            }
            (Quote::None, b'>') => {
                let operator_index = index;
                index = index.saturating_add(1);
                if bytes.get(index) == Some(&b'>') {
                    index = index.saturating_add(1);
                }

                let descriptor_duplication = bytes.get(index) == Some(&b'&');
                if descriptor_duplication {
                    index = index.saturating_add(1);
                }
                let Some((target, end)) = parse_shell_word(command, index, descriptor_duplication)
                else {
                    continue;
                };
                index = end.saturating_sub(1);

                // `2>&1` and `2>&-` only connect/close descriptors. Bash also
                // accepts `>&file`, which is a real file write and must pass.
                if descriptor_duplication
                    && (target == "-" || target.chars().all(|ch| ch.is_ascii_digit()))
                {
                    index = index.saturating_add(1);
                    continue;
                }

                let segment = &command[segment_start.min(operator_index)..operator_index];
                if (include_verification_output || !is_verification_output_segment(segment))
                    && is_substantive_redirection_target(&target)
                {
                    return true;
                }
            }
            _ => {}
        }
        index = index.saturating_add(1);
    }
    false
}

/// Parse one shell word beginning at (or after) `start`, stripping quotes but
/// preserving escaped bytes. The returned index is an ASCII operator or
/// whitespace boundary; descriptor duplication additionally stops at an
/// unquoted command/group closing delimiter.
fn parse_shell_word(
    command: &str,
    start: usize,
    descriptor_duplication: bool,
) -> Option<(String, usize)> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Quote {
        None,
        Single,
        Double,
    }

    let bytes = command.as_bytes();
    let mut index = start;
    while bytes.get(index).is_some_and(u8::is_ascii_whitespace) {
        index += 1;
    }
    if bytes.get(index) == Some(&b'(') {
        // Process substitution (`>(...)`) is not a workspace path.
        return None;
    }

    let mut quote = Quote::None;
    let mut word = Vec::new();
    while index < bytes.len() {
        let byte = bytes[index];
        match (quote, byte) {
            (Quote::Single, b'\'') => quote = Quote::None,
            (Quote::Double, b'"') => quote = Quote::None,
            (Quote::None, b'\'') => quote = Quote::Single,
            (Quote::None, b'"') => quote = Quote::Double,
            (Quote::None, b'\\') => {
                if let Some(escaped) = bytes.get(index + 1) {
                    word.push(*escaped);
                    index += 1;
                }
            }
            // A descriptor duplication at the end of command substitution or
            // a shell group ends before the unquoted closing delimiter. Keep
            // ordinary redirection targets unchanged: `> cache.tmp}` is a
            // filename and must not inherit the transient `.tmp` exemption.
            (Quote::None, b')' | b'}') if descriptor_duplication => break,
            (Quote::None, b';' | b'&' | b'|' | b'<' | b'>') => break,
            (Quote::None, byte) if byte.is_ascii_whitespace() => break,
            (_, byte) => word.push(byte),
        }
        index += 1;
    }
    (!word.is_empty()).then(|| (String::from_utf8_lossy(&word).into_owned(), index))
}

fn is_verification_output_segment(segment: &str) -> bool {
    let tokens = segment
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch: char| matches!(ch, '\'' | '"' | '(' | ')' | ';' | '&' | '|'))
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or("")
        })
        .collect::<Vec<_>>();

    let mut program_index = 0usize;
    while let Some(token) = tokens.get(program_index) {
        if token.contains('=') || matches!(*token, "env" | "command" | "nohup" | "time") {
            program_index += 1;
        } else {
            break;
        }
    }
    let Some(program) = tokens.get(program_index).copied() else {
        return false;
    };
    let arguments = &tokens[program_index.saturating_add(1)..];

    matches!(program, "pytest" | "py.test" | "ctest")
        || (program.starts_with("python")
            && arguments
                .windows(2)
                .any(|pair| matches!(pair, ["-m", "pytest"] | ["-m", "unittest"])))
        || matches!(
            program,
            "cargo" | "go" | "dotnet" | "mvn" | "gradle" | "make"
        ) && arguments.first() == Some(&"test")
        || program == "cargo" && arguments.first() == Some(&"nextest")
        || matches!(program, "npm" | "pnpm" | "yarn" | "bun")
            && (arguments.first() == Some(&"test") || matches!(arguments, ["run", "test", ..]))
}

fn is_substantive_redirection_target(target: &str) -> bool {
    let normalized = target.trim().replace('\\', "/").to_lowercase();
    if normalized.is_empty()
        || normalized == "-"
        || normalized == "nul"
        || normalized.starts_with("/dev/")
        || normalized.starts_with("/proc/self/fd/")
        || normalized == "$null"
    {
        return false;
    }
    !is_transient_workspace_effect_path(Path::new(&normalized))
}

/// Classify well-known transient redirection targets. This is intentionally
/// narrow and applies only to the shell prefilter: configurable workspace
/// exclusions remain authoritative, and generic `build/`, `dist/` or `.log`
/// deliverables are not suppressed merely because of their name.
fn is_transient_workspace_effect_path(path: &Path) -> bool {
    let normalized = path.to_string_lossy().replace('\\', "/").to_lowercase();
    let components = normalized.split('/').collect::<Vec<_>>();
    let transient_directories = [
        ".git",
        ".gliding_horse",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "__pycache__",
    ];
    if components
        .iter()
        .any(|component| transient_directories.contains(component))
    {
        return true;
    }

    let file_name = components.last().copied().unwrap_or("");
    let transient_suffixes = [
        ".tmp", ".temp", ".bak", ".backup", ".orig", ".pyc", ".pyo", ".swp", ".swo",
    ];
    if file_name.ends_with('~')
        || transient_suffixes
            .iter()
            .any(|suffix| file_name.ends_with(suffix))
    {
        return true;
    }

    let verification_artifact = file_name == ".coverage"
        || file_name.starts_with(".coverage.")
        || matches!(file_name, "coverage.xml" | "lcov.info" | "junit.xml")
        || [
            "pytest_run",
            "pytest-run",
            "pytest_output",
            "pytest-output",
            "test_results",
            "test-results",
            "test_output",
            "test-output",
        ]
        .iter()
        .any(|prefix| file_name.starts_with(prefix));
    verification_artifact
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompletionState {
    Complete,
    Incomplete,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingEffect {
    pub objective: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub effect_policy: EffectPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompletionEnvelope {
    pub completion_state: CompletionState,
    #[serde(default)]
    pub changes: Vec<String>,
    #[serde(default)]
    pub verification: Vec<String>,
    #[serde(default)]
    pub pending_effects: Vec<PendingEffect>,
    #[serde(default)]
    pub blockers: Vec<String>,
    /// True only when the model supplied the structured protocol. SA can use
    /// this to distinguish authoritative empty pending work from a legacy
    /// prose response.
    #[serde(skip)]
    pub structured: bool,
}

impl CompletionEnvelope {
    pub fn complete() -> Self {
        Self {
            completion_state: CompletionState::Complete,
            changes: Vec::new(),
            verification: Vec::new(),
            pending_effects: Vec::new(),
            blockers: Vec::new(),
            structured: false,
        }
    }

    pub fn from_result(status: &str, output: Option<&Value>, summary: &str) -> Self {
        if let Some(parsed) = output.and_then(Self::parse_value) {
            return parsed;
        }

        // Legacy fallback is intentionally conservative but deterministic:
        // successful prose without an explicit residual-work marker proceeds
        // to CA, while partial/failed work receives one bounded decomposition.
        let normalized = summary.to_lowercase();
        let blocked = status == "failed"
            || normalized.starts_with("failed:")
            || normalized.contains("blocked:");
        let residual_markers = [
            "remaining:",
            "pending:",
            "still needs",
            "not completed",
            "incomplete",
            "尚未完成",
            "仍需",
            "待完成",
            "剩余",
        ];
        let incomplete = status == "partial_success"
            || residual_markers
                .iter()
                .any(|marker| normalized.contains(marker));
        let completion_state = if blocked {
            CompletionState::Blocked
        } else if incomplete {
            CompletionState::Incomplete
        } else {
            CompletionState::Complete
        };
        Self {
            completion_state,
            changes: Vec::new(),
            verification: Vec::new(),
            pending_effects: if incomplete {
                vec![PendingEffect {
                    objective: summary.chars().take(1_000).collect(),
                    target: None,
                    reason: "legacy result reported residual work".to_string(),
                    effect_policy: EffectPolicy::None,
                }]
            } else {
                Vec::new()
            },
            blockers: if blocked {
                vec![summary.chars().take(1_000).collect()]
            } else {
                Vec::new()
            },
            structured: false,
        }
    }

    fn parse_value(value: &Value) -> Option<Self> {
        let candidate = match value {
            Value::Object(map) => map
                .get("completion")
                .or_else(|| map.get("completion_envelope"))
                .unwrap_or(value),
            Value::String(text) => {
                let parsed: Value = serde_json::from_str(text).ok().or_else(|| {
                    let start = text.find('{')?;
                    let end = text.rfind('}')?;
                    serde_json::from_str(&text[start..=end]).ok()
                })?;
                return Self::parse_value(&parsed);
            }
            _ => return None,
        };
        let mut envelope: Self = serde_json::from_value(candidate.clone()).ok()?;
        envelope.structured = true;
        Some(envelope)
    }

    pub fn needs_follow_up_execution(&self) -> bool {
        self.completion_state == CompletionState::Incomplete && !self.pending_effects.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_open_default_and_explicit_read_modes_are_not_mutations() {
        let cases = [
            (
                "bash",
                serde_json::json!({
                    "command": r#"python3 -B -c "open('design.md').read()""#
                }),
            ),
            (
                "bash",
                serde_json::json!({
                    "command": r#"python3 -c 'open("README.md", "r", encoding="utf-8").read()'"#
                }),
            ),
            (
                "bash",
                serde_json::json!({
                    "command": r#"python3 -c "open(\"README.md\", mode=\"rb\").read()""#
                }),
            ),
            (
                "powershell",
                serde_json::json!({
                    "command": r#"python -c "from pathlib import Path; Path('design.md').open().read()""#
                }),
            ),
            (
                "bash",
                serde_json::json!({
                    "command": r#"python3 -c "from pathlib import Path; Path('design.md').open(mode='rt').read()""#
                }),
            ),
            (
                "bash",
                serde_json::json!({
                    "command": r#"python3 -c "import io; io.open('design.md').read()""#
                }),
            ),
            (
                "code_execute",
                serde_json::json!({
                    "language": "python",
                    "code": "with open('calculator.py', 'br') as source:\n    print(source.read(1))"
                }),
            ),
        ];

        for (name, arguments) in cases {
            assert!(
                !is_substantive_workspace_effect(name, &arguments),
                "read-only open was classified as a substantive effect: {arguments}"
            );
            assert!(
                !is_workspace_mutation_candidate(name, &arguments),
                "read-only open was classified as a mutation candidate: {arguments}"
            );
        }
    }

    #[test]
    fn python_open_write_and_unknown_modes_remain_fail_closed() {
        let commands = [
            r#"python3 -c "open('report.md', 'w').write('report')""#,
            r#"python3 -c "import builtins; builtins.open('report.md', mode='wb')""#,
            r#"python3 -c "import io; io.open('report.md', 'x')""#,
            r#"python3 -c "from pathlib import Path; Path('report.md').open('a')""#,
            r#"python3 -c "path.open(mode='r+')""#,
            r#"python3 -c "open(path, mode)""#,
            r#"python3 -c "open(path, **options)""#,
            r#"python3 -c "open(path, opener=custom_opener)""#,
            r#"python3 -c "open(open('report.md', 'w'))""#,
            r#"python3 -c "database.open()""#,
            r#"python3 -c "open(path,""#,
        ];

        for command in commands {
            let arguments = serde_json::json!({"command": command});
            assert!(
                is_substantive_workspace_effect("bash", &arguments),
                "write or unknown open mode escaped classification: {command}"
            );
            assert!(
                is_workspace_mutation_candidate("bash", &arguments),
                "write or unknown open mode escaped the pre-execution guard: {command}"
            );
        }
    }

    #[test]
    fn open_relaxation_is_python_only_and_preserves_other_write_detectors() {
        let non_python = serde_json::json!({
            "command": r#"ruby -e 'open("report.md")'"#
        });
        assert!(is_workspace_mutation_candidate("bash", &non_python));
        let non_python_code = serde_json::json!({
            "language": "javascript",
            "code": "open('report.md')"
        });
        assert!(is_workspace_mutation_candidate(
            "code_execute",
            &non_python_code
        ));

        for command in [
            "python3 -c \"from pathlib import Path; Path('report').write_text('x')\"",
            "python3 -c \"from pathlib import Path; Path('report').write_bytes(b'x')\"",
            "python3 -c \"print('x')\" > report.md",
        ] {
            let arguments = serde_json::json!({"command": command});
            assert!(
                is_substantive_workspace_effect("bash", &arguments),
                "existing write detector regressed: {command}"
            );
            assert!(is_workspace_mutation_candidate("bash", &arguments));
        }
    }

    #[test]
    fn descriptor_duplication_at_substitution_boundaries_is_not_a_write() {
        for command in [
            r#"out=$(python3 -m unittest -v test_calculator.py 2>&1); status=$?"#,
            r#"out=$(python3 -m unittest -v test_calculator.py 2>&1}"#,
            r#"out=$(python3 -m unittest -v test_calculator.py 2>&-)"#,
            r#"out=$(python3 -m unittest -v test_calculator.py 2>&"1")"#,
        ] {
            let arguments = serde_json::json!({"command": command});
            assert!(
                !is_substantive_workspace_effect("bash", &arguments),
                "descriptor duplication was classified as substantive: {command}"
            );
            assert!(
                !is_workspace_mutation_candidate("bash", &arguments),
                "descriptor duplication was classified as a mutation: {command}"
            );
        }
    }

    #[test]
    fn command_substitution_file_redirection_remains_a_mutation() {
        let arguments = serde_json::json!({
            "command": "out=$(printf result > report.txt)"
        });
        assert!(is_substantive_workspace_effect("bash", &arguments));
        assert!(is_workspace_mutation_candidate("bash", &arguments));
    }

    #[test]
    fn verification_output_is_guarded_but_not_a_substantive_deliverable() {
        for command in [
            "pytest -q > report.txt",
            "python3 -m pytest -q > report.txt 2>&1",
            r#"PYTHONDONTWRITEBYTECODE=1 python -m pytest > "reports/test run.txt""#,
        ] {
            let arguments = serde_json::json!({"command": command});
            assert!(
                !is_substantive_workspace_effect("bash", &arguments),
                "verification output was classified as a deliverable: {command}"
            );
            assert!(
                is_workspace_mutation_candidate("bash", &arguments),
                "verification output escaped the mutation guard: {command}"
            );
        }
    }

    #[test]
    fn quoted_descriptor_and_transient_redirection_regressions() {
        for command in [
            r#"printf '%s\n' '2>&1)'"#,
            r#"printf '%s\n' "2>&1}""#,
            "pytest -q > /dev/null 2>&1",
            "pytest -q > /proc/self/fd/1 2>&1",
            "pytest -q > .pytest_cache/report.txt",
            "pytest -q > pytest_run_now.txt",
        ] {
            let arguments = serde_json::json!({"command": command});
            assert!(
                !is_substantive_workspace_effect("bash", &arguments),
                "quoted or transient redirection was substantive: {command}"
            );
            assert!(
                !is_workspace_mutation_candidate("bash", &arguments),
                "quoted or transient redirection was a mutation candidate: {command}"
            );
        }

        for command in [
            r#"printf result > "report).txt""#,
            "printf result > cache.tmp}",
            r#"printf result >&"report).txt""#,
        ] {
            let arguments = serde_json::json!({"command": command});
            assert!(
                is_substantive_workspace_effect("bash", &arguments),
                "quoted or delimiter-bearing target escaped classification: {command}"
            );
            assert!(is_workspace_mutation_candidate("bash", &arguments));
        }
    }

    #[test]
    fn workspace_lease_authorizes_only_the_exact_normalized_write_path() {
        let workspace = tempfile::tempdir().unwrap();
        let lease = WorkspaceResourceLease::new(
            "child-a",
            workspace.path(),
            vec![("reports/final.md".to_string(), WorkspaceLeaseAccess::Write)],
        )
        .unwrap();

        lease
            .authorizes_file_tool(
                "file_write",
                &serde_json::json!({"path": "reports/final.md"}),
            )
            .unwrap();
        lease
            .authorizes_file_tool(
                "file_edit",
                &serde_json::json!({"path": workspace.path().join("reports/final.md")}),
            )
            .unwrap();

        assert!(lease
            .authorizes_file_tool(
                "file_write",
                &serde_json::json!({"path": "reports/other.md"}),
            )
            .is_err());
        assert!(lease
            .authorizes_file_tool(
                "file_write",
                &serde_json::json!({"path": "reports/final.md/child"}),
            )
            .is_err());
        assert!(lease
            .authorizes_file_tool("file_write", &serde_json::json!({"path": "../outside.md"}),)
            .is_err());
    }

    #[test]
    fn workspace_lease_conflicts_are_path_and_access_sensitive() {
        let workspace = tempfile::tempdir().unwrap();
        let first = WorkspaceResourceLease::new(
            "first",
            workspace.path(),
            vec![("src/a.rs".to_string(), WorkspaceLeaseAccess::Write)],
        )
        .unwrap();
        let same = WorkspaceResourceLease::new(
            "same",
            workspace.path(),
            vec![("src/a.rs".to_string(), WorkspaceLeaseAccess::Write)],
        )
        .unwrap();
        let distinct = WorkspaceResourceLease::new(
            "distinct",
            workspace.path(),
            vec![("src/b.rs".to_string(), WorkspaceLeaseAccess::Write)],
        )
        .unwrap();

        assert!(first.conflicts_with(&same));
        assert!(!first.conflicts_with(&distinct));
        assert!(WorkspaceResourceLease::new(
            "broad",
            workspace.path(),
            vec![("src/".to_string(), WorkspaceLeaseAccess::Write)],
        )
        .is_err());
        assert!(WorkspaceResourceLease::new(
            "glob",
            workspace.path(),
            vec![("src/*.rs".to_string(), WorkspaceLeaseAccess::Write)],
        )
        .is_err());
    }

    #[cfg(unix)]
    #[test]
    fn workspace_lease_rejects_symlink_escape_created_after_lease() {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let lease = WorkspaceResourceLease::new(
            "symlink-swap",
            workspace.path(),
            vec![("switch/result.md".to_string(), WorkspaceLeaseAccess::Write)],
        )
        .unwrap();
        symlink(outside.path(), workspace.path().join("switch")).unwrap();

        assert!(lease
            .authorizes_file_tool(
                "file_write",
                &serde_json::json!({"path": "switch/result.md"}),
            )
            .is_err());
    }

    #[cfg(unix)]
    #[test]
    fn workspace_lease_rejects_dangling_symlink_as_existing_ancestor() {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir().unwrap();
        symlink(
            workspace.path().join("missing-target"),
            workspace.path().join("dangling"),
        )
        .unwrap();

        let result = WorkspaceResourceLease::new(
            "dangling-link",
            workspace.path(),
            vec![(
                "dangling/report.md".to_string(),
                WorkspaceLeaseAccess::Write,
            )],
        );
        assert!(result.is_err());
    }

    #[test]
    fn structured_complete_has_no_residual_work() {
        let value = serde_json::json!({
            "completion_state": "complete",
            "changes": ["a"],
            "verification": ["ok"],
            "pending_effects": [],
            "blockers": []
        });
        let parsed = CompletionEnvelope::from_result("success", Some(&value), "done");
        assert!(parsed.structured);
        assert!(!parsed.needs_follow_up_execution());
    }

    #[test]
    fn legacy_success_proceeds_to_independent_check() {
        let parsed = CompletionEnvelope::from_result("success", None, "implemented and tested");
        assert_eq!(parsed.completion_state, CompletionState::Complete);
        assert!(!parsed.needs_follow_up_execution());
    }

    #[test]
    fn conditional_policy_without_condition_remains_backward_compatible() {
        let parsed: EffectPolicy = serde_json::from_value(serde_json::json!({
            "mode": "conditional",
            "effect": "workspace_mutation"
        }))
        .unwrap();
        assert_eq!(
            parsed,
            EffectPolicy::Conditional {
                effect: EffectKind::WorkspaceMutation,
                condition: String::new(),
            }
        );
    }

    #[test]
    fn partial_result_gets_one_fallback_pending_effect() {
        let parsed = CompletionEnvelope::from_result("partial_success", None, "budget exhausted");
        assert!(parsed.needs_follow_up_execution());
        assert_eq!(parsed.pending_effects.len(), 1);
    }
}
