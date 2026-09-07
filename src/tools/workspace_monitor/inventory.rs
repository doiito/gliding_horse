use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, instrument, warn};
use walkdir::WalkDir;

use crate::memory::l2_blackboard::Blackboard;

/// Schema for the bounded, content-addressed workspace view used to attribute
/// one foreground tool invocation.  It is deliberately independent from the
/// watcher/inventory generation: those streams are asynchronous and cannot
/// establish which call caused a filesystem change.
pub const WORKSPACE_EFFECT_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceManifestIssueStage {
    Capture,
    Before,
    After,
    Diff,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceManifestIssueKind {
    InvalidRoot,
    RootNotDirectory,
    RootIdentityUnavailable,
    RootChanged,
    Traversal,
    FileLimitExceeded,
    DirectoryLimitExceeded,
    ByteLimitExceeded,
    InvalidRelativePath,
    Metadata,
    Read,
    ConcurrentMutation,
    UnsupportedEntryType,
    FileIdentityUnavailable,
    DuplicatePath,
    DuplicateIdentity,
    SchemaMismatch,
    RootMismatch,
    DigestMismatch,
    IncompleteManifest,
}

/// A fail-closed diagnostic from manifest capture or comparison.  `path`,
/// when present, is a normalized workspace-relative path.  A path which
/// cannot be represented safely is kept only in `message` for diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceManifestIssue {
    pub stage: WorkspaceManifestIssueStage,
    pub kind: WorkspaceManifestIssueKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub message: String,
}

/// One complete regular-file revision.  Hashes include the `sha256:` prefix
/// so callers cannot confuse them with another digest algorithm.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceFileRevision {
    pub path: String,
    pub size_bytes: u64,
    pub content_sha256: String,
    /// Stable identity of the underlying regular file, independent of path.
    /// Unix uses a domain-separated digest of `(st_dev, st_ino)`; Windows
    /// uses `(volume serial, file index)`. `None` is never accepted in a
    /// complete manifest containing files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_identity: Option<String>,
    /// Unix permission and special bits (`st_mode & 0o7777`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unix_mode: Option<u32>,
}

/// A content change at one stable workspace-relative path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceFileModification {
    pub path: String,
    pub before_size_bytes: u64,
    pub after_size_bytes: u64,
    pub before_content_sha256: String,
    pub after_content_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_file_identity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_file_identity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_unix_mode: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_unix_mode: Option<u32>,
}

/// Bounded point-in-time view of regular files and directories.  Partial data
/// is retained for operator diagnostics, but `complete=false` means it must
/// never be used as positive attribution evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceEffectManifest {
    pub schema_version: u32,
    pub workspace_root: PathBuf,
    pub workspace_root_identity: String,
    pub files: Vec<WorkspaceFileRevision>,
    pub directories: Vec<String>,
    pub total_bytes_hashed: u64,
    pub digest: String,
    pub complete: bool,
    pub errors: Vec<WorkspaceManifestIssue>,
}

/// Exact net difference between two compatible manifests.  Renames are
/// represented as one removed and one created path.  Directory metadata-only
/// changes are intentionally ignored; creation/removal is represented.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceEffectDelta {
    pub schema_version: u32,
    pub before_digest: String,
    pub after_digest: String,
    pub files_created: Vec<WorkspaceFileRevision>,
    pub files_modified: Vec<WorkspaceFileModification>,
    pub files_removed: Vec<WorkspaceFileRevision>,
    pub directories_created: Vec<String>,
    pub directories_removed: Vec<String>,
    pub complete: bool,
    pub errors: Vec<WorkspaceManifestIssue>,
}

fn update_len_prefixed(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}

fn valid_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    })
}

fn manifest_digest(
    workspace_root_identity: &str,
    files: &[WorkspaceFileRevision],
    directories: &[String],
    total_bytes_hashed: u64,
    complete: bool,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"glidinghorse.workspace-effect-manifest/v1\0");
    update_len_prefixed(&mut digest, workspace_root_identity.as_bytes());
    digest.update([u8::from(complete)]);
    digest.update(total_bytes_hashed.to_le_bytes());
    digest.update((files.len() as u64).to_le_bytes());
    for file in files {
        digest.update(b"file\0");
        update_len_prefixed(&mut digest, file.path.as_bytes());
        digest.update(file.size_bytes.to_le_bytes());
        update_len_prefixed(&mut digest, file.content_sha256.as_bytes());
        match &file.file_identity {
            Some(identity) => {
                digest.update([1]);
                update_len_prefixed(&mut digest, identity.as_bytes());
            }
            None => digest.update([0]),
        }
        match file.unix_mode {
            Some(mode) => {
                digest.update([1]);
                digest.update(mode.to_le_bytes());
            }
            None => digest.update([0]),
        }
    }
    digest.update((directories.len() as u64).to_le_bytes());
    for directory in directories {
        digest.update(b"directory\0");
        update_len_prefixed(&mut digest, directory.as_bytes());
    }
    format!("sha256:{}", hex::encode(digest.finalize()))
}

fn normalized_relative_path(root: &Path, path: &Path) -> Result<String, String> {
    let relative = path.strip_prefix(root).map_err(|_| {
        format!(
            "path '{}' is not below workspace root '{}'",
            path.display(),
            root.display()
        )
    })?;
    let components = relative
        .components()
        .map(|component| match component {
            Component::Normal(value) => value
                .to_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| "workspace path is not valid UTF-8".to_string()),
            _ => Err("workspace path contains a non-normal component".to_string()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if components.is_empty() {
        return Err("workspace root is not an attributable entry".to_string());
    }
    Ok(components.join("/"))
}

fn valid_normalized_relative_path(path: &str) -> bool {
    if path.is_empty() || path.chars().any(char::is_control) || path.contains('\\') {
        return false;
    }
    let components = Path::new(path).components().collect::<Vec<_>>();
    !components.is_empty()
        && components
            .iter()
            .all(|component| matches!(component, Component::Normal(_)))
        && components
            .iter()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")
            == path
}

fn workspace_root_identity(
    path: &Path,
    metadata: &std::fs::Metadata,
) -> std::io::Result<Option<String>> {
    let mut digest = Sha256::new();
    digest.update(b"glidinghorse.workspace-root-identity/v1\0");
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::MetadataExt;
        update_len_prefixed(&mut digest, b"unix-path-dev-ino");
        update_len_prefixed(&mut digest, path.as_os_str().as_bytes());
        digest.update(metadata.dev().to_le_bytes());
        digest.update(metadata.ino().to_le_bytes());
        return Ok(Some(format!("sha256:{}", hex::encode(digest.finalize()))));
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;

        let _ = metadata;
        update_len_prefixed(&mut digest, b"windows-path-volume-file-index");
        let mut path_bytes = Vec::new();
        for unit in path.as_os_str().encode_wide() {
            path_bytes.extend_from_slice(&unit.to_le_bytes());
        }
        update_len_prefixed(&mut digest, &path_bytes);
        let (volume_serial, file_index) = windows_path_identity_parts(path, true)?;
        digest.update(volume_serial.to_le_bytes());
        digest.update(file_index.to_le_bytes());
        return Ok(Some(format!("sha256:{}", hex::encode(digest.finalize()))));
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (path, metadata, digest);
        Ok(None)
    }
}

fn digest_file_identity(platform: &[u8], first: u64, second: u64) -> String {
    let mut digest = Sha256::new();
    digest.update(b"glidinghorse.workspace-file-identity/v1\0");
    update_len_prefixed(&mut digest, platform);
    digest.update(first.to_le_bytes());
    digest.update(second.to_le_bytes());
    format!("sha256:{}", hex::encode(digest.finalize()))
}

#[cfg(windows)]
#[repr(C)]
struct WindowsFileTime {
    low_date_time: u32,
    high_date_time: u32,
}

#[cfg(windows)]
#[repr(C)]
struct WindowsByHandleFileInformation {
    file_attributes: u32,
    creation_time: WindowsFileTime,
    last_access_time: WindowsFileTime,
    last_write_time: WindowsFileTime,
    volume_serial_number: u32,
    file_size_high: u32,
    file_size_low: u32,
    number_of_links: u32,
    file_index_high: u32,
    file_index_low: u32,
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    #[link_name = "GetFileInformationByHandle"]
    fn get_file_information_by_handle(
        file: *mut std::ffi::c_void,
        information: *mut WindowsByHandleFileInformation,
    ) -> i32;
}

#[cfg(windows)]
fn windows_handle_identity_parts(file: &std::fs::File) -> std::io::Result<(u64, u64)> {
    use std::mem::MaybeUninit;
    use std::os::windows::io::AsRawHandle;

    let mut information = MaybeUninit::<WindowsByHandleFileInformation>::uninit();
    // SAFETY: `file` owns a live Windows handle for the duration of this
    // call and `information` points to writable, correctly laid-out storage.
    let succeeded = unsafe {
        get_file_information_by_handle(file.as_raw_handle().cast(), information.as_mut_ptr())
    };
    if succeeded == 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: a non-zero result promises that the output was initialized.
    let information = unsafe { information.assume_init() };
    let file_index =
        (u64::from(information.file_index_high) << 32) | u64::from(information.file_index_low);
    Ok((u64::from(information.volume_serial_number), file_index))
}

#[cfg(windows)]
fn windows_path_identity_parts(path: &Path, directory: bool) -> std::io::Result<(u64, u64)> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    let mut options = std::fs::OpenOptions::new();
    options.read(true).custom_flags(
        FILE_FLAG_OPEN_REPARSE_POINT
            | if directory {
                FILE_FLAG_BACKUP_SEMANTICS
            } else {
                0
            },
    );
    let handle = options.open(path)?;
    let metadata = handle.metadata()?;
    if directory != metadata.file_type().is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "workspace path type changed while acquiring stable identity",
        ));
    }
    windows_handle_identity_parts(&handle)
}

fn stable_file_identity(
    file: &std::fs::File,
    metadata: &std::fs::Metadata,
) -> std::io::Result<Option<String>> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let _ = file;
        return Ok(Some(digest_file_identity(
            b"unix-dev-ino",
            metadata.dev(),
            metadata.ino(),
        )));
    }
    #[cfg(windows)]
    {
        let _ = metadata;
        let (volume_serial, file_index) = windows_handle_identity_parts(file)?;
        return Ok(Some(digest_file_identity(
            b"windows-volume-file-index",
            volume_serial,
            file_index,
        )));
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, metadata);
        Ok(None)
    }
}

fn stable_path_identity(
    path: &Path,
    metadata: &std::fs::Metadata,
) -> std::io::Result<Option<String>> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let _ = path;
        return Ok(Some(digest_file_identity(
            b"unix-dev-ino",
            metadata.dev(),
            metadata.ino(),
        )));
    }
    #[cfg(windows)]
    {
        let reopened = open_manifest_file(path)?;
        let reopened_metadata = reopened.metadata()?;
        if !reopened_metadata.file_type().is_file()
            || !same_file_revision(metadata, &reopened_metadata)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "workspace path identity changed while it was reopened",
            ));
        }
        return stable_file_identity(&reopened, &reopened_metadata);
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (path, metadata);
        Ok(None)
    }
}

fn captured_unix_mode(metadata: &std::fs::Metadata) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some(metadata.mode() & 0o7777)
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        None
    }
}

fn same_file_revision(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    if left.len() != right.len() || left.modified().ok() != right.modified().ok() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        left.dev() == right.dev()
            && left.ino() == right.ino()
            && left.mode() == right.mode()
            && left.nlink() == right.nlink()
            && left.mtime() == right.mtime()
            && left.mtime_nsec() == right.mtime_nsec()
            && left.ctime() == right.ctime()
            && left.ctime_nsec() == right.ctime_nsec()
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        left.file_attributes() == right.file_attributes()
            && left.creation_time() == right.creation_time()
            && left.last_write_time() == right.last_write_time()
            && left.file_size() == right.file_size()
    }
    #[cfg(not(any(unix, windows)))]
    {
        true
    }
}

fn restage_issue(
    issue: &WorkspaceManifestIssue,
    stage: WorkspaceManifestIssueStage,
) -> WorkspaceManifestIssue {
    let mut issue = issue.clone();
    issue.stage = stage;
    issue
}

impl WorkspaceEffectDelta {
    /// Compare two manifests captured around one foreground tool invocation.
    /// No positive changes are returned if either input is partial, corrupt,
    /// from a different root, or from an incompatible schema.
    pub fn between(before: &WorkspaceEffectManifest, after: &WorkspaceEffectManifest) -> Self {
        let mut delta = Self {
            schema_version: WORKSPACE_EFFECT_MANIFEST_SCHEMA_VERSION,
            before_digest: before.digest.clone(),
            after_digest: after.digest.clone(),
            files_created: Vec::new(),
            files_modified: Vec::new(),
            files_removed: Vec::new(),
            directories_created: Vec::new(),
            directories_removed: Vec::new(),
            complete: false,
            errors: before
                .errors
                .iter()
                .map(|issue| restage_issue(issue, WorkspaceManifestIssueStage::Before))
                .chain(
                    after
                        .errors
                        .iter()
                        .map(|issue| restage_issue(issue, WorkspaceManifestIssueStage::After)),
                )
                .collect(),
        };

        if before.schema_version != WORKSPACE_EFFECT_MANIFEST_SCHEMA_VERSION
            || after.schema_version != WORKSPACE_EFFECT_MANIFEST_SCHEMA_VERSION
            || before.schema_version != after.schema_version
        {
            delta.errors.push(WorkspaceManifestIssue {
                stage: WorkspaceManifestIssueStage::Diff,
                kind: WorkspaceManifestIssueKind::SchemaMismatch,
                path: None,
                message: format!(
                    "manifest schema mismatch: before={}, after={}, supported={}",
                    before.schema_version,
                    after.schema_version,
                    WORKSPACE_EFFECT_MANIFEST_SCHEMA_VERSION
                ),
            });
        }
        if before.workspace_root != after.workspace_root
            || before.workspace_root_identity != after.workspace_root_identity
        {
            delta.errors.push(WorkspaceManifestIssue {
                stage: WorkspaceManifestIssueStage::Diff,
                kind: WorkspaceManifestIssueKind::RootMismatch,
                path: None,
                message: "before and after manifests do not identify the same workspace root"
                    .to_string(),
            });
        }
        for (stage, manifest) in [
            (WorkspaceManifestIssueStage::Before, before),
            (WorkspaceManifestIssueStage::After, after),
        ] {
            if !manifest.complete && manifest.errors.is_empty() {
                delta.errors.push(WorkspaceManifestIssue {
                    stage,
                    kind: WorkspaceManifestIssueKind::IncompleteManifest,
                    path: None,
                    message: "manifest is incomplete without a diagnostic".to_string(),
                });
            }
            if !valid_sha256(&manifest.workspace_root_identity)
                || !valid_sha256(&manifest.digest)
                || manifest.digest
                    != manifest_digest(
                        &manifest.workspace_root_identity,
                        &manifest.files,
                        &manifest.directories,
                        manifest.total_bytes_hashed,
                        manifest.complete,
                    )
            {
                delta.errors.push(WorkspaceManifestIssue {
                    stage,
                    kind: WorkspaceManifestIssueKind::DigestMismatch,
                    path: None,
                    message: "manifest/root digest is non-canonical or does not match its entries"
                        .to_string(),
                });
            }
            let mut prior_file = None;
            let mut computed_total_bytes = 0u64;
            let mut identities = BTreeMap::<&str, &str>::new();
            for file in &manifest.files {
                if !valid_normalized_relative_path(&file.path)
                    || !valid_sha256(&file.content_sha256)
                {
                    delta.errors.push(WorkspaceManifestIssue {
                        stage,
                        kind: WorkspaceManifestIssueKind::InvalidRelativePath,
                        path: Some(file.path.clone()),
                        message: "manifest contains an invalid file path or content digest"
                            .to_string(),
                    });
                    break;
                }
                if prior_file.is_some_and(|prior: &str| prior >= file.path.as_str()) {
                    delta.errors.push(WorkspaceManifestIssue {
                        stage,
                        kind: WorkspaceManifestIssueKind::DuplicatePath,
                        path: Some(file.path.clone()),
                        message: "manifest file paths are not strictly sorted and unique"
                            .to_string(),
                    });
                    break;
                }
                let identity = match file.file_identity.as_deref() {
                    Some(identity) if valid_sha256(identity) => identity,
                    _ => {
                        delta.errors.push(WorkspaceManifestIssue {
                            stage,
                            kind: WorkspaceManifestIssueKind::FileIdentityUnavailable,
                            path: Some(file.path.clone()),
                            message: "manifest file has no canonical stable identity".to_string(),
                        });
                        break;
                    }
                };
                if let Some(previous_path) = identities.insert(identity, file.path.as_str()) {
                    delta.errors.push(WorkspaceManifestIssue {
                        stage,
                        kind: WorkspaceManifestIssueKind::DuplicateIdentity,
                        path: Some(file.path.clone()),
                        message: format!(
                            "manifest paths share one regular-file identity: previous='{previous_path}'"
                        ),
                    });
                    break;
                }
                #[cfg(unix)]
                if file.unix_mode.is_none_or(|mode| mode > 0o7777) {
                    delta.errors.push(WorkspaceManifestIssue {
                        stage,
                        kind: WorkspaceManifestIssueKind::Metadata,
                        path: Some(file.path.clone()),
                        message: "manifest file has no canonical Unix mode".to_string(),
                    });
                    break;
                }
                #[cfg(not(unix))]
                if file.unix_mode.is_some() {
                    delta.errors.push(WorkspaceManifestIssue {
                        stage,
                        kind: WorkspaceManifestIssueKind::Metadata,
                        path: Some(file.path.clone()),
                        message: "non-Unix manifest unexpectedly contains a Unix mode".to_string(),
                    });
                    break;
                }
                prior_file = Some(file.path.as_str());
                computed_total_bytes = match computed_total_bytes.checked_add(file.size_bytes) {
                    Some(total) => total,
                    None => {
                        delta.errors.push(WorkspaceManifestIssue {
                            stage,
                            kind: WorkspaceManifestIssueKind::DigestMismatch,
                            path: None,
                            message: "manifest file sizes overflow the aggregate byte count"
                                .to_string(),
                        });
                        break;
                    }
                };
            }
            if computed_total_bytes != manifest.total_bytes_hashed {
                delta.errors.push(WorkspaceManifestIssue {
                    stage,
                    kind: WorkspaceManifestIssueKind::DigestMismatch,
                    path: None,
                    message: format!(
                        "manifest byte total mismatch: entries={computed_total_bytes}, recorded={}",
                        manifest.total_bytes_hashed
                    ),
                });
            }
            let mut prior_directory = None;
            for directory in &manifest.directories {
                if !valid_normalized_relative_path(directory) {
                    delta.errors.push(WorkspaceManifestIssue {
                        stage,
                        kind: WorkspaceManifestIssueKind::InvalidRelativePath,
                        path: Some(directory.clone()),
                        message: "manifest contains an invalid directory path".to_string(),
                    });
                    break;
                }
                if prior_directory.is_some_and(|prior: &str| prior >= directory.as_str()) {
                    delta.errors.push(WorkspaceManifestIssue {
                        stage,
                        kind: WorkspaceManifestIssueKind::DuplicatePath,
                        path: Some(directory.clone()),
                        message: "manifest directory paths are not strictly sorted and unique"
                            .to_string(),
                    });
                    break;
                }
                prior_directory = Some(directory.as_str());
            }
            if let Some(file) = manifest
                .files
                .iter()
                .find(|file| manifest.directories.binary_search(&file.path).is_ok())
            {
                delta.errors.push(WorkspaceManifestIssue {
                    stage,
                    kind: WorkspaceManifestIssueKind::DuplicatePath,
                    path: Some(file.path.clone()),
                    message: "manifest path is both a regular file and a directory".to_string(),
                });
            }
        }

        if !before.complete || !after.complete || !delta.errors.is_empty() {
            return delta;
        }

        let before_files = before
            .files
            .iter()
            .map(|file| (file.path.as_str(), file))
            .collect::<BTreeMap<_, _>>();
        let after_files = after
            .files
            .iter()
            .map(|file| (file.path.as_str(), file))
            .collect::<BTreeMap<_, _>>();
        for (path, after_file) in &after_files {
            match before_files.get(path) {
                None => delta.files_created.push((*after_file).clone()),
                Some(before_file)
                    if before_file.size_bytes != after_file.size_bytes
                        || before_file.content_sha256 != after_file.content_sha256
                        || before_file.file_identity != after_file.file_identity
                        || before_file.unix_mode != after_file.unix_mode =>
                {
                    delta.files_modified.push(WorkspaceFileModification {
                        path: (*path).to_string(),
                        before_size_bytes: before_file.size_bytes,
                        after_size_bytes: after_file.size_bytes,
                        before_content_sha256: before_file.content_sha256.clone(),
                        after_content_sha256: after_file.content_sha256.clone(),
                        before_file_identity: before_file.file_identity.clone(),
                        after_file_identity: after_file.file_identity.clone(),
                        before_unix_mode: before_file.unix_mode,
                        after_unix_mode: after_file.unix_mode,
                    });
                }
                Some(_) => {}
            }
        }
        for (path, before_file) in &before_files {
            if !after_files.contains_key(path) {
                delta.files_removed.push((*before_file).clone());
            }
        }

        let before_directories = before
            .directories
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let after_directories = after
            .directories
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        delta.directories_created = after_directories
            .difference(&before_directories)
            .map(|path| (*path).to_string())
            .collect();
        delta.directories_removed = before_directories
            .difference(&after_directories)
            .map(|path| (*path).to_string())
            .collect();
        delta.complete = true;
        delta
    }
}

fn finish_effect_manifest(
    workspace_root: PathBuf,
    workspace_root_identity: String,
    files: BTreeMap<String, WorkspaceFileRevision>,
    directories: BTreeSet<String>,
    total_bytes_hashed: u64,
    errors: Vec<WorkspaceManifestIssue>,
) -> WorkspaceEffectManifest {
    let files = files.into_values().collect::<Vec<_>>();
    let directories = directories.into_iter().collect::<Vec<_>>();
    let complete = errors.is_empty();
    let digest = manifest_digest(
        &workspace_root_identity,
        &files,
        &directories,
        total_bytes_hashed,
        complete,
    );
    WorkspaceEffectManifest {
        schema_version: WORKSPACE_EFFECT_MANIFEST_SCHEMA_VERSION,
        workspace_root,
        workspace_root_identity,
        files,
        directories,
        total_bytes_hashed,
        digest,
        complete,
        errors,
    }
}

fn capture_issue(
    kind: WorkspaceManifestIssueKind,
    path: Option<String>,
    message: impl Into<String>,
) -> WorkspaceManifestIssue {
    WorkspaceManifestIssue {
        stage: WorkspaceManifestIssueStage::Capture,
        kind,
        path,
        message: message.into(),
    }
}

fn open_manifest_file(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Do not follow a reparse point introduced between traversal and
        // open. This is FILE_FLAG_OPEN_REPARSE_POINT from WinBase.h.
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    options.open(path)
}

/// Workspace-local process state is never project input.  This boundary is
/// intentionally independent of user-configurable ignore patterns: the
/// monitor stores its own databases below this directory and indexing them
/// would make the monitor observe (and potentially cache) its own writes.
pub(crate) const WORKSPACE_RUNTIME_DIR: &str = ".gliding_horse";

pub(crate) fn is_workspace_runtime_path(path: &Path) -> bool {
    // `Path::components` handles native paths.  The normalized fallback keeps
    // event payloads produced on another platform safe as well.
    path.components().any(|component| {
        matches!(
            component,
            std::path::Component::Normal(name) if name == WORKSPACE_RUNTIME_DIR
        )
    }) || path
        .to_string_lossy()
        .replace('\\', "/")
        .split('/')
        .any(|component| component == WORKSPACE_RUNTIME_DIR)
}

fn is_excluded_with_patterns(path: &Path, patterns: &[String]) -> bool {
    if is_workspace_runtime_path(path) {
        return true;
    }
    let normalized = path.to_string_lossy().replace('\\', "/");
    patterns.iter().any(|pattern| {
        let normalized_pattern = pattern.replace('\\', "/");
        match_glob_pattern(&normalized, &normalized_pattern)
    })
}

fn normalize_exact_relative_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Err(format!(
            "exact workspace exclusion must be relative: '{}'",
            path.display()
        ));
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => normalized.push(value),
            _ => {
                return Err(format!(
                    "exact workspace exclusion contains a non-normal component: '{}'",
                    path.display()
                ))
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err("exact workspace exclusion must not be empty".to_string());
    }
    Ok(normalized)
}

pub(crate) fn validate_exact_exclude_paths(
    workspace_root: &Path,
    paths: &[PathBuf],
) -> Result<(PathBuf, BTreeSet<PathBuf>), String> {
    let canonical_root = std::fs::canonicalize(workspace_root).map_err(|error| {
        format!(
            "workspace root '{}' cannot be canonicalized for exact exclusions: {error}",
            workspace_root.display()
        )
    })?;
    if !canonical_root.is_dir() {
        return Err(format!(
            "workspace root '{}' is not a directory",
            canonical_root.display()
        ));
    }

    let mut validated = BTreeSet::new();
    for path in paths {
        let relative = normalize_exact_relative_path(path)?;
        let candidate = canonical_root.join(&relative);
        let metadata = std::fs::symlink_metadata(&candidate).map_err(|error| {
            format!(
                "exact workspace exclusion '{}' is unavailable: {error}",
                relative.display()
            )
        })?;
        if !metadata.file_type().is_file() {
            return Err(format!(
                "exact workspace exclusion '{}' must identify a regular file",
                relative.display()
            ));
        }
        let canonical_candidate = std::fs::canonicalize(&candidate).map_err(|error| {
            format!(
                "exact workspace exclusion '{}' cannot be canonicalized: {error}",
                relative.display()
            )
        })?;
        let canonical_relative = canonical_candidate
            .strip_prefix(&canonical_root)
            .map_err(|_| {
                format!(
                    "exact workspace exclusion '{}' resolves outside workspace '{}'",
                    relative.display(),
                    canonical_root.display()
                )
            })?
            .to_path_buf();
        if canonical_relative != relative {
            return Err(format!(
                "exact workspace exclusion '{}' must not traverse a symlink or alias",
                relative.display()
            ));
        }
        validated.insert(relative);
    }
    Ok((canonical_root, validated))
}

fn normalized_path_relative_to_root(root: &Path, path: &Path) -> Option<PathBuf> {
    let relative = if path.is_absolute() {
        path.strip_prefix(root).ok()?
    } else {
        path
    };
    normalize_exact_relative_path(relative).ok()
}

fn is_excluded_with_policy(
    workspace_root: Option<&Path>,
    path: &Path,
    patterns: &[String],
    exact_paths: &BTreeSet<PathBuf>,
) -> bool {
    if is_excluded_with_patterns(path, patterns) {
        return true;
    }
    workspace_root
        .and_then(|root| normalized_path_relative_to_root(root, path))
        .is_some_and(|relative| exact_paths.contains(&relative))
}

/// File state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileState {
    /// File exists on disk but not yet tracked.
    Undiscovered,
    /// Discovered via full_scan().
    Discovered,
    /// File has been read and is up-to-date.
    ReadFresh,
    /// File was read but has been modified externally.
    ReadStale,
    /// File was written by the Agent but not yet re-read.
    WrittenUnread,
}

impl FileState {
    pub fn as_str(&self) -> &'static str {
        match self {
            FileState::Undiscovered => "undiscovered",
            FileState::Discovered => "discovered",
            FileState::ReadFresh => "read_fresh",
            FileState::ReadStale => "read_stale",
            FileState::WrittenUnread => "written_unread",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "undiscovered" => FileState::Undiscovered,
            "discovered" => FileState::Discovered,
            "read_fresh" => FileState::ReadFresh,
            "read_stale" => FileState::ReadStale,
            "written_unread" => FileState::WrittenUnread,
            _ => FileState::Undiscovered,
        }
    }
}

/// A single file entry in the inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub file_size: u64,
    pub file_ext: String,
    pub language: String,
    pub mtime: i64,
    pub content_hash: String,
    pub state: FileState,
    pub last_read_at: Option<i64>,
    pub last_read_version: u64,
    pub current_version: u64,
    pub read_count: u64,
}

impl FileEntry {
    /// The IRI used for this file in L2 Named Graph.
    pub fn iri(&self) -> String {
        format!("iri://workspace/file/{}", self.path)
    }

    /// The parent directory IRI.
    pub fn parent_dir_iri(&self) -> String {
        let parent = Path::new(&self.path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        format!("iri://workspace/dir/{}/", parent)
    }
}

/// Classification of a language from a file extension.
fn classify_language(ext: &str) -> &'static str {
    match ext {
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "py" => "python",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "hpp" | "cc" | "cxx" => "cpp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "kt" | "kts" => "kotlin",
        "scala" => "scala",
        "toml" => "toml",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "md" | "mdx" => "markdown",
        "html" | "htm" => "html",
        "css" | "scss" | "less" => "css",
        _ => "unknown",
    }
}

/// Shared workspace RDF constant.
pub const WORKSPACE_GRAPH: &str = "iri://workspace";

/// Inventory cache table: key = file path (str), value = serialized FileEntry.
const INVENTORY_CACHE: TableDefinition<&str, &[u8]> = TableDefinition::new("inventory_cache");

/// Maximum number of entries in the in-memory inventory.
/// Prevents unbounded memory growth in large workspaces.
/// In tests, uses a small value so limit enforcement is testable without 50k files.
#[cfg(not(test))]
pub(crate) const MAX_INVENTORY_ENTRIES: usize = 50_000;
#[cfg(test)]
pub(crate) const MAX_INVENTORY_ENTRIES: usize = 10;

/// FileInventory — thin facade over L2 Blackboard (RDF) with redb hot cache.
///
/// The authority data source is L2 (Oxigraph RDF named graph `iri://workspace`).
/// redb serves as a hot cache for fast metadata lookups.
pub struct FileInventory {
    /// L2 Blackboard (RDF triple store) for authority data.
    blackboard: Option<Arc<Blackboard>>,
    /// redb hot cache: path → serialized FileEntry.
    cache: Option<Database>,
    /// In-memory cache for fastest access (no redb deserialization).
    mem_cache: RwLock<HashMap<String, FileEntry>>,
    /// Exclude patterns for scanning. Uses RwLock for post-construction updates (gitignore sync).
    exclude_patterns: RwLock<Vec<String>>,
    /// Canonical root used to convert absolute watcher/inventory paths into
    /// the normalized relative namespace of `exact_exclude_paths`.
    exact_exclude_workspace_root: Option<PathBuf>,
    /// Validated exact regular-file exclusions. Unlike legacy patterns these
    /// never match by basename, substring, suffix, or directory prefix.
    exact_exclude_paths: BTreeSet<PathBuf>,
}

impl FileInventory {
    /// Create a new FileInventory.
    ///
    /// * `blackboard` - Optional L2 Blackboard for RDF storage.
    /// * `db` - Optional redb database for hot cache.
    /// * `exclude_patterns` - Glob patterns to exclude from scanning (e.g., "node_modules/").
    pub fn new(
        blackboard: Option<Arc<Blackboard>>,
        db: Option<Database>,
        exclude_patterns: Vec<String>,
    ) -> Self {
        Self::new_with_validated_exact_exclude_paths(
            blackboard,
            db,
            exclude_patterns,
            None,
            BTreeSet::new(),
        )
    }

    pub(crate) fn new_with_validated_exact_exclude_paths(
        blackboard: Option<Arc<Blackboard>>,
        db: Option<Database>,
        exclude_patterns: Vec<String>,
        exact_exclude_workspace_root: Option<PathBuf>,
        exact_exclude_paths: BTreeSet<PathBuf>,
    ) -> Self {
        let mut mem_cache = HashMap::new();

        // Pre-warm from redb if available
        if let Some(ref database) = db {
            if let Ok(read_txn) = database.begin_read() {
                if let Ok(table) = read_txn.open_table(INVENTORY_CACHE) {
                    if let Ok(iter) = table.iter() {
                        for result in iter {
                            if let Ok((key, value)) = result {
                                let path = key.value().to_string();
                                if let Ok(entry) =
                                    serde_json::from_slice::<FileEntry>(value.value())
                                {
                                    mem_cache.insert(path, entry);
                                }
                            }
                        }
                    }
                }
            }
        }

        let inventory = Self {
            blackboard,
            cache: db,
            mem_cache: RwLock::new(mem_cache),
            exclude_patterns: RwLock::new(exclude_patterns),
            exact_exclude_workspace_root,
            exact_exclude_paths,
        };

        // A previous version could persist workspace-local runtime files when
        // callers supplied a custom exclusion list.  Remove those entries at
        // construction so they cannot reappear after an upgrade/restart.
        inventory.purge_excluded_entries();
        inventory
    }

    /// Perform a full directory scan of `root`, discovering all files.
    ///
    /// Returns the number of discovered files.
    #[instrument(skip(self))]
    pub fn full_scan(&self, root: &str) -> usize {
        let mut count = 0;
        let mut seen = std::collections::HashSet::new();

        for entry in WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| !self.is_excluded(e.path()))
        {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path().to_string_lossy().to_string();
            seen.insert(path.clone());
            if let Some(existing) = self.get_entry(&path) {
                // Reconcile files modified while the process/watch service was
                // offline using metadata only. Content and hashes stay lazy.
                if let Ok(metadata) = entry.metadata() {
                    let mtime = metadata
                        .modified()
                        .ok()
                        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|duration| duration.as_millis() as i64)
                        .unwrap_or(0);
                    if existing.file_size != metadata.len() || existing.mtime != mtime {
                        self.mark_stale(&path);
                    }
                }
            } else if self.add_entry(&path).is_some() {
                // add_entry owns the capacity check. Even when no new entry
                // can be admitted, the scan must continue so persisted files
                // modified/deleted while the watcher was offline are still
                // reconciled.
                count += 1;
            }
        }

        // Remove persisted inventory entries deleted while no watcher was
        // running. Restrict reconciliation to this scan root.
        let root_path = Path::new(root);
        let removed = self
            .list_all()
            .into_iter()
            .filter(|entry| {
                Path::new(&entry.path).starts_with(root_path) && !seen.contains(&entry.path)
            })
            .map(|entry| entry.path)
            .collect::<Vec<_>>();
        for path in removed {
            self.remove(&path);
        }

        debug!(root = %root, discovered = count, "FileInventory: full scan completed");
        count
    }

    /// Hash the visible workspace paths and file contents under explicit
    /// resource limits.  The result is used only to confirm an actual tool
    /// effect; it is not persisted as memory or injected into an LLM prompt.
    pub fn semantic_fingerprint(
        &self,
        root: &Path,
        max_files: usize,
        max_bytes: u64,
    ) -> Result<String, String> {
        use std::io::Read;

        let mut files = Vec::new();
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|entry| !self.is_excluded(entry.path()))
        {
            let entry = entry.map_err(|error| error.to_string())?;
            if entry.file_type().is_file() {
                files.push(entry.into_path());
                if files.len() > max_files {
                    return Err(format!(
                        "semantic effect snapshot exceeds configured file limit {max_files}"
                    ));
                }
            }
        }
        files.sort();

        let mut total_bytes = 0u64;
        let mut digest = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        for path in files {
            let metadata = std::fs::metadata(&path).map_err(|error| error.to_string())?;
            total_bytes = total_bytes.saturating_add(metadata.len());
            if total_bytes > max_bytes {
                return Err(format!(
                    "semantic effect snapshot exceeds configured byte limit {max_bytes}"
                ));
            }
            let relative = path
                .strip_prefix(root)
                .unwrap_or(path.as_path())
                .to_string_lossy()
                .replace('\\', "/");
            digest.update((relative.len() as u64).to_le_bytes());
            digest.update(relative.as_bytes());
            digest.update(metadata.len().to_le_bytes());

            let mut file = std::fs::File::open(&path).map_err(|error| error.to_string())?;
            loop {
                let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
                if read == 0 {
                    break;
                }
                digest.update(&buffer[..read]);
            }
        }
        Ok(format!("sha256:{}", hex::encode(digest.finalize())))
    }

    /// Hash the bounded directory topology independently from file contents.
    /// The semantic fingerprint intentionally contains only files, so this
    /// companion receipt is required to distinguish a real `mkdir`/`rmdir`
    /// from a successful command that made no workspace change.
    pub fn structural_fingerprint(
        &self,
        root: &Path,
        max_entries: usize,
    ) -> Result<String, String> {
        let mut directories = Vec::new();
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|entry| !self.is_excluded(entry.path()))
        {
            let entry = entry.map_err(|error| error.to_string())?;
            if entry.depth() > 0 && entry.file_type().is_dir() {
                directories.push(entry.into_path());
                if directories.len() > max_entries {
                    return Err(format!(
                        "structural effect snapshot exceeds configured entry limit {max_entries}"
                    ));
                }
            }
        }
        directories.sort();

        let mut digest = Sha256::new();
        for path in directories {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(path.as_path())
                .to_string_lossy()
                .replace('\\', "/");
            digest.update((relative.len() as u64).to_le_bytes());
            digest.update(relative.as_bytes());
        }
        Ok(format!("sha256:{}", hex::encode(digest.finalize())))
    }

    /// Capture a deterministic, bounded map of every visible regular file and
    /// directory below `root`.  This performs direct filesystem I/O and does
    /// not consult watcher events, the inventory cache, or generation state.
    ///
    /// The caller is responsible for placing captures immediately before and
    /// after the same foreground tool invocation and for serializing other
    /// agent-owned mutations across that window.  Files changing during this
    /// individual scan are detected best-effort and make the manifest
    /// incomplete.  `max_files` independently bounds regular files and
    /// directories, preventing an empty-directory tree from bypassing the
    /// memory bound; `max_bytes` bounds bytes read and hashed.
    pub fn capture_effect_manifest(
        &self,
        root: &Path,
        max_files: usize,
        max_bytes: u64,
    ) -> WorkspaceEffectManifest {
        // Exclusions may be refreshed from .gitignore after initialization.
        // Snapshot them once so one manifest never combines two visibility
        // policies, and release the lock before performing filesystem I/O.
        let exclude_patterns = self.exclude_patterns.read().clone();
        let exact_exclude_paths = self.exact_exclude_paths.clone();
        let requested_root = root.to_path_buf();
        let canonical_root = match std::fs::canonicalize(root) {
            Ok(root) => root,
            Err(error) => {
                return finish_effect_manifest(
                    requested_root,
                    "unavailable".to_string(),
                    BTreeMap::new(),
                    BTreeSet::new(),
                    0,
                    vec![capture_issue(
                        WorkspaceManifestIssueKind::InvalidRoot,
                        None,
                        format!("workspace root cannot be canonicalized: {error}"),
                    )],
                );
            }
        };
        let root_metadata = match std::fs::symlink_metadata(&canonical_root) {
            Ok(metadata) => metadata,
            Err(error) => {
                return finish_effect_manifest(
                    canonical_root,
                    "unavailable".to_string(),
                    BTreeMap::new(),
                    BTreeSet::new(),
                    0,
                    vec![capture_issue(
                        WorkspaceManifestIssueKind::Metadata,
                        None,
                        format!("workspace root metadata is unavailable: {error}"),
                    )],
                );
            }
        };
        if !root_metadata.file_type().is_dir() {
            return finish_effect_manifest(
                canonical_root,
                "unavailable".to_string(),
                BTreeMap::new(),
                BTreeSet::new(),
                0,
                vec![capture_issue(
                    WorkspaceManifestIssueKind::RootNotDirectory,
                    None,
                    "workspace effect manifest root is not a directory",
                )],
            );
        }
        let initial_root_identity = match workspace_root_identity(&canonical_root, &root_metadata) {
            Ok(Some(identity)) => identity,
            Ok(None) => {
                return finish_effect_manifest(
                    canonical_root,
                    "unavailable".to_string(),
                    BTreeMap::new(),
                    BTreeSet::new(),
                    0,
                    vec![capture_issue(
                        WorkspaceManifestIssueKind::RootIdentityUnavailable,
                        None,
                        "this platform cannot provide a stable workspace-root identity",
                    )],
                );
            }
            Err(error) => {
                return finish_effect_manifest(
                    canonical_root,
                    "unavailable".to_string(),
                    BTreeMap::new(),
                    BTreeSet::new(),
                    0,
                    vec![capture_issue(
                        WorkspaceManifestIssueKind::RootIdentityUnavailable,
                        None,
                        format!("workspace-root identity is unavailable: {error}"),
                    )],
                );
            }
        };
        let mut files = BTreeMap::<String, WorkspaceFileRevision>::new();
        let mut file_identities = BTreeMap::<String, String>::new();
        let mut directories = BTreeSet::<String>::new();
        let mut total_bytes_hashed = 0u64;
        let mut errors = Vec::new();

        let walker = WalkDir::new(&canonical_root)
            .follow_links(false)
            .sort_by_file_name()
            .into_iter()
            .filter_entry(|entry| {
                entry.depth() == 0
                    || entry
                        .path()
                        .strip_prefix(&canonical_root)
                        .map(|relative| {
                            !is_excluded_with_policy(
                                Some(&canonical_root),
                                relative,
                                &exclude_patterns,
                                &exact_exclude_paths,
                            )
                        })
                        .unwrap_or(true)
            });
        for entry in walker {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    let path = error
                        .path()
                        .and_then(|path| normalized_relative_path(&canonical_root, path).ok());
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::Traversal,
                        path,
                        format!("workspace traversal failed: {error}"),
                    ));
                    break;
                }
            };
            if entry.depth() == 0 {
                continue;
            }
            let relative = match normalized_relative_path(&canonical_root, entry.path()) {
                Ok(relative) => relative,
                Err(error) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::InvalidRelativePath,
                        None,
                        format!(
                            "workspace entry '{}' has no safe relative representation: {error}",
                            entry.path().display()
                        ),
                    ));
                    break;
                }
            };
            // `WalkDir::filter_entry` prevents descent but still yields the
            // rejected directory entry itself. Apply the same policy before
            // recording it so ignored/runtime directory names cannot enter
            // the manifest topology.
            if is_excluded_with_policy(
                Some(&canonical_root),
                Path::new(&relative),
                &exclude_patterns,
                &exact_exclude_paths,
            ) {
                continue;
            }

            let file_type = entry.file_type();
            if file_type.is_dir() {
                if directories.contains(&relative) {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::DuplicatePath,
                        Some(relative),
                        "workspace traversal returned the same directory more than once",
                    ));
                    break;
                }
                if directories.len() >= max_files {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::DirectoryLimitExceeded,
                        Some(relative),
                        format!("workspace effect manifest exceeds directory limit {max_files}"),
                    ));
                    break;
                }
                directories.insert(relative);
                continue;
            }
            if !file_type.is_file() {
                errors.push(capture_issue(
                    WorkspaceManifestIssueKind::UnsupportedEntryType,
                    Some(relative),
                    "workspace contains a symlink or special entry that this file/directory manifest cannot safely attribute",
                ));
                break;
            }
            if files.contains_key(&relative) {
                errors.push(capture_issue(
                    WorkspaceManifestIssueKind::DuplicatePath,
                    Some(relative),
                    "workspace traversal returned the same file more than once",
                ));
                break;
            }
            if files.len() >= max_files {
                errors.push(capture_issue(
                    WorkspaceManifestIssueKind::FileLimitExceeded,
                    Some(relative),
                    format!("workspace effect manifest exceeds file limit {max_files}"),
                ));
                break;
            }

            let path = entry.path();
            let path_metadata_before = match std::fs::symlink_metadata(path) {
                Ok(metadata) if metadata.file_type().is_file() => metadata,
                Ok(_) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::ConcurrentMutation,
                        Some(relative),
                        "workspace file type changed during manifest capture",
                    ));
                    break;
                }
                Err(error) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::Metadata,
                        Some(relative),
                        format!("workspace file metadata is unavailable: {error}"),
                    ));
                    break;
                }
            };
            let projected_total = total_bytes_hashed.checked_add(path_metadata_before.len());
            if projected_total.is_none_or(|total| total > max_bytes) {
                errors.push(capture_issue(
                    WorkspaceManifestIssueKind::ByteLimitExceeded,
                    Some(relative),
                    format!("workspace effect manifest exceeds byte limit {max_bytes}"),
                ));
                break;
            }

            let mut file = match open_manifest_file(path) {
                Ok(file) => file,
                Err(error) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::Read,
                        Some(relative),
                        format!("workspace file cannot be opened safely: {error}"),
                    ));
                    break;
                }
            };
            let handle_metadata_before = match file.metadata() {
                Ok(metadata) => metadata,
                Err(error) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::Metadata,
                        Some(relative),
                        format!("opened workspace file metadata is unavailable: {error}"),
                    ));
                    break;
                }
            };
            if !same_file_revision(&path_metadata_before, &handle_metadata_before) {
                errors.push(capture_issue(
                    WorkspaceManifestIssueKind::ConcurrentMutation,
                    Some(relative),
                    "workspace file identity changed before it could be hashed",
                ));
                break;
            }
            let file_identity = match stable_file_identity(&file, &handle_metadata_before) {
                Ok(Some(identity)) => identity,
                Ok(None) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::FileIdentityUnavailable,
                        Some(relative),
                        "this platform cannot provide a stable regular-file identity; refusing incomplete attribution evidence",
                    ));
                    break;
                }
                Err(error) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::FileIdentityUnavailable,
                        Some(relative),
                        format!("workspace file identity is unavailable: {error}"),
                    ));
                    break;
                }
            };
            if let Some(previous_path) = file_identities.get(&file_identity) {
                errors.push(capture_issue(
                    WorkspaceManifestIssueKind::DuplicateIdentity,
                    Some(relative),
                    format!(
                        "workspace paths share one regular-file identity: previous='{previous_path}'"
                    ),
                ));
                break;
            }
            let unix_mode = captured_unix_mode(&handle_metadata_before);

            let mut file_digest = Sha256::new();
            let mut file_bytes = 0u64;
            let mut buffer = [0u8; 64 * 1024];
            loop {
                // The pre-open metadata check handles stable files. Limiting
                // every read as well keeps the physical I/O bound strict if a
                // file grows concurrently; the post-read metadata check then
                // reports that mutation without reading beyond `max_bytes`.
                let remaining = max_bytes
                    .saturating_sub(total_bytes_hashed)
                    .saturating_sub(file_bytes);
                if remaining == 0 {
                    break;
                }
                let read_limit = usize::try_from(remaining)
                    .unwrap_or(usize::MAX)
                    .min(buffer.len());
                let read = match std::io::Read::read(&mut file, &mut buffer[..read_limit]) {
                    Ok(read) => read,
                    Err(error) => {
                        errors.push(capture_issue(
                            WorkspaceManifestIssueKind::Read,
                            Some(relative.clone()),
                            format!("workspace file read failed: {error}"),
                        ));
                        break;
                    }
                };
                if read == 0 {
                    break;
                }
                file_bytes = file_bytes.saturating_add(read as u64);
                file_digest.update(&buffer[..read]);
            }
            if !errors.is_empty() {
                break;
            }

            let handle_metadata_after = match file.metadata() {
                Ok(metadata) => metadata,
                Err(error) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::Metadata,
                        Some(relative),
                        format!("workspace file metadata cannot be re-read: {error}"),
                    ));
                    break;
                }
            };
            let path_metadata_after = match std::fs::symlink_metadata(path) {
                Ok(metadata) if metadata.file_type().is_file() => metadata,
                Ok(_) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::ConcurrentMutation,
                        Some(relative),
                        "workspace file type changed while it was hashed",
                    ));
                    break;
                }
                Err(error) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::ConcurrentMutation,
                        Some(relative),
                        format!("workspace file disappeared while it was hashed: {error}"),
                    ));
                    break;
                }
            };
            if file_bytes != handle_metadata_after.len()
                || !same_file_revision(&handle_metadata_before, &handle_metadata_after)
                || !same_file_revision(&handle_metadata_after, &path_metadata_after)
            {
                errors.push(capture_issue(
                    WorkspaceManifestIssueKind::ConcurrentMutation,
                    Some(relative),
                    "workspace file changed while its content was hashed",
                ));
                break;
            }
            let handle_identity_after = match stable_file_identity(&file, &handle_metadata_after) {
                Ok(Some(identity)) => identity,
                Ok(None) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::FileIdentityUnavailable,
                        Some(relative),
                        "stable file identity became unavailable after hashing",
                    ));
                    break;
                }
                Err(error) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::FileIdentityUnavailable,
                        Some(relative),
                        format!("workspace file identity cannot be re-read: {error}"),
                    ));
                    break;
                }
            };
            let path_identity_after = match stable_path_identity(path, &path_metadata_after) {
                Ok(Some(identity)) => identity,
                Ok(None) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::FileIdentityUnavailable,
                        Some(relative),
                        "stable workspace path identity is unavailable after hashing",
                    ));
                    break;
                }
                Err(error) => {
                    errors.push(capture_issue(
                        WorkspaceManifestIssueKind::ConcurrentMutation,
                        Some(relative),
                        format!("workspace path identity cannot be confirmed: {error}"),
                    ));
                    break;
                }
            };
            if handle_identity_after != file_identity
                || path_identity_after != file_identity
                || captured_unix_mode(&handle_metadata_after) != unix_mode
                || captured_unix_mode(&path_metadata_after) != unix_mode
            {
                errors.push(capture_issue(
                    WorkspaceManifestIssueKind::ConcurrentMutation,
                    Some(relative),
                    "workspace file identity or mode changed while its content was hashed",
                ));
                break;
            }

            total_bytes_hashed = total_bytes_hashed.saturating_add(file_bytes);
            file_identities.insert(file_identity.clone(), relative.clone());
            files.insert(
                relative.clone(),
                WorkspaceFileRevision {
                    path: relative,
                    size_bytes: file_bytes,
                    content_sha256: format!("sha256:{}", hex::encode(file_digest.finalize())),
                    file_identity: Some(file_identity),
                    unix_mode,
                },
            );
        }

        match std::fs::canonicalize(root).and_then(|current_root| {
            std::fs::symlink_metadata(&current_root).map(|m| (current_root, m))
        }) {
            Ok((current_root, current_metadata)) if current_root == canonical_root => {
                match workspace_root_identity(&current_root, &current_metadata) {
                    Ok(Some(identity)) if identity == initial_root_identity => {}
                    Ok(Some(_)) => errors.push(capture_issue(
                        WorkspaceManifestIssueKind::RootChanged,
                        None,
                        "workspace root identity changed during manifest capture",
                    )),
                    Ok(None) => errors.push(capture_issue(
                        WorkspaceManifestIssueKind::RootIdentityUnavailable,
                        None,
                        "stable workspace-root identity became unavailable after capture",
                    )),
                    Err(error) => errors.push(capture_issue(
                        WorkspaceManifestIssueKind::RootIdentityUnavailable,
                        None,
                        format!("workspace-root identity cannot be confirmed: {error}"),
                    )),
                }
            }
            Ok(_) => errors.push(capture_issue(
                WorkspaceManifestIssueKind::RootChanged,
                None,
                "workspace root canonical path changed during manifest capture",
            )),
            Err(error) => errors.push(capture_issue(
                WorkspaceManifestIssueKind::RootChanged,
                None,
                format!("workspace root became unavailable during manifest capture: {error}"),
            )),
        }

        finish_effect_manifest(
            canonical_root,
            initial_root_identity,
            files,
            directories,
            total_bytes_hashed,
            errors,
        )
    }

    /// Add or update a single file entry by scanning the file on disk.
    pub fn add_or_update(&self, path: &str) -> Option<FileEntry> {
        let path_obj = Path::new(path);
        if self.is_excluded(path_obj) {
            self.remove(path);
            return None;
        }
        if !path_obj.is_file() {
            // File doesn't exist — treat as removal
            self.remove_internal(path);
            return None;
        }

        // Enforce maximum entries limit — only reject genuinely NEW entries
        if self.get_entry(path).is_none() {
            let mem = self.mem_cache.read();
            if mem.len() >= MAX_INVENTORY_ENTRIES {
                enforce_max_entries(mem.len());
                return None;
            }
        }

        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => return None,
        };

        let file_size = metadata.len();
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Discovery is metadata-only. Reading and hashing every file during
        // workspace initialization delayed TUI startup and duplicated the
        // ContentStore's lazy read path. The hash is populated when content is
        // actually consumed or when a change requires verification.
        let content_hash = String::new();

        let ext = path_obj
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_string();

        let language = classify_language(&ext).to_string();
        let state = FileState::Discovered;
        let version = 0;

        let entry = FileEntry {
            path: path.to_string(),
            file_size,
            file_ext: ext,
            language,
            mtime,
            content_hash,
            state,
            last_read_at: None,
            last_read_version: 0,
            current_version: version,
            read_count: 0,
        };

        self.store_entry(&entry);
        self.sync_to_l2(&entry);

        debug!(path = %path, "FileInventory: file added/updated");
        Some(entry)
    }

    /// Mark a file as stale (externally modified).
    pub fn mark_stale(&self, path: &str) {
        if self.is_excluded(Path::new(path)) {
            self.remove(path);
            return;
        }
        let mut mem = self.mem_cache.write();
        if let Some(entry) = mem.get_mut(path) {
            entry.state = FileState::ReadStale;
            // Keep invalidation metadata-only. ContentStore computes the hash
            // lazily on the next targeted read.
            entry.content_hash.clear();
            if let Ok(meta) = std::fs::metadata(path) {
                if let Ok(t) = meta.modified() {
                    if let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                        entry.mtime = d.as_millis() as i64;
                    }
                }
                entry.file_size = meta.len();
            }
            entry.current_version += 1;
            let cloned = entry.clone();
            drop(mem);
            self.persist_to_cache(&cloned);
            self.sync_to_l2(&cloned);
            debug!(path = %path, version = cloned.current_version, "FileInventory: marked stale");
        }
    }

    /// Mark a file as read (fresh).
    pub fn mark_read(&self, path: &str, version: u64) {
        self.mark_read_with_hash(path, version, None);
    }

    pub fn mark_read_with_hash(&self, path: &str, version: u64, content_hash: Option<String>) {
        if self.is_excluded(Path::new(path)) {
            self.remove(path);
            return;
        }
        let mut mem = self.mem_cache.write();
        if let Some(entry) = mem.get_mut(path) {
            entry.state = FileState::ReadFresh;
            entry.last_read_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64,
            );
            entry.last_read_version = version;
            if let Some(content_hash) = content_hash {
                entry.content_hash = content_hash;
            }
            entry.read_count += 1;
            let cloned = entry.clone();
            drop(mem);
            self.persist_to_cache(&cloned);
            self.sync_to_l2(&cloned);
        }
    }

    /// Mark a file as written (but not re-read) by the agent.
    pub fn mark_written(&self, path: &str) {
        if self.is_excluded(Path::new(path)) {
            self.remove(path);
            return;
        }
        let mut mem = self.mem_cache.write();
        if let Some(entry) = mem.get_mut(path) {
            entry.state = FileState::WrittenUnread;
            entry.current_version += 1;
            let cloned = entry.clone();
            drop(mem);
            self.persist_to_cache(&cloned);
            self.sync_to_l2(&cloned);
            debug!(path = %path, "FileInventory: marked written_unread");
        } else {
            // New file written by agent
            drop(mem);
            self.add_or_update(path);
        }
    }

    /// Mark a file as externally read (e.g., via read_full_result micro-tool).
    /// Lightweight: no disk I/O, just updates in-memory state so subsequent file_read calls
    /// recognize the file as already-read and return cached/diff response instead of full content.
    pub fn mark_external_read(&self, path: &str) {
        if self.is_excluded(Path::new(path)) {
            self.remove(path);
            return;
        }
        let mut mem = self.mem_cache.write();
        if let Some(entry) = mem.get_mut(path) {
            entry.state = FileState::ReadFresh;
            entry.read_count += 1;
            entry.last_read_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64,
            );
            let cloned = entry.clone();
            drop(mem);
            self.persist_to_cache(&cloned);
            self.sync_to_l2(&cloned);
        } else {
            // Entry doesn't exist yet — add a minimal placeholder
            drop(mem);
            let minimal = FileEntry {
                path: path.to_string(),
                file_size: 0,
                file_ext: Path::new(path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_string(),
                language: "unknown".to_string(),
                mtime: 0,
                content_hash: String::new(),
                state: FileState::ReadFresh,
                last_read_at: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as i64,
                ),
                last_read_version: 0,
                current_version: 0,
                read_count: 1,
            };
            self.store_entry(&minimal);
        }
    }

    /// Remove a file from the inventory (e.g., on deletion).
    pub fn remove(&self, path: &str) -> bool {
        self.remove_internal(path);
        self.remove_from_l2(path);
        debug!(path = %path, "FileInventory: file removed");
        true
    }

    /// Clear all tracked files from the inventory.
    pub fn clear_all(&self) {
        let mut mem = self.mem_cache.write();
        mem.clear();
    }

    /// Get a file entry by path.
    pub fn get_entry(&self, path: &str) -> Option<FileEntry> {
        if self.is_excluded(Path::new(path)) {
            return None;
        }
        let mem = self.mem_cache.read();
        mem.get(path).cloned()
    }

    /// Update exclude patterns after construction (e.g., after WatchEngine loads .gitignore).
    pub fn set_exclude_patterns(&self, patterns: Vec<String>) {
        let mut ep = self.exclude_patterns.write();
        for p in patterns {
            if !ep.contains(&p) {
                ep.push(p);
            }
        }
        drop(ep);
        self.purge_excluded_entries();
    }

    /// List all files matching a state filter.
    pub fn list_by_state(&self, state: FileState) -> Vec<FileEntry> {
        let mem = self.mem_cache.read();
        mem.values()
            .filter(|entry| entry.state == state && !self.is_excluded(Path::new(&entry.path)))
            .cloned()
            .collect()
    }

    /// List all tracked files.
    pub fn list_all(&self) -> Vec<FileEntry> {
        let mem = self.mem_cache.read();
        mem.values()
            .filter(|entry| !self.is_excluded(Path::new(&entry.path)))
            .cloned()
            .collect()
    }

    /// List files under a directory prefix.
    pub fn list_dir(&self, dir_prefix: &str) -> Vec<FileEntry> {
        let prefix = if dir_prefix.ends_with('/') {
            dir_prefix.to_string()
        } else {
            format!("{}/", dir_prefix)
        };
        let mem = self.mem_cache.read();
        mem.values()
            .filter(|e| {
                !self.is_excluded(Path::new(&e.path))
                    && (e.path.starts_with(&prefix) || e.path.starts_with(dir_prefix))
            })
            .cloned()
            .collect()
    }

    /// Count files by state.
    pub fn state_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        let mem = self.mem_cache.read();
        for entry in mem
            .values()
            .filter(|entry| !self.is_excluded(Path::new(&entry.path)))
        {
            *counts.entry(entry.state.as_str().to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Total number of tracked files.
    pub fn total_count(&self) -> usize {
        self.mem_cache
            .read()
            .values()
            .filter(|entry| !self.is_excluded(Path::new(&entry.path)))
            .count()
    }

    /// Get stale files (for prompting re-read).
    pub fn stale_files(&self) -> Vec<FileEntry> {
        self.list_by_state(FileState::ReadStale)
    }

    /// Get files with state ReadStale or WrittenUnread.
    pub fn unread_files(&self) -> Vec<FileEntry> {
        let mut result = self.list_by_state(FileState::ReadStale);
        result.extend(self.list_by_state(FileState::WrittenUnread));
        result
    }

    // ── Private helpers ──

    pub(crate) fn is_excluded(&self, path: &std::path::Path) -> bool {
        let ep = self.exclude_patterns.read();
        is_excluded_with_policy(
            self.exact_exclude_workspace_root.as_deref(),
            path,
            &ep,
            &self.exact_exclude_paths,
        )
    }

    fn add_entry(&self, path: &str) -> Option<FileEntry> {
        if self.is_excluded(Path::new(path)) {
            self.remove(path);
            return None;
        }
        // Enforce maximum entries limit
        {
            let mem = self.mem_cache.read();
            if mem.len() >= MAX_INVENTORY_ENTRIES {
                enforce_max_entries(mem.len());
                return None;
            }
        }

        let path_obj = Path::new(path);
        // Initial discovery is metadata-only; ContentStore hashes lazily on
        // the first task-relevant read.
        let content_hash = String::new();
        let metadata = std::fs::metadata(path).ok()?;

        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let ext = path_obj
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_string();

        let language = classify_language(&ext).to_string();

        let entry = FileEntry {
            path: path.to_string(),
            file_size: metadata.len(),
            file_ext: ext,
            language,
            mtime,
            content_hash,
            state: FileState::Discovered,
            last_read_at: None,
            last_read_version: 0,
            current_version: 0,
            read_count: 0,
        };

        self.store_entry(&entry);
        self.sync_to_l2(&entry);
        Some(entry)
    }

    /// Remove entries that became invisible after ignore-policy changes, or
    /// that were persisted by an older version which did not enforce the
    /// workspace-runtime boundary on incremental events.
    fn purge_excluded_entries(&self) -> usize {
        let paths = {
            let mem = self.mem_cache.read();
            mem.values()
                .filter(|entry| self.is_excluded(Path::new(&entry.path)))
                .map(|entry| entry.path.clone())
                .collect::<Vec<_>>()
        };
        for path in &paths {
            self.remove(path);
        }
        if !paths.is_empty() {
            debug!(
                count = paths.len(),
                "FileInventory: purged excluded persisted entries"
            );
        }
        paths.len()
    }

    fn store_entry(&self, entry: &FileEntry) {
        // In-memory cache
        {
            let mut mem = self.mem_cache.write();
            mem.insert(entry.path.clone(), entry.clone());
        }

        // redb persistence
        if let Some(ref db) = self.cache {
            if let Ok(encoded) = serde_json::to_vec(entry) {
                if let Ok(write_txn) = db.begin_write() {
                    if let Ok(mut table) = write_txn.open_table(INVENTORY_CACHE) {
                        if let Err(e) = table.insert(entry.path.as_str(), encoded.as_slice()) {
                            warn!(path = %entry.path, error = %e, "FileInventory: redb insert failed");
                        }
                    }
                    let _ = write_txn.commit();
                }
            }
        }
    }

    fn remove_internal(&self, path: &str) {
        {
            let mut mem = self.mem_cache.write();
            mem.remove(path);
        }
        if let Some(ref db) = self.cache {
            if let Ok(write_txn) = db.begin_write() {
                if let Ok(mut table) = write_txn.open_table(INVENTORY_CACHE) {
                    let _ = table.remove(path);
                }
                let _ = write_txn.commit();
            }
        }
    }

    fn persist_to_cache(&self, entry: &FileEntry) {
        if let Some(ref db) = self.cache {
            if let Ok(encoded) = serde_json::to_vec(entry) {
                if let Ok(write_txn) = db.begin_write() {
                    if let Ok(mut table) = write_txn.open_table(INVENTORY_CACHE) {
                        if let Err(e) = table.insert(entry.path.as_str(), encoded.as_slice()) {
                            warn!(path = %entry.path, error = %e, "FileInventory: redb persist failed");
                        }
                    }
                    let _ = write_txn.commit();
                }
            }
        }
    }

    fn sync_to_l2(&self, entry: &FileEntry) {
        let blackboard = match self.blackboard.as_ref() {
            Some(b) => b,
            None => return,
        };

        let iri = entry.iri();
        let parent_dir_iri = entry.parent_dir_iri();

        let json_ld = serde_json::json!({
            "@id": &iri,
            "@type": ["ws:File"],
            "ws:filePath": entry.path,
            "ws:fileSize": entry.file_size,
            "ws:fileExt": entry.file_ext,
            "ws:language": entry.language,
            "ws:mtime": entry.mtime,
            "ws:contentHash": entry.content_hash,
            "ws:state": entry.state.as_str(),
            "ws:lastReadAt": entry.last_read_at.unwrap_or(0),
            "ws:lastReadVersion": entry.last_read_version,
            "ws:currentVersion": entry.current_version,
            "ws:readCount": entry.read_count,
            "ws:parentDir": parent_dir_iri,
        });

        let config = crate::CoreConfig {
            max_node_size: 65536,
            ..crate::CoreConfig::default()
        };

        if let Err(e) =
            blackboard.write_node_to_graph(&iri, &json_ld.to_string(), WORKSPACE_GRAPH, &config)
        {
            warn!(path = %entry.path, error = %e, "FileInventory: L2 sync failed");
        }
    }

    fn remove_from_l2(&self, path: &str) {
        let blackboard = match self.blackboard.as_ref() {
            Some(b) => b,
            None => return,
        };

        let iri = format!("iri://workspace/file/{}", path);
        let _ = blackboard.delete_node(&iri);
    }
}

/// Match a path against a gitignore-style glob pattern.
///
/// Supports:
/// - `*.ext` (extension glob)
/// - `name/` (directory prefix)
/// - `path/name` (exact path segment)
/// - Substring match as fallback (backward compatibility)
pub fn match_glob_pattern(path: &str, pattern: &str) -> bool {
    // Exact match
    if path == pattern || path.trim_end_matches('/') == pattern.trim_end_matches('/') {
        return true;
    }

    // Directory prefix: pattern like "node_modules/" or "target/"
    let pat_dir = if pattern.ends_with('/') {
        pattern.to_string()
    } else if !pattern.contains('.') {
        format!("{}/", pattern)
    } else {
        // Check if it's a glob extension pattern like *.log
        if let Some(ext) = pattern.strip_prefix("*.") {
            if ext.contains('*') || ext.contains('/') {
                // Complex pattern — fall back to substring match
                return path.contains(pattern) || path.ends_with(pattern);
            }
            // Match file extension
            let ext_dot = format!(".{}", ext);
            return path.ends_with(&ext_dot) || path.contains(&format!("/{}", ext_dot));
        }
        return path.contains(pattern) || path.ends_with(pattern);
    };

    // Directory match: the path starts with the dir or contains /<dir> or ends with /<dir>
    path.starts_with(&pat_dir)
        || path.contains(&format!("/{}", pat_dir))
        || path.ends_with(&format!("/{}", pat_dir.trim_end_matches('/')))
}

fn enforce_max_entries(count: usize) {
    if count >= MAX_INVENTORY_ENTRIES {
        warn!(
            "FileInventory has reached {} entries (max {}). Further files will not be tracked.",
            count, MAX_INVENTORY_ENTRIES
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_workspace(dir: &TempDir, files: &[(&str, &str)]) {
        for (name, content) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&path, content).unwrap();
        }
    }

    #[test]
    fn test_full_scan() {
        let dir = TempDir::new().unwrap();
        create_workspace(
            &dir,
            &[
                ("src/main.rs", "fn main() {}"),
                ("src/lib.rs", "pub fn hello() {}"),
                ("README.md", "# Hello"),
            ],
        );

        let inventory = FileInventory::new(None, None, vec![]);
        let count = inventory.full_scan(&dir.path().to_string_lossy());
        assert_eq!(count, 3);
    }

    #[test]
    fn full_scan_reconciles_offline_modification_and_deletion_without_reading_content() {
        let dir = TempDir::new().unwrap();
        create_workspace(&dir, &[("changed.rs", "v1"), ("deleted.rs", "gone")]);
        let changed = dir.path().join("changed.rs");
        let deleted = dir.path().join("deleted.rs");
        let inventory = FileInventory::new(None, None, vec![]);
        assert_eq!(inventory.full_scan(&dir.path().to_string_lossy()), 2);
        inventory.mark_read(&changed.to_string_lossy(), 0);

        std::fs::write(&changed, "version two is larger").unwrap();
        std::fs::remove_file(&deleted).unwrap();
        assert_eq!(inventory.full_scan(&dir.path().to_string_lossy()), 0);

        let changed_entry = inventory.get_entry(&changed.to_string_lossy()).unwrap();
        assert_eq!(changed_entry.state, FileState::ReadStale);
        assert!(changed_entry.content_hash.is_empty());
        assert!(inventory.get_entry(&deleted.to_string_lossy()).is_none());
    }

    #[test]
    fn test_exclude_pattern() {
        let dir = TempDir::new().unwrap();
        create_workspace(
            &dir,
            &[
                ("src/main.rs", "fn main() {}"),
                ("node_modules/pkg/index.js", "module.exports = {};"),
                ("target/debug/app", "binary"),
            ],
        );

        let inventory =
            FileInventory::new(None, None, vec!["node_modules/".into(), "target/".into()]);
        let count = inventory.full_scan(&dir.path().to_string_lossy());
        assert_eq!(count, 1);
    }

    #[test]
    fn effect_manifest_diff_reports_exact_content_and_directory_changes() {
        let dir = TempDir::new().unwrap();
        create_workspace(
            &dir,
            &[
                ("unchanged.txt", "stable"),
                ("modified.txt", "AAAA"),
                ("removed.txt", "removed"),
                ("old_name.txt", "renamed"),
            ],
        );
        std::fs::create_dir(dir.path().join("empty_removed")).unwrap();
        std::fs::create_dir(dir.path().join("stable_dir")).unwrap();
        let inventory = FileInventory::new(None, None, vec![]);

        let before = inventory.capture_effect_manifest(dir.path(), 100, 1_000_000);
        assert!(before.complete, "{:?}", before.errors);
        assert!(before.files.iter().all(|file| {
            valid_normalized_relative_path(&file.path)
                && !Path::new(&file.path).is_absolute()
                && valid_sha256(&file.content_sha256)
        }));

        // Same byte length and potentially the same coarse filesystem mtime:
        // content hashing, rather than metadata, must identify this change.
        std::fs::write(dir.path().join("modified.txt"), "BBBB").unwrap();
        std::fs::remove_file(dir.path().join("removed.txt")).unwrap();
        std::fs::rename(
            dir.path().join("old_name.txt"),
            dir.path().join("renamed.txt"),
        )
        .unwrap();
        std::fs::remove_dir(dir.path().join("empty_removed")).unwrap();
        create_workspace(&dir, &[("new/nested/created.txt", "created")]);

        let after = inventory.capture_effect_manifest(dir.path(), 100, 1_000_000);
        assert!(after.complete, "{:?}", after.errors);
        assert_ne!(before.digest, after.digest);

        let delta = WorkspaceEffectDelta::between(&before, &after);
        assert!(delta.complete, "{:?}", delta.errors);
        assert_eq!(
            delta
                .files_created
                .iter()
                .map(|file| file.path.as_str())
                .collect::<Vec<_>>(),
            vec!["new/nested/created.txt", "renamed.txt"]
        );
        assert_eq!(
            delta
                .files_modified
                .iter()
                .map(|file| file.path.as_str())
                .collect::<Vec<_>>(),
            vec!["modified.txt"]
        );
        assert_ne!(
            delta.files_modified[0].before_content_sha256,
            delta.files_modified[0].after_content_sha256
        );
        assert_eq!(
            delta
                .files_removed
                .iter()
                .map(|file| file.path.as_str())
                .collect::<Vec<_>>(),
            vec!["old_name.txt", "removed.txt"]
        );
        assert_eq!(
            delta.directories_created,
            vec!["new".to_string(), "new/nested".to_string()]
        );
        assert_eq!(delta.directories_removed, vec!["empty_removed".to_string()]);

        let unchanged = inventory.capture_effect_manifest(dir.path(), 100, 1_000_000);
        let no_delta = WorkspaceEffectDelta::between(&after, &unchanged);
        assert!(no_delta.complete);
        assert_eq!(no_delta.before_digest, no_delta.after_digest);
        assert!(no_delta.files_created.is_empty());
        assert!(no_delta.files_modified.is_empty());
        assert!(no_delta.files_removed.is_empty());
        assert!(no_delta.directories_created.is_empty());
        assert!(no_delta.directories_removed.is_empty());
    }

    #[test]
    fn effect_manifest_hard_excludes_configured_and_runtime_trees() {
        let dir = TempDir::new().unwrap();
        create_workspace(
            &dir,
            &[
                ("visible.txt", "visible"),
                ("ignored/hidden.txt", "before"),
                (".gliding_horse/state/data", "before"),
                ("nested/.gliding_horse/state/data", "before"),
            ],
        );
        let inventory = FileInventory::new(None, None, vec!["ignored/".into()]);
        let before = inventory.capture_effect_manifest(dir.path(), 100, 1_000_000);
        assert!(before.complete, "{:?}", before.errors);
        assert_eq!(
            before
                .files
                .iter()
                .map(|file| file.path.as_str())
                .collect::<Vec<_>>(),
            vec!["visible.txt"]
        );
        assert!(
            before
                .directories
                .iter()
                .all(|path| !path.contains("ignored") && !path.contains(".gliding_horse")),
            "{:?}",
            before.directories
        );

        std::fs::write(dir.path().join("ignored/hidden.txt"), "after").unwrap();
        std::fs::write(dir.path().join(".gliding_horse/state/data"), "after").unwrap();
        let after = inventory.capture_effect_manifest(dir.path(), 100, 1_000_000);
        let delta = WorkspaceEffectDelta::between(&before, &after);
        assert!(delta.complete, "{:?}", delta.errors);
        assert_eq!(before.digest, after.digest);
        assert!(delta.files_created.is_empty());
        assert!(delta.files_modified.is_empty());
        assert!(delta.files_removed.is_empty());
    }

    #[test]
    fn effect_manifest_limits_and_incomplete_diffs_fail_closed() {
        let dir = TempDir::new().unwrap();
        create_workspace(&dir, &[("a.txt", "aa"), ("b.txt", "bb")]);
        let inventory = FileInventory::new(None, None, vec![]);

        let file_limited = inventory.capture_effect_manifest(dir.path(), 1, 1_000_000);
        assert!(!file_limited.complete);
        assert_eq!(
            file_limited.errors[0].kind,
            WorkspaceManifestIssueKind::FileLimitExceeded
        );
        assert!(file_limited.files.len() <= 1);
        assert!(valid_sha256(&file_limited.digest));

        let byte_limited = inventory.capture_effect_manifest(dir.path(), 100, 1);
        assert!(!byte_limited.complete);
        assert_eq!(
            byte_limited.errors[0].kind,
            WorkspaceManifestIssueKind::ByteLimitExceeded
        );
        assert_eq!(byte_limited.total_bytes_hashed, 0);

        let complete = inventory.capture_effect_manifest(dir.path(), 100, 1_000_000);
        let delta = WorkspaceEffectDelta::between(&file_limited, &complete);
        assert!(!delta.complete);
        assert!(delta.files_created.is_empty());
        assert!(delta.files_modified.is_empty());
        assert!(delta.files_removed.is_empty());
        assert!(delta.errors.iter().any(|issue| {
            issue.stage == WorkspaceManifestIssueStage::Before
                && issue.kind == WorkspaceManifestIssueKind::FileLimitExceeded
        }));

        std::fs::remove_file(dir.path().join("a.txt")).unwrap();
        std::fs::remove_file(dir.path().join("b.txt")).unwrap();
        std::fs::create_dir(dir.path().join("d1")).unwrap();
        std::fs::create_dir(dir.path().join("d2")).unwrap();
        let directory_limited = inventory.capture_effect_manifest(dir.path(), 1, 1_000_000);
        assert!(!directory_limited.complete);
        assert_eq!(
            directory_limited.errors[0].kind,
            WorkspaceManifestIssueKind::DirectoryLimitExceeded
        );
        assert!(directory_limited.directories.len() <= 1);
    }

    #[test]
    fn effect_manifest_rejects_corrupt_or_cross_root_inputs() {
        let first_root = TempDir::new().unwrap();
        let second_root = TempDir::new().unwrap();
        create_workspace(&first_root, &[("same.txt", "same")]);
        create_workspace(&second_root, &[("same.txt", "same")]);
        let inventory = FileInventory::new(None, None, vec![]);
        let first = inventory.capture_effect_manifest(first_root.path(), 10, 1_000);
        let second = inventory.capture_effect_manifest(second_root.path(), 10, 1_000);

        let cross_root = WorkspaceEffectDelta::between(&first, &second);
        assert!(!cross_root.complete);
        assert!(cross_root
            .errors
            .iter()
            .any(|issue| issue.kind == WorkspaceManifestIssueKind::RootMismatch));

        let mut corrupt = first.clone();
        corrupt.digest = format!("sha256:{}", "0".repeat(64));
        let corrupt_delta = WorkspaceEffectDelta::between(&first, &corrupt);
        assert!(!corrupt_delta.complete);
        assert!(corrupt_delta
            .errors
            .iter()
            .any(|issue| issue.kind == WorkspaceManifestIssueKind::DigestMismatch));

        let mut corrupt_total = first.clone();
        corrupt_total.total_bytes_hashed += 1;
        corrupt_total.digest = manifest_digest(
            &corrupt_total.workspace_root_identity,
            &corrupt_total.files,
            &corrupt_total.directories,
            corrupt_total.total_bytes_hashed,
            corrupt_total.complete,
        );
        let corrupt_total_delta = WorkspaceEffectDelta::between(&first, &corrupt_total);
        assert!(!corrupt_total_delta.complete);
        assert!(corrupt_total_delta.errors.iter().any(|issue| {
            issue.kind == WorkspaceManifestIssueKind::DigestMismatch
                && issue.message.contains("byte total mismatch")
        }));

        let mut file_directory_collision = first.clone();
        file_directory_collision
            .directories
            .push(file_directory_collision.files[0].path.clone());
        file_directory_collision.directories.sort();
        file_directory_collision.digest = manifest_digest(
            &file_directory_collision.workspace_root_identity,
            &file_directory_collision.files,
            &file_directory_collision.directories,
            file_directory_collision.total_bytes_hashed,
            file_directory_collision.complete,
        );
        let collision_delta = WorkspaceEffectDelta::between(&first, &file_directory_collision);
        assert!(!collision_delta.complete);
        assert!(collision_delta.errors.iter().any(|issue| {
            issue.kind == WorkspaceManifestIssueKind::DuplicatePath
                && issue.path.as_deref() == Some("same.txt")
        }));

        let mut unsafe_path = first.clone();
        unsafe_path.files[0].path = "../escape.txt".to_string();
        unsafe_path.digest = manifest_digest(
            &unsafe_path.workspace_root_identity,
            &unsafe_path.files,
            &unsafe_path.directories,
            unsafe_path.total_bytes_hashed,
            unsafe_path.complete,
        );
        let unsafe_path_delta = WorkspaceEffectDelta::between(&first, &unsafe_path);
        assert!(!unsafe_path_delta.complete);
        assert!(unsafe_path_delta.errors.iter().any(|issue| {
            issue.kind == WorkspaceManifestIssueKind::InvalidRelativePath
                && issue.path.as_deref() == Some("../escape.txt")
        }));
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn effect_manifest_detects_workspace_root_replacement_at_same_path() {
        let parent = TempDir::new().unwrap();
        let root = parent.path().join("workspace");
        let old_root = parent.path().join("old-workspace");
        std::fs::create_dir(&root).unwrap();
        let inventory = FileInventory::new(None, None, vec![]);

        let before = inventory.capture_effect_manifest(&root, 10, 1_000);
        assert!(before.complete, "{:?}", before.errors);
        std::fs::rename(&root, &old_root).unwrap();
        std::fs::create_dir(&root).unwrap();
        let after = inventory.capture_effect_manifest(&root, 10, 1_000);
        assert!(after.complete, "{:?}", after.errors);

        let delta = WorkspaceEffectDelta::between(&before, &after);
        assert!(!delta.complete);
        assert!(delta
            .errors
            .iter()
            .any(|issue| issue.kind == WorkspaceManifestIssueKind::RootMismatch));
    }

    #[cfg(unix)]
    #[test]
    fn effect_manifest_fails_closed_on_symlink_entries() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().unwrap();
        create_workspace(&dir, &[("target.txt", "target")]);
        symlink("target.txt", dir.path().join("alias.txt")).unwrap();
        let inventory = FileInventory::new(None, None, vec![]);
        let manifest = inventory.capture_effect_manifest(dir.path(), 10, 1_000);

        assert!(!manifest.complete);
        assert!(manifest.errors.iter().any(|issue| {
            issue.kind == WorkspaceManifestIssueKind::UnsupportedEntryType
                && issue.path.as_deref() == Some("alias.txt")
        }));
    }

    #[cfg(unix)]
    #[test]
    fn effect_manifest_detects_mode_and_file_identity_changes() {
        use std::os::unix::fs::PermissionsExt;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tracked.txt");
        std::fs::write(&path, "stable bytes").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        let inventory = FileInventory::new(None, None, vec![]);

        let before = inventory.capture_effect_manifest(dir.path(), 10, 1_000);
        assert!(before.complete, "{:?}", before.errors);
        assert!(before.files[0]
            .file_identity
            .as_deref()
            .is_some_and(valid_sha256));
        assert_eq!(before.files[0].unix_mode, Some(0o640));

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let mode_after = inventory.capture_effect_manifest(dir.path(), 10, 1_000);
        let mode_delta = WorkspaceEffectDelta::between(&before, &mode_after);
        assert!(mode_delta.complete, "{:?}", mode_delta.errors);
        assert_eq!(mode_delta.files_modified.len(), 1);
        let mode_change = &mode_delta.files_modified[0];
        assert_eq!(
            mode_change.before_content_sha256,
            mode_change.after_content_sha256
        );
        assert_eq!(
            mode_change.before_file_identity,
            mode_change.after_file_identity
        );
        assert_eq!(mode_change.before_unix_mode, Some(0o640));
        assert_eq!(mode_change.after_unix_mode, Some(0o600));

        let replacement = dir.path().join("replacement.tmp");
        std::fs::write(&replacement, "stable bytes").unwrap();
        std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        let identity_after = inventory.capture_effect_manifest(dir.path(), 10, 1_000);
        let identity_delta = WorkspaceEffectDelta::between(&mode_after, &identity_after);
        assert!(identity_delta.complete, "{:?}", identity_delta.errors);
        assert_eq!(identity_delta.files_modified.len(), 1);
        let identity_change = &identity_delta.files_modified[0];
        assert_eq!(
            identity_change.before_content_sha256,
            identity_change.after_content_sha256
        );
        assert_eq!(
            identity_change.before_unix_mode,
            identity_change.after_unix_mode
        );
        assert_ne!(
            identity_change.before_file_identity,
            identity_change.after_file_identity
        );
    }

    #[cfg(unix)]
    #[test]
    fn effect_manifest_fails_closed_on_hardlink_aliases() {
        let dir = TempDir::new().unwrap();
        let first = dir.path().join("a.txt");
        let alias = dir.path().join("b.txt");
        std::fs::write(&first, "shared inode").unwrap();
        std::fs::hard_link(&first, &alias).unwrap();
        let inventory = FileInventory::new(None, None, vec![]);

        let manifest = inventory.capture_effect_manifest(dir.path(), 10, 1_000);

        assert!(!manifest.complete);
        assert!(manifest.errors.iter().any(|issue| {
            issue.kind == WorkspaceManifestIssueKind::DuplicateIdentity
                && issue.path.as_deref() == Some("b.txt")
                && issue.message.contains("a.txt")
        }));
    }

    #[test]
    fn workspace_runtime_tree_is_hard_excluded_from_empty_workspace() {
        let dir = TempDir::new().unwrap();
        create_workspace(
            &dir,
            &[
                (".gliding_horse/ws_monitor/metadata", "runtime metadata"),
                (".gliding_horse/ws_monitor/content", "runtime content"),
                ("nested/.gliding_horse/cache/blob", "nested runtime"),
            ],
        );

        // An explicitly empty configurable ignore list must not disable the
        // process-state boundary.
        let inventory = FileInventory::new(None, None, vec![]);
        assert_eq!(inventory.full_scan(&dir.path().to_string_lossy()), 0);
        assert_eq!(inventory.total_count(), 0);
        assert!(inventory.list_all().is_empty());

        let runtime_file = dir.path().join(".gliding_horse/ws_monitor/content");
        assert!(inventory
            .add_or_update(&runtime_file.to_string_lossy())
            .is_none());
        assert!(inventory
            .get_entry(&runtime_file.to_string_lossy())
            .is_none());
    }

    #[test]
    fn runtime_path_boundary_matches_components_not_similar_names() {
        assert!(is_workspace_runtime_path(Path::new(
            "/workspace/.gliding_horse/ws_monitor/content"
        )));
        assert!(is_workspace_runtime_path(Path::new(
            "/workspace/project/.gliding_horse/nested/state"
        )));
        assert!(is_workspace_runtime_path(Path::new(
            r"C:\workspace\.gliding_horse\ws_monitor\metadata"
        )));
        assert!(!is_workspace_runtime_path(Path::new(
            "/workspace/.gliding_horse_notes/report.md"
        )));
        assert!(!is_workspace_runtime_path(Path::new(
            "/workspace/src/gliding_horse/module.rs"
        )));
    }

    #[test]
    fn constructor_purges_legacy_runtime_entries_from_persistent_cache() {
        let dir = TempDir::new().unwrap();
        let runtime_file = dir.path().join(".gliding_horse/ws_monitor/content");
        create_workspace(
            &dir,
            &[(".gliding_horse/ws_monitor/content", "legacy runtime")],
        );
        let db_path = dir.path().join("legacy-inventory.redb");
        let db = Database::create(&db_path).unwrap();
        let legacy = FileEntry {
            path: runtime_file.to_string_lossy().to_string(),
            file_size: 14,
            file_ext: String::new(),
            language: "unknown".into(),
            mtime: 0,
            content_hash: String::new(),
            state: FileState::Discovered,
            last_read_at: None,
            last_read_version: 0,
            current_version: 0,
            read_count: 0,
        };
        {
            let write_txn = db.begin_write().unwrap();
            {
                let mut table = write_txn.open_table(INVENTORY_CACHE).unwrap();
                let encoded = serde_json::to_vec(&legacy).unwrap();
                table
                    .insert(legacy.path.as_str(), encoded.as_slice())
                    .unwrap();
            }
            write_txn.commit().unwrap();
        }

        let inventory = FileInventory::new(None, Some(db), vec![]);
        assert!(inventory.mem_cache.read().is_empty());
        let read_txn = inventory.cache.as_ref().unwrap().begin_read().unwrap();
        let table = read_txn.open_table(INVENTORY_CACHE).unwrap();
        assert!(table.get(legacy.path.as_str()).unwrap().is_none());
    }

    #[test]
    fn test_add_and_get() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        std::fs::write(&path, "fn test() {}").unwrap();

        let inventory = FileInventory::new(None, None, vec![]);
        let entry = inventory.add_or_update(&path.to_string_lossy()).unwrap();

        assert_eq!(entry.file_ext, "rs");
        assert_eq!(entry.language, "rust");
        assert_eq!(entry.state, FileState::Discovered);

        let fetched = inventory.get_entry(&path.to_string_lossy()).unwrap();
        assert_eq!(fetched.path, entry.path);
    }

    #[test]
    fn test_mark_stale_and_read() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        std::fs::write(&path, "v1").unwrap();

        let inventory = FileInventory::new(None, None, vec![]);
        inventory.add_or_update(&path.to_string_lossy());

        inventory.mark_read(&path.to_string_lossy(), 0);
        assert_eq!(
            inventory.get_entry(&path.to_string_lossy()).unwrap().state,
            FileState::ReadFresh
        );

        inventory.mark_stale(&path.to_string_lossy());
        assert_eq!(
            inventory.get_entry(&path.to_string_lossy()).unwrap().state,
            FileState::ReadStale
        );
    }

    #[test]
    fn test_list_by_state() {
        let dir = TempDir::new().unwrap();
        let p1 = dir.path().join("a.rs");
        let p2 = dir.path().join("b.rs");
        std::fs::write(&p1, "a").unwrap();
        std::fs::write(&p2, "b").unwrap();

        let inventory = FileInventory::new(None, None, vec![]);
        inventory.add_or_update(&p1.to_string_lossy());
        inventory.add_or_update(&p2.to_string_lossy());
        inventory.mark_read(&p1.to_string_lossy(), 0);

        assert_eq!(inventory.list_by_state(FileState::ReadFresh).len(), 1);
        assert_eq!(inventory.list_by_state(FileState::Discovered).len(), 1);
    }

    #[test]
    fn test_state_counts() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("f.rs");
        std::fs::write(&p, "fn main() {}").unwrap();

        let inventory = FileInventory::new(None, None, vec![]);
        inventory.add_or_update(&p.to_string_lossy());
        inventory.mark_read(&p.to_string_lossy(), 0);

        let counts = inventory.state_counts();
        assert_eq!(*counts.get("read_fresh").unwrap_or(&0), 1);
    }

    // ── match_glob_pattern tests ──

    #[test]
    fn test_glob_exact_match() {
        assert!(match_glob_pattern(
            "node_modules/pkg/index.js",
            "node_modules/"
        ));
        assert!(match_glob_pattern("target/debug/app", "target/"));
        assert!(match_glob_pattern("src/main.rs", "src/main.rs"));
    }

    #[test]
    fn test_glob_extension_wildcard() {
        // *.log should match .log files
        assert!(match_glob_pattern("server.log", "*.log"));
        assert!(match_glob_pattern("logs/app.log", "*.log"));
        assert!(match_glob_pattern("src/error.log", "*.log"));
        // Should NOT match files without .log ending
        assert!(!match_glob_pattern("server.logger", "*.log"));
        assert!(!match_glob_pattern("log.txt", "*.log"));
        assert!(!match_glob_pattern("logs", "*.log"));
    }

    #[test]
    fn test_glob_extension_wildcard_other() {
        // *.rs should match .rs files
        assert!(match_glob_pattern("src/main.rs", "*.rs"));
        assert!(match_glob_pattern("lib.rs", "*.rs"));
        assert!(!match_glob_pattern("main.rs.bak", "*.rs"));
    }

    #[test]
    fn test_glob_directory_prefix() {
        // Directory patterns - with trailing slash
        assert!(match_glob_pattern(
            "node_modules/pkg/index.js",
            "node_modules/"
        ));
        assert!(match_glob_pattern(
            "project/node_modules/pkg.js",
            "node_modules/"
        ));

        // Without trailing slash, no dot → treated as directory
        assert!(match_glob_pattern(
            "node_modules/pkg/index.js",
            "node_modules"
        ));
        assert!(match_glob_pattern("build/output.o", "build"));

        // Should NOT match partial directory names
        assert!(!match_glob_pattern(
            "src/node_modules_test/helper.js",
            "node_modules/"
        ));
    }

    #[test]
    fn test_glob_path_specific() {
        // Exact path segments in the middle
        assert!(match_glob_pattern(
            "/home/user/project/data/rag/doc.json",
            "data/"
        ));
        assert!(!match_glob_pattern(
            "/home/user/project/database/schema.sql",
            "data/"
        ));
    }

    #[test]
    fn test_glob_gitignore_patterns() {
        // Common .gitignore patterns
        let patterns = vec![
            ".env",            // dotfile
            "*.pyc",           // compiled python
            "__pycache__/",    // cache dir
            ".next/",          // build dir
            "dist/",           // output dir
            ".gliding_horse/", // app data
        ];

        for pat in &patterns {
            assert!(
                match_glob_pattern(&format!("/workspace/{}", pat.trim_end_matches('/')), pat),
                "Pattern '{}' should match itself",
                pat
            );
        }
    }

    #[test]
    fn test_glob_exclude_all_variants() {
        let inventory = FileInventory::new(
            None,
            None,
            vec!["node_modules/".into(), "*.pyc".into(), "build/".into()],
        );

        // Must exclude
        assert!(inventory.is_excluded(Path::new("/project/node_modules/pkg/index.js")));
        assert!(inventory.is_excluded(Path::new("/project/src/__pycache__/cache.pyc")));
        assert!(inventory.is_excluded(Path::new("/project/build/o.app")));

        // Must NOT exclude
        assert!(!inventory.is_excluded(Path::new("/project/src/main.rs")));
        assert!(!inventory.is_excluded(Path::new("/project/Cargo.toml")));
        assert!(!inventory.is_excluded(Path::new("/project/src/pycache/api.py")));
    }

    #[test]
    fn test_glob_no_false_positive_substring() {
        // Ensure "target" doesn't match "targeting.rs"
        let inventory = FileInventory::new(None, None, vec!["target/".into()]);
        assert!(!inventory.is_excluded(Path::new("/project/src/targeting.rs")));
        assert!(inventory.is_excluded(Path::new("/project/target/debug/app")));
    }

    #[test]
    fn test_set_exclude_patterns() {
        let inventory = FileInventory::new(None, None, vec!["node_modules/".into()]);
        assert!(inventory.is_excluded(Path::new("node_modules/pkg/index.js")));
        assert!(!inventory.is_excluded(Path::new("build/output.o")));

        // Add patterns after construction (simulates gitignore sync)
        inventory.set_exclude_patterns(vec!["build/".into(), "*.o".into()]);
        assert!(inventory.is_excluded(Path::new("build/output.o")));
        // Original patterns preserved
        assert!(inventory.is_excluded(Path::new("node_modules/pkg/index.js")));
    }

    #[test]
    fn test_max_entries_limit() {
        let dir = TempDir::new().unwrap();
        // Create files that would exceed limit
        // We use a fresh inventory and lower the effective limit by
        // directly testing add_entry behavior
        let inventory = FileInventory::new(None, None, vec![]);

        // Fill up to MAX_INVENTORY_ENTRIES
        for i in 0..super::MAX_INVENTORY_ENTRIES {
            let path = dir.path().join(format!("file_{}.rs", i));
            std::fs::write(&path, "fn f() {}").unwrap();
            let result = inventory.add_or_update(&path.to_string_lossy());
            assert!(result.is_some(), "Entry {} should be added", i);
        }

        // Next entry should be rejected
        let overflow = dir.path().join("overflow.rs");
        std::fs::write(&overflow, "fn overflow() {}").unwrap();
        let result = inventory.add_or_update(&overflow.to_string_lossy());
        assert!(
            result.is_none(),
            "Entry beyond MAX_INVENTORY_ENTRIES should be rejected"
        );

        assert_eq!(inventory.total_count(), super::MAX_INVENTORY_ENTRIES);
    }

    #[test]
    fn test_full_scan_honors_exclude_patterns() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join("vendor")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("vendor/lib.rs"), "pub fn lib() {}").unwrap();

        let inventory = FileInventory::new(None, None, vec!["vendor/".into()]);
        let count = inventory.full_scan(&dir.path().to_string_lossy());
        assert_eq!(count, 1, "full_scan should exclude vendor/");
        assert!(inventory
            .get_entry(&dir.path().join("src/main.rs").to_string_lossy())
            .is_some());
        assert!(inventory
            .get_entry(&dir.path().join("vendor/lib.rs").to_string_lossy())
            .is_none());
    }
}
