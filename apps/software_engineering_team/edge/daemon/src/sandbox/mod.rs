//! Heavy-weight container-level sandbox (reserved).
//!
//! This module manages Docker container lifecycle (`SandboxManager`) for
//! heavy isolation: an independent rootfs and network stack per task.
//! It complements — not replaces — the process-level `unshare` sandbox in
//! the main repository (`src/tools/builtin/sandbox.rs`), which protects
//! every bash command in the main agent execution path.
//!
//! Status: **reserved** — the manager is fully implemented (create / exec /
//! destroy / list) but is not yet wired into the daemon agent execution
//! path (`server.rs`, `api/`, `agent/` do not instantiate it). Enable by
//! constructing a `SandboxManager` and routing command execution through
//! `exec_in_sandbox` once heavy isolation is required.
pub mod manage;