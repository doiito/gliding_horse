mod agent;
mod execution;
mod intervention;
mod planning;
mod process;
mod stats;
mod types;

/// A recoverable tool error is feedback to the active BizAgent, not proof
/// that the agent is blocked. Only an explicit blocked event may ask SA to
/// coordinate an intervention.
fn event_requires_blocked_intervention(event_type: &str) -> bool {
    event_type == "AGENT_BLOCKED"
}

// Re-export all types so existing callers' imports continue to work
pub use agent::SupervisorAgent;
pub use types::*;

// Action handler registry (already exists, no changes needed)
mod actions;

#[cfg(test)]
mod tests;
