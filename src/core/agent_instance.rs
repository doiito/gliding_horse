use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AgentRole {
    Plan,
    Do,
    Check,
    Act,
}

impl AgentRole {
    /// Semantic task-type key used by [`UnifiedGateway`](crate::gateway::unified_gateway::UnifiedGateway)
    /// model routing.
    ///
    /// Keep this separate from `Display`: PA/DA/CA/AA are human-facing role
    /// labels, while gateway configuration uses stable capability keys.
    pub const fn model_routing_key(self) -> &'static str {
        match self {
            Self::Plan => "planning",
            Self::Do => "execution",
            Self::Check | Self::Act => "analysis",
        }
    }
}

impl std::fmt::Display for AgentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentRole::Plan => write!(f, "PA"),
            AgentRole::Do => write!(f, "DA"),
            AgentRole::Check => write!(f, "CA"),
            AgentRole::Act => write!(f, "AA"),
        }
    }
}

impl std::str::FromStr for AgentRole {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "PA" | "PLAN" => Ok(Self::Plan),
            "DA" | "DO" => Ok(Self::Do),
            "CA" | "CHECK" => Ok(Self::Check),
            "AA" | "ACT" => Ok(Self::Act),
            _ => Err(format!("Unknown agent role: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Idle,
    Running,
    Completed,
    Failed,
}

/// A single agent instance with isolated state
///
/// All PA/DA/CA/AA agents are instances of this struct.
/// Differences are:
/// - role (determines system prompt template)
/// - status (lifecycle tracking)
/// - L1 session (isolated via MemoryManager)
#[derive(Debug, Clone)]
pub struct AgentInstance {
    pub agent_id: String,
    pub role: AgentRole,
    pub status: AgentStatus,
}

impl AgentInstance {
    pub fn new(agent_id: String, role: AgentRole) -> Self {
        Self {
            agent_id,
            role,
            status: AgentStatus::Idle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AgentRole;

    #[test]
    fn every_agent_role_has_the_expected_gateway_model_routing_key() {
        assert_eq!(AgentRole::Plan.model_routing_key(), "planning");
        assert_eq!(AgentRole::Do.model_routing_key(), "execution");
        assert_eq!(AgentRole::Check.model_routing_key(), "analysis");
        assert_eq!(AgentRole::Act.model_routing_key(), "analysis");
    }
}
