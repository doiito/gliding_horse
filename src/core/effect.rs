//! Generic effect and completion contracts used by the orchestration kernel.
//!
//! The kernel deliberately does not classify software-engineering language.
//! Applications map their domain tasks onto these generic policies, while SA
//! and BizAgent use the same protocol for execution, evidence and recovery.

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
