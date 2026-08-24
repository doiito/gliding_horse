//! Layered prompt contracts shared by the runtime kernel and applications.
//!
//! The kernel owns non-overridable execution rules. Applications add a
//! domain-specific contract without replacing role, security, or lifecycle
//! rules. The rendered text is deliberately small and injected as a distinct
//! system-prompt region so it can be observed and tested independently.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromptVariant {
    Baseline,
    Optimized,
}

impl PromptVariant {
    pub fn from_env() -> Self {
        match std::env::var("GLIDING_PROMPT_VARIANT")
            .unwrap_or_else(|_| "optimized".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "baseline" | "a" => Self::Baseline,
            _ => Self::Optimized,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Optimized => "optimized",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplicationPromptProfile {
    pub application_id: String,
    pub contract_version: String,
    pub contract: String,
    pub optimized_contract: Option<String>,
}

impl ApplicationPromptProfile {
    pub fn new(
        application_id: impl Into<String>,
        version: impl Into<String>,
        contract: impl Into<String>,
    ) -> Self {
        Self {
            application_id: application_id.into(),
            contract_version: version.into(),
            contract: contract.into(),
            optimized_contract: None,
        }
    }

    pub fn render(&self) -> String {
        self.render_for(PromptVariant::Optimized)
    }

    pub fn with_optimized_contract(mut self, contract: impl Into<String>) -> Self {
        self.optimized_contract = Some(contract.into());
        self
    }

    pub fn render_for(&self, variant: PromptVariant) -> String {
        let contract = match variant {
            PromptVariant::Baseline => &self.contract,
            PromptVariant::Optimized => self.optimized_contract.as_ref().unwrap_or(&self.contract),
        };
        format!(
            "Application: {}\nContract version: {}\n\n{}",
            self.application_id, self.contract_version, contract
        )
    }
}

/// Stable, low-cardinality information emitted for every assembled prompt.
/// It is intentionally metadata only; prompt contents and secrets are not logged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptAssemblyReport {
    pub variant: PromptVariant,
    pub role: String,
    pub application_id: Option<String>,
    pub sections: BTreeMap<String, usize>,
    pub total_chars: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_application_contract_without_replacing_role() {
        let profile = ApplicationPromptProfile::new("glidingcode", "v1", "Use code evidence.");
        let rendered = profile.render();
        assert!(rendered.contains("Application: glidingcode"));
        assert!(rendered.contains("Contract version: v1"));
        assert!(rendered.contains("Use code evidence."));
    }

    #[test]
    fn selects_baseline_and_optimized_application_contracts() {
        let profile = ApplicationPromptProfile::new("app", "v1", "baseline")
            .with_optimized_contract("optimized");
        assert!(profile
            .render_for(PromptVariant::Baseline)
            .ends_with("baseline"));
        assert!(profile
            .render_for(PromptVariant::Optimized)
            .ends_with("optimized"));
    }
}
