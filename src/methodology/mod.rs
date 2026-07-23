pub mod evolution;
/// Methodology Layer (L2) — Methodology definitions, registry, and activation conditions.
///
/// A methodology is an on-demand behavioral protocol extracted from Superpowers skills.
/// Methodologies sit between the Constitution (L3, always-on rules) and Enforcement (L1, code-level gates).
///
/// Architecture Layer: L2 — Methodology (On-Demand)
/// See design: PR-res/superpowers-skills-full-integration-design.md §0
pub mod gate;
pub mod integration;

use crate::core::constitution::{ActivationCondition, ConstitutionRole};

/// The nature of a methodology — determines how it's communicated to the agent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodologyType {
    /// Hard rules with authority language (YOU MUST, Never) — for TDD, verification, etc.
    Discipline,
    /// Guidance with collaborative framing — for brainstorming, reviews, etc.
    Guidance,
    /// Reference information — for tool mappings, skill descriptions
    Reference,
    /// Process flows — for multi-step workflows (plan→execute→review)
    Process,
}

/// A red flag entry — pattern the methodology should watch for
#[derive(Debug, Clone)]
pub struct RedFlagEntry {
    pub pattern: &'static str,
    pub severity: RedFlagSeverity,
    pub rationalization_check: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedFlagSeverity {
    /// Always blocks — must be addressed before proceeding
    Critical,
    /// Should be addressed but not blocking
    Warning,
    /// Advisory — best practice reminder
    Info,
}

/// An anti-pattern entry — with gate function for enforcement
#[derive(Debug, Clone)]
pub struct AntiPatternEntry {
    pub name: &'static str,
    pub description: &'static str,
    /// The action BEFORE which the gate check triggers
    pub gate_before: &'static str,
    /// The question the agent must ask itself
    pub gate_ask: &'static str,
    /// What happens if the anti-pattern is detected
    pub gate_action: &'static str,
}

/// Persuasion profile — how to frame the methodology in system prompts
#[derive(Debug, Clone)]
pub struct PersuasionProfile {
    /// Primary persuasion principles to use
    pub principles: &'static [&'static str],
    /// Example phrasing (evaluated at prompt build time)
    pub phrasing_examples: &'static [&'static str],
}

/// A full methodology definition
#[derive(Debug, Clone)]
pub struct MethodologyDefinition {
    /// Unique identifier like "methodology:index-priority"
    pub id: &'static str,
    /// Human-readable name
    pub name: &'static str,
    /// One-line description
    pub description: &'static str,
    /// Methodology type (determines framing)
    pub methodology_type: MethodologyType,
    /// Domain this applies to ("general", "programming", "debugging", etc.)
    pub domain: &'static str,
    /// Source skill file in superpowers-main
    pub source: &'static str,
    /// Red flags to watch for
    pub red_flags: &'static [RedFlagEntry],
    /// Anti-patterns with gate functions
    pub anti_patterns: &'static [AntiPatternEntry],
    /// Persuasion profile for injection
    pub persuasion: PersuasionProfile,
    /// When to auto-activate
    pub activation: ActivationCondition,
    /// Related methodology IDs
    pub related: &'static [&'static str],
}

// ════════════════════════════════════════════════════════════════════════
// Methodology Registry
// ════════════════════════════════════════════════════════════════════════

/// Registry of all available methodologies.
///
/// Can load built-in definitions or be populated dynamically.
pub struct MethodologyRegistry {
    entries: Vec<MethodologyDefinition>,
}

impl MethodologyRegistry {
    /// Create registry with all built-in methodology definitions
    pub fn new() -> Self {
        Self {
            entries: builtin_methodologies(),
        }
    }

    /// Create with custom set
    pub fn with_entries(entries: Vec<MethodologyDefinition>) -> Self {
        Self { entries }
    }

    /// Get a methodology by its ID
    pub fn get(&self, id: &str) -> Option<&MethodologyDefinition> {
        self.entries.iter().find(|m| m.id == id)
    }

    /// Get all methodologies
    pub fn all(&self) -> &[MethodologyDefinition] {
        &self.entries
    }

    /// Find methodologies matching an activation condition
    pub fn for_activation(&self, condition: &ActivationCondition) -> Vec<&MethodologyDefinition> {
        self.entries
            .iter()
            .filter(|m| std::mem::discriminant(&m.activation) == std::mem::discriminant(condition))
            .collect()
    }

    /// Get methodologies for a specific domain
    pub fn for_domain(&self, domain: &str) -> Vec<&MethodologyDefinition> {
        self.entries
            .iter()
            .filter(|m| m.domain == domain || m.domain == "general")
            .collect()
    }

    /// Number of registered methodologies
    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for MethodologyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global singleton accessor — caches the registry on first call.
pub fn global_registry() -> &'static MethodologyRegistry {
    use std::sync::OnceLock;
    static REGISTRY: OnceLock<MethodologyRegistry> = OnceLock::new();
    REGISTRY.get_or_init(MethodologyRegistry::new)
}

// ════════════════════════════════════════════════════════════════════════
// Built-in Methodology Definitions
// ════════════════════════════════════════════════════════════════════════

/// Returns all built-in methodology definitions.
///
/// These correspond to the 14 superpowers-main skills plus 5 new methodologies
/// that fill gaps identified during Constitution analysis:
/// - Index-Priority (constitution rule "index-priority" had no Superpowers equivalent)
/// - Cost-Awareness (constitution rule "cost-awareness" was only implicit in DA/PA)
/// - Least-Privilege (constitution rule "least-privilege" had no formal protocol)
/// - Complexity-Assessment (constitution rule "honest-complexity-assessment" had no Superpowers equivalent)
/// - Boundary-Enforcement (constitution rule "boundary-rejection" + "boundary-principle" aggregation)
pub fn builtin_methodologies() -> Vec<MethodologyDefinition> {
    vec![
        // ── 1. Index-Priority (NEW) ──
        MethodologyDefinition {
            id: "methodology:index-priority",
            name: "Index-First Strategy",
            description: "When facing large volumes of files or data, first use search tools to get an index/overview, then read precisely as needed",
            methodology_type: MethodologyType::Discipline,
            domain: "general",
            source: "new — constitution gap fill for uni-perception-2",
            red_flags: &[
                RedFlagEntry {
                    pattern: "blind directory traversal",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"The directory is small, faster to read it all\" — searching then reading is always better"),
                },
                RedFlagEntry {
                    pattern: "guessing content by filename",
                    severity: RedFlagSeverity::Warning,
                    rationalization_check: Some("\"I know what it is from the name\" — actual content may not match the filename"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "full traversal",
                    description: "Using ls -R / find . to traverse entire directories instead of precise search",
                    gate_before: "before executing directory traversal tools",
                    gate_ask: "Can you narrow the scope with grep/glob before traversing?",
                    gate_action: "STOP — search index first, then read on demand",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "commitment"],
                phrasing_examples: &["YOU MUST search before traverse", "Always get the index first"],
            },
            activation: ActivationCondition::OnToolCategory(&["file_search", "directory_list"]),
            related: &["methodology:using-superpowers", "methodology:cost-awareness"],
        },

        // ── 2. Cost-Awareness (NEW) ──
        MethodologyDefinition {
            id: "methodology:cost-awareness",
            name: "Cost-Awareness Protocol",
            description: "Explicitly evaluate token, time, and compute resource costs in all decisions; choose the lowest overall cost path",
            methodology_type: MethodologyType::Discipline,
            domain: "general",
            source: "new — constitution gap fill for pa-4, da-6, aa-3, sa-decision-3, uni-perception-3",
            red_flags: &[
                RedFlagEntry {
                    pattern: "unnecessary large output",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"Safer to look at everything\" — just use grep to filter what's needed"),
                },
                RedFlagEntry {
                    pattern: "ignoring token budget",
                    severity: RedFlagSeverity::Warning,
                    rationalization_check: None,
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "blind full scan",
                    description: "Large-output tool calls without specifying scope or grep filtering",
                    gate_before: "before executing bash / file_read or other large-output tools",
                    gate_ask: "Could output exceed 100 lines? Can you use | head / | grep to limit?",
                    gate_action: "STOP — use precise search instead of full scan",
                },
                AntiPatternEntry {
                    name: "no alternative comparison",
                    description: "Proposing a plan with only one option and no cost comparison",
                    gate_before: "before submitting a plan/decision",
                    gate_ask: "Is there a lower-cost alternative?",
                    gate_action: "STOP — provide at least 2 options with cost comparison",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "social_proof"],
                phrasing_examples: &["YOU MUST control token usage", "Always prefer the lowest-cost path"],
            },
            activation: ActivationCondition::OnHookPoint("PreToolCall"),
            related: &["methodology:index-priority", "methodology:verification-before-completion"],
        },

        // ── 3. Least-Privilege (NEW) ──
        MethodologyDefinition {
            id: "methodology:least-privilege",
            name: "Least-Privilege Protocol",
            description: "Tool calls and data access strictly limited to the minimum scope required by the task; no access to unrelated resources",
            methodology_type: MethodologyType::Discipline,
            domain: "general",
            source: "new — constitution gap fill for uni-boundary-1",
            red_flags: &[
                RedFlagEntry {
                    pattern: "accessing unrelated directories/files",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"Just a peek won't hurt\" — permissions should be minimized; task-irrelevant access is prohibited"),
                },
                RedFlagEntry {
                    pattern: "using dangerous commands",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"Just this once\" — high-risk operations must go through approval"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "privilege escalation",
                    description: "Executing tool calls or data access unrelated to the current task goal",
                    gate_before: "before executing any tool",
                    gate_ask: "Is this tool/data necessary for the current task goal?",
                    gate_action: "STOP — remove unnecessary operations",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "security"],
                phrasing_examples: &["YOU MUST restrict to task scope", "No access outside task boundary"],
            },
            activation: ActivationCondition::OnToolCategory(&["shell", "file_write", "network"]),
            related: &["methodology:boundary-enforcement"],
        },

        // ── 4. Complexity-Assessment (NEW) ──
        MethodologyDefinition {
            id: "methodology:complexity-assessment",
            name: "Honest Complexity Assessment",
            description: "Objectively select the complexity level based on task reality; no downgrading for convenience or upgrading for show",
            methodology_type: MethodologyType::Guidance,
            domain: "general",
            source: "new — constitution gap fill for sa-perception-3, uni-boundary-4",
            red_flags: &[
                RedFlagEntry {
                    pattern: "convenience downgrade",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"This is easy, just keep it simple\" — complexity should be based on objective task characteristics"),
                },
                RedFlagEntry {
                    pattern: "show-off upgrade",
                    severity: RedFlagSeverity::Warning,
                    rationalization_check: Some("\"Using advanced features looks professional\" — the simplest solution that works is best"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "complexity bias",
                    description: "Selected complexity level does not match actual task requirements",
                    gate_before: "before SA selects complexity level",
                    gate_ask: "Evaluation factors: 1) Goal clarity 2) Step count 3) Risk level 4) Resource constraints",
                    gate_action: "STOP — re-evaluate with complexity_matrix",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["social_proof", "commitment"],
                phrasing_examples: &["Always match complexity to facts", "Be honest about difficulty"],
            },
            activation: ActivationCondition::OnAgentRole(&[ConstitutionRole::Supervisor, ConstitutionRole::Plan]),
            related: &["methodology:cost-awareness", "methodology:boundary-enforcement"],
        },

        // ── 5. Boundary-Enforcement (NEW) ──
        MethodologyDefinition {
            id: "methodology:boundary-enforcement",
            name: "Boundary Enforcement",
            description: "When encountering safety, capability, or ethical boundaries, must reject, warn, or exit",
            methodology_type: MethodologyType::Discipline,
            domain: "general",
            source: "new — constitution gap fill for uni-boundary-2, uni-boundary-3, sa-safety-4",
            red_flags: &[
                RedFlagEntry {
                    pattern: "overextending within capability",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"One more try might work\" — seek help when exceeding capability boundaries"),
                },
                RedFlagEntry {
                    pattern: "ignoring risk",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"Should be fine\" — risk assessment is mandatory"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "illegal request",
                    description: "Requests involving illegal, unsafe, or unethical content",
                    gate_before: "before responding to any request",
                    gate_ask: "Is this a safe/legal/ethical request?",
                    gate_action: "ABORT — explicitly refuse and explain why",
                },
                AntiPatternEntry {
                    name: "boundary overstep",
                    description: "Executing operations beyond your capability or task authorization",
                    gate_before: "before executing potentially overstepping operations",
                    gate_ask: "Am I authorized/capable of performing this operation?",
                    gate_action: "STOP — suggest narrowing scope or requesting authorization",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "unity"],
                phrasing_examples: &["YOU MUST refuse unsafe requests", "We are responsible for safety"],
            },
            activation: ActivationCondition::Always,
            related: &["methodology:least-privilege", "methodology:complexity-assessment"],
        },

        // ── 6. Using-Superpowers (existing skill → methodology) ──
        MethodologyDefinition {
            id: "methodology:using-superpowers",
            name: "Using Superpowers Methodology",
            description: "Invoke relevant skills before any response or operation; check red-flag lists to prevent common errors",
            methodology_type: MethodologyType::Discipline,
            domain: "general",
            source: "superpowers-main/skills/using-superpowers/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "skipping skill check",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"This is just a simple question\" — a question is a task; check the methodology"),
                },
                RedFlagEntry {
                    pattern: "guessing content by filename",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"I know what that means\" — knowing the concept ≠ skipping the methodology"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "assumption as answer",
                    description: "Making judgments based solely on filenames or partial information",
                    gate_before: "before making judgments about files/code",
                    gate_ask: "Have you read it completely?",
                    gate_action: "STOP — Read first, then judge",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "commitment", "social_proof"],
                phrasing_examples: &["Always check methodology first", "Never skip the checklist"],
            },
            activation: ActivationCondition::Always,
            related: &["methodology:index-priority"],
        },

        // ── 7. Brainstorming ──
        MethodologyDefinition {
            id: "methodology:brainstorming",
            name: "Brainstorming Methodology",
            description: "Must use before any creative work — first explore user intent, requirements, and design",
            methodology_type: MethodologyType::Process,
            domain: "general",
            source: "superpowers-main/skills/brainstorming/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "designing without exploration",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"Requirements are clear, no exploration needed\" — explore first, then design"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "skipping clarification",
                    description: "Implementing directly when requirements are ambiguous instead of asking first",
                    gate_before: "before entering implementation",
                    gate_ask: "Are all requirements clarified without ambiguity?",
                    gate_action: "STOP — clarify first, then implement",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["unity", "commitment"],
                phrasing_examples: &["Let's explore before building", "We are colleagues working together"],
            },
            activation: ActivationCondition::OnHookPoint("PrePlanCreation"),
            related: &["methodology:writing-plans", "methodology:complexity-assessment"],
        },

        // ── 8. TDD ──
        MethodologyDefinition {
            id: "methodology:test-driven-development",
            name: "Test-Driven Development",
            description: "When implementing any feature or fixing a bug, write tests first then implementation code",
            methodology_type: MethodologyType::Discipline,
            domain: "programming",
            source: "superpowers-main/skills/test-driven-development/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "implement-then-test",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"Write code first, tests later\" — must write tests first"),
                },
                RedFlagEntry {
                    pattern: "delete-to-pass",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"Tests are wrong, delete and restart\" — first check why the test failed"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "implementation first",
                    description: "Starting implementation code without writing tests first",
                    gate_before: "before writing any implementation code",
                    gate_ask: "Are tests written? Are they failing (red)?",
                    gate_action: "STOP — write test, see red, then implement",
                },
                AntiPatternEntry {
                    name: "mock replacing real behavior",
                    description: "Using mocks instead of verifying real behavior",
                    gate_before: "before asserting on mocks",
                    gate_ask: "Are you testing real behavior or mock behavior?",
                    gate_action: "STOP — test real behavior, not mocks",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "commitment"],
                phrasing_examples: &["YOU MUST test before code", "Never skip the RED phase"],
            },
            activation: ActivationCondition::OnToolCategory(&["file_write", "code_generation"]),
            related: &["methodology:verification-before-completion", "methodology:systematic-debugging"],
        },

        // ── 9. Systematic Debugging ──
        MethodologyDefinition {
            id: "methodology:systematic-debugging",
            name: "Systematic Debugging",
            description: "When encountering any bug, test failure, or unexpected behavior, systematically analyze root cause before fixing",
            methodology_type: MethodologyType::Process,
            domain: "programming",
            source: "superpowers-main/skills/systematic-debugging/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "blind retry",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"One more retry might work\" — analyze logs to find root cause first"),
                },
                RedFlagEntry {
                    pattern: "random modification",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"Let's try changing this\" — change one variable at a time, verify, then change"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "Shotgun Debugging",
                    description: "Modifying multiple things at once hoping one will fix the issue",
                    gate_before: "before modifying multiple files/variables simultaneously",
                    gate_ask: "Does each change target a root cause? One change at a time?",
                    gate_action: "STOP — change one thing at a time, verify, then move to the next",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "discipline"],
                phrasing_examples: &["Always find root cause first", "One change at a time, verify each"],
            },
            activation: ActivationCondition::OnTaskError,
            related: &["methodology:test-driven-development", "methodology:verification-before-completion"],
        },

        // ── 10. Verification Before Completion ──
        MethodologyDefinition {
            id: "methodology:verification-before-completion",
            name: "Verification Before Completion",
            description: "Before claiming work is done, must run verification commands and confirm output before declaring success",
            methodology_type: MethodologyType::Discipline,
            domain: "general",
            source: "superpowers-main/skills/verification-before-completion/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "claiming pass without running",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"The code is simple, it must be fine\" — verification must be run"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "no-evidence claim",
                    description: "Claiming work is complete without providing any verification evidence",
                    gate_before: "before reporting task completion",
                    gate_ask: "Did you run verification? What was the output? Did diagnostics pass?",
                    gate_action: "STOP — run verification and attach output evidence",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "commitment"],
                phrasing_examples: &["No evidence = not complete", "Always verify before claiming done"],
            },
            activation: ActivationCondition::OnPhaseEnd("ACT"),
            related: &["methodology:test-driven-development"],
        },
    ]
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_methodologies_loaded() {
        let registry = MethodologyRegistry::new();
        assert!(
            registry.count() >= 10,
            "Expected 10+ built-in methodologies, got {}",
            registry.count()
        );
    }

    #[test]
    fn test_get_methodology_by_id() {
        let registry = MethodologyRegistry::new();
        let idx = registry.get("methodology:index-priority").unwrap();
        assert_eq!(idx.name, "Index-First Strategy");
    }

    #[test]
    fn test_new_methodologies_present() {
        let registry = MethodologyRegistry::new();
        for id in &[
            "methodology:index-priority",
            "methodology:cost-awareness",
            "methodology:least-privilege",
            "methodology:complexity-assessment",
            "methodology:boundary-enforcement",
        ] {
            assert!(
                registry.get(id).is_some(),
                "Missing new methodology: {}",
                id
            );
        }
    }

    #[test]
    fn test_methodology_has_red_flags() {
        let registry = MethodologyRegistry::new();
        for method in registry.all() {
            assert!(
                !method.red_flags.is_empty(),
                "Methodology {} has no red flags",
                method.id
            );
        }
    }

    #[test]
    fn test_methodology_has_anti_patterns() {
        let registry = MethodologyRegistry::new();
        for method in registry.all() {
            assert!(
                !method.anti_patterns.is_empty(),
                "Methodology {} has no anti-patterns",
                method.id
            );
        }
    }

    #[test]
    fn test_resolver_bindings() {
        // binding count covered in constitution tests; MethodologyResolver removed (P9)
    }

    #[test]
    fn test_constitutions_for_methodology() {
        // MethodologyResolver removed as unused (P9)
    }
}
