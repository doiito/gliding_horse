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

use std::path::{Path, PathBuf};
use tracing::{debug, warn};

use crate::core::constitution::{ActivationCondition, ConstitutionRole};
use crate::CoreError;

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

    /// Load methodology definitions from all `*.jsonld` files in a directory.
    ///
    /// Entries replace same-ID builtins when present; unknown IDs are appended.
    pub fn load_from_jsonld_dir<P: AsRef<Path>>(&mut self, dir: P) -> Result<usize, CoreError> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err(CoreError::Internal {
                message: format!("Methodology nodes directory not found: {}", dir.display()),
            });
        }
        let mut loaded = 0;
        for entry in std::fs::read_dir(dir).map_err(|e| CoreError::Internal {
            message: format!(
                "Failed to read methodology nodes dir {}: {}",
                dir.display(),
                e
            ),
        })? {
            let entry = entry.map_err(|e| CoreError::Internal {
                message: format!("read_dir entry: {}", e),
            })?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonld") {
                continue;
            }
            loaded += self.load_from_jsonld_file(&path)?;
        }
        Ok(loaded)
    }

    /// Load a single methodology definition from a JSON-LD file, replacing the
    /// builtin entry with the same ID if present.
    pub fn load_from_jsonld_file<P: AsRef<Path>>(&mut self, path: P) -> Result<usize, CoreError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| CoreError::Internal {
            message: format!(
                "Failed to read methodology JSON-LD {}: {}",
                path.display(),
                e
            ),
        })?;
        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| CoreError::InvalidJsonLd {
                message: format!("Invalid JSON in {}: {}", path.display(), e),
            })?;
        let def = parse_methodology_from_jsonld(&json)?;
        if let Some(existing) = self.entries.iter_mut().find(|m| m.id == def.id) {
            *existing = def;
        } else {
            self.entries.push(def);
        }
        Ok(1)
    }

    /// Best-effort load of the bundled `nodes/` definitions. Falls back to the
    /// hardcoded builtins (no-op) when the directory is absent, logging a warning.
    pub fn load_bundled_nodes(&mut self) {
        let candidates: [PathBuf; 2] = [
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/methodology/nodes"),
            PathBuf::from("src/methodology/nodes"),
        ];
        for dir in &candidates {
            if dir.is_dir() {
                match self.load_from_jsonld_dir(dir) {
                    Ok(n) => {
                        debug!(count = n, path = %dir.display(), "Methodologies loaded from JSON-LD nodes");
                        return;
                    }
                    Err(e) => {
                        warn!(path = %dir.display(), error = %e, "Failed to load methodology nodes, falling back to builtins");
                    }
                }
            }
        }
        warn!("Methodology nodes/ dir not found; using builtin fallback");
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
// JSON-LD Parsing
// ════════════════════════════════════════════════════════════════════════

fn get_json_str<'a>(json: &'a serde_json::Value, key: &str) -> Result<&'a str, CoreError> {
    json.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| CoreError::InvalidJsonLd {
            message: format!("Missing string field `{}` in methodology node", key),
        })
}

fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn parse_methodology_type_str(s: &str) -> Result<MethodologyType, CoreError> {
    match s {
        "Discipline" => Ok(MethodologyType::Discipline),
        "Guidance" => Ok(MethodologyType::Guidance),
        "Reference" => Ok(MethodologyType::Reference),
        "Process" => Ok(MethodologyType::Process),
        other => Err(CoreError::InvalidJsonLd {
            message: format!("Unknown methodology type `{}`", other),
        }),
    }
}

fn parse_severity_str(s: &str) -> Result<RedFlagSeverity, CoreError> {
    match s {
        "Critical" => Ok(RedFlagSeverity::Critical),
        "Warning" => Ok(RedFlagSeverity::Warning),
        "Info" => Ok(RedFlagSeverity::Info),
        other => Err(CoreError::InvalidJsonLd {
            message: format!("Unknown red flag severity `{}`", other),
        }),
    }
}

fn parse_role_str(s: &str) -> Result<ConstitutionRole, CoreError> {
    match s.trim() {
        "Universal" => Ok(ConstitutionRole::Universal),
        "Supervisor" => Ok(ConstitutionRole::Supervisor),
        "Plan" => Ok(ConstitutionRole::Plan),
        "Do" => Ok(ConstitutionRole::Do),
        "Check" => Ok(ConstitutionRole::Check),
        "Act" => Ok(ConstitutionRole::Act),
        other => Err(CoreError::InvalidJsonLd {
            message: format!("Unknown constitution role `{}`", other),
        }),
    }
}

/// Parse a Debug-formatted list such as `["a", "b"]` or `[Universal, Supervisor]`.
fn parse_dbg_list(inner: &str, quoted_items: bool) -> Result<Vec<String>, CoreError> {
    let t = inner.trim();
    let list = t
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| CoreError::InvalidJsonLd {
            message: format!("Malformed list in activation condition: `{}`", inner),
        })?;
    if list.trim().is_empty() {
        return Ok(Vec::new());
    }
    list.split(',')
        .map(|item| {
            let item = item.trim();
            if quoted_items {
                Ok(item.trim_matches('"').to_string())
            } else {
                Ok(item.to_string())
            }
        })
        .collect()
}

/// Parse the Debug-string serialization of [`ActivationCondition`] produced by
/// [`MethodologyDefinition::to_json_ld`] (which uses `format!("{:?}")`).
fn parse_activation_str(raw: &str) -> Result<ActivationCondition, CoreError> {
    let t = raw.trim();
    match t {
        "Always" => return Ok(ActivationCondition::Always),
        "OnTaskError" => return Ok(ActivationCondition::OnTaskError),
        _ => {}
    }

    if let Some(inner) = t
        .strip_prefix("OnHookPoint(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let hook = inner.trim().trim_matches('"');
        if hook.is_empty() {
            return Err(CoreError::InvalidJsonLd {
                message: "OnHookPoint with empty hook point".into(),
            });
        }
        return Ok(ActivationCondition::OnHookPoint(leak_str(hook.to_string())));
    }

    if let Some(inner) = t
        .strip_prefix("OnPhaseEnd(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let phase = inner.trim().trim_matches('"');
        if phase.is_empty() {
            return Err(CoreError::InvalidJsonLd {
                message: "OnPhaseEnd with empty phase".into(),
            });
        }
        return Ok(ActivationCondition::OnPhaseEnd(leak_str(phase.to_string())));
    }

    if let Some(inner) = t
        .strip_prefix("OnToolCategory(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let items = parse_dbg_list(inner, true)?;
        let leaked: &'static [&'static str] = Box::leak(
            items
                .into_iter()
                .map(leak_str)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        return Ok(ActivationCondition::OnToolCategory(leaked));
    }

    if let Some(inner) = t
        .strip_prefix("OnAgentRole(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let items = parse_dbg_list(inner, false)?;
        let mut roles = Vec::with_capacity(items.len());
        for item in items {
            roles.push(parse_role_str(&item)?);
        }
        let leaked: &'static [ConstitutionRole] = Box::leak(roles.into_boxed_slice());
        return Ok(ActivationCondition::OnAgentRole(leaked));
    }

    Err(CoreError::InvalidJsonLd {
        message: format!("Unrecognized activation condition: `{}`", raw),
    })
}

/// Parse a single methodology definition from a JSON-LD node in the format
/// produced by [`MethodologyDefinition::to_json_ld`].
///
/// All returned string fields are leaked to `'static` to match
/// [`MethodologyDefinition`]'s interned representation.
pub fn parse_methodology_from_jsonld(
    json: &serde_json::Value,
) -> Result<MethodologyDefinition, CoreError> {
    let id = leak_str(get_json_str(json, "methodology:id")?.to_string());
    let name = leak_str(get_json_str(json, "methodology:name")?.to_string());
    let description = leak_str(get_json_str(json, "methodology:description")?.to_string());
    let methodology_type = parse_methodology_type_str(get_json_str(json, "methodology:type")?)?;
    let domain = leak_str(get_json_str(json, "methodology:domain")?.to_string());
    let source = leak_str(get_json_str(json, "methodology:source")?.to_string());
    let activation = parse_activation_str(get_json_str(json, "methodology:activation")?)?;

    // Red flags
    let mut red_flags: Vec<RedFlagEntry> = Vec::new();
    if let Some(arr) = json.get("methodology:redFlags").and_then(|v| v.as_array()) {
        for f in arr {
            let pattern = leak_str(get_json_str(f, "methodology:pattern")?.to_string());
            let severity = parse_severity_str(get_json_str(f, "methodology:severity")?)?;
            let rationalization_check = f
                .get("methodology:rationalizationCheck")
                .and_then(|v| v.as_str())
                .map(|s| leak_str(s.to_string()));
            red_flags.push(RedFlagEntry {
                pattern,
                severity,
                rationalization_check,
            });
        }
    }

    // Anti-patterns
    let mut anti_patterns: Vec<AntiPatternEntry> = Vec::new();
    if let Some(arr) = json
        .get("methodology:antiPatterns")
        .and_then(|v| v.as_array())
    {
        for ap in arr {
            anti_patterns.push(AntiPatternEntry {
                name: leak_str(get_json_str(ap, "methodology:antiPatternName")?.to_string()),
                description: leak_str(
                    get_json_str(ap, "methodology:antiPatternDescription")?.to_string(),
                ),
                gate_before: leak_str(get_json_str(ap, "methodology:gateBefore")?.to_string()),
                gate_ask: leak_str(get_json_str(ap, "methodology:gateAsk")?.to_string()),
                gate_action: leak_str(get_json_str(ap, "methodology:gateAction")?.to_string()),
            });
        }
    }

    // Persuasion
    let mut principles: Vec<&'static str> = Vec::new();
    let mut phrasing_examples: Vec<&'static str> = Vec::new();
    if let Some(p) = json.get("methodology:persuasion") {
        if let Some(arr) = p.get("methodology:principles").and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    principles.push(leak_str(s.to_string()));
                }
            }
        }
        if let Some(arr) = p
            .get("methodology:phrasingExamples")
            .and_then(|v| v.as_array())
        {
            for item in arr {
                if let Some(s) = item.as_str() {
                    phrasing_examples.push(leak_str(s.to_string()));
                }
            }
        }
    }

    // Related
    let mut related: Vec<&'static str> = Vec::new();
    if let Some(arr) = json.get("methodology:related").and_then(|v| v.as_array()) {
        for r in arr {
            if let Some(s) = r.get("methodology:id").and_then(|v| v.as_str()) {
                related.push(leak_str(s.to_string()));
            }
        }
    }

    Ok(MethodologyDefinition {
        id,
        name,
        description,
        methodology_type,
        domain,
        source,
        red_flags: Box::leak(red_flags.into_boxed_slice()),
        anti_patterns: Box::leak(anti_patterns.into_boxed_slice()),
        persuasion: PersuasionProfile {
            principles: Box::leak(principles.into_boxed_slice()),
            phrasing_examples: Box::leak(phrasing_examples.into_boxed_slice()),
        },
        activation,
        related: Box::leak(related.into_boxed_slice()),
    })
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
            activation: ActivationCondition::OnHookPoint("skill_before"),
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
            activation: ActivationCondition::OnHookPoint("phase_start"),
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

        // ── 11. Writing Plans ──
        MethodologyDefinition {
            id: "methodology:writing-plans",
            name: "Writing Plans",
            description: "When you have a spec or requirements for a multi-step task, write an implementation plan before touching code",
            methodology_type: MethodologyType::Process,
            domain: "planning",
            source: "superpowers-main/skills/writing-plans/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "implementing before planning",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"The task is simple enough to start immediately\" — multi-step tasks need a written plan"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "code-before-plan",
                    description: "Jumping straight into implementation without a written plan for a multi-step task",
                    gate_before: "before implementing a multi-step task",
                    gate_ask: "Do you have a written plan covering all steps and checkpoints?",
                    gate_action: "STOP — write the plan first, then implement",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["commitment", "authority"],
                phrasing_examples: &["Plan before you code", "A written plan is a commitment to the outcome"],
            },
            activation: ActivationCondition::OnHookPoint("phase_start"),
            related: &["methodology:brainstorming", "methodology:executing-plans"],
        },

        // ── 12. Dispatching Parallel Agents ──
        MethodologyDefinition {
            id: "methodology:dispatching-parallel-agents",
            name: "Dispatching Parallel Agents",
            description: "When facing 2+ independent tasks that can be worked on without shared state or sequential dependencies, dispatch them in parallel",
            methodology_type: MethodologyType::Process,
            domain: "general",
            source: "superpowers-main/skills/dispatching-parallel-agents/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "serializing independent tasks",
                    severity: RedFlagSeverity::Warning,
                    rationalization_check: Some("\"I can do them one by one\" — independent tasks should run in parallel"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "sequential-only-execution",
                    description: "Executing independent tasks one after another instead of dispatching parallel agents",
                    gate_before: "before executing a sequence of independent tasks",
                    gate_ask: "Are there 2+ independent tasks that could run in parallel without shared state?",
                    gate_action: "STOP — dispatch independent tasks to parallel agents",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "commitment"],
                phrasing_examples: &["Parallelize independent work", "Independent tasks go to parallel agents"],
            },
            activation: ActivationCondition::OnAgentRole(&[ConstitutionRole::Supervisor]),
            related: &["methodology:subagent-driven-development", "methodology:executing-plans"],
        },

        // ── 13. Executing Plans ──
        MethodologyDefinition {
            id: "methodology:executing-plans",
            name: "Executing Plans",
            description: "When you have a written implementation plan to execute in a separate session with review checkpoints",
            methodology_type: MethodologyType::Process,
            domain: "planning",
            source: "superpowers-main/skills/executing-plans/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "plan deviation",
                    severity: RedFlagSeverity::Warning,
                    rationalization_check: Some("\"I found a better way mid-flight\" — deviations from the plan need explicit justification"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "unplanned-deviation",
                    description: "Executing plan steps without following the written plan's checkpoints",
                    gate_before: "before each plan step",
                    gate_ask: "Does this step match the written plan? Were previous steps verified?",
                    gate_action: "STOP — follow the plan; record deviations explicitly",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["commitment", "authority"],
                phrasing_examples: &["Follow the plan step by step", "Verify each checkpoint before moving on"],
            },
            activation: ActivationCondition::OnHookPoint("phase_start"),
            related: &["methodology:writing-plans", "methodology:subagent-driven-development"],
        },

        // ── 14. Finishing a Development Branch ──
        MethodologyDefinition {
            id: "methodology:finishing-a-development-branch",
            name: "Finishing a Development Branch",
            description: "When implementation is complete and all tests pass, decide how to integrate the work: merge, PR, or cleanup",
            methodology_type: MethodologyType::Process,
            domain: "git",
            source: "superpowers-main/skills/finishing-a-development-branch/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "merge without integration check",
                    severity: RedFlagSeverity::Critical,
                    rationalization_check: Some("\"All tests pass, just merge it\" — integration requires deciding merge vs PR vs cleanup"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "automatic-merge",
                    description: "Merging without evaluating whether the work should be merged, PR'd, or cleaned up",
                    gate_before: "before merging a development branch",
                    gate_ask: "Have you decided merge vs PR vs cleanup, and verified all tests pass?",
                    gate_action: "STOP — present integration options and verify before merging",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "commitment"],
                phrasing_examples: &["Choose merge, PR, or cleanup deliberately", "Verify before integrating"],
            },
            activation: ActivationCondition::OnPhaseEnd("ACT"),
            related: &["methodology:verification-before-completion", "methodology:requesting-code-review"],
        },

        // ── 15. Receiving Code Review ──
        MethodologyDefinition {
            id: "methodology:receiving-code-review",
            name: "Receiving Code Review",
            description: "When receiving code review feedback, evaluate suggestions technically before implementing; require rigor, not performative agreement",
            methodology_type: MethodologyType::Guidance,
            domain: "review",
            source: "superpowers-main/skills/receiving-code-review/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "blind implementation of feedback",
                    severity: RedFlagSeverity::Warning,
                    rationalization_check: Some("\"The reviewer said so, just do it\" — feedback must be verified for technical correctness"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "performative-agreement",
                    description: "Agreeing with and implementing all review feedback without verifying it is technically correct",
                    gate_before: "before implementing review feedback",
                    gate_ask: "Is this feedback technically correct and consistent with the codebase?",
                    gate_action: "STOP — verify feedback; implement only what is correct",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "commitment"],
                phrasing_examples: &["Verify review feedback before implementing", "Rigor over agreement"],
            },
            activation: ActivationCondition::OnAgentRole(&[ConstitutionRole::Check]),
            related: &["methodology:requesting-code-review"],
        },

        // ── 16. Requesting Code Review ──
        MethodologyDefinition {
            id: "methodology:requesting-code-review",
            name: "Requesting Code Review",
            description: "When completing tasks, implementing major features, or before merging, verify work meets requirements and request review",
            methodology_type: MethodologyType::Guidance,
            domain: "review",
            source: "superpowers-main/skills/requesting-code-review/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "review request without self-verification",
                    severity: RedFlagSeverity::Warning,
                    rationalization_check: Some("\"The reviewer will catch issues\" — self-verify first, then request review"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "unverified-review-request",
                    description: "Requesting review before verifying your own work meets the requirements",
                    gate_before: "before requesting a code review",
                    gate_ask: "Did you verify your work against the requirements and run the checks?",
                    gate_action: "STOP — self-verify first, then request review",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "commitment"],
                phrasing_examples: &["Self-verify before requesting review", "Review requests follow verification"],
            },
            activation: ActivationCondition::OnAgentRole(&[ConstitutionRole::Check]),
            related: &["methodology:verification-before-completion", "methodology:receiving-code-review"],
        },

        // ── 17. Subagent-Driven Development ──
        MethodologyDefinition {
            id: "methodology:subagent-driven-development",
            name: "Subagent-Driven Development",
            description: "When executing implementation plans with independent tasks in the current session, dispatch work to subagents",
            methodology_type: MethodologyType::Process,
            domain: "development",
            source: "superpowers-main/skills/subagent-driven-development/SKILL.md",
            red_flags: &[
                RedFlagEntry {
                    pattern: "doing independent tasks inline",
                    severity: RedFlagSeverity::Warning,
                    rationalization_check: Some("\"I'll just do it myself\" — independent plan tasks belong in subagents"),
                },
            ],
            anti_patterns: &[
                AntiPatternEntry {
                    name: "inline-serial-execution",
                    description: "Executing independent plan tasks inline in the main session instead of dispatching to subagents",
                    gate_before: "before executing an independent plan task",
                    gate_ask: "Is this task independent enough to dispatch to a subagent?",
                    gate_action: "STOP — dispatch the task to a subagent",
                },
            ],
            persuasion: PersuasionProfile {
                principles: &["authority", "commitment"],
                phrasing_examples: &["Dispatch independent tasks to subagents", "The main session orchestrates, subagents execute"],
            },
            activation: ActivationCondition::OnAgentRole(&[ConstitutionRole::Do]),
            related: &["methodology:dispatching-parallel-agents", "methodology:executing-plans"],
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

    #[test]
    fn test_jsonld_roundtrip_preserves_definition() {
        let registry = MethodologyRegistry::new();
        for m in registry.all() {
            let json = m.to_json_ld();
            let parsed = parse_methodology_from_jsonld(&json).expect("round-trip parse");
            assert_eq!(parsed.id, m.id, "id mismatch for {}", m.id);
            assert_eq!(parsed.name, m.name);
            assert_eq!(parsed.description, m.description);
            assert_eq!(parsed.methodology_type, m.methodology_type);
            assert_eq!(parsed.domain, m.domain);
            assert_eq!(parsed.source, m.source);
            assert_eq!(
                format!("{:?}", parsed.activation),
                format!("{:?}", m.activation)
            );
            assert_eq!(parsed.red_flags.len(), m.red_flags.len());
            for (p, orig) in parsed.red_flags.iter().zip(m.red_flags.iter()) {
                assert_eq!(p.pattern, orig.pattern);
                assert_eq!(p.severity, orig.severity);
                assert_eq!(p.rationalization_check, orig.rationalization_check);
            }
            assert_eq!(parsed.anti_patterns.len(), m.anti_patterns.len());
            for (p, orig) in parsed.anti_patterns.iter().zip(m.anti_patterns.iter()) {
                assert_eq!(p.name, orig.name);
                assert_eq!(p.gate_action, orig.gate_action);
            }
            assert_eq!(
                parsed.persuasion.principles.as_ref(),
                m.persuasion.principles.as_ref()
            );
            assert_eq!(
                parsed.persuasion.phrasing_examples.as_ref(),
                m.persuasion.phrasing_examples.as_ref()
            );
            assert_eq!(parsed.related.as_ref(), m.related.as_ref());
        }
    }

    #[test]
    fn test_load_bundled_nodes_resolves_all() {
        let mut registry = MethodologyRegistry::new();
        registry.load_bundled_nodes();
        assert_eq!(registry.count(), 17);
        for id in [
            "methodology:index-priority",
            "methodology:cost-awareness",
            "methodology:least-privilege",
            "methodology:complexity-assessment",
            "methodology:boundary-enforcement",
            "methodology:using-superpowers",
            "methodology:brainstorming",
            "methodology:test-driven-development",
            "methodology:systematic-debugging",
            "methodology:verification-before-completion",
            "methodology:writing-plans",
            "methodology:dispatching-parallel-agents",
            "methodology:executing-plans",
            "methodology:finishing-a-development-branch",
            "methodology:receiving-code-review",
            "methodology:requesting-code-review",
            "methodology:subagent-driven-development",
        ] {
            assert!(
                registry.get(id).is_some(),
                "{} must resolve after load_bundled_nodes",
                id
            );
        }
    }

    #[test]
    fn test_load_bundled_nodes_overrides_builtin() {
        let mut registry = MethodologyRegistry::new();
        registry.load_bundled_nodes();
        let wp = registry.get("methodology:writing-plans").unwrap();
        assert_eq!(wp.source, "superpowers-main/skills/writing-plans/SKILL.md");
        assert_eq!(wp.domain, "planning");
    }

    #[test]
    fn test_load_jsonld_missing_dir_is_err() {
        let mut registry = MethodologyRegistry::new();
        let res = registry.load_from_jsonld_dir("/nonexistent/methodology/nodes");
        assert!(res.is_err(), "missing dir must error, not silently pass");
    }

    #[test]
    fn test_parse_activation_variants() {
        use crate::core::constitution::ConstitutionRole;
        let cases: Vec<(ActivationCondition, &str)> = vec![
            (ActivationCondition::Always, "Always"),
            (ActivationCondition::OnTaskError, "OnTaskError"),
            (
                ActivationCondition::OnHookPoint("skill_before"),
                "OnHookPoint(\"skill_before\")",
            ),
            (
                ActivationCondition::OnPhaseEnd("ACT"),
                "OnPhaseEnd(\"ACT\")",
            ),
            (
                ActivationCondition::OnToolCategory(&["file_search", "shell"]),
                "OnToolCategory([\"file_search\", \"shell\"])",
            ),
            (
                ActivationCondition::OnAgentRole(&[
                    ConstitutionRole::Supervisor,
                    ConstitutionRole::Plan,
                ]),
                "OnAgentRole([Supervisor, Plan])",
            ),
        ];
        for (condition, raw) in cases {
            let parsed =
                parse_activation_str(raw).unwrap_or_else(|e| panic!("parse {}: {}", raw, e));
            assert_eq!(
                format!("{:?}", parsed),
                format!("{:?}", condition),
                "parsed {:?} must equal {:?}",
                parsed,
                condition
            );
        }
    }

    #[test]
    fn test_parse_activation_rejects_garbage() {
        assert!(parse_activation_str("OnBogus([x])").is_err());
        assert!(parse_activation_str("").is_err());
        assert!(parse_activation_str("OnHookPoint(\"\")").is_err());
    }
}
