use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tracing::debug;

use crate::core::constitution::{ActivationCondition, ConstitutionRegistry};
use crate::methodology::{
    evolution::{EvolutionEngineHandle, ViolationReporter},
    MethodologyDefinition, MethodologyRegistry, MethodologyType, RedFlagEntry,
};
use crate::tools::hooks::{FunctionHook, HookContext, HookManager, HookPoint, HookResult};
use crate::tools::tool_guard::ToolCategory;

// ════════════════════════════════════════════════════════════════════════
// Core Types
// ════════════════════════════════════════════════════════════════════════

/// Source that triggered a methodology activation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerSource {
    HookPoint(HookPoint),
    ConstitutionRule(String),
    AgentRole(String),
    PhaseEnd(String),
    TaskError,
    ToolCategory(ToolCategory),
    Manual,
    Always,
}

/// A methodology that is currently active with runtime context
#[derive(Debug, Clone)]
pub struct ActivatedMethodology {
    /// Index into the registry's methodology list (for lookups)
    pub registry_index: usize,
    /// Methodology ID for cross-referencing
    pub methodology_id: String,
    /// What triggered this activation
    pub source: TriggerSource,
    /// Timestamp of activation (unix seconds)
    pub activated_at: u64,
}

/// Result of an anti-pattern gate check — indicates a potential violation
#[derive(Debug, Clone)]
pub struct AntiPatternGateResult {
    /// Which methodology's anti-pattern fired
    pub methodology_id: String,
    /// Which anti-pattern matched
    pub anti_pattern_name: String,
    /// Unique gate description
    pub description: String,
    /// The question the agent must ask itself
    pub gate_ask: String,
    /// Action to take on match
    pub gate_action: String,
    /// Whether this should block execution
    pub should_block: bool,
    /// Formatted warning message for injection
    pub message: String,
}

/// A constitution→methodology binding registered for dynamic triggering
#[derive(Debug, Clone)]
pub struct ConstitutionMethodologyBinding {
    pub constitution_id: String,
    pub methodology_id: String,
    pub condition: ActivationCondition,
}

// ════════════════════════════════════════════════════════════════════════
// MethodologyGate
// ════════════════════════════════════════════════════════════════════════

/// Gate that evaluates methodology activation conditions against runtime context
/// and provides anti-pattern checks, red flags, and persuasion injection.
pub struct MethodologyGate {
    /// Reference to the global methodology registry
    registry: MethodologyRegistry,
    /// Active methodologies with runtime context
    active: Vec<ActivatedMethodology>,
    /// Constitution bindings for dynamic trigger registration
    bindings: Vec<ConstitutionMethodologyBinding>,
    /// Maximum number of concurrent active methodologies
    max_active: usize,
}

impl MethodologyGate {
    /// Create a new gate with the given methodology registry.
    ///
    /// `max_active` limits concurrent active methodologies.
    pub fn new(registry: MethodologyRegistry, max_active: usize) -> Self {
        Self {
            registry,
            active: Vec::new(),
            bindings: Vec::new(),
            max_active,
        }
    }

    /// Set the maximum number of concurrently active methodologies.
    pub fn with_max_active(mut self, max: usize) -> Self {
        self.max_active = max;
        self
    }

    // ─── Activation Condition Evaluation ───

    /// Evaluate activation conditions against a hook context and update active set.
    ///
    /// Returns the newly activated methodologies (for immediate use).
    /// Call this from a hook handler to keep the gate in sync with execution context.
    pub fn on_hook_trigger(
        &mut self,
        point: HookPoint,
        context: &HookContext,
    ) -> Vec<ActivatedMethodology> {
        let mut newly_active = Vec::new();

        for (idx, methodology) in self.registry.all().iter().enumerate() {
            if self.active.iter().any(|a| a.registry_index == idx) {
                continue;
            }

            let source = match self.evaluate_activation(&methodology.activation, point, context) {
                Some(source) => source,
                None => continue,
            };

            let activated = ActivatedMethodology {
                registry_index: idx,
                methodology_id: methodology.id.to_string(),
                source,
                activated_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            debug!(
                methodology = %methodology.id,
                point = ?point,
                "MethodologyGate: Activated"
            );

            if self.active.len() < self.max_active {
                self.active.push(activated.clone());
                newly_active.push(activated);
            }
        }

        for binding in &self.bindings {
            let mid = &binding.methodology_id;
            if self.active.iter().any(|a| a.methodology_id == *mid) {
                continue;
            }

            let idx = match self.registry.all().iter().position(|m| m.id == *mid) {
                Some(i) => i,
                None => continue,
            };

            let source = match self.evaluate_activation(&binding.condition, point, context) {
                Some(s) => s,
                None => continue,
            };

            let activated = ActivatedMethodology {
                registry_index: idx,
                methodology_id: mid.clone(),
                source,
                activated_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            if self.active.len() < self.max_active {
                self.active.push(activated.clone());
                newly_active.push(activated);
            }
        }

        newly_active
    }

    /// Evaluate a single ActivationCondition against context.
    fn evaluate_activation(
        &self,
        condition: &ActivationCondition,
        point: HookPoint,
        context: &HookContext,
    ) -> Option<TriggerSource> {
        match condition {
            ActivationCondition::Always => Some(TriggerSource::Always),
            ActivationCondition::OnToolCategory(categories) => {
                let tool_name = context
                    .data
                    .get("tool_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if categories
                    .iter()
                    .any(|c| tool_name.contains(c) || category_matches_tool(c, tool_name))
                {
                    Some(TriggerSource::ToolCategory(ToolCategory::Meta))
                } else {
                    None
                }
            }
            ActivationCondition::OnHookPoint(hook_str) => {
                if point.as_str() == *hook_str {
                    Some(TriggerSource::HookPoint(point))
                } else {
                    None
                }
            }
            ActivationCondition::OnPhaseEnd(phase) => {
                if point == HookPoint::PhaseEnd {
                    let ctx_phase = context
                        .data
                        .get("phase")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if ctx_phase == *phase {
                        Some(TriggerSource::PhaseEnd(phase.to_string()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            ActivationCondition::OnTaskError => {
                if point == HookPoint::TaskError && context.error.is_some() {
                    Some(TriggerSource::TaskError)
                } else {
                    None
                }
            }
            ActivationCondition::OnAgentRole(roles) => {
                if roles.iter().any(|r| {
                    let role_str = r.as_str();
                    context.agent_role == role_str
                        || context.agent_role.to_lowercase() == role_str.to_lowercase()
                }) {
                    Some(TriggerSource::AgentRole(context.agent_role.clone()))
                } else {
                    None
                }
            }
        }
    }

    // ─── Activation Lifecycle ───

    /// Manually activate a methodology by ID.
    pub fn activate(&mut self, methodology_id: &str, source: TriggerSource) -> Option<usize> {
        let idx = self
            .registry
            .all()
            .iter()
            .position(|m| m.id == methodology_id)?;
        if self.active.iter().any(|a| a.registry_index == idx) {
            return Some(idx);
        }
        if self.active.len() >= self.max_active {
            return None;
        }
        self.active.push(ActivatedMethodology {
            registry_index: idx,
            methodology_id: methodology_id.to_string(),
            source,
            activated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        });
        Some(idx)
    }

    /// Manually deactivate a methodology by ID.
    pub fn deactivate(&mut self, methodology_id: &str) {
        self.active.retain(|a| a.methodology_id != methodology_id);
    }

    /// Clear all active methodologies.
    pub fn reset(&mut self) {
        self.active.clear();
    }

    /// Get the definition for an activated methodology.
    pub fn get_definition(
        &self,
        activated: &ActivatedMethodology,
    ) -> Option<&MethodologyDefinition> {
        self.registry.all().get(activated.registry_index)
    }

    // ─── Query Active Methodologies ───

    /// Get all currently active methodologies.
    pub fn active_methodologies(&self) -> &[ActivatedMethodology] {
        &self.active
    }

    /// Get the number of active methodologies.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Check if a specific methodology is active.
    pub fn is_active(&self, methodology_id: &str) -> bool {
        self.active
            .iter()
            .any(|a| a.methodology_id == methodology_id)
    }

    /// Get active methodologies for a specific domain.
    pub fn active_for_domain(
        &self,
        domain: &str,
    ) -> Vec<(&ActivatedMethodology, &MethodologyDefinition)> {
        self.active
            .iter()
            .filter_map(|a| {
                self.registry
                    .all()
                    .get(a.registry_index)
                    .map(|def| (a, def))
            })
            .filter(|(_, def)| def.domain == domain || def.domain == "general")
            .collect()
    }

    // ─── Red Flags & Rationalizations ───

    /// Collect all red flags from currently active methodologies.
    pub fn active_red_flags(&self) -> Vec<(&ActivatedMethodology, &RedFlagEntry)> {
        let mut flags = Vec::new();
        for activated in &self.active {
            if let Some(def) = self.registry.all().get(activated.registry_index) {
                for red_flag in def.red_flags {
                    flags.push((activated, red_flag));
                }
            }
        }
        flags
    }

    /// Collect all rationalization checks (red flags with rationalization text) from active methodologies.
    pub fn active_rationalizations(&self) -> Vec<(&ActivatedMethodology, &RedFlagEntry, &str)> {
        let mut result = Vec::new();
        for activated in &self.active {
            if let Some(def) = self.registry.all().get(activated.registry_index) {
                for red_flag in def.red_flags {
                    if let Some(check) = red_flag.rationalization_check {
                        result.push((activated, red_flag, check));
                    }
                }
            }
        }
        result
    }

    // ─── Anti-Pattern Gate Checks ───

    /// Check all active methodologies for anti-patterns relevant to the current tool context.
    ///
    /// Returns a list of gate results. Any result with `should_block: true` should
    /// block the tool call until the check is addressed.
    pub fn check_anti_patterns_for_tool(&self, tool_name: &str) -> Vec<AntiPatternGateResult> {
        let mut results = Vec::new();

        for activated in &self.active {
            let def = match self.registry.all().get(activated.registry_index) {
                Some(d) => d,
                None => continue,
            };

            for ap in def.anti_patterns {
                let is_match = ap
                    .gate_before
                    .to_lowercase()
                    .contains(&tool_name.to_lowercase())
                    || tool_name_matches_gate(tool_name, ap.gate_before);

                if !is_match {
                    continue;
                }

                let should_block =
                    ap.gate_action.starts_with("STOP") || ap.gate_action.starts_with("ABORT");

                results.push(AntiPatternGateResult {
                    methodology_id: def.id.to_string(),
                    anti_pattern_name: ap.name.to_string(),
                    description: ap.description.to_string(),
                    gate_ask: ap.gate_ask.to_string(),
                    gate_action: ap.gate_action.to_string(),
                    should_block,
                    message: format!(
                        "⚠️ Anti-pattern [{}]: {} — {}\nAsk yourself: {}\nAction: {}",
                        def.name, ap.name, ap.description, ap.gate_ask, ap.gate_action
                    ),
                });
            }
        }

        results
    }

    // ─── Persuasion Injection ───

    /// Generate persuasion directives from active methodologies.
    ///
    /// These are formatted for inclusion in system prompts to frame the agent's mindset.
    pub fn persuasive_directives(&self) -> Vec<String> {
        let mut directives = Vec::new();
        for activated in &self.active {
            if let Some(def) = self.registry.all().get(activated.registry_index) {
                let prefix = match def.methodology_type {
                    MethodologyType::Discipline => "📜 [Strict Discipline]",
                    MethodologyType::Guidance => "💡 [Guidance Suggestion]",
                    MethodologyType::Process => "📋 [Process Standard]",
                    MethodologyType::Reference => "📖 [Reference Material]",
                };
                for example in def.persuasion.phrasing_examples {
                    directives.push(format!("{} {} ({})", prefix, example, def.name));
                }
            }
        }
        directives
    }

    // ─── Constitution Binding ───

    /// Register a binding from a constitution rule to a methodology with trigger condition.
    pub fn bind_constitution_to_methodology(
        &mut self,
        constitution_id: &str,
        methodology_id: &str,
        condition: ActivationCondition,
    ) {
        self.bindings
            .retain(|b| b.constitution_id != constitution_id || b.methodology_id != methodology_id);
        self.bindings.push(ConstitutionMethodologyBinding {
            constitution_id: constitution_id.to_string(),
            methodology_id: methodology_id.to_string(),
            condition,
        });
    }

    /// Load all constitution→methodology bindings from a ConstitutionRegistry.
    ///
    /// This registers triggers so that when a constitution rule's trigger condition
    /// is met, the associated methodology is automatically activated.
    pub fn register_constitution_bindings(&mut self, constitution: &ConstitutionRegistry) {
        let mut count = 0;
        for entry in constitution.all() {
            if let Some(bindings) = constitution.get_bindings(entry.id) {
                for binding in bindings {
                    self.bind_constitution_to_methodology(
                        entry.id,
                        binding.methodology_id,
                        binding.trigger.clone(),
                    );
                    count += 1;
                }
            }
        }
        debug!(
            bindings_registered = count,
            "MethodologyGate: Constitution bindings registered"
        );
    }

    /// Get all registered constitution bindings.
    pub fn get_bindings(&self) -> &[ConstitutionMethodologyBinding] {
        &self.bindings
    }

    /// Get the number of registered bindings.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Get a reference to the underlying registry.
    pub fn registry(&self) -> &MethodologyRegistry {
        &self.registry
    }
}

// ════════════════════════════════════════════════════════════════════════
// Thread-Safe Handle for Hook Registration
// ════════════════════════════════════════════════════════════════════════

/// Thread-safe handle wrapping MethodologyGate behind Arc<RwLock>,
/// with optional EvolutionEngine for recording violation history.
///
/// This is the recommended way to register MethodologyGate with HookManager,
/// since the gate maintains mutable active-state tracking.
#[derive(Clone)]
pub struct MethodologyGateHandle {
    inner: std::sync::Arc<parking_lot::RwLock<MethodologyGate>>,
    evolution: Option<EvolutionEngineHandle>,
}

impl MethodologyGateHandle {
    pub fn new(gate: MethodologyGate) -> Self {
        Self {
            inner: std::sync::Arc::new(parking_lot::RwLock::new(gate)),
            evolution: None,
        }
    }

    pub fn with_evolution(mut self, evolution: EvolutionEngineHandle) -> Self {
        self.evolution = Some(evolution);
        self
    }

    /// Get a clone of the inner Arc for hook closures.
    pub fn inner(&self) -> std::sync::Arc<parking_lot::RwLock<MethodologyGate>> {
        self.inner.clone()
    }

    /// Get a clone of the evolution engine handle.
    pub fn evolution_handle(&self) -> Option<EvolutionEngineHandle> {
        self.evolution.clone()
    }

    /// Register hooks into HookManager.
    ///
    /// This method registers FunctionHooks that call into the MethodologyGate
    /// at each relevant hook point:
    /// - TaskError: auto-activates task-error methodologies
    /// - PhaseEnd: auto-activates phase-end methodologies  
    /// - SkillBefore: checks anti-pattern gates before tool calls
    /// - AgentInit: activates always-on methodologies
    pub fn register_hooks(&self, hook_manager: &HookManager) {
        let gate = self.inner.clone();
        let evolution = self.evolution.clone();

        // ── AgentInit: Activate "Always" methodologies ──
        let gate_always = gate.clone();
        let evo_init = evolution.clone();
        let always_hook = FunctionHook::new(
            "methodology_gate::agent_init",
            vec![HookPoint::AgentInit],
            50,
            move |ctx: &mut HookContext| {
                let mut g = gate_always.write();
                let context_clone = ctx.clone();
                let activated = g.on_hook_trigger(HookPoint::AgentInit, &context_clone);
                if let Some(ref evo) = evo_init {
                    let inner = evo.inner();
                    let mut e = inner.write();
                    for a in &activated {
                        e.record_activation(&a.methodology_id);
                    }
                }
                HookResult::Continue
            },
        );
        hook_manager.register_arc(std::sync::Arc::new(always_hook));

        // ── TaskError: Activate task-error + debugging methodologies ──
        let gate_error = gate.clone();
        let evo_error = evolution.clone();
        let error_hook = FunctionHook::new(
            "methodology_gate::task_error",
            vec![HookPoint::TaskError],
            50,
            move |ctx: &mut HookContext| {
                let mut g = gate_error.write();
                let context_clone = ctx.clone();
                let activated = g.on_hook_trigger(HookPoint::TaskError, &context_clone);
                if let Some(ref evo) = evo_error {
                    let inner = evo.inner();
                    let mut e = inner.write();
                    for a in &activated {
                        e.record_activation(&a.methodology_id);
                    }
                }
                HookResult::Continue
            },
        );
        hook_manager.register_arc(std::sync::Arc::new(error_hook));

        // ── PhaseEnd: Activate phase-end methodologies ──
        let gate_phase = gate.clone();
        let evo_phase = evolution.clone();
        let phase_hook = FunctionHook::new(
            "methodology_gate::phase_end",
            vec![HookPoint::PhaseEnd],
            50,
            move |ctx: &mut HookContext| {
                let mut g = gate_phase.write();
                let context_clone = ctx.clone();
                let activated = g.on_hook_trigger(HookPoint::PhaseEnd, &context_clone);
                if let Some(ref evo) = evo_phase {
                    let inner = evo.inner();
                    let mut e = inner.write();
                    for a in &activated {
                        e.record_activation(&a.methodology_id);
                    }
                }
                HookResult::Continue
            },
        );
        hook_manager.register_arc(std::sync::Arc::new(phase_hook));

        // ── SkillBefore: Check anti-pattern gates + record violations ──
        let gate_tool = gate.clone();
        let evo_tool = evolution.clone();
        let tool_hook = FunctionHook::new(
            "methodology_gate::skill_before",
            vec![HookPoint::SkillBefore],
            60,
            move |ctx: &mut HookContext| {
                let g = gate_tool.read();
                let tool_name = match ctx.data.get("tool_name").and_then(|v| v.as_str()) {
                    Some(name) => name,
                    None => return HookResult::Continue,
                };

                let anti_patterns = g.check_anti_patterns_for_tool(tool_name);
                let role = &ctx.agent_role;
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                if let Some(ref evo) = evo_tool {
                    let inner = evo.inner();
                    let mut e = inner.write();
                    for ap in &anti_patterns {
                        let record =
                            ViolationReporter::from_anti_pattern(ap, role, None, timestamp);
                        e.record_violation(record);
                    }
                }

                let blocking: Vec<_> = anti_patterns.iter().filter(|r| r.should_block).collect();

                if !blocking.is_empty() {
                    let messages: Vec<String> =
                        blocking.iter().map(|r| r.message.clone()).collect();
                    ctx.metadata.insert(
                        "methodology_anti_patterns".to_string(),
                        Value::Array(messages.into_iter().map(Value::String).collect()),
                    );

                    let first = &blocking[0];
                    ctx.error = Some(format!(
                        "MethodologyGate anti-pattern block [{}]: {}",
                        first.anti_pattern_name, first.description
                    ));
                    debug!(
                        methodology = %first.methodology_id,
                        anti_pattern = %first.anti_pattern_name,
                        "MethodologyGate: Anti-pattern blocked tool call"
                    );
                    return HookResult::Abort;
                }

                let warnings: Vec<String> = anti_patterns
                    .iter()
                    .filter(|r| !r.should_block)
                    .map(|r| r.message.clone())
                    .collect();
                if !warnings.is_empty() {
                    ctx.metadata.insert(
                        "methodology_anti_pattern_warnings".to_string(),
                        Value::Array(warnings.into_iter().map(Value::String).collect()),
                    );
                }

                HookResult::Continue
            },
        );
        hook_manager.register_arc(std::sync::Arc::new(tool_hook));
    }
}

// ════════════════════════════════════════════════════════════════════════
// Helper Functions
// ════════════════════════════════════════════════════════════════════════

fn tool_name_matches_gate(tool_name: &str, gate_before: &str) -> bool {
    let tl = tool_name.to_lowercase();
    let gb = gate_before.to_lowercase();

    if gb.contains(&tl) {
        return true;
    }

    let tool_keywords: &[(&str, &[&str])] = &[
        (
            "glob",
            &[
                "glob",
                "目录遍历",
                "文件搜索",
                "find",
                "directory traversal",
                "file search",
            ],
        ),
        (
            "grep_search",
            &["grep", "搜索", "search", "内容查找", "content search"],
        ),
        (
            "bash",
            &[
                "bash", "shell", "命令", "终端", "执行", "command", "terminal",
            ],
        ),
        (
            "file_read",
            &["read", "文件读取", "file_read", "reading file"],
        ),
        (
            "file_write",
            &[
                "write",
                "文件写入",
                "创建文件",
                "file_write",
                "writing file",
            ],
        ),
        (
            "file_edit",
            &["edit", "编辑", "修改文件", "file_edit", "editing file"],
        ),
        ("web_fetch", &["fetch", "网络", "http", "web", "website"]),
        (
            "web_search",
            &["web_search", "网络搜索", "搜索", "internet search"],
        ),
    ];

    for (key, keywords) in tool_keywords {
        if tl == *key || tl.contains(key) {
            if keywords.iter().any(|k| gb.contains(k)) {
                return true;
            }
        }
    }

    false
}

fn category_matches_tool(category: &str, tool_name: &str) -> bool {
    let tl = tool_name.to_lowercase();
    let cat = category.to_lowercase();

    let mappings: &[(&str, &[&str])] = &[
        ("file_read", &["file_read", "file_list", "glob", "grep"]),
        ("file_write", &["file_write", "file_edit"]),
        ("file_search", &["grep", "glob", "search"]),
        ("search", &["grep", "glob", "search"]),
        ("shell", &["bash", "shell", "sh", "zsh"]),
        ("network", &["http", "fetch", "web", "url", "curl"]),
        ("code", &["bash", "code", "exec"]),
    ];

    for (cat_key, tools) in mappings {
        if tl.as_str() == *cat_key || cat.contains(cat_key) {
            if tools.iter().any(|t| tl.contains(t)) {
                return true;
            }
        }
    }

    cat.contains(&tl) || tl.contains(&cat)
}

// ════════════════════════════════════════════════════════════════════════
// JSON-LD Serialization for Methodology Definitions
// ════════════════════════════════════════════════════════════════════════

/// Convert a MethodologyDefinition to a JSON-LD node.
///
/// This enables persistence, query, and dynamic loading of methodology definitions
/// through the Gliding Horse JSON-LD data bus.
impl MethodologyDefinition {
    pub fn to_json_ld(&self) -> serde_json::Value {
        let mut node = serde_json::json!({
            "@context": {
                "schema": "https://schema.org/",
                "methodology": "https://gliding.horse/ontology/methodology#"
            },
            "@id": format!("https://gliding.horse/methodology/{}", self.id.replace(':', "/")),
            "@type": "methodology:Methodology",
            "methodology:id": self.id,
            "methodology:name": self.name,
            "methodology:description": self.description,
            "methodology:type": format!("{:?}", self.methodology_type),
            "methodology:domain": self.domain,
            "methodology:source": self.source,
            "methodology:activation": format!("{:?}", self.activation),
        });

        // Red flags
        if !self.red_flags.is_empty() {
            let flags: Vec<serde_json::Value> = self
                .red_flags
                .iter()
                .map(|rf| {
                    let mut f = serde_json::json!({
                        "methodology:pattern": rf.pattern,
                        "methodology:severity": format!("{:?}", rf.severity),
                    });
                    if let Some(check) = rf.rationalization_check {
                        f["methodology:rationalizationCheck"] =
                            serde_json::Value::String(check.to_string());
                    }
                    f
                })
                .collect();
            node["methodology:redFlags"] = serde_json::Value::Array(flags);
        }

        // Anti-patterns
        if !self.anti_patterns.is_empty() {
            let aps: Vec<serde_json::Value> = self
                .anti_patterns
                .iter()
                .map(|ap| {
                    serde_json::json!({
                        "methodology:antiPatternName": ap.name,
                        "methodology:antiPatternDescription": ap.description,
                        "methodology:gateBefore": ap.gate_before,
                        "methodology:gateAsk": ap.gate_ask,
                        "methodology:gateAction": ap.gate_action,
                    })
                })
                .collect();
            node["methodology:antiPatterns"] = serde_json::Value::Array(aps);
        }

        // Persuasion
        node["methodology:persuasion"] = serde_json::json!({
            "methodology:principles": self.persuasion.principles,
            "methodology:phrasingExamples": self.persuasion.phrasing_examples,
        });

        // Related methodologies
        if !self.related.is_empty() {
            let related: Vec<serde_json::Value> = self
                .related
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "@id": format!("https://gliding.horse/methodology/{}", r.replace(':', "/")),
                        "methodology:id": r,
                    })
                })
                .collect();
            node["methodology:related"] = serde_json::Value::Array(related);
        }

        node
    }

    pub fn to_kg_quads(&self) -> Vec<crate::knowledge_graph::types::RdfQuad> {
        use crate::knowledge_graph::types::{RdfQuad, RdfValue};
        let id_iri = format!(
            "https://gliding.horse/methodology/{}",
            self.id.replace(':', "/")
        );

        let mut quads = vec![
            RdfQuad {
                subject: id_iri.clone(),
                predicate: "rdf:type".into(),
                object: RdfValue::Iri("methodology:Methodology".into()),
                graph: None,
            },
            RdfQuad {
                subject: id_iri.clone(),
                predicate: "rdfs:label".into(),
                object: RdfValue::Literal(self.name.to_string()),
                graph: None,
            },
            RdfQuad {
                subject: id_iri.clone(),
                predicate: "methodology:id".into(),
                object: RdfValue::Literal(self.id.to_string()),
                graph: None,
            },
            RdfQuad {
                subject: id_iri.clone(),
                predicate: "schema:description".into(),
                object: RdfValue::Literal(self.description.to_string()),
                graph: None,
            },
            RdfQuad {
                subject: id_iri.clone(),
                predicate: "methodology:type".into(),
                object: RdfValue::Literal(format!("{:?}", self.methodology_type)),
                graph: None,
            },
            RdfQuad {
                subject: id_iri.clone(),
                predicate: "methodology:domain".into(),
                object: RdfValue::Literal(self.domain.to_string()),
                graph: None,
            },
            RdfQuad {
                subject: id_iri.clone(),
                predicate: "methodology:source".into(),
                object: RdfValue::Literal(self.source.to_string()),
                graph: None,
            },
            RdfQuad {
                subject: id_iri.clone(),
                predicate: "methodology:activation".into(),
                object: RdfValue::Literal(format!("{:?}", self.activation)),
                graph: None,
            },
        ];

        for rf in self.red_flags {
            quads.push(RdfQuad {
                subject: id_iri.clone(),
                predicate: "methodology:hasRedFlag".into(),
                object: RdfValue::Literal(rf.pattern.to_string()),
                graph: None,
            });
        }

        for ap in self.anti_patterns {
            quads.push(RdfQuad {
                subject: id_iri.clone(),
                predicate: "methodology:hasAntiPattern".into(),
                object: RdfValue::Literal(ap.name.to_string()),
                graph: None,
            });
        }

        quads
    }
}

#[cfg(test)]
mod kg_bridge_tests {
    use super::*;

    #[test]
    fn test_methodology_to_kg_quads() {
        let registry = MethodologyRegistry::new();
        for m in registry.all() {
            let quads = m.to_kg_quads();
            assert!(!quads.is_empty(), "{} should produce quads", m.id);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// ConstitutionRole helper
// ════════════════════════════════════════════════════════════════════════

impl crate::core::constitution::ConstitutionRole {
    /// Get the string representation of the role.
    pub fn as_str(&self) -> &'static str {
        match self {
            crate::core::constitution::ConstitutionRole::Universal => "Universal",
            crate::core::constitution::ConstitutionRole::Supervisor => "SA",
            crate::core::constitution::ConstitutionRole::Plan => "PA",
            crate::core::constitution::ConstitutionRole::Do => "DA",
            crate::core::constitution::ConstitutionRole::Check => "CA",
            crate::core::constitution::ConstitutionRole::Act => "AA",
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_gate() -> MethodologyGate {
        MethodologyGate::new(MethodologyRegistry::new(), 20)
    }

    fn test_context(point: HookPoint, role: &str, tool_name: Option<&str>) -> HookContext {
        let mut ctx = HookContext::new(point, "test_agent", role);
        if let Some(name) = tool_name {
            ctx = ctx.with_data("tool_name", Value::String(name.to_string()));
        }
        ctx
    }

    // ─── Basic Activation Tests ───

    #[test]
    fn test_gate_always_active() {
        let mut gate = test_gate();
        let ctx = test_context(HookPoint::AgentInit, "SA", None);
        let activated = gate.on_hook_trigger(HookPoint::AgentInit, &ctx);

        let always_ids: Vec<&str> = gate
            .registry()
            .all()
            .iter()
            .filter(|m| matches!(m.activation, ActivationCondition::Always))
            .map(|m| m.id)
            .collect();

        let activated_ids: Vec<&str> = activated
            .iter()
            .map(|a| a.methodology_id.as_str())
            .collect();

        for id in &always_ids {
            assert!(
                activated_ids.contains(id),
                "Always-active methodology {} was not activated",
                id
            );
        }
    }

    #[test]
    fn test_gate_task_error_triggers_debugging() {
        let mut gate = test_gate();
        let mut ctx = test_context(HookPoint::TaskError, "DA", None);
        ctx.error = Some("test error".to_string());

        let activated = gate.on_hook_trigger(HookPoint::TaskError, &ctx);

        let activated_ids: Vec<&str> = activated
            .iter()
            .map(|a| a.methodology_id.as_str())
            .collect();

        assert!(
            activated_ids.contains(&"methodology:systematic-debugging"),
            "systematic-debugging should activate on TaskError"
        );
    }

    #[test]
    fn test_agent_role_activation() {
        let mut gate = test_gate();
        let ctx = test_context(HookPoint::AgentInit, "SA", None);
        let activated = gate.on_hook_trigger(HookPoint::AgentInit, &ctx);

        let activated_ids: Vec<&str> = activated
            .iter()
            .map(|a| a.methodology_id.as_str())
            .collect();

        assert!(
            activated_ids.contains(&"methodology:complexity-assessment"),
            "complexity-assessment should activate for SA role"
        );
    }

    #[test]
    fn test_manual_activate_deactivate() {
        let mut gate = test_gate();

        assert!(!gate.is_active("methodology:index-priority"));
        let result = gate.activate("methodology:index-priority", TriggerSource::Manual);
        assert!(result.is_some());
        assert!(gate.is_active("methodology:index-priority"));
        assert_eq!(gate.active_count(), 1);

        gate.deactivate("methodology:index-priority");
        assert!(!gate.is_active("methodology:index-priority"));
        assert_eq!(gate.active_count(), 0);
    }

    #[test]
    fn test_reset_clears_active() {
        let mut gate = test_gate();
        gate.activate("methodology:index-priority", TriggerSource::Manual);
        gate.activate("methodology:cost-awareness", TriggerSource::Manual);
        assert_eq!(gate.active_count(), 2);

        gate.reset();
        assert_eq!(gate.active_count(), 0);
    }

    // ─── Red Flags & Rationalizations ───

    #[test]
    fn test_active_red_flags() {
        let mut gate = test_gate();
        gate.activate("methodology:using-superpowers", TriggerSource::Manual);

        let flags = gate.active_red_flags();
        assert!(!flags.is_empty(), "using-superpowers should have red flags");
        assert!(flags.iter().any(|(_, rf)| rf.pattern.contains("skipping")));
    }

    #[test]
    fn test_active_rationalizations() {
        let mut gate = test_gate();
        gate.activate("methodology:index-priority", TriggerSource::Manual);

        let rationalizations = gate.active_rationalizations();
        assert!(
            !rationalizations.is_empty(),
            "index-priority should have rationalization checks"
        );
        assert!(rationalizations
            .iter()
            .any(|(_, _, check)| check.contains("directory")));
    }

    // ─── Anti-Pattern Gate Checks ───

    #[test]
    fn test_anti_pattern_check_for_tool() {
        let mut gate = test_gate();
        gate.activate("methodology:index-priority", TriggerSource::Manual);

        let results = gate.check_anti_patterns_for_tool("glob");
        assert!(
            !results.is_empty(),
            "glob should trigger index-priority anti-pattern"
        );

        gate.activate("methodology:cost-awareness", TriggerSource::Manual);
        let bash_results = gate.check_anti_patterns_for_tool("bash");
        assert!(
            !bash_results.is_empty(),
            "bash should trigger cost-awareness anti-pattern"
        );
    }

    #[test]
    fn test_anti_pattern_blocking() {
        let mut gate = test_gate();
        gate.activate("methodology:boundary-enforcement", TriggerSource::Manual);

        let results = gate.check_anti_patterns_for_tool("bash");
        let blocking: Vec<_> = results.iter().filter(|r| r.should_block).collect();
        assert!(
            blocking.is_empty()
                || blocking[0].gate_action.contains("STOP")
                || blocking[0].gate_action.contains("ABORT"),
            "boundary enforcement anti-patterns should block"
        );
    }

    #[test]
    fn test_no_false_positive_anti_pattern() {
        let gate = test_gate();
        let results = gate.check_anti_patterns_for_tool("bash");
        assert!(
            results.is_empty(),
            "Without active methodologies, no anti-patterns should fire"
        );
    }

    // ─── Persuasion ───

    #[test]
    fn test_persuasive_directives() {
        let mut gate = test_gate();
        gate.activate(
            "methodology:verification-before-completion",
            TriggerSource::Manual,
        );

        let directives = gate.persuasive_directives();
        assert!(
            !directives.is_empty(),
            "verification should have persuasion directives"
        );
        assert!(directives.iter().any(|d| d.contains("Always verify")));
    }

    #[test]
    fn test_discipline_persuasion_prefix() {
        let mut gate = test_gate();
        gate.activate("methodology:test-driven-development", TriggerSource::Manual);

        let directives = gate.persuasive_directives();
        assert!(
            directives.iter().any(|d| d.contains("Strict Discipline")),
            "Discipline-type methodologies should use Strict Discipline prefix"
        );
    }

    // ─── Constitution Binding ───

    #[test]
    fn test_constitution_binding() {
        let mut gate = test_gate();
        let constitution = ConstitutionRegistry::new();

        gate.register_constitution_bindings(&constitution);

        assert!(
            gate.binding_count() > 30,
            "Should register 30+ bindings from constitution, got {}",
            gate.binding_count()
        );
    }

    #[test]
    fn test_bind_constitution_to_methodology() {
        let mut gate = test_gate();
        gate.bind_constitution_to_methodology(
            "uni-verification-2",
            "methodology:systematic-debugging",
            ActivationCondition::OnTaskError,
        );

        assert_eq!(gate.binding_count(), 1);
    }

    #[test]
    fn test_dedup_bindings() {
        let mut gate = test_gate();
        gate.bind_constitution_to_methodology(
            "uni-verification-2",
            "methodology:systematic-debugging",
            ActivationCondition::OnTaskError,
        );
        gate.bind_constitution_to_methodology(
            "uni-verification-2",
            "methodology:systematic-debugging",
            ActivationCondition::OnTaskError,
        );

        assert_eq!(
            gate.binding_count(),
            1,
            "Duplicate bindings should be deduplicated"
        );
    }

    // ─── Max Active Limit ───

    #[test]
    fn test_max_active_limit() {
        let mut gate = test_gate();
        gate.max_active = 2;

        assert!(gate
            .activate("methodology:index-priority", TriggerSource::Manual)
            .is_some());
        assert!(gate
            .activate("methodology:cost-awareness", TriggerSource::Manual)
            .is_some());
        assert!(
            gate.activate("methodology:least-privilege", TriggerSource::Manual)
                .is_none(),
            "Should hit max active limit"
        );

        assert_eq!(gate.active_count(), 2);
    }

    // ─── Domain Filtering ───

    #[test]
    fn test_active_for_domain() {
        let mut gate = test_gate();
        gate.activate("methodology:test-driven-development", TriggerSource::Manual);
        gate.activate("methodology:index-priority", TriggerSource::Manual);

        let prog = gate.active_for_domain("programming");
        let general = gate.active_for_domain("general");

        assert!(
            !prog.is_empty(),
            "Should have programming domain methodologies"
        );
        assert!(
            !general.is_empty(),
            "Should have general domain methodologies"
        );
    }

    // ─── JSON-LD Serialization ───

    #[test]
    fn test_methodology_to_json_ld() {
        let registry = MethodologyRegistry::new();
        let methodology = registry.get("methodology:index-priority").unwrap();

        let json = methodology.to_json_ld();

        assert_eq!(json["@type"], "methodology:Methodology");
        assert_eq!(json["methodology:id"], "methodology:index-priority");
        assert!(json.get("methodology:redFlags").is_some());
        assert!(json.get("methodology:antiPatterns").is_some());
        assert!(json.get("methodology:persuasion").is_some());
    }

    #[test]
    fn test_all_methodologies_serializable() {
        let registry = MethodologyRegistry::new();
        for methodology in registry.all() {
            let json = methodology.to_json_ld();
            assert_eq!(
                json["methodology:id"], methodology.id,
                "JSON-LD id mismatch for {}",
                methodology.id
            );
        }
    }

    // ─── Hook Context ───

    #[test]
    fn test_agent_role_tracking() {
        let mut gate = test_gate();
        let ctx = test_context(HookPoint::AgentInit, "SA", None);
        gate.on_hook_trigger(HookPoint::AgentInit, &ctx);

        assert!(
            gate.is_active("methodology:complexity-assessment"),
            "complexity-assessment should be active for SA role"
        );
    }

    #[test]
    fn test_phase_end_activation() {
        let mut gate = test_gate();
        let mut ctx = test_context(HookPoint::PhaseEnd, "DA", None);
        ctx = ctx.with_data("phase", Value::String("ACT".to_string()));

        gate.on_hook_trigger(HookPoint::PhaseEnd, &ctx);

        assert!(
            gate.is_active("methodology:verification-before-completion"),
            "verification-before-completion should activate on PhaseEnd(ACT)"
        );
    }

    // ─── Edge Cases ───

    #[test]
    fn test_deactivate_nonexistent() {
        let mut gate = test_gate();
        gate.deactivate("methodology:nonexistent");
        assert_eq!(gate.active_count(), 0);
    }

    #[test]
    fn test_activate_nonexistent() {
        let mut gate = test_gate();
        let result = gate.activate("methodology:nonexistent", TriggerSource::Manual);
        assert!(
            result.is_none(),
            "Activating nonexistent methodology should return None"
        );
    }

    #[test]
    fn test_get_definition_for_activated() {
        let mut gate = test_gate();
        gate.activate("methodology:index-priority", TriggerSource::Manual);

        let activated = &gate.active[0];
        let def = gate.get_definition(activated);
        assert!(def.is_some());
        assert_eq!(def.unwrap().id, "methodology:index-priority");
    }

    #[test]
    fn test_handle_double_activation() {
        let mut gate = test_gate();
        gate.activate("methodology:index-priority", TriggerSource::Manual);
        gate.activate("methodology:index-priority", TriggerSource::Manual);
        assert_eq!(gate.active_count(), 1);
    }
}
