use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::memory::l0_store::{L0Store, MesiState};
use crate::CoreError;

/// Cosine similarity calculation
///
/// Computes cosine similarity between two equal-length f32 vectors.
/// Range: [-1.0, 1.0], 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite.
/// Used in L1 eviction policy for semantic relevance evaluation between turns and queries.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0) as f64
}

/// L1 eviction policy weight configuration
///
/// Controls the weights of three evaluation metrics in `evict_by_policy()`.
/// Different agent roles use different configurations to optimize retained context.
///
/// Formula: `score = recency_weight * (1/time_since) + relevance_weight * (1/semantic_relevance) + cost_weight * token_cost`
///
/// Where `semantic_relevance = beta * query_sim + (1-beta) * task_relevance`
///
/// Enhancement: Added hard threshold filtering (relevance_threshold + safe_window_seconds),
/// entries with low relevance beyond the safe window are directly evicted without score ranking.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct EvictionConfig {
    pub recency_weight: f64,
    pub relevance_weight: f64,
    pub cost_weight: f64,
    /// Hard threshold for low relevance: relevance_score < this AND beyond safe window → direct eviction
    pub relevance_threshold: f64,
    /// Safe window in seconds: minimum time to keep even low-relevance entries
    pub safe_window_seconds: i64,
    /// Beta fusion weight: β * query_sim + (1-β) * task_relevance
    pub beta: f64,
}

impl EvictionConfig {
    /// Default config — for Supervisor (SA), broad perspective
    pub const fn default_sa() -> Self {
        Self {
            recency_weight: 0.30,
            relevance_weight: 0.40,
            cost_weight: 0.30,
            relevance_threshold: 0.3,
            safe_window_seconds: 300,
            beta: 0.7,
        }
    }

    /// Plan (PA) — prioritize plan-structure-related history
    pub const fn plan() -> Self {
        Self {
            recency_weight: 0.20,
            relevance_weight: 0.60,
            cost_weight: 0.20,
            relevance_threshold: 0.3,
            safe_window_seconds: 300,
            beta: 0.7,
        }
    }

    /// Do (DA) — prioritize recent technical details, balance token cost
    pub const fn do_agent() -> Self {
        Self {
            recency_weight: 0.35,
            relevance_weight: 0.30,
            cost_weight: 0.35,
            relevance_threshold: 0.3,
            safe_window_seconds: 300,
            beta: 0.7,
        }
    }

    /// Check (CA) — prioritize audit standards and verification relevance
    pub const fn check() -> Self {
        Self {
            recency_weight: 0.15,
            relevance_weight: 0.65,
            cost_weight: 0.20,
            relevance_threshold: 0.3,
            safe_window_seconds: 300,
            beta: 0.7,
        }
    }

    /// Act (AA) — balanced config, slightly biased toward decision context
    pub const fn act() -> Self {
        Self {
            recency_weight: 0.25,
            relevance_weight: 0.45,
            cost_weight: 0.30,
            relevance_threshold: 0.3,
            safe_window_seconds: 300,
            beta: 0.7,
        }
    }

    pub fn for_role(role: &str) -> Self {
        match role {
            "Plan" | "PA" => Self::plan(),
            "Do" | "DA" | "Executor" => Self::do_agent(),
            "Check" | "CA" | "Reviewer" => Self::check(),
            "Act" | "AA" | "Decision" => Self::act(),
            _ => Self::default_sa(),
        }
    }
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self::default_sa()
    }
}

/// L1 single-turn summary record
///
/// L1 only stores the `summary` field of LLM responses.
/// Full `thought` + `content` is archived to L0 via `archive_full()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1Turn {
    pub role: String,
    pub summary: String,
    pub timestamp: DateTime<Utc>,
    /// IRI of archived full thought+content in L0
    pub l0_archive_iri: Option<String>,
    /// Semantic vector for relevance computation
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    /// Task relevance coefficient [0,1], used for enhanced eviction strategy
    #[serde(default)]
    pub relevance_score: Option<f64>,
    /// Last access time (used for safe window calculation)
    #[serde(default)]
    pub last_access: Option<DateTime<Utc>>,
    /// Supplement flag: true = user mid-turn supplement, not subject to hard threshold eviction
    #[serde(default)]
    pub is_supplement: bool,
}

/// L1 session — single agent summary chain
///
/// Design:
/// - Only `summary` field stored per LLM response
/// - Full `thought` + `content` archived to L0
/// - Summary-only context building (token-efficient)
/// - Full details reloadable from L0 on demand
/// - Built-in token budget with automatic policy-driven eviction
///
/// Multi-turn conversation summary chain format:
/// ```text
/// [Session History]
/// [agent_A] Step 1 completed: found the main issue
/// [agent_A] Step 2 completed: applied the fix
/// ```
#[derive(Debug, Clone)]
pub struct L1Session {
    session_id: String,
    agent_id: String,
    agent_role: String,
    task_iri: String,
    turns: Vec<L1Turn>,
    created_at: DateTime<Utc>,
    token_budget: usize,
    current_tokens: usize,
    /// Evicted IRI weak reference list for page-fault reload
    weak_refs: Vec<String>,
    /// MESI cache coherence state (L1 as S/I state holder)
    mesi_state: MesiState,
    eviction_config: EvictionConfig,
    /// Task-level semantic vector (generated from 5W2H.what+why or objective)
    /// Used as fallback query_embedding for evict_with_query
    task_embedding: Option<Vec<f32>>,
}

impl L1Session {
    pub fn new(agent_id: &str, agent_role: &str, task_iri: &str) -> Self {
        Self::with_budget(agent_id, agent_role, task_iri, 4000)
    }

    pub fn with_budget(
        agent_id: &str,
        agent_role: &str,
        task_iri: &str,
        token_budget: usize,
    ) -> Self {
        let eviction_config = EvictionConfig::for_role(agent_role);
        Self {
            session_id: format!("l1_{}", uuid::Uuid::new_v4().hyphenated()),
            agent_id: agent_id.to_string(),
            agent_role: agent_role.to_string(),
            task_iri: task_iri.to_string(),
            turns: Vec::new(),
            created_at: Utc::now(),
            token_budget,
            current_tokens: 0,
            weak_refs: Vec::new(),
            mesi_state: MesiState::Shared,
            eviction_config,
            task_embedding: None,
        }
    }

    pub fn with_config(
        agent_id: &str,
        agent_role: &str,
        task_iri: &str,
        token_budget: usize,
        eviction_config: EvictionConfig,
    ) -> Self {
        Self {
            session_id: format!("l1_{}", uuid::Uuid::new_v4().hyphenated()),
            agent_id: agent_id.to_string(),
            agent_role: agent_role.to_string(),
            task_iri: task_iri.to_string(),
            turns: Vec::new(),
            created_at: Utc::now(),
            token_budget,
            current_tokens: 0,
            weak_refs: Vec::new(),
            mesi_state: MesiState::Shared,
            eviction_config,
            task_embedding: None,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }
    pub fn agent_role(&self) -> &str {
        &self.agent_role
    }
    pub fn task_iri(&self) -> &str {
        &self.task_iri
    }
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }
    pub fn created_at(&self) -> &DateTime<Utc> {
        &self.created_at
    }
    pub fn duration(&self) -> chrono::Duration {
        Utc::now() - self.created_at
    }
    pub fn token_budget(&self) -> usize {
        self.token_budget
    }

    pub fn set_token_budget(&mut self, budget: usize) {
        self.token_budget = budget;
        if self.current_tokens > self.token_budget {
            self.evict_by_policy();
        }
    }

    /// Set task-level embedding for evict_with_query semantic fallback
    pub fn set_task_embedding(&mut self, embedding: Vec<f32>) {
        self.task_embedding = Some(embedding);
    }

    pub fn get_task_embedding(&self) -> Option<&[f32]> {
        self.task_embedding.as_deref()
    }

    /// Evict turns exceeding token budget using eviction policy
    ///
    /// Strategy: keep first turn, evict lowest-scored turns.
    /// Score = recency_weight * (1 / seconds_since_last_access) + relevance_weight * (1 / semantic_relevance) + cost_weight * token_cost
    /// Lower score means more likely to be evicted.
    pub fn evict_by_policy(&mut self) -> usize {
        self.evict_with_query(None)
    }

    /// Evict using optional query_embedding for semantic relevance evaluation
    ///
    /// Strategy (two-phase):
    /// 1. Hard threshold phase: relevance < threshold AND beyond safe window → direct eviction (skips is_supplement entries)
    /// 2. Scoring phase: weighted score eviction by recency/relevance/cost
    ///
    /// semantic_relevance = beta * cosine_sim(query, turn_embedding) + (1-beta) * turn.relevance_score
    pub fn evict_with_query(&mut self, query_embedding: Option<&[f32]>) -> usize {
        if self.current_tokens <= self.token_budget || self.turns.len() <= 1 {
            return 0;
        }

        let now = Utc::now();
        let mut evicted = 0;
        let cfg = &self.eviction_config;

        // Use passed query_embedding, fallback to self.task_embedding
        let query = query_embedding.or(self.task_embedding.as_deref());

        // Phase 1: Hard threshold eviction — low relevance + beyond safe window → direct eviction
        // is_supplement entries skip this phase, only participate in scoring phase
        if cfg.relevance_threshold > 0.0 {
            let mut i = 1;
            while i < self.turns.len()
                && self.current_tokens > self.token_budget
                && self.turns.len() > 1
            {
                let t = &self.turns[i];
                if !t.is_supplement {
                    let time_since = (now - t.timestamp).num_seconds();
                    let relevance = t.relevance_score.unwrap_or(0.5);
                    if relevance < cfg.relevance_threshold && time_since > cfg.safe_window_seconds {
                        let removed = self.turns.remove(i);
                        self.current_tokens -= (removed.summary.len() as f64 * 0.3) as usize;
                        if let Some(iri) = removed.l0_archive_iri {
                            self.weak_refs.push(iri);
                        }
                        evicted += 1;
                        continue; // i not incremented because remove shifts subsequent elements forward
                    }
                }
                i += 1;
            }
        }

        // Phase 2: Scoring eviction — evict lowest score by β fusion
        while self.current_tokens > self.token_budget && self.turns.len() > 1 {
            let mut min_idx = None;
            let mut min_score = f64::MAX;
            for (i, t) in self.turns.iter().enumerate().skip(1) {
                let time_since = (now - t.timestamp).num_seconds().max(1) as f64;
                let token_cost = (t.summary.len() as f64 * 0.3) as f64;

                let query_sim = match (query, t.embedding.as_ref()) {
                    (Some(q), Some(e)) if q.len() == e.len() && !q.is_empty() => {
                        cosine_similarity(q, e).abs().max(0.001)
                    }
                    _ => 0.5,
                };
                // β fusion: query relevance × β + task relevance × (1-β)
                let task_relevance = t.relevance_score.unwrap_or(query_sim);
                let semantic_relevance =
                    (cfg.beta * query_sim + (1.0 - cfg.beta) * task_relevance).max(0.001);

                let score = (1.0 / time_since) * cfg.recency_weight
                    + (1.0 / semantic_relevance) * cfg.relevance_weight
                    + token_cost * cfg.cost_weight;
                if score < min_score {
                    min_score = score;
                    min_idx = Some(i);
                }
            }

            if let Some(idx) = min_idx {
                let removed = self.turns.remove(idx);
                self.current_tokens -= (removed.summary.len() as f64 * 0.3) as usize;

                if let Some(iri) = removed.l0_archive_iri {
                    self.weak_refs.push(iri);
                }

                evicted += 1;
            } else {
                break;
            }
        }

        evicted
    }

    /// Attempt to reload content from L0 into L1 session by IRI
    pub fn try_reload_from_l0(&mut self, l0_store: &L0Store, iri: &str) -> bool {
        if let Ok(Some(entry)) = l0_store.retrieve(iri) {
            let summary = if entry.content.len() > 200 {
                entry.content.chars().take(200).collect()
            } else {
                entry.content.clone()
            };
            self.add_summary(
                "system",
                &format!("[Reloaded] {}", summary),
                Some(iri.to_string()),
            );
            true
        } else {
            false
        }
    }

    /// Store supplement input to L1 (called by AgentRunner on CycleStart injection)
    ///
    /// Unlike add_summary:
    /// - is_supplement = true (not subject to hard threshold eviction)
    /// - Preserves embedding and relevance_score for eviction policy use
    pub fn add_supplement(
        &mut self,
        role: &str,
        summary: &str,
        embedding: Option<Vec<f32>>,
        relevance_score: Option<f64>,
    ) -> &mut L1Turn {
        let turn = L1Turn {
            role: role.to_string(),
            summary: summary.to_string(),
            timestamp: Utc::now(),
            l0_archive_iri: None,
            embedding,
            relevance_score,
            last_access: Some(Utc::now()),
            is_supplement: true,
        };
        let token_cost = (summary.len() as f64 * 0.3) as usize;
        self.current_tokens += token_cost;
        self.turns.push(turn);

        if self.current_tokens > self.token_budget {
            self.evict_with_query(None);
        }

        self.turns.last_mut().unwrap()
    }

    /// Store LLM `summary` field to L1.
    /// thought+content should be separately archived to L0 via archive_full().
    /// Automatically checks token budget after adding, triggers eviction if exceeded.
    pub fn add_summary(
        &mut self,
        role: &str,
        summary: &str,
        l0_archive_iri: Option<String>,
    ) -> &mut L1Turn {
        let turn = L1Turn {
            role: role.to_string(),
            summary: summary.to_string(),
            timestamp: Utc::now(),
            l0_archive_iri,
            embedding: None,
            relevance_score: None,
            last_access: Some(Utc::now()),
            is_supplement: false,
        };
        let token_cost = (summary.len() as f64 * 0.3) as usize;
        self.current_tokens += token_cost;
        self.turns.push(turn);

        if self.current_tokens > self.token_budget {
            self.evict_by_policy();
        }

        self.turns.last_mut().expect("turn was just pushed above")
    }

    /// Archive full thought+content to L0 and return archive IRI.
    /// Called after adding an assistant turn.
    pub fn archive_full_to_l0(
        &self,
        l0_store: &L0Store,
        role: &str,
        thought: &str,
        content_json: &str,
    ) -> Result<String, CoreError> {
        let iri = format!(
            "iri://archive/{}/{}/{}",
            self.task_iri
                .strip_prefix("iri://")
                .unwrap_or(&self.task_iri),
            role,
            uuid::Uuid::new_v4().hyphenated()
        );
        let payload = serde_json::json!({
            "@id": &iri,
            "@type": "LLMResponse",
            "role": role,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "thought": thought,
            "content": serde_json::from_str::<serde_json::Value>(content_json).ok(),
            "timestamp": Utc::now().to_rfc3339(),
        });
        l0_store.store(&iri, &payload.to_string())?;
        debug!(iri = %iri, "Archived full LLM response to L0");
        Ok(iri)
    }

    /// Get summary chain for LLM context building.
    /// Ensures token budget is met before returning.
    pub fn get_summary_chain(&mut self) -> Vec<serde_json::Value> {
        if self.turns.is_empty() {
            return Vec::new();
        }

        if self.current_tokens > self.token_budget {
            self.evict_by_policy();
        }

        let threshold = self.eviction_config.relevance_threshold;

        // Split by relevance: high-relevance + supplement go to main, low-relevance to reference
        let main: Vec<String> = self
            .turns
            .iter()
            .filter(|t| t.is_supplement || t.relevance_score.unwrap_or(0.5) >= threshold)
            .map(|t| format!("[{}] {}", t.role, t.summary))
            .collect();

        let mut content = format!(
            "[Previous context from {} ({})]\n{}",
            self.agent_id,
            self.agent_role,
            main.join("\n")
        );

        // Low-relevance turns appended as reference section (only when meaningful and low_rel entries exist)
        let low: Vec<String> = self
            .turns
            .iter()
            .filter(|t| !t.is_supplement && t.relevance_score.unwrap_or(0.5) < threshold)
            .map(|t| {
                let truncated: String = t.summary.chars().take(80).collect();
                let score = t.relevance_score.unwrap_or(0.0);
                format!("[{}] {} (relevance: {:.2})", t.role, truncated, score)
            })
            .collect();

        if !low.is_empty() {
            content.push_str("\n\n[Historical Reference - Low Relevance]\n");
            content.push_str(&low.join("\n"));
        }

        vec![serde_json::json!({
            "role": "system",
            "content": content
        })]
    }

    /// Get summary chain with IRIs, for building structured reference summaries on message truncation.
    /// Each turn's summary is truncated to summary_length characters, with L0 archive IRI attached.
    pub fn get_summary_chain_with_iris(
        &self,
        max_turns: usize,
        summary_length: usize,
    ) -> Vec<String> {
        self.turns
            .iter()
            .rev()
            .take(max_turns)
            .map(|t| {
                let truncated: String = t.summary.chars().take(summary_length).collect();
                match t.l0_archive_iri {
                    Some(ref iri) => format!("[{}] {} | {}", t.role, truncated, iri),
                    None => format!("[{}] {}", t.role, truncated),
                }
            })
            .collect()
    }

    /// Build compact summary string for agent handoff (L1→next L1)
    pub fn handoff_summary(&self) -> String {
        if self.turns.is_empty() {
            return format!(
                "Agent {} ({}) ran with {} turns.",
                self.agent_id,
                self.agent_role,
                self.turns.len()
            );
        }
        let summaries: Vec<String> = self
            .turns
            .iter()
            .map(|t| format!("[{}] {}", t.role, t.summary))
            .collect();
        format!(
            "From {} ({}):\n{}",
            self.agent_id,
            self.agent_role,
            summaries.join("\n")
        )
    }

    /// Estimated token consumption of the current session
    pub fn estimated_tokens(&self) -> u32 {
        self.current_tokens as u32
    }

    /// Summarize session state
    pub fn summarize(&self) -> SessionSummary {
        SessionSummary {
            session_id: self.session_id.clone(),
            agent_id: self.agent_id.clone(),
            agent_role: self.agent_role.clone(),
            task_iri: self.task_iri.clone(),
            turn_count: self.turns.len(),
            created_at: self.created_at,
            summary_text: self.handoff_summary(),
        }
    }

    pub fn clear(&mut self) {
        self.turns.clear();
        self.weak_refs.clear();
        self.current_tokens = 0;
    }

    /// Get weak reference list
    pub fn get_weak_refs(&self) -> &[String] {
        &self.weak_refs
    }

    /// Reload from weak reference list into L1
    pub fn reload_from_weak_refs(&mut self, l0_store: &L0Store) -> usize {
        let mut reloaded = 0;
        let refs_to_reload: Vec<String> = self.weak_refs.drain(..).collect();

        for iri in refs_to_reload {
            if self.try_reload_from_l0(l0_store, &iri) {
                reloaded += 1;
            }
        }

        reloaded
    }

    /// Set turn embedding (for semantic relevance computation).
    ///
    /// DEPRECATED: turn embeddings are set directly at the call sites
    /// (`execution.rs` ReAct loop and `utils.rs` tool path assign
    /// `l1_turn.embedding = Some(emb)` right after `add_summary`), so this
    /// method has no runtime callers. Kept as a public API for programmatic
    /// use; do not route the execution paths through it.
    #[deprecated(
        note = "turn embeddings are assigned directly at add_summary call sites in agent_runner"
    )]
    pub fn set_turn_embedding(&mut self, turn_idx: usize, embedding: Vec<f32>) {
        if let Some(turn) = self.turns.get_mut(turn_idx) {
            turn.embedding = Some(embedding);
        }
    }

    /// Get MESI state
    pub fn mesi_state(&self) -> MesiState {
        self.mesi_state
    }

    /// Set MESI state
    pub fn set_mesi_state(&mut self, state: MesiState) {
        self.mesi_state = state;
    }

    /// Invalidate cache (set state to Invalid)
    pub fn invalidate(&mut self) {
        self.mesi_state = MesiState::Invalid;
    }

    pub fn eviction_config(&self) -> &EvictionConfig {
        &self.eviction_config
    }

    pub fn set_eviction_config(&mut self, config: EvictionConfig) {
        self.eviction_config = config;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub agent_id: String,
    pub agent_role: String,
    pub task_iri: String,
    pub turn_count: usize,
    pub created_at: DateTime<Utc>,
    pub summary_text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_only_session() {
        let mut session = L1Session::new("agent_1", "DA", "iri://task/abc");
        session.add_summary("assistant", "Found the root cause in config.rs", None);
        session.add_summary("assistant", "Applied the fix and verified", None);
        assert_eq!(session.turn_count(), 2);

        let chain = session.get_summary_chain();
        assert_eq!(chain.len(), 1);
        let content = chain[0]["content"].as_str().unwrap();
        assert!(content.contains("Found the root cause"));
        assert!(content.contains("Applied the fix"));
    }

    #[test]
    fn test_handoff_is_compact() {
        let mut session = L1Session::new("agent_1", "DA", "iri://task/abc");
        session.add_summary("assistant", "Completed analysis", None);
        let handoff = session.handoff_summary();
        assert_eq!(handoff.lines().count(), 2);
        assert!(handoff.contains("agent_1"));
        assert!(handoff.contains("Completed analysis"));
    }

    #[test]
    fn test_default_token_budget() {
        let session = L1Session::new("agent_1", "DA", "iri://task/abc");
        assert_eq!(session.token_budget(), 4000);
        assert_eq!(session.estimated_tokens(), 0);
    }

    #[test]
    fn test_with_budget() {
        let session = L1Session::with_budget("agent_1", "DA", "iri://task/abc", 1000);
        assert_eq!(session.token_budget(), 1000);
    }

    #[test]
    fn test_token_tracking() {
        let mut session = L1Session::new("agent_1", "DA", "iri://task/abc");
        session.add_summary("assistant", "Hello world", None);
        assert!(session.estimated_tokens() > 0);
    }

    #[test]
    fn test_eviction_on_budget_exceeded() {
        let mut session = L1Session::with_budget("agent_1", "DA", "iri://task/abc", 10);
        session.add_summary("assistant", "First turn that stays", None);
        session.add_summary("assistant", "Second turn with content", None);
        session.add_summary("assistant", "Third turn more content here", None);
        assert!(session.current_tokens <= session.token_budget || session.turns.len() <= 1);
    }

    #[test]
    fn test_set_token_budget_triggers_eviction() {
        let mut session = L1Session::with_budget("agent_1", "DA", "iri://task/abc", 10000);
        session.add_summary("assistant", "First turn content here", None);
        session.add_summary("assistant", "Second turn content here", None);
        session.add_summary("assistant", "Third turn content here", None);
        session.set_token_budget(10);
        assert!(session.current_tokens <= session.token_budget || session.turns.len() <= 1);
    }

    #[test]
    fn test_clear_resets_tokens() {
        let mut session = L1Session::new("agent_1", "DA", "iri://task/abc");
        session.add_summary("assistant", "Some content", None);
        assert!(session.estimated_tokens() > 0);
        session.clear();
        assert_eq!(session.estimated_tokens(), 0);
    }

    // ========== Cosine Similarity Tests ==========

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical vectors should have similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 0.0).abs() < 1e-6,
            "orthogonal vectors should have similarity 0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "opposite vectors should have similarity -1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_empty_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 0.0).abs() < 1e-6,
            "zero vector should give 0.0, got {}",
            sim
        );
    }

    // ========== Eviction Config Tests ==========

    #[test]
    fn test_eviction_config_default() {
        let cfg = EvictionConfig::default();
        assert!((cfg.recency_weight - 0.30).abs() < 1e-6);
        assert!((cfg.relevance_weight - 0.40).abs() < 1e-6);
        assert!((cfg.cost_weight - 0.30).abs() < 1e-6);
    }

    #[test]
    fn test_eviction_config_for_role() {
        let sa = EvictionConfig::for_role("Supervisor");
        assert!((sa.recency_weight - 0.30).abs() < 1e-6);

        let pa = EvictionConfig::for_role("PA");
        assert!(
            pa.relevance_weight > pa.recency_weight,
            "PA should prioritize relevance over recency"
        );
        assert!((pa.relevance_weight - 0.60).abs() < 1e-6);

        let da = EvictionConfig::for_role("DA");
        assert!(
            da.recency_weight >= da.cost_weight.min(da.relevance_weight),
            "DA should balance recency and cost"
        );

        let ca = EvictionConfig::for_role("CA");
        assert!(
            ca.relevance_weight > 0.5,
            "CA should heavily prioritize relevance"
        );
    }

    #[test]
    fn test_eviction_config_with_config() {
        let custom = EvictionConfig {
            recency_weight: 0.5,
            relevance_weight: 0.3,
            cost_weight: 0.2,
            ..Default::default()
        };
        let session = L1Session::with_config("agent_1", "DA", "iri://task/abc", 1000, custom);
        assert!((session.eviction_config().recency_weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_evict_with_query_embedding() {
        let mut session = L1Session::with_budget("agent_1", "DA", "iri://task/abc", 10);
        session.add_summary("assistant", "short", None);

        let q_emb = vec![1.0, 0.0, 0.0];
        let match_emb = vec![0.99, 0.01, 0.01];
        let diff_emb = vec![0.0, 1.0, 0.0];

        session.add_summary(
            "assistant",
            "matching content",
            Some("iri://match".to_string()),
        );
        if let Some(t) = session.turns.last_mut() {
            t.embedding = Some(match_emb.clone());
        }

        session.add_summary(
            "assistant",
            "different content",
            Some("iri://diff".to_string()),
        );
        if let Some(t) = session.turns.last_mut() {
            t.embedding = Some(diff_emb.clone());
        }

        let _evicted = session.evict_with_query(Some(&q_emb));
        assert!(session.current_tokens <= session.token_budget || session.turns.len() <= 1);
        let remaining: Vec<&str> = session.turns.iter().map(|t| t.summary.as_str()).collect();
        let still_has_matching = remaining.iter().any(|s| *s == "matching content");
        assert!(still_has_matching, "matching content should be retained");
    }

    // ========== Supplement Input Tests ==========

    #[test]
    fn test_add_supplement_preserves_fields() {
        let mut session = L1Session::with_budget("agent_1", "DA", "iri://task/abc", 10000);
        let emb = Some(vec![0.5, 0.5]);
        session.add_supplement("user", "supplement note", emb.clone(), Some(0.85));

        assert_eq!(session.turns.len(), 1);
        let t = &session.turns[0];
        assert!(
            t.is_supplement,
            "add_supplement should set is_supplement = true"
        );
        assert_eq!(t.role, "user");
        assert_eq!(t.summary, "supplement note");
        assert_eq!(t.embedding, emb);
        assert!((t.relevance_score.unwrap() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_supplement_protected_from_hard_threshold_eviction() {
        let mut session = L1Session::with_budget("agent_1", "DA", "iri://task/abc", 10000);
        // Add summary as first turn, add_supplement for subsequent turns
        session.add_summary("assistant", "the first real assistant turn", None);

        // Add supplement with low relevance + old timestamp (simulating scenario that triggers hard threshold eviction)
        session.add_supplement("user", "old supplement", None, Some(0.1));
        // Force this turn's timestamp to be older
        let old_time = chrono::Utc::now() - chrono::Duration::seconds(600);
        if let Some(t) = session.turns.last_mut() {
            t.timestamp = old_time;
        }

        // Increase budget pressure to trigger eviction
        session.token_budget = 100;

        // Hard threshold eviction does not remove is_supplement entries
        let _evicted = session.evict_with_query(None);
        // Supplements should not be evicted by hard threshold
        let has_supplement = session.turns.iter().any(|t| t.is_supplement);
        assert!(
            has_supplement,
            "supplement should be protected from hard threshold eviction"
        );
    }

    #[test]
    fn test_beta_fusion_influences_eviction() {
        let mut session = L1Session::with_config(
            "agent_1",
            "DA",
            "iri://task/abc",
            10000,
            EvictionConfig {
                recency_weight: 0.0,
                relevance_weight: 1.0,
                cost_weight: 0.0,
                relevance_threshold: 0.0,
                safe_window_seconds: 0,
                beta: 0.5,
            },
        );
        // Keep first turn (always kept), add padding turns to create budget pressure
        session.add_summary(
            "assistant",
            "first long padding text to generate token cost xxxxxx",
            None,
        );
        session.add_summary(
            "assistant",
            "second long padding text to generate more cost yyyyyy",
            None,
        );

        // Two turns: same query_sim but different task_relevance
        let emb = Some(vec![1.0, 0.0]);
        session.add_summary("assistant", "high_rel_turn", None);
        if let Some(t) = session.turns.last_mut() {
            t.embedding = emb.clone();
            t.relevance_score = Some(0.9);
        }
        session.add_summary("assistant", "low_rel_turn", None);
        if let Some(t) = session.turns.last_mut() {
            t.embedding = emb.clone();
            t.relevance_score = Some(0.1);
        }

        // Tighten budget to trigger evict (1 token less, ensures only 1 turn evicted)
        session.token_budget = session.current_tokens - 1;
        let q_emb = vec![1.0, 0.0];
        let evicted = session.evict_with_query(Some(&q_emb));
        assert!(
            evicted > 0,
            "eviction should occur when tokens exceed budget"
        );

        // β=0.5: high_rel semantic = 0.5*1.0+0.5*0.9=0.95, low_rel = 0.5*1.0+0.5*0.1=0.55
        // score = (1/semantic)*1.0, so high_rel ≈ 1.05, low_rel ≈ 1.82
        // min score wins eviction → high_rel evicted
        let has_low = session.turns.iter().any(|t| t.summary == "low_rel_turn");
        assert!(
            has_low,
            "low relevance turn (higher score) should survive eviction"
        );
        let has_high = session.turns.iter().any(|t| t.summary == "high_rel_turn");
        assert!(
            !has_high,
            "high relevance turn (lower score) should be evicted first"
        );
    }
}
