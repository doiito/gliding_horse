use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use super::agent::SupervisorAgent;
use super::types::*;

impl SupervisorAgent {
    #[allow(dead_code)]
    pub fn get_cycle_status(&self, cycle_id: &str) -> Option<&CycleState> {
        self.active_cycles.get(cycle_id)
    }

    pub fn active_cycles(&self) -> Vec<&CycleState> {
        self.active_cycles.values().collect()
    }

    pub fn cleanup_expired_cycles(&mut self, max_age_secs: i64) {
        let now = chrono::Utc::now();
        self.active_cycles.retain(|_, cycle| {
            now.signed_duration_since(cycle.started_at).num_seconds() < max_age_secs
                || !cycle.task_completed
        });
    }

    /// Try to read L1 session count from the memory manager using its atomic
    /// counter — does not block if the memory_manager lock is contended.
    pub fn try_l1_session_count(&self) -> Option<u64> {
        self.runner
            .memory_manager
            .try_lock()
            .ok()
            .map(|mm| mm.l1_session_count())
    }

    /// Returns the atomic token counters from the agent runner.
    /// Returns (total_prompt, total_completion, last_prompt, last_completion).
    pub fn token_usage_arcs(
        &self,
    ) -> (
        Arc<AtomicU64>,
        Arc<AtomicU64>,
        Arc<AtomicU64>,
        Arc<AtomicU64>,
    ) {
        (
            self.runner.total_prompt_tokens.clone(),
            self.runner.total_completion_tokens.clone(),
            self.runner.last_prompt_tokens.clone(),
            self.runner.last_completion_tokens.clone(),
        )
    }

    /// Query L1 session count and L3 projection cache count from the memory manager.
    pub fn memory_stats(&self) -> (usize, usize) {
        let mm = self.runner.memory_manager.blocking_lock();
        let l1 = mm.session_count();
        let l3 = mm.projection().cache_stats().total_views;
        (l1, l3)
    }

    #[allow(dead_code)]
    fn query_historical_5w2h(
        &self,
        limit: usize,
    ) -> Vec<(String, crate::core::five_w2h::Task5W2H)> {
        let mut results = Vec::new();
        let tags = vec!["5w2h".to_string(), "frozen".to_string()];
        if let Ok(entries) = self.runner.l0_store.search_by_tags(&tags) {
            for entry in entries.into_iter().take(limit) {
                if let Ok(node) = serde_json::from_str::<serde_json::Value>(&entry.content) {
                    if let Ok(w2h) = crate::core::five_w2h::Task5W2H::from_json_ld(&node) {
                        if w2h.frozen {
                            results.push((entry.iri.clone(), w2h));
                        }
                    }
                }
            }
        }
        results
    }

    #[allow(dead_code)]
    fn match_similar_tasks(
        &self,
        current_what: &str,
        current_why: &str,
        historical: &[(String, crate::core::five_w2h::Task5W2H)],
        top_k: usize,
    ) -> Vec<(String, crate::core::five_w2h::Task5W2H, f32)> {
        let mut scored: Vec<_> = historical
            .iter()
            .map(|(iri, w2h)| {
                let what_sim = Self::text_similarity(&w2h.what, current_what);
                let why_sim = Self::text_similarity(&w2h.why.description, current_why);
                let combined = what_sim * 0.6 + why_sim * 0.4;
                (iri.clone(), w2h.clone(), combined)
            })
            .collect();

        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).collect()
    }

    #[allow(dead_code)]
    fn text_similarity(a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        let a_words: std::collections::HashSet<&str> = a_lower.split_whitespace().collect();
        let b_words: std::collections::HashSet<&str> = b_lower.split_whitespace().collect();

        if a_words.is_empty() || b_words.is_empty() {
            return 0.0;
        }

        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    #[allow(dead_code)]
    fn format_historical_experience(
        &self,
        similar: &[(String, crate::core::five_w2h::Task5W2H, f32)],
    ) -> String {
        if similar.is_empty() {
            return String::new();
        }

        let mut experience_section =
            String::from("\n## Historical Experience Reference (Similar Tasks)\n");
        experience_section.push_str("The following historical tasks are similar to the current task, for reference only:\n\n");

        for (i, (iri, w2h, score)) in similar.iter().enumerate() {
            experience_section.push_str(&format!(
                "### Similar Task {} (Similarity: {:.0}%)\n",
                i + 1,
                score * 100.0
            ));
            experience_section.push_str(&format!("- **What**: {}\n", w2h.what));
            experience_section.push_str(&format!("- **Why**: {}\n", w2h.why.description));
            if let Some(ref how) = w2h.how {
                if let Some(ref steps) = how.required_steps {
                    experience_section.push_str(&format!("- **Execution Steps**: {}\n", steps));
                }
            }
            experience_section.push_str(&format!("- **Source**: {}\n\n", iri));
        }

        experience_section.push_str("**Note**: Historical experience is for reference only. Please adjust based on the actual current task.\n");
        experience_section
    }
}
