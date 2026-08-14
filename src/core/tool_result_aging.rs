use parking_lot::RwLock;
use tracing::debug;

use crate::config::settings::ToolResultAgingSettings;
use crate::gateway::unified_gateway::ChatMessage;

/// Auto-degrade old tool results in messages[] by turn depth.
///
/// Strategy:
/// - Keep the most recent N full tool results (keep_full)
/// - Older tool results with a micro-tool → replace with reference-style compression
/// - Oldest tool results → replace with brief summary
///
/// Basis: messages[] are chronologically ordered, scanning front to back = old to new.
#[derive(Clone)]
pub struct ToolResultAging {
    /// Number of full results to keep (most recent stay intact)
    keep_full: usize,
    /// Number of old results to try micro-tool reference compression on
    try_microtool: usize,
    /// Compression threshold: only process tool messages exceeding this byte count
    compress_threshold: usize,
}

impl ToolResultAging {
    pub fn new(settings: &ToolResultAgingSettings) -> Self {
        Self {
            keep_full: settings.keep_full,
            try_microtool: settings.try_microtool,
            compress_threshold: settings.compress_threshold,
        }
    }

    pub fn keep_full(&self) -> usize {
        self.keep_full
    }

    pub fn try_microtool(&self) -> usize {
        self.try_microtool
    }

    /// Auto-degrade and compress tool messages in messages by staleness.
    ///
    /// Returns (aged_count, freed_bytes)
    pub fn age_tool_results(
        &self,
        messages: &mut Vec<ChatMessage>,
        tool_executor: &RwLock<crate::tools::tool_executor::ToolExecutor>,
    ) -> (usize, usize) {
        // Collect all tool message indices (skip non-tool messages like system/perception)
        let tool_indices: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "tool")
            .map(|(i, _)| i)
            .collect();

        let total = tool_indices.len();
        if total <= self.keep_full {
            return (0, 0);
        }

        let mut aged = 0usize;
        let mut freed = 0usize;

        // Process chronologically from old to new: keep N most recent intact, degrade older ones
        // batch_idx=0=oldest, batch_idx=N-1=newest
        let microtool_end = self.keep_full + self.try_microtool;

        for (batch_idx, &msg_idx) in tool_indices.iter().enumerate() {
            // Position counting from newest: rev_position=0=newest
            let rev_position = total - 1 - batch_idx;
            if rev_position < self.keep_full {
                continue; // Keep N most recent intact
            }

            let msg = &messages[msg_idx];
            if msg.content.len() < self.compress_threshold {
                continue; // Skip small results
            }

            let call_id = match msg.tool_call_id.as_deref() {
                Some(id) if !id.is_empty() => id.to_string(),
                _ => continue,
            };

            let original_len = msg.content.len();

            if rev_position < microtool_end {
                // Second-oldest batch: try micro-tool reference compression
                let micro_tool_name = format!("read_full_result_{}", call_id);
                let has_micro_tool = tool_executor
                    .read()
                    .try_get_handler(&micro_tool_name)
                    .is_some();

                if has_micro_tool {
                    let iri = format!("iri://tool-result/{}", call_id);
                    messages[msg_idx].content = format!(
                        "[Compressed {} bytes] Full result available via `{}` tool\nIRI: {}",
                        original_len, micro_tool_name, iri,
                    );
                } else {
                    // No micro-tool, use brief summary
                    let preview: String = msg.content.chars().take(150).collect();
                    messages[msg_idx].content =
                        format!("[Old result {} bytes] {}...", original_len, preview);
                }
            } else {
                // Oldest batch: replace with brief summary directly
                let preview: String = msg.content.chars().take(100).collect();
                messages[msg_idx].content =
                    format!("[Historical result {} bytes] {}...", original_len, preview);
            }

            freed += original_len.saturating_sub(messages[msg_idx].content.len());
            aged += 1;
        }

        if aged > 0 {
            debug!(
                "[tool_aging] Aged {} tool results, freed {} bytes",
                aged, freed
            );
        }

        (aged, freed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tool_executor::ToolExecutor;

    fn make_tool_msg(content: &str, call_id: &str) -> ChatMessage {
        ChatMessage {
            role: "tool".to_string(),
            content: content.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: Some(call_id.to_string()),
            reasoning_content: None,
        }
    }

    fn make_system_msg() -> ChatMessage {
        ChatMessage {
            role: "system".to_string(),
            content: "sys".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn default_settings() -> ToolResultAgingSettings {
        ToolResultAgingSettings {
            enabled: true,
            keep_full: 3,
            try_microtool: 3,
            compress_threshold: 200,
        }
    }

    #[test]
    fn test_aging_keeps_recent_full() {
        let aging = ToolResultAging::new(&default_settings());
        let executor = RwLock::new(ToolExecutor::new());

        let mut msgs = vec![make_system_msg()];
        for i in 0..5 {
            msgs.push(make_tool_msg(&"x".repeat(500), &format!("call_{}", i)));
        }

        let (aged, _) = aging.age_tool_results(&mut msgs, &executor);

        // keep_full=3: keep newest 3 (call_4/3/2), oldest 2 are compressed (call_0/1)
        // call_0/1 rev_position 4/3 < microtool_end(6) → [Old result] prefix
        assert_eq!(aged, 2, "should age 2 oldest results");

        let contents: Vec<&str> = msgs
            .iter()
            .filter(|m| m.role == "tool")
            .map(|m| m.content.as_str())
            .collect();
        assert!(
            contents[0].starts_with("[Old result"),
            "call_0 oldest should be compressed"
        );
        assert!(
            contents[1].starts_with("[Old result"),
            "call_1 oldest should be compressed"
        );
        assert!(
            contents[2].starts_with("x"),
            "call_2 should remain full (recent)"
        );
        assert!(
            contents[3].starts_with("x"),
            "call_3 should remain full (recent)"
        );
        assert!(
            contents[4].starts_with("x"),
            "call_4 should remain full (recent)"
        );
    }

    #[test]
    fn test_aging_microtool_reference() {
        let aging = ToolResultAging::new(&ToolResultAgingSettings {
            enabled: true,
            keep_full: 1,
            try_microtool: 2,
            compress_threshold: 50,
        });
        let executor = RwLock::new(ToolExecutor::new());

        let mut msgs = vec![make_system_msg()];
        for i in 0..4 {
            msgs.push(make_tool_msg(&"y".repeat(200), &format!("call_{}", i)));
        }

        let (aged, _) = aging.age_tool_results(&mut msgs, &executor);

        // total=4, keep_full=1, microtool_end=3
        // rev_positions: call_0=3, call_1=2, call_2=1, call_3=0
        // call_3(rev=0) < keep_full(1) → kept full
        // call_2(rev=1) < keep_full? NO, < microtool_end(3)? YES → microtool → [Old result]
        // call_1(rev=2) < keep_full? NO, < microtool_end(3)? YES → microtool → [Old result]
        // call_0(rev=3) < keep_full? NO, < microtool_end(3)? NO → oldest → [Historical result]
        assert_eq!(aged, 3);

        let contents: Vec<&str> = msgs
            .iter()
            .filter(|m| m.role == "tool")
            .map(|m| m.content.as_str())
            .collect();
        // call_0 (idx 0 in msgs, oldest): rev=3 >= microtool_end → [Historical result]
        assert!(
            contents[0].starts_with("[Historical result"),
            "call_0 oldest should be brief summary"
        );
        // call_1 (idx 1): rev=2 < microtool_end → [Old result]
        assert!(
            contents[1].starts_with("[Old result"),
            "call_1 should be in microtool range"
        );
        // call_2 (idx 2): rev=1 < microtool_end → [Old result]
        assert!(
            contents[2].starts_with("[Old result"),
            "call_2 should be in microtool range"
        );
        // call_3 (idx 3, newest): rev=0 < keep_full → kept full
        assert!(contents[3].starts_with("y"), "call_3 newest should be full");
    }

    #[test]
    fn test_aging_skips_small_results() {
        let aging = ToolResultAging::new(&ToolResultAgingSettings {
            enabled: true,
            keep_full: 1,
            try_microtool: 2,
            compress_threshold: 500, // Skip results smaller than 500 bytes
        });
        let executor = RwLock::new(ToolExecutor::new());

        let mut msgs = vec![make_system_msg()];
        for i in 0..4 {
            msgs.push(make_tool_msg(&"small".to_string(), &format!("call_{}", i)));
        }

        let (aged, _) = aging.age_tool_results(&mut msgs, &executor);
        assert_eq!(aged, 0, "small results should not be aged");
    }

    #[test]
    fn test_aging_frees_bytes() {
        let aging = ToolResultAging::new(&default_settings());
        let executor = RwLock::new(ToolExecutor::new());

        let mut msgs = vec![make_system_msg()];
        for i in 0..4 {
            msgs.push(make_tool_msg(&"x".repeat(500), &format!("call_{}", i)));
        }

        let (_, freed) = aging.age_tool_results(&mut msgs, &executor);
        assert!(freed > 0, "should free bytes from aging");
    }
}
