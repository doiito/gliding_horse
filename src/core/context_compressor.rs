use crate::config::settings::{ContextWindowSettings, ToolResultCompressorSettings};
use crate::gateway::unified_gateway::ChatMessage;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultEntry {
    pub turn: u32,
    pub tool_name: String,
    /// Exact tool_call_id for reliable mapping back to tool messages in messages
    pub tool_call_id: String,
    pub content: String,
    pub is_compressed: bool,
}

pub struct ToolResultCompressor {
    enabled: bool,
    max_full_results: usize,
    max_summary_length: usize,
    compression_trigger: usize,
    /// Tool messages exceeding this byte threshold attempt micro-tool reference replacement
    compress_tool_result_threshold: usize,
    results: VecDeque<ToolResultEntry>,
}

impl ToolResultCompressor {
    pub fn new(settings: &ToolResultCompressorSettings) -> Self {
        Self {
            enabled: settings.enabled,
            max_full_results: settings.max_full_results,
            max_summary_length: settings.max_summary_length,
            compression_trigger: settings.compression_trigger,
            compress_tool_result_threshold: settings.compress_tool_result_threshold,
            results: VecDeque::new(),
        }
    }

    pub fn add_result(&mut self, turn: u32, tool_name: &str, tool_call_id: &str, content: &str) {
        let entry = ToolResultEntry {
            turn,
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            content: content.to_string(),
            is_compressed: false,
        };
        self.results.push_back(entry);

        if self.results.len() >= self.compression_trigger {
            self.compress_old_results();
        }
    }

    fn compress_old_results(&mut self) {
        if self.results.len() <= self.max_full_results {
            return;
        }

        let to_compress = self.results.len() - self.max_full_results;
        let summaries: Vec<(usize, String)> = self
            .results
            .iter()
            .take(to_compress)
            .enumerate()
            .filter(|(_, entry)| {
                !entry.is_compressed && entry.content.len() > self.max_summary_length
            })
            .map(|(i, entry)| (i, self.summarize_content(&entry.content)))
            .collect();

        for (i, summary) in summaries {
            if let Some(entry) = self.results.get_mut(i) {
                entry.content = summary;
                entry.is_compressed = true;
            }
        }
    }

    fn summarize_content(&self, content: &str) -> String {
        if content.len() <= self.max_summary_length {
            return content.to_string();
        }

        let lines: Vec<&str> = content.lines().take(5).collect();
        let preview = if lines.len() > 3 {
            lines[..3].join("\n")
        } else {
            lines.join("\n")
        };

        format!(
            "[Summary {} bytes] {}... (total {} chars)",
            self.max_summary_length,
            preview,
            content.len()
        )
    }

    /// Compress tool result content in messages.
    /// Used together with compress_old_results(): the latter compresses entries inside the compressor,
    /// this method writes compressed results back to the corresponding tool messages via tool_call_id matching.
    pub fn compress_tool_messages(&self, messages: &mut Vec<ChatMessage>) {
        if !self.enabled {
            return;
        }
        // Build compressed entry map: tool_call_id -> compressed_content
        let compressed_map: std::collections::HashMap<&str, &str> = self
            .results
            .iter()
            .filter(|e| e.is_compressed)
            .map(|e| (e.tool_call_id.as_str(), e.content.as_str()))
            .collect();

        if compressed_map.is_empty() {
            return;
        }

        // Match tool messages in messages by exact tool_call_id
        for msg in messages.iter_mut() {
            if msg.role != "tool" {
                continue;
            }
            let call_id = match msg.tool_call_id.as_deref() {
                Some(id) if !id.is_empty() => id,
                _ => continue,
            };
            if let Some(compressed_content) = compressed_map.get(call_id) {
                msg.content = compressed_content.to_string();
            }
        }
    }

    pub fn get_results(&self) -> &VecDeque<ToolResultEntry> {
        &self.results
    }

    pub fn clear(&mut self) {
        self.results.clear();
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn compress_tool_result_threshold(&self) -> usize {
        self.compress_tool_result_threshold
    }
}

pub struct ContextWindowManager {
    max_messages: usize,
    max_tokens: usize,
    compression_ratio: f32,
    preserve_recent: usize,
}

impl ContextWindowManager {
    pub fn new(settings: &ContextWindowSettings) -> Self {
        Self {
            max_messages: settings.max_messages,
            max_tokens: settings.max_tokens,
            compression_ratio: settings.compression_ratio,
            preserve_recent: settings.preserve_recent,
        }
    }

    /// Estimate token consumption of a message list (4 chars ≈ 1 token, mixed CJK/Latin estimation)
    pub fn estimate_tokens(messages: &[ChatMessage]) -> usize {
        messages
            .iter()
            .map(|m| {
                let mut total = m.content.len() / 4 + m.role.len() / 4;
                if let Some(ref calls) = m.tool_calls {
                    for call in calls {
                        total += call.function.name.len() / 4;
                        total += call.function.arguments.len() / 4;
                        // Include tool_call_id (~36 chars per UUID)
                        total += call.id.len() / 4;
                    }
                }
                if let Some(ref id) = m.tool_call_id {
                    total += id.len() / 4;
                }
                total
            })
            .sum()
    }

    /// Determine whether compression is needed. Checks both message count and estimated token count.
    pub fn should_compress(&self, message_count: usize, messages: &[ChatMessage]) -> bool {
        if message_count > self.max_messages {
            return true;
        }
        if Self::estimate_tokens(messages) > self.max_tokens {
            return true;
        }
        false
    }

    pub fn compress_messages(&self, messages: &[ChatMessage]) -> (Vec<ChatMessage>, String) {
        if messages.len() <= self.max_messages {
            return (messages.to_vec(), String::new());
        }

        let system_msg = messages.first().filter(|m| m.role == "system").cloned();
        let mut recent_start = messages.len().saturating_sub(self.preserve_recent);

        // OpenAI/DeepSeek require every `role: "tool"` message to be preceded
        // by an `assistant` message whose `tool_calls` array contains a
        // matching id.  Adjust the boundary so tool_call groups stay intact.
        recent_start = Self::adjust_boundary_for_tool_calls(messages, recent_start);
        let recent: Vec<_> = messages[recent_start..].to_vec();

        let middle_start = if system_msg.is_some() { 1 } else { 0 };
        let middle: Vec<_> = messages[middle_start..recent_start].to_vec();

        let keep_count = (middle.len() as f32 * self.compression_ratio) as usize;
        let keep_count = keep_count.min(middle.len());
        let empty: &[ChatMessage] = &[];
        let (to_summarize, to_keep) = if keep_count > 0 && keep_count < middle.len() {
            let mut split = middle.len() - keep_count;
            // Adjust split to avoid splitting tool_call groups within middle
            split = Self::adjust_boundary_for_tool_calls(&middle, split);
            (&middle[..split], &middle[split..])
        } else if keep_count >= middle.len() {
            (empty, &middle[..])
        } else {
            (&middle[..], empty)
        };

        let summary = self.summarize_middle_messages(to_summarize);

        let mut compressed = Vec::new();
        if let Some(sys) = system_msg {
            compressed.push(sys);
        }

        if !summary.is_empty() {
            compressed.push(ChatMessage {
                role: "user".to_string(),
                content: format!("[History Summary] {}", summary),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }

        compressed.extend(to_keep.iter().cloned());
        compressed.extend(recent);

        // Safety: remove any orphaned tool messages that slipped through
        let cleaned = Self::remove_orphaned_tool_messages(compressed);
        (cleaned, summary)
    }

    /// OpenAI/DeepSeek require every `role: "tool"` message to be preceded
    /// by an `assistant` with a matching `tool_calls` entry.  Adjust a
    /// message-array boundary so that these groups are never split.
    fn adjust_boundary_for_tool_calls(messages: &[ChatMessage], boundary: usize) -> usize {
        if boundary == 0 || boundary >= messages.len() {
            return boundary;
        }
        if messages[boundary].role != "tool" {
            return boundary;
        }
        let tool_call_id = match messages[boundary].tool_call_id.as_deref() {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => return boundary,
        };
        for j in (0..boundary).rev() {
            if let Some(ref calls) = messages[j].tool_calls {
                if calls.iter().any(|c| c.id == tool_call_id) {
                    return j;
                }
            }
        }
        boundary
    }

    /// Safety net: convert orphaned `role: "tool"` messages (no preceding
    /// assistant with matching `tool_calls`) to `user` messages so the
    /// content is preserved but the API-invalid role is removed.
    pub fn remove_orphaned_tool_messages(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        let mut known_tool_call_ids: Vec<String> = Vec::new();
        let mut result = Vec::with_capacity(messages.len());

        for msg in messages {
            if msg.role == "assistant" {
                if let Some(ref calls) = msg.tool_calls {
                    for call in calls {
                        known_tool_call_ids.push(call.id.clone());
                    }
                }
                result.push(msg);
            } else if msg.role == "tool" {
                let is_orphaned = match msg.tool_call_id.as_deref() {
                    Some(id) if !id.is_empty() => !known_tool_call_ids.iter().any(|kid| kid == id),
                    _ => true,
                };
                if is_orphaned {
                    result.push(ChatMessage {
                        role: "user".to_string(),
                        content: msg.content,
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    });
                } else {
                    result.push(msg);
                }
            } else {
                result.push(msg);
            }
        }
        result
    }

    fn summarize_middle_messages(&self, messages: &[ChatMessage]) -> String {
        let mut tool_calls = Vec::new();
        let mut summaries = Vec::new();
        let mut errors = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "assistant" => {
                    if let Some(ref tool_calls_data) = msg.tool_calls {
                        for tc in tool_calls_data {
                            tool_calls.push(tc.function.name.clone());
                        }
                    }
                    if msg.content.len() > 50 && msg.content.len() < 200 {
                        summaries.push(msg.content.clone());
                    }
                }
                "tool" => {
                    if msg.content.contains("error") || msg.content.contains("Error") {
                        errors.push(msg.content.chars().take(100).collect::<String>());
                    }
                }
                _ => {}
            }
        }

        let mut parts = Vec::new();

        if !tool_calls.is_empty() {
            let unique_tools: std::collections::HashSet<_> = tool_calls.into_iter().collect();
            parts.push(format!(
                "Tools called: {}",
                unique_tools.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }

        if !errors.is_empty() {
            parts.push(format!("Errors: {}", errors.len()));
        }

        if !summaries.is_empty() {
            parts.push(format!("Key content: {}", summaries.join("; ")));
        }

        parts.join(" | ")
    }

    pub fn max_messages(&self) -> usize {
        self.max_messages
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_settings() -> ToolResultCompressorSettings {
        ToolResultCompressorSettings {
            enabled: true,
            max_full_results: 2,
            max_summary_length: 200,
            compression_trigger: 5,
            compress_tool_result_threshold: 500,
        }
    }

    fn default_context_settings() -> ContextWindowSettings {
        ContextWindowSettings {
            max_messages: 15,
            max_tokens: 16000,
            compression_ratio: 0.3,
            preserve_recent: 4,
        }
    }

    #[test]
    fn test_compressor_add_result() {
        let mut compressor = ToolResultCompressor::new(&default_settings());

        compressor.add_result(1, "file_read", "call_001", "test content");
        assert_eq!(compressor.get_results().len(), 1);
        assert_eq!(compressor.get_results()[0].tool_call_id, "call_001");
    }

    #[test]
    fn test_compressor_compress() {
        let mut compressor = ToolResultCompressor::new(&default_settings());

        let long_content = "x".repeat(500);

        for i in 1..=6 {
            compressor.add_result(i, "file_read", &format!("call_{}", i), &long_content);
        }

        let results = compressor.get_results();
        assert!(results.front().unwrap().is_compressed);
        assert!(!results.back().unwrap().is_compressed);
    }

    #[test]
    fn test_compress_tool_messages_by_call_id() {
        let mut compressor = ToolResultCompressor::new(&default_settings());

        // Add results and trigger compression
        let long = "y".repeat(500);
        for i in 1..=6 {
            compressor.add_result(i, "file_read", &format!("call_{}", i), &long);
        }
        assert!(compressor.get_results().front().unwrap().is_compressed);

        // Build messages: system + several tool messages
        let mut msgs = vec![ChatMessage {
            role: "system".to_string(),
            content: "sys".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        for i in 1..=4 {
            msgs.push(ChatMessage {
                role: "tool".to_string(),
                content: long.clone(),
                name: None,
                tool_calls: None,
                tool_call_id: Some(format!("call_{}", i)),
                reasoning_content: None,
            });
        }

        compressor.compress_tool_messages(&mut msgs);

        // call_1 and call_2 are compressed (first two entries)
        let compressed_ids: std::collections::HashSet<String> = compressor
            .results
            .iter()
            .filter(|e| e.is_compressed)
            .map(|e| e.tool_call_id.clone())
            .collect();
        for msg in msgs.iter().filter(|m| m.role == "tool") {
            let cid = msg.tool_call_id.as_ref().unwrap();
            if compressed_ids.contains(cid) {
                assert!(
                    msg.content.starts_with("[Summary"),
                    "tool_call_id={} should be compressed",
                    cid
                );
            } else {
                assert_eq!(
                    msg.content.len(),
                    long.len(),
                    "tool_call_id={} should remain full",
                    cid
                );
            }
        }
    }

    #[test]
    fn test_context_window_should_compress() {
        let manager = ContextWindowManager::new(&default_context_settings());
        let empty: Vec<ChatMessage> = Vec::new();

        assert!(!manager.should_compress(10, &empty));
        assert!(manager.should_compress(20, &empty));
    }
}
