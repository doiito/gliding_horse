use std::collections::VecDeque;

use chrono::{DateTime, Utc};
use tracing::debug;

use crate::batch::error::BatchError;
use crate::batch::types::{TriggerReason, WindowEntry, WindowStatus};

impl WindowConfig {
    pub fn hybrid(max_messages: usize, max_seconds: u64) -> Self {
        Self {
            max_entries: max_messages,
            min_entries: 1,
            time_window_secs: max_seconds,
            intent_shift_threshold: 0.6,
        }
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            max_entries: 10,
            min_entries: 1,
            time_window_secs: 600,
            intent_shift_threshold: 0.6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub max_entries: usize,
    pub min_entries: usize,
    pub time_window_secs: u64,
    pub intent_shift_threshold: f64,
}

pub struct SlidingWindow {
    entries: VecDeque<WindowEntry>,
    config: WindowConfig,
    last_trigger_time: DateTime<Utc>,
    running_summary: Option<String>,
}

impl SlidingWindow {
    pub fn new(config: WindowConfig) -> Self {
        Self {
            entries: VecDeque::new(),
            config,
            last_trigger_time: Utc::now(),
            running_summary: None,
        }
    }

    pub fn push(&mut self, entry: WindowEntry) -> Result<(), BatchError> {
        // Enforce max_entries by dropping oldest
        if self.entries.len() >= self.config.max_entries {
            if let Some(dropped) = self.entries.pop_front() {
                debug!(
                    dropped_msg = %dropped.message_id,
                    window_size = %self.entries.len(),
                    "Window overflow — dropped oldest entry"
                );
            }
        }
        self.entries.push_back(entry);
        Ok(())
    }

    pub fn drain(&mut self) -> Vec<WindowEntry> {
        let drained: Vec<WindowEntry> = self.entries.drain(..).collect();
        self.last_trigger_time = Utc::now();
        self.running_summary = None;
        drained
    }

    pub fn should_trigger(&self) -> TriggerReason {
        let count = self.entries.len();

        // Not enough data yet
        if count < self.config.min_entries {
            return TriggerReason::NotReady;
        }

        // Message count threshold
        if count >= self.config.max_entries {
            return TriggerReason::WindowFull {
                count,
                max: self.config.max_entries,
            };
        }

        // Time window
        if let Some(oldest) = self.entries.front().map(|e| e.timestamp) {
            let elapsed = (Utc::now() - oldest).num_seconds() as u64;
            if elapsed >= self.config.time_window_secs {
                return TriggerReason::TimeElapsed {
                    elapsed_secs: elapsed,
                    max_secs: self.config.time_window_secs,
                };
            }
        }

        TriggerReason::NotReady
    }

    pub fn status(&self) -> WindowStatus {
        WindowStatus {
            entry_count: self.entries.len(),
            oldest: self.entries.front().map(|e| e.timestamp),
            newest: self.entries.back().map(|e| e.timestamp),
            has_summary: self.running_summary.is_some(),
            last_trigger: Some(self.last_trigger_time),
        }
    }

    pub fn get_running_summary(&self) -> Option<&str> {
        self.running_summary.as_deref()
    }

    pub fn set_running_summary(&mut self, summary: String) {
        self.running_summary = Some(summary);
    }

    pub fn config(&self) -> &WindowConfig {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn last_trigger_time(&self) -> DateTime<Utc> {
        self.last_trigger_time
    }

    pub fn entries(&self) -> &VecDeque<WindowEntry> {
        &self.entries
    }
}

// Lightweight intent-shift detection using keyword overlap
impl SlidingWindow {
    pub fn detect_intent_shift(&self, threshold: f64) -> Option<(String, String)> {
        if self.entries.len() < 3 {
            return None;
        }

        let halves = self.entries.len() / 2;
        let first_half: Vec<&str> = self
            .entries
            .iter()
            .take(halves)
            .map(|e| e.content.as_str())
            .collect();
        let second_half: Vec<&str> = self
            .entries
            .iter()
            .skip(halves)
            .map(|e| e.content.as_str())
            .collect();

        let first_keywords = extract_keywords(&first_half.join(" "));
        let second_keywords = extract_keywords(&second_half.join(" "));

        let overlap: f64 = if first_keywords.is_empty() && second_keywords.is_empty() {
            1.0
        } else if first_keywords.is_empty() || second_keywords.is_empty() {
            0.0
        } else {
            let intersection: f64 = first_keywords
                .iter()
                .filter(|k| second_keywords.contains(k))
                .count() as f64;
            let union = (first_keywords.len() + second_keywords.len()) as f64 - intersection;
            if union == 0.0 {
                1.0
            } else {
                intersection / union
            }
        };

        if overlap < threshold {
            let from = first_keywords
                .into_iter()
                .take(3)
                .collect::<Vec<_>>()
                .join(", ");
            let to = second_keywords
                .into_iter()
                .take(3)
                .collect::<Vec<_>>()
                .join(", ");
            return Some((from, to));
        }

        None
    }
}

fn extract_keywords(text: &str) -> Vec<String> {
    let stop_words = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "each", "every", "both", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just", "because", "and", "but",
        "or", "if", "while", "that", "this", "it", "its", "i", "me", "my", "we", "our", "you",
        "your", "he", "she", "they", "what", "which", "who", "about", "use", "used",
    ];

    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| w.len() > 2 && !stop_words.contains(&w.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_entry(content: &str) -> WindowEntry {
        WindowEntry {
            message_id: format!("msg_{}", uuid::Uuid::new_v4().hyphenated()),
            role: "user".into(),
            content: content.to_string(),
            timestamp: Utc::now(),
            estimated_intent: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_push_and_drain() {
        let mut w = SlidingWindow::new(WindowConfig {
            max_entries: 5,
            min_entries: 1,
            time_window_secs: 60,
            intent_shift_threshold: 0.6,
        });

        w.push(make_entry("hello")).unwrap();
        w.push(make_entry("world")).unwrap();
        assert_eq!(w.len(), 2);

        let drained = w.drain();
        assert_eq!(drained.len(), 2);
        assert!(w.is_empty());
    }

    #[test]
    fn test_should_trigger_window_full() {
        let mut w = SlidingWindow::new(WindowConfig {
            max_entries: 3,
            min_entries: 1,
            time_window_secs: 600,
            intent_shift_threshold: 0.6,
        });

        w.push(make_entry("a")).unwrap();
        assert!(matches!(w.should_trigger(), TriggerReason::NotReady));

        w.push(make_entry("b")).unwrap();
        w.push(make_entry("c")).unwrap();
        assert!(matches!(
            w.should_trigger(),
            TriggerReason::WindowFull { .. }
        ));
    }

    #[test]
    fn test_intent_shift_detection() {
        let mut w = SlidingWindow::new(WindowConfig {
            max_entries: 10,
            min_entries: 3,
            time_window_secs: 600,
            intent_shift_threshold: 0.3,
        });

        w.push(make_entry("Rust is great for backend web services"))
            .unwrap();
        w.push(make_entry("I like the Rust web framework ecosystem"))
            .unwrap();
        w.push(make_entry("Rust has excellent performance characteristics"))
            .unwrap();
        w.push(make_entry("Let me check which Rust web framework to use"))
            .unwrap();
        assert!(w.detect_intent_shift(0.3).is_none());

        let mut w2 = SlidingWindow::new(WindowConfig {
            max_entries: 10,
            min_entries: 3,
            time_window_secs: 600,
            intent_shift_threshold: 0.3,
        });

        // Different topics
        w2.push(make_entry("The database schema needs an index on user_id"))
            .unwrap();
        w2.push(make_entry("We should normalize the orders table"))
            .unwrap();
        w2.push(make_entry(
            "Let me deploy the docker container to kubernetes",
        ))
        .unwrap();
        let shift = w2.detect_intent_shift(0.3);
        assert!(
            shift.is_some(),
            "Expected intent shift between DB and k8s topics"
        );
    }

    #[test]
    fn test_overflow_drops_oldest() {
        let mut w = SlidingWindow::new(WindowConfig {
            max_entries: 2,
            min_entries: 1,
            time_window_secs: 60,
            intent_shift_threshold: 0.6,
        });

        w.push(make_entry("first")).unwrap();
        w.push(make_entry("second")).unwrap();
        w.push(make_entry("third")).unwrap();

        assert_eq!(w.len(), 2);
        assert_eq!(w.entries().back().unwrap().content, "third");
        assert_eq!(w.entries().front().unwrap().content, "second");
    }
}
