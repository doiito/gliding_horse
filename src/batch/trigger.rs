use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use tokio::sync::broadcast;
use tracing::{debug, warn};

use crate::batch::types::{TriggerConfig, TriggerReason, TriggerType};
use crate::batch::window::SlidingWindow;
use crate::core::event_bus::EventBus;

#[derive(Debug, Clone)]
pub struct CronSchedule {
    pub job_id: String,
    pub expression: String,
    pub one_shot: bool,
}

impl CronSchedule {
    /// Parse a cron-ish expression and compute the next trigger time.
    /// Supports simplified format: "*/N * * * * *" (seconds granularity)
    pub fn next_after(&self, from: DateTime<Utc>) -> Option<DateTime<Utc>> {
        parse_cron_next(&self.expression, from)
    }
}

/// Parse a simplified cron expression and return the next occurrence after `from`.
/// Format: "sec min hour day month weekday"
/// We only support "*/N" and fixed numbers at the seconds position for batch usage.
fn parse_cron_next(expression: &str, from: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let parts: Vec<&str> = expression.split_whitespace().collect();
    if parts.len() < 1 {
        return None;
    }

    // Only parse seconds field: "*/N" or fixed second
    let sec_part = parts[0];
    let interval_secs = if let Some(n) = sec_part.strip_prefix("*/") {
        n.parse::<i64>().ok()
    } else if let Ok(n) = sec_part.parse::<i64>() {
        Some(n)
    } else {
        None
    }?;

    if interval_secs <= 0 {
        return None;
    }

    // Find the next multiple of interval_secs from the epoch
    let timestamp = from.timestamp();
    let remainder = timestamp % interval_secs;
    let next = if remainder == 0 {
        timestamp
    } else {
        timestamp + interval_secs - remainder
    };

    // If the calculated time is in the past or too close (< 1s), add one interval
    let next = if next <= timestamp {
        next + interval_secs
    } else {
        next
    };

    DateTime::from_timestamp(next, 0)
}

pub struct TriggerSystem {
    triggers: Vec<TriggerConfig>,
    window: Arc<RwLock<SlidingWindow>>,
    last_execution: DateTime<Utc>,
    cron_schedules: Vec<CronSchedule>,
}

impl TriggerSystem {
    pub fn new(triggers: Vec<TriggerConfig>, window: Arc<RwLock<SlidingWindow>>) -> Self {
        let cron_schedules: Vec<CronSchedule> = triggers
            .iter()
            .filter_map(|t| match &t.trigger_type {
                TriggerType::CronSchedule(expr) => Some(CronSchedule {
                    job_id: format!("cron_{}", uuid::Uuid::new_v4().hyphenated()),
                    expression: expr.clone(),
                    one_shot: false,
                }),
                _ => None,
            })
            .collect();

        Self {
            triggers,
            window,
            last_execution: Utc::now(),
            cron_schedules,
        }
    }

    pub async fn evaluate(&self) -> Vec<TriggerReason> {
        let mut reasons = Vec::new();

        for trigger in &self.triggers {
            match &trigger.trigger_type {
                TriggerType::WindowFull => {
                    let win = self.window.read();
                    let reason = win.should_trigger();
                    match reason {
                        TriggerReason::NotReady => continue,
                        r => reasons.push(r),
                    }
                }

                TriggerType::CronSchedule(_) => {
                    for cron in &self.cron_schedules {
                        if let Some(next) = cron.next_after(self.last_execution) {
                            if next <= Utc::now() {
                                reasons.push(TriggerReason::TimeElapsed {
                                    elapsed_secs: (Utc::now() - self.last_execution).num_seconds()
                                        as u64,
                                    max_secs: 0,
                                });
                            }
                        }
                    }
                }

                TriggerType::IntentShift => {
                    let win = self.window.read();
                    let threshold = trigger
                        .params
                        .get("threshold")
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(0.6);

                    if let Some((from, to)) = win.detect_intent_shift(threshold) {
                        reasons.push(TriggerReason::IntentShift { from, to });
                    }
                }

                TriggerType::MessageThreshold(threshold) => {
                    let win = self.window.read();
                    if win.len() >= *threshold {
                        reasons.push(TriggerReason::WindowFull {
                            count: win.len(),
                            max: *threshold,
                        });
                    }
                }

                TriggerType::CustomEvent(event_type) => {
                    // Custom events are handled externally via listen_to
                    // The evaluation here just checks if any pending custom event exists
                    // Currently a no-op; events are received via the listener
                    debug!("CustomEvent trigger {} pending evaluation", event_type);
                }
            }
        }

        reasons
    }

    pub fn listen_to(&mut self, event_bus: &EventBus, event_types: Vec<String>) {
        if event_types.is_empty() {
            return;
        }

        let window = self.window.clone();
        let mut rx = event_bus.subscribe();

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        if event_types.contains(&event.event_type) {
                            debug!(
                                event_type = %event.event_type,
                                "TriggerSystem received custom event"
                            );
                            // Add a synthetic entry so the window can detect it.
                            // Actual re-evaluation is triggered by the manager polling.
                            let mut win = window.write();
                            let entry = crate::batch::types::WindowEntry {
                                message_id: event.event_id.clone(),
                                role: "event".to_string(),
                                content: event.payload.clone(),
                                timestamp: event.timestamp,
                                estimated_intent: None,
                                metadata: std::collections::HashMap::from([
                                    ("event_type".to_string(), event.event_type.clone()),
                                    ("task_iri".to_string(), event.task_iri.clone()),
                                    (
                                        "source_agent_iri".to_string(),
                                        event.source_agent_iri.clone(),
                                    ),
                                ]),
                            };
                            if let Err(error) = win.push(entry) {
                                warn!(%error, "Failed to enqueue custom event into window");
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("TriggerSystem lagged by {} events", n);
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    pub fn next_scheduled_run(&self) -> Option<DateTime<Utc>> {
        let now = Utc::now();
        self.cron_schedules
            .iter()
            .filter_map(|c| c.next_after(now))
            .min()
    }

    pub fn last_execution(&self) -> DateTime<Utc> {
        self.last_execution
    }

    pub fn update_last_execution(&mut self, time: DateTime<Utc>) {
        self.last_execution = time;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cron_next_after() {
        let base = DateTime::from_timestamp(1000, 0).unwrap(); // some known timestamp

        // */5 seconds
        let next = parse_cron_next("*/5", base).unwrap();
        let diff = next.timestamp() - base.timestamp();
        assert!(
            diff > 0 && diff <= 5,
            "Expected next within 5s, got diff={}",
            diff
        );

        // */10 seconds
        let base2 = DateTime::from_timestamp(1003, 0).unwrap();
        let next2 = parse_cron_next("*/10", base2).unwrap();
        assert_eq!(next2.timestamp() % 10, 0, "Should land on a multiple of 10");
    }

    #[test]
    fn test_cron_at_exact_second() {
        let base = DateTime::from_timestamp(100, 0).unwrap();
        let next = parse_cron_next("*/30", base).unwrap();
        assert_eq!(
            next.timestamp(),
            120,
            "Next multiple of 30 after 100 is 120"
        );
    }

    #[test]
    fn test_cron_invalid_expression() {
        let base = Utc::now();
        assert!(parse_cron_next("", base).is_none());
        assert!(parse_cron_next("invalid", base).is_none());
    }

    #[tokio::test]
    async fn test_listen_to_enqueues_custom_event_into_window() {
        let window = Arc::new(parking_lot::RwLock::new(SlidingWindow::new(
            crate::batch::WindowConfig {
                max_entries: 10,
                min_entries: 1,
                time_window_secs: 600,
                intent_shift_threshold: 0.6,
            },
        )));
        let mut system = TriggerSystem::new(vec![], window.clone());
        let bus = crate::core::event_bus::EventBus::new(64);

        system.listen_to(&bus, vec!["custom.alert".to_string()]);
        bus.emit("iri://task/t1", "custom.alert", "iri://agent/a1", "payload-1")
            .await;

        // Give the spawned listener a moment to drain the broadcast channel.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let win = window.read();
        assert_eq!(win.len(), 1, "custom event must be enqueued into the window");
        let entry = win.entries().back().unwrap();
        assert_eq!(entry.role, "event");
        assert_eq!(entry.content, "payload-1");
        assert_eq!(
            entry.metadata.get("event_type").map(|s| s.as_str()),
            Some("custom.alert")
        );
    }
}
