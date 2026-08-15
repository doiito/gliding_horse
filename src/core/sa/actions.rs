use std::future::Future;
use std::pin::Pin;

use tracing::{info, warn};

use crate::core::event_bus::EventPriority;
use crate::core::sa::{
    ActionParams, CycleState, InterventionAction, SupervisorAgent, SupplementaryInputAction,
};
use crate::CoreError;

type ActionHandler = Box<
    dyn for<'a> Fn(
            &'a mut SupervisorAgent,
            &'a mut CycleState,
            ActionParams,
            &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<(), CoreError>> + Send + 'a>>
        + Send,
>;

/// Intervention action registry: look up handler by action type.
///
/// Handlers receive `&mut CycleState` so effective actions can write back
/// into `cycle.intervention`.
pub(super) fn get_action_handler(action: &InterventionAction) -> Option<ActionHandler> {
    match action {
        InterventionAction::Continue => Some(Box::new(|_sa, _cycle, _params, _task_iri| {
            Box::pin(async move {
                info!("Intervention: continue execution");
                Ok(())
            })
        })),
        InterventionAction::ContinueWithMonitor => Some(Box::new(|_sa, cycle, _params, _task_iri| {
            Box::pin(async move {
                cycle.intervention.monitor = true;
                warn!("Intervention: continue with monitoring");
                Ok(())
            })
        })),
        InterventionAction::IncreaseRetry { .. } => Some(Box::new(|_sa, cycle, params, _task_iri| {
            Box::pin(async move {
                let retries = params.additional_retries.unwrap_or(3);
                cycle.intervention.max_iterations_delta += retries as i32;
                info!(
                    "Intervention: increase retries by {} (max_iterations_delta={})",
                    retries, cycle.intervention.max_iterations_delta
                );
                Ok(())
            })
        })),
        InterventionAction::IncreaseTimeout { .. } => Some(Box::new(|_sa, cycle, params, _task_iri| {
            Box::pin(async move {
                let secs = params.additional_seconds.unwrap_or(60);
                cycle.intervention.timeout_delta_secs += secs as i64;
                info!(
                    "Intervention: increase timeout by {}s (timeout_delta_secs={})",
                    secs, cycle.intervention.timeout_delta_secs
                );
                Ok(())
            })
        })),
        InterventionAction::ReduceComplexity => Some(Box::new(|_sa, _cycle, _params, _task_iri| {
            Box::pin(async move {
                info!("Intervention: reduce complexity expectation");
                Ok(())
            })
        })),
        InterventionAction::RestrictTools { .. } => Some(Box::new(|_sa, cycle, params, _task_iri| {
            Box::pin(async move {
                let tools = params.allowed_tools.clone().unwrap_or_default();
                cycle.intervention.tool_allowlist_override = Some(tools.clone());
                info!("Intervention: restrict tools to {:?}", tools);
                Ok(())
            })
        })),
        InterventionAction::SkipStep { .. } => Some(Box::new(|_sa, _cycle, params, _task_iri| {
            Box::pin(async move {
                let step = params.step_id.as_deref().unwrap_or("unknown");
                info!("Intervention: skip step {}", step);
                Ok(())
            })
        })),
        InterventionAction::RetryStep { .. } => Some(Box::new(|_sa, _cycle, params, _task_iri| {
            Box::pin(async move {
                let step = params.step_id.as_deref().unwrap_or("unknown");
                info!("Intervention: retry step {}", step);
                Ok(())
            })
        })),
        InterventionAction::Parallelize => Some(Box::new(|_sa, _cycle, _params, _task_iri| {
            Box::pin(async move {
                info!("Intervention: parallelize execution");
                Ok(())
            })
        })),
        InterventionAction::SplitStep { .. } => Some(Box::new(|_sa, _cycle, params, _task_iri| {
            Box::pin(async move {
                let step = params.step_id.as_deref().unwrap_or("unknown");
                let sub_steps = params.sub_steps.clone().unwrap_or_default();
                info!(
                    "Intervention: split step {} into {:?} sub-steps",
                    step,
                    sub_steps.len()
                );
                Ok(())
            })
        })),
        InterventionAction::InsertExtraStep { .. } => {
            Some(Box::new(|_sa, _cycle, params, _task_iri| {
                Box::pin(async move {
                    let desc = params.description.as_deref().unwrap_or("unknown");
                    info!("Intervention: insert extra step: {}", desc);
                    Ok(())
                })
            }))
        }
        InterventionAction::FallbackToShallow => Some(Box::new(|_sa, _cycle, _params, _task_iri| {
            Box::pin(async move {
                info!("Intervention: fall back to shallow mode");
                Ok(())
            })
        })),
        InterventionAction::EmergencyMode => Some(Box::new(|_sa, _cycle, _params, _task_iri| {
            Box::pin(async move {
                warn!("Intervention: entering emergency mode");
                Ok(())
            })
        })),
        InterventionAction::IncreaseBudget { .. } => Some(Box::new(|_sa, _cycle, params, _task_iri| {
            Box::pin(async move {
                let tokens = params.additional_tokens.unwrap_or(1000);
                let secs = params.additional_time_secs.unwrap_or(120);
                info!(
                    "Intervention: increase budget {} tokens + {}s (human approved)",
                    tokens, secs
                );
                Ok(())
            })
        })),
        InterventionAction::FreezeAndReport => {
            Some(Box::new(|sa, _cycle, _params, task_iri| {
                Box::pin(async move {
                    info!("Intervention: freeze state and generate report");
                    let _ = sa.event_bus.emit(task_iri, "TASK_FROZEN", "SA",
                    &serde_json::json!({"action": "freeze_and_report", "task_iri": task_iri}).to_string()).await;
                    Ok(())
                })
            }))
        }
        InterventionAction::AbortTask { .. } => Some(Box::new(|sa, _cycle, params, task_iri| {
            Box::pin(async move {
                let reason = params.reason.as_deref().unwrap_or("no specific reason");
                warn!("Intervention: abort task, reason: {}", reason);
                let _ = sa
                    .event_bus
                    .emit(
                        task_iri,
                        "TASK_ABORTED",
                        "SA",
                        &serde_json::json!({"reason": reason}).to_string(),
                    )
                    .await;
                Ok(())
            })
        })),
        InterventionAction::NotifyHuman { .. } => Some(Box::new(|sa, _cycle, params, task_iri| {
            Box::pin(async move {
                let msg = params
                    .message
                    .as_deref()
                    .unwrap_or("Human intervention needed");
                info!("Intervention: notify human: {}", msg);
                let _ = sa
                    .event_bus
                    .emit_with_priority(
                        task_iri,
                        "NOTIFY_HUMAN",
                        "SA",
                        &serde_json::json!({"message": msg, "task_iri": task_iri}).to_string(),
                        EventPriority::Critical,
                    )
                    .await;
                Ok(())
            })
        })),
    }
}

/// Extract JSON from LLM output
#[allow(dead_code)]
fn extract_json(content: &str) -> &str {
    if content.starts_with('{') {
        content
    } else if let Some(start) = content.find('{') {
        if let Some(end) = content.rfind('}') {
            &content[start..=end]
        } else {
            content
        }
    } else {
        content
    }
}

/// Supplementary input action registry
#[allow(dead_code)]
type SupplementaryInputHandler = Box<
    dyn for<'a> Fn(
            &'a SupervisorAgent,
            SupplementaryInputAction,
            ActionParams,
            &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<(), CoreError>> + Send + 'a>>
        + Send,
>;

#[allow(dead_code)]
fn get_supplementary_handler(
    action: &SupplementaryInputAction,
) -> Option<SupplementaryInputHandler> {
    match action {
        SupplementaryInputAction::AddContext => {
            Some(Box::new(|sa, _action, _params, supplement| {
                Box::pin(async move {
                    info!("Supplementary input: add context");
                    sa.inject_to_current_agent("", supplement).await;
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::RefineObjective => {
            Some(Box::new(|_sa, _action, _params, supplement| {
                Box::pin(async move {
                    info!("Supplementary input: refine objective - {}", supplement);
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::ProvideConstraint => {
            Some(Box::new(|_sa, _action, _params, supplement| {
                Box::pin(async move {
                    info!("Supplementary input: provide constraint - {}", supplement);
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::GuideDirection => {
            Some(Box::new(|sa, _action, _params, supplement| {
                Box::pin(async move {
                    info!("Supplementary input: guide direction - {}", supplement);
                    sa.inject_to_current_agent("", supplement).await;
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::PrioritizeStep => {
            Some(Box::new(|_sa, _action, params, _supplement| {
                Box::pin(async move {
                    let step = params.step_id.as_deref().unwrap_or("next");
                    info!("Supplementary input: prioritize step - {}", step);
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::SuggestApproach => {
            Some(Box::new(|sa, _action, _params, supplement| {
                Box::pin(async move {
                    info!("Supplementary input: suggest approach - {}", supplement);
                    sa.inject_to_current_agent("", supplement).await;
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::PauseExecution => {
            Some(Box::new(|_sa, _action, _params, _supplement| {
                Box::pin(async move {
                    warn!("Supplementary input: pause execution");
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::ResumeExecution => {
            Some(Box::new(|_sa, _action, _params, _supplement| {
                Box::pin(async move {
                    info!("Supplementary input: resume execution");
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::SkipCurrentStep => {
            Some(Box::new(|_sa, _action, _params, _supplement| {
                Box::pin(async move {
                    info!("Supplementary input: skip current step");
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::ConfirmDirection => {
            Some(Box::new(|sa, _action, _params, supplement| {
                Box::pin(async move {
                    info!("Supplementary input: confirm direction");
                    sa.inject_to_current_agent("", supplement).await;
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::CorrectApproach => {
            Some(Box::new(|sa, _action, _params, supplement| {
                Box::pin(async move {
                    info!("Supplementary input: correct direction - {}", supplement);
                    sa.inject_to_current_agent("", supplement).await;
                    Ok(())
                })
            }))
        }
        SupplementaryInputAction::AbortCurrentStep => {
            Some(Box::new(|_sa, _action, _params, _supplement| {
                Box::pin(async move {
                    warn!("Supplementary input: abort current step");
                    Ok(())
                })
            }))
        }
    }
}

pub(super) fn parse_or_repair_json<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, String> {
    if let Ok(v) = serde_json::from_str(raw) {
        return Ok(v);
    }

    let mut repaired = String::with_capacity(raw.len() + 8);
    let mut in_string = false;
    let mut escaped = false;
    let mut brace_depth: i32 = 0;
    let mut bracket_depth: i32 = 0;

    for c in raw.chars() {
        if escaped {
            escaped = false;
        } else if c == '\\' && in_string {
            escaped = true;
        } else if c == '"' {
            in_string = !in_string;
        } else if !in_string {
            match c {
                '{' => brace_depth += 1,
                '}' => brace_depth -= 1,
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                _ => {}
            }
        }
        repaired.push(c);
    }

    if in_string {
        repaired.push('"');
    }
    while repaired.ends_with(',') {
        repaired.pop();
    }
    for _ in 0..brace_depth.max(0) {
        repaired.push('}');
    }
    for _ in 0..bracket_depth.max(0) {
        repaired.push(']');
    }

    serde_json::from_str(&repaired).map_err(|e| format!("{}", e))
}
