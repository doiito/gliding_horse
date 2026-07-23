//! Batch Agent Error Types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum BatchError {
    #[error("Agent '{name}' not found")]
    AgentNotFound { name: String },

    #[error("Agent '{name}' already registered")]
    AgentAlreadyExists { name: String },

    #[error("Agent '{name}' is not running")]
    AgentNotRunning { name: String },

    #[error("Agent '{name}' is already running")]
    AgentAlreadyRunning { name: String },

    #[error("Invalid window configuration: {message}")]
    InvalidWindowConfig { message: String },

    #[error("Invalid trigger configuration: {message}")]
    InvalidTriggerConfig { message: String },

    #[error("Invalid cron expression '{expression}': {detail}")]
    InvalidCronExpression { expression: String, detail: String },

    #[error("LLM extraction failed after {attempts} attempts: {message}")]
    ExtractionFailed { attempts: u32, message: String },

    #[error("Output validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("Template not found: {name}")]
    TemplateNotFound { name: String },

    #[error("Knowledge graph write failed: {message}")]
    KgWriteFailed { message: String },

    #[error("Memory operation failed: {message}")]
    MemoryOperationFailed { message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("Event emission failed: {message}")]
    EventEmitFailed { message: String },

    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl From<serde_json::Error> for BatchError {
    fn from(e: serde_json::Error) -> Self {
        BatchError::SerializationError {
            message: e.to_string(),
        }
    }
}

impl From<std::io::Error> for BatchError {
    fn from(e: std::io::Error) -> Self {
        BatchError::Internal {
            message: e.to_string(),
        }
    }
}
