pub mod emitter;
pub mod error;
pub mod manager;
pub mod trigger;
pub mod types;
pub mod vocabulary;
pub mod window;

pub mod bridge;
pub mod context;
pub mod extractor;
pub mod handlers;
pub mod journal;
pub mod persister;
pub mod prompt;
pub mod validator;

pub use bridge::BatchEventBridge;
pub use context::ContextCollector;
pub use emitter::BatchEventEmitter;
pub use error::BatchError;
pub use extractor::ExtractorPipeline;
pub use journal::{BatchEventEnvelope, BatchEventJournal};
pub use manager::{BatchAgentInstance, BatchAgentManager};
pub use persister::KnowledgePersister;
pub use prompt::DynamicPromptEngine;
pub use types::{
    BatchAgentConfig, BatchAgentStatus, BatchMetrics, DetectedIntent, EmitCondition,
    ExtractedDecision, ExtractedEntity, ExtractedRelation, ExtractionResult, PersistReport,
    PromptContext, PromptSource, RdfQuad, RdfValue, TriggerConfig, TriggerReason, TriggerType,
    WindowEntry, WindowStatus, WindowType,
};
pub use validator::OutputValidator;
pub use vocabulary::{EntityTypeConfig, IntentTypeConfig, RelationTypeConfig};
pub use window::{SlidingWindow, WindowConfig};
