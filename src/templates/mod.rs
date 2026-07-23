pub mod schemas;
pub mod template_engine;

pub use template_engine::{build_system_prompt, Schema, Template, TemplateEngine, TemplateManager};

pub use schemas::{
    create_aa_template, create_ca_template, create_da_template, create_pa_template,
    validate_template, AgentTemplate, PromptSegment, SegmentType, TemplateRegistry,
};
