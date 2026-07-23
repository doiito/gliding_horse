//! JSON-LD module - JSON-LD context, types, and utility functions
//!
//! Provides global unified @context definitions, JSON-LD node types, and IRI processing tools

pub mod context;
pub mod framing;
pub mod registry;
pub mod type_router;
pub mod types;
pub mod utils;

pub use context::JsonLdContext;
pub use framing::{
    apply_frame, embed_node, estimate_tokens, filter_properties, fit_to_budget, EmbedDirective,
    FrameTemplate, AA_DECISION_FRAME, CA_REVIEW_FRAME, DA_INPUT_FRAME, PLAN_CONTEXT_FRAME,
    SUMMARY_ONLY_FRAME,
};
pub use type_router::{TypeRoute, TypeRouter};
pub use types::{JsonLdKeyword, JsonLdNode};
pub use utils::{generate_iri, is_iri_reference, parse_iri, validate_jsonld_node};
