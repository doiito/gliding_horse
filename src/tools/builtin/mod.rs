pub mod bash;
pub mod file_ops;
pub mod hooks;
pub mod knowledge;
pub mod mcp;
pub mod permissions;
pub mod rag;
pub mod sandbox;

#[cfg(feature = "ontology")]
pub mod ontology_tools;

pub use permissions::{
    PermissionContext, PermissionMode, PermissionOutcome, PermissionOverride, PermissionPolicy,
    PermissionPromptDecision, PermissionPrompter, PermissionRequest,
};

pub use hooks::{
    HookAbortSignal, HookEvent, HookPermissionDecision, HookProgressEvent, HookProgressReporter,
    HookRunResult, HookRunner,
};

pub use knowledge::{
    execute_knowledge_delete, execute_knowledge_import_directory, execute_knowledge_import_file,
    execute_knowledge_import_url, execute_knowledge_list, execute_knowledge_search,
    execute_knowledge_update,
};

#[cfg(feature = "ontology")]
pub use ontology_tools::{
    execute_ontology_diff_turtle, execute_ontology_lint_turtle, execute_ontology_reason,
    execute_ontology_validate_shacl, execute_ontology_validate_turtle,
};
