pub mod graphify;
pub mod micro_tools;
pub mod router;
pub mod summary;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SESSION_NONCE_HEX_LEN: usize = 16;
const LEGACY_CALL_NONCE_HEX_LEN: usize = 16;
const CALL_NONCE_HEX_LEN: usize = 24;
const ENTITY_DISCRIMINATOR_HEX_LEN: usize = 12;

/// Collision-resistant identity for one provider tool call inside one L1
/// execution session. Provider call IDs are commonly reused (`call_0`) by
/// independent requests, so they are observability metadata rather than safe
/// process-wide storage or tool-registration keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultRoutingIdentity {
    pub provider_call_id: String,
    pub session_scope: String,
    pub routing_call_key: String,
    pub storage_iri: String,
    pub reader_name: String,
    pub graph_name: String,
}

impl ResultRoutingIdentity {
    /// Legacy two-component constructor retained for isolated tests and
    /// migration tooling. Production agent execution must use
    /// [`Self::for_tool_call`], because provider call IDs can be reused by
    /// consecutive model requests in the same L1 session.
    pub fn new(session_id: &str, provider_call_id: &str) -> Self {
        let session_scope = Self::session_scope_for(session_id);
        let call_nonce = digest_nonce("provider-call", provider_call_id, LEGACY_CALL_NONCE_HEX_LEN);
        Self::from_parts(provider_call_id, session_scope, call_nonce)
    }

    /// Build the production routing identity from the complete durable tool
    /// call identity. The raw provider ID is copied verbatim for protocol and
    /// diagnostics; only process-wide storage/tool names use the digest key.
    pub fn for_tool_call(
        agent_id: &str,
        l1_session_id: &str,
        llm_request_id: &str,
        provider_call_id: &str,
    ) -> Self {
        let session_scope = Self::session_scope_for(l1_session_id);
        let call_nonce = digest_nonce_parts(
            "tool-call-v2",
            &[agent_id, l1_session_id, llm_request_id, provider_call_id],
            CALL_NONCE_HEX_LEN,
        );
        Self::from_parts(provider_call_id, session_scope, call_nonce)
    }

    pub fn session_scope_for(session_id: &str) -> String {
        format!(
            "s{}",
            digest_nonce("session", session_id, SESSION_NONCE_HEX_LEN)
        )
    }

    /// Rehydrate a validated router-issued reference found in historical
    /// content. This intentionally does not attempt to reconstruct a key from
    /// a raw provider ID, because old ChatMessage values do not carry the LLM
    /// request ID needed to disambiguate reused IDs.
    pub fn from_routing_call_key(routing_call_key: &str, provider_call_id: &str) -> Option<Self> {
        if !valid_routing_call_key(routing_call_key) {
            return None;
        }
        let session_scope = routing_call_key.split_once("_c")?.0.to_string();
        Some(Self {
            provider_call_id: provider_call_id.to_string(),
            session_scope,
            storage_iri: format!("iri://tool-result/{routing_call_key}"),
            reader_name: format!("read_full_result_{routing_call_key}"),
            graph_name: format!("graph:tool-result:{routing_call_key}"),
            routing_call_key: routing_call_key.to_string(),
        })
    }

    fn from_parts(provider_call_id: &str, session_scope: String, call_nonce: String) -> Self {
        let routing_call_key = format!("{session_scope}_c{call_nonce}");
        Self {
            provider_call_id: provider_call_id.to_string(),
            session_scope,
            storage_iri: format!("iri://tool-result/{routing_call_key}"),
            reader_name: format!("read_full_result_{routing_call_key}"),
            graph_name: format!("graph:tool-result:{routing_call_key}"),
            routing_call_key,
        }
    }

    pub fn query_name(&self, entity_type: &str) -> String {
        format!(
            "query_{}_{}",
            self.routing_call_key,
            digest_nonce("entity-type", entity_type, ENTITY_DISCRIMINATOR_HEX_LEN)
        )
    }

    pub fn entity_details_name(&self) -> String {
        format!("get_entity_details_{}", self.routing_call_key)
    }

    pub fn relation_expansion_name(&self) -> String {
        format!("expand_relation_{}", self.routing_call_key)
    }
}

fn digest_nonce(domain: &str, value: &str, hex_len: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(value.as_bytes());
    let digest = hex::encode(hasher.finalize());
    digest[..hex_len].to_string()
}

fn digest_nonce_parts(domain: &str, values: &[&str], hex_len: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    for value in values {
        // Length-prefix every component so no pair of tuples can alias by
        // concatenation (for example ["ab", "c"] vs ["a", "bc"]).
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value.as_bytes());
    }
    let digest = hex::encode(hasher.finalize());
    digest[..hex_len].to_string()
}

pub(crate) fn valid_routing_call_key(value: &str) -> bool {
    let Some((session, call)) = value.split_once("_c") else {
        return false;
    };
    session.len() == SESSION_NONCE_HEX_LEN + 1
        && session.starts_with('s')
        && session[1..].bytes().all(|byte| byte.is_ascii_hexdigit())
        && matches!(call.len(), LEGACY_CALL_NONCE_HEX_LEN | CALL_NONCE_HEX_LEN)
        && call.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Extract one exact router-issued result reference from delivered historical
/// content. A reference is accepted only when its key is valid, belongs to the
/// expected L1 session, and all recognized IRI/reader fields agree. Ambiguous
/// or attacker-supplied mixed references fail closed.
pub fn routing_identity_from_content(
    content: &str,
    session_id: &str,
    provider_call_id: &str,
) -> Option<ResultRoutingIdentity> {
    let expected_scope = ResultRoutingIdentity::session_scope_for(session_id);
    let mut keys = std::collections::BTreeSet::<String>::new();

    for line in content.lines() {
        let line = line.trim();
        if let Some(iri) = line.strip_prefix("IRI: ") {
            if let Some(key) = iri.strip_prefix("iri://tool-result/") {
                if valid_routing_call_key(key) {
                    keys.insert(key.to_string());
                }
            }
        }
        if let Some(reader) = line.strip_prefix("Session reader: ") {
            let reader = reader.trim_matches('`');
            if let Some(key) = reader.strip_prefix("read_full_result_") {
                if valid_routing_call_key(key) {
                    keys.insert(key.to_string());
                }
            }
        }
    }

    if let Ok(serde_json::Value::Object(object)) =
        serde_json::from_str::<serde_json::Value>(content)
    {
        if let Some(key) = object
            .get("result_iri")
            .and_then(serde_json::Value::as_str)
            .and_then(|iri| iri.strip_prefix("iri://tool-result/"))
            .filter(|key| valid_routing_call_key(key))
        {
            keys.insert(key.to_string());
        }
        if let Some(key) = object
            .get("session_reader")
            .and_then(serde_json::Value::as_str)
            .and_then(|reader| reader.strip_prefix("read_full_result_"))
            .filter(|key| valid_routing_call_key(key))
        {
            keys.insert(key.to_string());
        }
    }

    if keys.len() != 1 {
        return None;
    }
    let key = keys.into_iter().next()?;
    let identity = ResultRoutingIdentity::from_routing_call_key(&key, provider_call_id)?;
    (identity.session_scope == expected_scope).then_some(identity)
}

/// Recognize only router-issued session tools. In particular, ordinary text
/// or business tools beginning with `query_` must not be treated as ephemeral.
pub fn is_session_scoped_micro_tool_name(name: &str) -> bool {
    if let Some(key) = name.strip_prefix("read_full_result_") {
        return valid_routing_call_key(key);
    }
    if let Some(key) = name.strip_prefix("get_entity_details_") {
        return valid_routing_call_key(key);
    }
    if let Some(key) = name.strip_prefix("expand_relation_") {
        return valid_routing_call_key(key);
    }
    let Some(rest) = name.strip_prefix("query_") else {
        return false;
    };
    let Some((key, discriminator)) = rest.rsplit_once('_') else {
        return false;
    };
    valid_routing_call_key(key)
        && matches!(
            discriminator.len(),
            ENTITY_DISCRIMINATOR_HEX_LEN | LEGACY_CALL_NONCE_HEX_LEN
        )
        && discriminator.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq)]
pub enum RouteDecision {
    PassThrough,
    Truncate {
        max_chars: usize,
    },
    Graphify {
        call_id: String,
        graph_name: String,
    },
    Summarize {
        call_id: String,
        preview_size: usize,
    },
    /// file_read over the small-file exemption: inline JSON skeleton + first `max_lines` lines; full content via `read_full_result_{call_id}` micro-tool.
    FileReadPreview {
        call_id: String,
        max_lines: usize,
        max_chars: usize,
    },
    /// A large shell/execution response. Keep the exit status and bounded
    /// stdout/stderr inline while archiving the complete response behind the
    /// exact session reader. Generic JSON graphification is a poor fit here:
    /// the often very large `command` field sorts before the actual outcome
    /// and can consume the whole preview.
    ExecutionPreview {
        call_id: String,
        max_chars: usize,
    },
}

#[derive(Debug, Clone)]
pub struct ToolResultMeta {
    pub tool_name: String,
    pub call_id: String,
    pub size_bytes: usize,
    pub is_json: bool,
    pub is_structured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub tool_type: MicroToolType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MicroToolType {
    EntityTypeQuery {
        entity_type: String,
        graph_name: String,
    },
    EntityDetails {
        graph_name: String,
    },
    RelationTraversal {
        graph_name: String,
    },
    FullTextRead {
        storage_key: String,
    },
}

#[derive(Debug, Clone)]
pub struct GraphifyResult {
    pub graph_name: String,
    pub entity_count: usize,
    pub relation_count: usize,
    pub entity_types: Vec<String>,
    pub summary: String,
    pub micro_tools: Vec<MicroToolSchema>,
}

#[derive(Debug, Clone)]
pub struct SchemaAnalysis {
    pub entity_types: Vec<(String, usize)>,
    pub relation_types: Vec<String>,
    pub property_names: Vec<String>,
    pub total_entities: usize,
    pub total_relations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_identity_is_session_scoped_and_provider_safe() {
        let pa = ResultRoutingIdentity::new("l1-pa-session", "call_0");
        let da = ResultRoutingIdentity::new("l1-da-session", "call_0");

        assert_eq!(pa.provider_call_id, "call_0");
        assert_eq!(da.provider_call_id, "call_0");
        assert_ne!(pa.session_scope, da.session_scope);
        assert_ne!(pa.routing_call_key, da.routing_call_key);
        assert_ne!(pa.storage_iri, da.storage_iri);
        assert_ne!(pa.reader_name, da.reader_name);
        assert_ne!(pa.graph_name, da.graph_name);
    }

    #[test]
    fn full_routing_identity_separates_reused_raw_id_across_requests() {
        let first = ResultRoutingIdentity::for_tool_call(
            "agent-1",
            "l1-shared",
            "request-1",
            "provider/call:原样-0",
        );
        let second = ResultRoutingIdentity::for_tool_call(
            "agent-1",
            "l1-shared",
            "request-2",
            "provider/call:原样-0",
        );

        assert_eq!(first.provider_call_id, "provider/call:原样-0");
        assert_eq!(second.provider_call_id, "provider/call:原样-0");
        assert_eq!(first.session_scope, second.session_scope);
        assert_ne!(first.routing_call_key, second.routing_call_key);
        assert_ne!(first.storage_iri, second.storage_iri);
        assert_ne!(first.reader_name, second.reader_name);
        assert_eq!(first.routing_call_key.len(), 43);
        assert_eq!(first.reader_name.len(), 60);
    }

    #[test]
    fn routed_reference_extraction_is_exact_session_scoped_and_ambiguity_safe() {
        let routing = ResultRoutingIdentity::for_tool_call("agent", "l1-own", "request", "call_0");
        let routed = format!(
            "preview\nIRI: {}\nSession reader: {}",
            routing.storage_iri, routing.reader_name
        );
        assert_eq!(
            routing_identity_from_content(&routed, "l1-own", "call_0")
                .expect("matching router reference")
                .routing_call_key,
            routing.routing_call_key
        );
        assert!(routing_identity_from_content(&routed, "l1-foreign", "call_0").is_none());

        let other = ResultRoutingIdentity::for_tool_call("agent", "l1-own", "other", "call_0");
        let ambiguous = format!("{routed}\nIRI: {}", other.storage_iri);
        assert!(routing_identity_from_content(&ambiguous, "l1-own", "call_0").is_none());
    }

    #[test]
    fn generated_tool_names_meet_provider_contract_and_are_precisely_recognized() {
        let routing = ResultRoutingIdentity::for_tool_call(
            "agent/with arbitrary unsafe characters",
            "L1/会话/with arbitrary and very long provider-unsafe characters",
            "request/also unsafe and long",
            "call_0/供应商/also extremely long and unsafe",
        );
        let names = [
            routing.reader_name.clone(),
            routing.query_name("https://example.test/ontology/VeryLongEntityType"),
            routing.entity_details_name(),
            routing.relation_expansion_name(),
        ];

        for name in names {
            assert!(name.len() <= 64, "tool name too long: {name}");
            assert!(
                name.bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')),
                "tool name contains provider-unsafe characters: {name}"
            );
            assert!(is_session_scoped_micro_tool_name(&name));
        }

        assert!(!is_session_scoped_micro_tool_name("query_customer_status"));
        assert!(!is_session_scoped_micro_tool_name("get_entity_details"));
        assert!(!is_session_scoped_micro_tool_name(
            "expand_relation_business"
        ));
        assert!(!is_session_scoped_micro_tool_name(
            "read_full_result_fabricated"
        ));
    }
}
