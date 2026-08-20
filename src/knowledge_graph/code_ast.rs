use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::debug;
use tree_sitter::{Language, Node, Parser, Tree};

use super::rdf_mapper::RdfMapper;
use super::store::KnowledgeGraphStore;
use super::types::{EdgeDef, LLMExtractionOutput, NodeDef, RdfMappingResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Go,
    Java,
    C,
    Cpp,
    Unknown,
}

impl CodeLanguage {
    pub fn from_extension(ext: &str) -> Self {
        match ext {
            "rs" => Self::Rust,
            "py" | "pyi" | "pyw" => Self::Python,
            "js" | "mjs" | "cjs" => Self::JavaScript,
            "ts" => Self::TypeScript,
            "tsx" | "jsx" => Self::Tsx,
            "go" => Self::Go,
            "java" => Self::Java,
            "c" | "h" => Self::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Self::Cpp,
            _ => Self::Unknown,
        }
    }

    pub fn from_path(path: &Path) -> Self {
        path.extension()
            .and_then(|e| e.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    pub fn to_language(&self) -> Option<Language> {
        match self {
            Self::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
            Self::Python => Some(tree_sitter_python::LANGUAGE.into()),
            Self::JavaScript => Some(tree_sitter_javascript::LANGUAGE.into()),
            Self::TypeScript => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            Self::Tsx => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
            Self::Go => Some(tree_sitter_go::LANGUAGE.into()),
            Self::Java => Some(tree_sitter_java::LANGUAGE.into()),
            Self::C => Some(tree_sitter_c::LANGUAGE.into()),
            Self::Cpp => Some(tree_sitter_cpp::LANGUAGE.into()),
            Self::Unknown => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
            Self::Go => "go",
            Self::Java => "java",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
struct AstEntity {
    id: String,
    label: String,
    entity_type: String,
    description: Option<String>,
    properties: HashMap<String, serde_json::Value>,
    #[allow(dead_code)]
    source_location: Option<String>,
}

#[derive(Debug, Clone)]
struct AstRelation {
    source: String,
    target: String,
    relation: String,
}

#[derive(Debug, Clone)]
struct AstExtraction {
    entities: Vec<AstEntity>,
    relations: Vec<AstRelation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncrementalResult {
    Unchanged,
    Updated {
        entity_count: usize,
        relation_count: usize,
        quad_count: usize,
        deleted_quads: usize,
    },
    Created {
        entity_count: usize,
        relation_count: usize,
        quad_count: usize,
    },
}

fn compute_sha256(content: &str) -> String {
    use std::fmt::Write;
    let hash = <sha2::Sha256 as sha2::Digest>::digest(content.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in hash {
        write!(&mut hex, "{:02x}", byte).expect("write! to String is infallible");
    }
    hex
}

fn get_cached_hash(store: &KnowledgeGraphStore, file_path: &str, graph: &str) -> Option<String> {
    let subject_iri = format!("iri://entity/file:{}", file_path);
    let sparql = format!(
        "SELECT ?hash WHERE {{ GRAPH <{}> {{ <{}> <https://agent-os.org/ontology/meta/contentHash> ?hash . }} }}",
        graph, subject_iri
    );
    match store.query_sparql(&sparql, None) {
        Ok(results) if !results.is_empty() => results[0]
            .get("?hash")
            .and_then(|v| v.as_str())
            .map(String::from),
        _ => None,
    }
}

pub struct CodeAstExtractor;

impl CodeAstExtractor {
    pub fn extract_incremental(
        path: &str,
        graph: &str,
        store: &KnowledgeGraphStore,
    ) -> Result<IncrementalResult, String> {
        let file_path = Path::new(path);
        let lang = CodeLanguage::from_path(file_path);

        if lang == CodeLanguage::Unknown {
            return Err(format!(
                "unsupported file type: {}",
                file_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("?")
            ));
        }

        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read file {}: {}", path, e))?;

        let current_hash = compute_sha256(&source);
        let cached_hash = get_cached_hash(store, path, graph);

        if cached_hash.as_ref() == Some(&current_hash) {
            debug!(file = path, hash = %current_hash, "file content unchanged, skipping AST extraction");
            return Ok(IncrementalResult::Unchanged);
        }

        let is_update = cached_hash.is_some();

        let deleted_quads = if is_update {
            store.delete_code_file_subgraph(&format!("iri://entity/file:{}", path), graph)?
        } else {
            0
        };

        let result =
            Self::extract_from_source_with_hash(&source, &lang, path, graph, &current_hash)?;

        store.write_quads(&result.quads, graph)?;

        debug!(
            file = path,
            hash = %current_hash,
            entities = result.entity_count,
            relations = result.relation_count,
            quads = result.quads.len(),
            deleted = deleted_quads,
            is_update = is_update,
            "incremental AST update complete"
        );

        if is_update {
            Ok(IncrementalResult::Updated {
                entity_count: result.entity_count,
                relation_count: result.relation_count,
                quad_count: result.quads.len(),
                deleted_quads,
            })
        } else {
            Ok(IncrementalResult::Created {
                entity_count: result.entity_count,
                relation_count: result.relation_count,
                quad_count: result.quads.len(),
            })
        }
    }
    pub fn extract_from_file(path: &str, graph: &str) -> Result<RdfMappingResult, String> {
        let file_path = Path::new(path);
        let lang = CodeLanguage::from_path(file_path);

        if lang == CodeLanguage::Unknown {
            return Err(format!(
                "unsupported file type: {}",
                file_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("?")
            ));
        }

        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read file {}: {}", path, e))?;

        let hash = compute_sha256(&source);
        Self::extract_from_source_with_hash(&source, &lang, path, graph, &hash)
    }

    pub fn extract_from_source(
        source: &str,
        lang: &CodeLanguage,
        file_path: &str,
        graph: &str,
    ) -> Result<RdfMappingResult, String> {
        let hash = compute_sha256(source);
        Self::extract_from_source_with_hash(source, lang, file_path, graph, &hash)
    }

    fn extract_from_source_with_hash(
        source: &str,
        lang: &CodeLanguage,
        file_path: &str,
        graph: &str,
        content_hash: &str,
    ) -> Result<RdfMappingResult, String> {
        let language = lang
            .to_language()
            .ok_or_else(|| format!("language {} does not support AST extraction", lang.name()))?;

        let mut parser = Parser::new();
        parser
            .set_language(&language)
            .map_err(|e| format!("failed to set tree-sitter language: {}", e))?;

        let tree = parser
            .parse(source, None)
            .ok_or_else(|| "failed to parse source code".to_string())?;

        let extraction = match lang {
            CodeLanguage::Rust => Self::extract_rust(&tree, source, file_path),
            CodeLanguage::Python => Self::extract_python(&tree, source, file_path),
            CodeLanguage::JavaScript | CodeLanguage::TypeScript | CodeLanguage::Tsx => {
                Self::extract_js_ts(&tree, source, file_path)
            }
            CodeLanguage::Go => Self::extract_go(&tree, source, file_path),
            CodeLanguage::Java => Self::extract_java(&tree, source, file_path),
            CodeLanguage::C | CodeLanguage::Cpp => Self::extract_c_cpp(&tree, source, file_path),
            CodeLanguage::Unknown => AstExtraction {
                entities: vec![],
                relations: vec![],
            },
        };

        let file_entity_id = format!("file:{}", file_path);
        let mut entities = extraction.entities;
        let mut relations = extraction.relations;

        entities.push(AstEntity {
            id: file_entity_id.clone(),
            label: file_path.to_string(),
            entity_type: "https://agent-os.org/ontology/code/SourceFile".to_string(),
            description: Some(format!("{} source file", lang.name())),
            properties: {
                let mut props = HashMap::new();
                props.insert(
                    "language".to_string(),
                    serde_json::Value::String(lang.name().to_string()),
                );
                props.insert(
                    "path".to_string(),
                    serde_json::Value::String(file_path.to_string()),
                );
                props.insert(
                    "contentHash".to_string(),
                    serde_json::Value::String(content_hash.to_string()),
                );
                props.insert(
                    "sourceFile".to_string(),
                    serde_json::Value::String(file_path.to_string()),
                );
                props
            },
            source_location: None,
        });

        for entity in &entities {
            if entity.id != file_entity_id {
                relations.push(AstRelation {
                    source: file_entity_id.clone(),
                    target: entity.id.clone(),
                    relation: "https://agent-os.org/ontology/code/contains".to_string(),
                });
            }
        }

        let output = LLMExtractionOutput {
            nodes: entities
                .iter()
                .map(|e| NodeDef {
                    id: e.id.clone(),
                    node_type: e.entity_type.clone(),
                    label: e.label.clone(),
                    description: e.description.clone(),
                    properties: e.properties.clone(),
                })
                .collect(),
            edges: relations
                .iter()
                .map(|r| EdgeDef {
                    source: r.source.clone(),
                    target: r.target.clone(),
                    relation: r.relation.clone(),
                    properties: HashMap::new(),
                })
                .collect(),
        };

        let result = RdfMapper::map_extraction(&output, graph);

        debug!(
            language = lang.name(),
            entities = result.entity_count,
            relations = result.relation_count,
            quads = result.quads.len(),
            "AST extraction complete"
        );

        Ok(result)
    }

    fn node_text<'a>(node: &Node, source: &'a str) -> &'a str {
        &source[node.byte_range()]
    }

    fn make_id(prefix: &str, file_path: &str, name: &str, line: usize) -> String {
        let safe_name = name.replace(' ', "_").replace('<', "_").replace('>', "_");
        format!("{}:{}:{}:{}", prefix, file_path, safe_name, line)
    }

    fn extract_rust(tree: &Tree, source: &str, file_path: &str) -> AstExtraction {
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let root = tree.root_node();

        let mut cursor = root.walk();
        Self::walk_rust(
            &mut cursor,
            source,
            file_path,
            &mut entities,
            &mut relations,
        );

        AstExtraction {
            entities,
            relations,
        }
    }

    fn walk_rust(
        cursor: &mut tree_sitter::TreeCursor,
        source: &str,
        file_path: &str,
        entities: &mut Vec<AstEntity>,
        relations: &mut Vec<AstRelation>,
    ) {
        loop {
            let node = cursor.node();
            match node.kind() {
                "function_item" | "function_signature" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id =
                            Self::make_id("fn", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Function".to_string(),
                            description: Some(format!("fn {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                        Self::extract_rust_calls(&node, source, file_path, &id, relations);
                    }
                }
                "struct_item" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id = Self::make_id(
                            "struct",
                            file_path,
                            &name,
                            node.start_position().row + 1,
                        );
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Struct".to_string(),
                            description: Some(format!("struct {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                    }
                }
                "enum_item" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id =
                            Self::make_id("enum", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Enum".to_string(),
                            description: Some(format!("enum {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                    }
                }
                "trait_item" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id =
                            Self::make_id("trait", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Trait".to_string(),
                            description: Some(format!("trait {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                    }
                }
                "impl_item" => {
                    if let Some(type_node) = node.child_by_field_name("type") {
                        let type_name = Self::node_text(&type_node, source).to_string();
                        let trait_name = node
                            .child_by_field_name("trait")
                            .map(|n| Self::node_text(&n, source).to_string());

                        if let Some(trait_name) = trait_name {
                            let impl_id = Self::make_id(
                                "impl",
                                file_path,
                                &format!("{}+{}", trait_name, type_name),
                                node.start_position().row + 1,
                            );
                            let type_id = Self::make_id("struct", file_path, &type_name, 0);
                            let trait_id = Self::make_id("trait", file_path, &trait_name, 0);
                            relations.push(AstRelation {
                                source: type_id,
                                target: trait_id,
                                relation: "https://agent-os.org/ontology/code/implements".to_string(),
                            });
                            entities.push(AstEntity {
                                id: impl_id,
                                label: format!("impl {} for {}", trait_name, type_name),
                                entity_type: "https://agent-os.org/ontology/code/Impl".to_string(),
                                description: Some(format!("impl {} for {}", trait_name, type_name)),
                                properties: HashMap::new(),
                                source_location: Some(format!(
                                    "{}:{}",
                                    file_path,
                                    node.start_position().row + 1
                                )),
                            });
                        }
                    }

                    let mut impl_cursor = node.walk();
                    for child in node.children(&mut impl_cursor) {
                        if child.kind() == "function_item" || child.kind() == "function_signature" {
                            if let Some(name_node) = child.child_by_field_name("name") {
                                let name = Self::node_text(&name_node, source).to_string();
                                let id = Self::make_id(
                                    "fn",
                                    file_path,
                                    &name,
                                    child.start_position().row + 1,
                                );
                                entities.push(AstEntity {
                                    id: id.clone(),
                                    label: name.clone(),
                                    entity_type: "https://agent-os.org/ontology/code/Method".to_string(),
                                    description: Some(format!("method {}", name)),
                                    properties: HashMap::new(),
                                    source_location: Some(format!(
                                        "{}:{}",
                                        file_path,
                                        child.start_position().row + 1
                                    )),
                                });
                                Self::extract_rust_calls(&child, source, file_path, &id, relations);
                            }
                        }
                    }
                    if !cursor.goto_next_sibling() {
                        break;
                    }
                    continue;
                }
                "use_declaration" => {
                    let use_text = Self::node_text(&node, source).to_string();
                    let id = Self::make_id(
                        "use",
                        file_path,
                        &use_text.replace(' ', ""),
                        node.start_position().row + 1,
                    );
                    entities.push(AstEntity {
                        id: id.clone(),
                        label: use_text.clone(),
                        entity_type: "https://agent-os.org/ontology/code/Import".to_string(),
                        description: Some(use_text),
                        properties: HashMap::new(),
                        source_location: Some(format!(
                            "{}:{}",
                            file_path,
                            node.start_position().row + 1
                        )),
                    });
                }
                _ => {}
            }

            if cursor.goto_first_child() {
                Self::walk_rust(cursor, source, file_path, entities, relations);
                cursor.goto_parent();
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    fn extract_rust_calls(
        node: &Node,
        source: &str,
        file_path: &str,
        caller_id: &str,
        relations: &mut Vec<AstRelation>,
    ) {
        let mut call_cursor = node.walk();
        for child in node.children(&mut call_cursor) {
            Self::find_rust_calls_recursive(&child, source, file_path, caller_id, relations);
        }
    }

    fn find_rust_calls_recursive(
        node: &Node,
        source: &str,
        file_path: &str,
        caller_id: &str,
        relations: &mut Vec<AstRelation>,
    ) {
        if node.kind() == "call_expression" {
            if let Some(func_node) = node.child_by_field_name("function") {
                let call_name = Self::node_text(&func_node, source).to_string();
                let callee_id = Self::make_id("fn", file_path, &call_name, 0);
                relations.push(AstRelation {
                    source: caller_id.to_string(),
                    target: callee_id,
                    relation: "https://agent-os.org/ontology/code/calls".to_string(),
                });
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::find_rust_calls_recursive(&child, source, file_path, caller_id, relations);
        }
    }

    fn extract_python(tree: &Tree, source: &str, file_path: &str) -> AstExtraction {
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let root = tree.root_node();

        let mut cursor = root.walk();
        Self::walk_python(
            &mut cursor,
            source,
            file_path,
            &mut entities,
            &mut relations,
        );

        AstExtraction {
            entities,
            relations,
        }
    }

    fn walk_python(
        cursor: &mut tree_sitter::TreeCursor,
        source: &str,
        file_path: &str,
        entities: &mut Vec<AstEntity>,
        relations: &mut Vec<AstRelation>,
    ) {
        loop {
            let node = cursor.node();
            match node.kind() {
                "function_definition" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id =
                            Self::make_id("fn", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Function".to_string(),
                            description: Some(format!("def {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                        Self::extract_generic_calls(
                            &node, source, file_path, &id, relations, "call",
                        );
                    }
                }
                "class_definition" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id =
                            Self::make_id("class", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Class".to_string(),
                            description: Some(format!("class {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });

                        if let Some(arg_list) = node.child_by_field_name("superclasses") {
                            let mut arg_cursor = arg_list.walk();
                            for child in arg_list.children(&mut arg_cursor) {
                                if child.kind() == "identifier" || child.kind() == "attribute" {
                                    let parent_name = Self::node_text(&child, source).to_string();
                                    let parent_id =
                                        Self::make_id("class", file_path, &parent_name, 0);
                                    relations.push(AstRelation {
                                        source: id.clone(),
                                        target: parent_id,
                                        relation: "https://agent-os.org/ontology/code/inherits"
                                            .to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
                "import_statement" | "import_from_statement" => {
                    let import_text = Self::node_text(&node, source).to_string();
                    let id = Self::make_id(
                        "import",
                        file_path,
                        &import_text.replace(' ', ""),
                        node.start_position().row + 1,
                    );
                    entities.push(AstEntity {
                        id: id.clone(),
                        label: import_text.clone(),
                        entity_type: "https://agent-os.org/ontology/code/Import".to_string(),
                        description: Some(import_text),
                        properties: HashMap::new(),
                        source_location: Some(format!(
                            "{}:{}",
                            file_path,
                            node.start_position().row + 1
                        )),
                    });
                }
                _ => {}
            }

            if cursor.goto_first_child() {
                Self::walk_python(cursor, source, file_path, entities, relations);
                cursor.goto_parent();
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    fn extract_js_ts(tree: &Tree, source: &str, file_path: &str) -> AstExtraction {
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let root = tree.root_node();

        let mut cursor = root.walk();
        Self::walk_js_ts(
            &mut cursor,
            source,
            file_path,
            &mut entities,
            &mut relations,
        );

        AstExtraction {
            entities,
            relations,
        }
    }

    fn walk_js_ts(
        cursor: &mut tree_sitter::TreeCursor,
        source: &str,
        file_path: &str,
        entities: &mut Vec<AstEntity>,
        relations: &mut Vec<AstRelation>,
    ) {
        loop {
            let node = cursor.node();
            match node.kind() {
                "function_declaration" | "generator_function_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id =
                            Self::make_id("fn", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Function".to_string(),
                            description: Some(format!("function {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                        Self::extract_generic_calls(
                            &node,
                            source,
                            file_path,
                            &id,
                            relations,
                            "call_expression",
                        );
                    }
                }
                "class_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id =
                            Self::make_id("class", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Class".to_string(),
                            description: Some(format!("class {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });

                        if let Some(heritage) = node.child_by_field_name("heritage") {
                            let heritage_text = Self::node_text(&heritage, source);
                            for part in heritage_text.split(',') {
                                let trimmed = part.trim();
                                if !trimmed.is_empty() {
                                    let parent_id = Self::make_id("class", file_path, trimmed, 0);
                                    relations.push(AstRelation {
                                        source: id.clone(),
                                        target: parent_id,
                                        relation: "https://agent-os.org/ontology/code/inherits"
                                            .to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
                "method_definition" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id = Self::make_id(
                            "method",
                            file_path,
                            &name,
                            node.start_position().row + 1,
                        );
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Method".to_string(),
                            description: Some(format!("method {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                    }
                }
                "import_statement" | "import_clause" => {
                    let import_text = Self::node_text(&node, source).to_string();
                    let id = Self::make_id(
                        "import",
                        file_path,
                        &import_text.replace(' ', ""),
                        node.start_position().row + 1,
                    );
                    entities.push(AstEntity {
                        id: id.clone(),
                        label: import_text.clone(),
                        entity_type: "https://agent-os.org/ontology/code/Import".to_string(),
                        description: Some(import_text),
                        properties: HashMap::new(),
                        source_location: Some(format!(
                            "{}:{}",
                            file_path,
                            node.start_position().row + 1
                        )),
                    });
                }
                "variable_declarator" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let value_node = node.child_by_field_name("value");
                        let is_arrow_fn =
                            value_node.map_or(false, |v| v.kind() == "arrow_function");
                        if is_arrow_fn {
                            let id = Self::make_id(
                                "fn",
                                file_path,
                                &name,
                                node.start_position().row + 1,
                            );
                            entities.push(AstEntity {
                                id: id.clone(),
                                label: name.clone(),
                                entity_type: "https://agent-os.org/ontology/code/Function".to_string(),
                                description: Some(format!("const {} = () => ...", name)),
                                properties: HashMap::new(),
                                source_location: Some(format!(
                                    "{}:{}",
                                    file_path,
                                    node.start_position().row + 1
                                )),
                            });
                        }
                    }
                }
                "interface_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id = Self::make_id(
                            "interface",
                            file_path,
                            &name,
                            node.start_position().row + 1,
                        );
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Interface".to_string(),
                            description: Some(format!("interface {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                    }
                }
                "type_alias_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id =
                            Self::make_id("type", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/TypeAlias".to_string(),
                            description: Some(format!("type {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                    }
                }
                _ => {}
            }

            if cursor.goto_first_child() {
                Self::walk_js_ts(cursor, source, file_path, entities, relations);
                cursor.goto_parent();
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    fn extract_go(tree: &Tree, source: &str, file_path: &str) -> AstExtraction {
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let root = tree.root_node();

        let mut cursor = root.walk();
        Self::walk_go(
            &mut cursor,
            source,
            file_path,
            &mut entities,
            &mut relations,
        );

        AstExtraction {
            entities,
            relations,
        }
    }

    fn walk_go(
        cursor: &mut tree_sitter::TreeCursor,
        source: &str,
        file_path: &str,
        entities: &mut Vec<AstEntity>,
        relations: &mut Vec<AstRelation>,
    ) {
        loop {
            let node = cursor.node();
            match node.kind() {
                "function_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id =
                            Self::make_id("fn", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Function".to_string(),
                            description: Some(format!("func {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                        Self::extract_generic_calls(
                            &node,
                            source,
                            file_path,
                            &id,
                            relations,
                            "call_expression",
                        );
                    }
                }
                "method_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id = Self::make_id(
                            "method",
                            file_path,
                            &name,
                            node.start_position().row + 1,
                        );
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Method".to_string(),
                            description: Some(format!("method {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                    }
                }
                "type_declaration" => {
                    let mut type_cursor = node.walk();
                    for child in node.children(&mut type_cursor) {
                        if child.kind() == "type_spec" {
                            if let Some(name_node) = child.child_by_field_name("name") {
                                let name = Self::node_text(&name_node, source).to_string();
                                let type_kind = child
                                    .child_by_field_name("type")
                                    .map(|t| t.kind().to_string())
                                    .unwrap_or_default();
                                let entity_type = match type_kind.as_str() {
                                    "struct_type" => "https://agent-os.org/ontology/code/Struct",
                                    "interface_type" => "https://agent-os.org/ontology/code/Interface",
                                    _ => "https://agent-os.org/ontology/code/Type",
                                };
                                let id = Self::make_id(
                                    "type",
                                    file_path,
                                    &name,
                                    node.start_position().row + 1,
                                );
                                entities.push(AstEntity {
                                    id: id.clone(),
                                    label: name.clone(),
                                    entity_type: entity_type.to_string(),
                                    description: Some(format!("type {}", name)),
                                    properties: HashMap::new(),
                                    source_location: Some(format!(
                                        "{}:{}",
                                        file_path,
                                        node.start_position().row + 1
                                    )),
                                });
                            }
                        }
                    }
                }
                "import_declaration" => {
                    let import_text = Self::node_text(&node, source).to_string();
                    let id = Self::make_id(
                        "import",
                        file_path,
                        &import_text.replace(' ', "").replace('"', ""),
                        node.start_position().row + 1,
                    );
                    entities.push(AstEntity {
                        id: id.clone(),
                        label: import_text.clone(),
                        entity_type: "https://agent-os.org/ontology/code/Import".to_string(),
                        description: Some(import_text),
                        properties: HashMap::new(),
                        source_location: Some(format!(
                            "{}:{}",
                            file_path,
                            node.start_position().row + 1
                        )),
                    });
                }
                _ => {}
            }

            if cursor.goto_first_child() {
                Self::walk_go(cursor, source, file_path, entities, relations);
                cursor.goto_parent();
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    fn extract_java(tree: &Tree, source: &str, file_path: &str) -> AstExtraction {
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let root = tree.root_node();

        let mut cursor = root.walk();
        Self::walk_java(
            &mut cursor,
            source,
            file_path,
            &mut entities,
            &mut relations,
        );

        AstExtraction {
            entities,
            relations,
        }
    }

    fn walk_java(
        cursor: &mut tree_sitter::TreeCursor,
        source: &str,
        file_path: &str,
        entities: &mut Vec<AstEntity>,
        relations: &mut Vec<AstRelation>,
    ) {
        loop {
            let node = cursor.node();
            match node.kind() {
                "class_declaration" | "record_declaration" | "enum_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let entity_type = match node.kind() {
                            "enum_declaration" => "https://agent-os.org/ontology/code/Enum",
                            "record_declaration" => "https://agent-os.org/ontology/code/Record",
                            _ => "https://agent-os.org/ontology/code/Class",
                        };
                        let id =
                            Self::make_id("class", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: entity_type.to_string(),
                            description: Some(format!(
                                "{} {}",
                                node.kind().replace("_declaration", ""),
                                name
                            )),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });

                        if let Some(superclass) = node.child_by_field_name("superclass") {
                            let parent_name = Self::node_text(&superclass, source).to_string();
                            let parent_id = Self::make_id("class", file_path, &parent_name, 0);
                            relations.push(AstRelation {
                                source: id.clone(),
                                target: parent_id,
                                relation: "https://agent-os.org/ontology/code/inherits".to_string(),
                            });
                        }

                        if let Some(interfaces) = node.child_by_field_name("interfaces") {
                            let mut iface_cursor = interfaces.walk();
                            for child in interfaces.children(&mut iface_cursor) {
                                if child.kind() == "type_identifier" {
                                    let iface_name = Self::node_text(&child, source).to_string();
                                    let iface_id =
                                        Self::make_id("interface", file_path, &iface_name, 0);
                                    relations.push(AstRelation {
                                        source: id.clone(),
                                        target: iface_id,
                                        relation: "https://agent-os.org/ontology/code/implements"
                                            .to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
                "interface_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id = Self::make_id(
                            "interface",
                            file_path,
                            &name,
                            node.start_position().row + 1,
                        );
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Interface".to_string(),
                            description: Some(format!("interface {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                    }
                }
                "method_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let id = Self::make_id(
                            "method",
                            file_path,
                            &name,
                            node.start_position().row + 1,
                        );
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Method".to_string(),
                            description: Some(format!("method {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                        Self::extract_generic_calls(
                            &node,
                            source,
                            file_path,
                            &id,
                            relations,
                            "method_invocation",
                        );
                    }
                }
                "import_declaration" => {
                    let import_text = Self::node_text(&node, source).to_string();
                    let id = Self::make_id(
                        "import",
                        file_path,
                        &import_text.replace(' ', ""),
                        node.start_position().row + 1,
                    );
                    entities.push(AstEntity {
                        id: id.clone(),
                        label: import_text.clone(),
                        entity_type: "https://agent-os.org/ontology/code/Import".to_string(),
                        description: Some(import_text),
                        properties: HashMap::new(),
                        source_location: Some(format!(
                            "{}:{}",
                            file_path,
                            node.start_position().row + 1
                        )),
                    });
                }
                _ => {}
            }

            if cursor.goto_first_child() {
                Self::walk_java(cursor, source, file_path, entities, relations);
                cursor.goto_parent();
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    fn extract_c_cpp(tree: &Tree, source: &str, file_path: &str) -> AstExtraction {
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let root = tree.root_node();

        let mut cursor = root.walk();
        Self::walk_c_cpp(
            &mut cursor,
            source,
            file_path,
            &mut entities,
            &mut relations,
        );

        AstExtraction {
            entities,
            relations,
        }
    }

    fn walk_c_cpp(
        cursor: &mut tree_sitter::TreeCursor,
        source: &str,
        file_path: &str,
        entities: &mut Vec<AstEntity>,
        relations: &mut Vec<AstRelation>,
    ) {
        loop {
            let node = cursor.node();
            match node.kind() {
                "function_definition" => {
                    let decl = node.child_by_field_name("declarator");
                    let name = decl.and_then(|d| {
                        let mut cur = d.walk();
                        for child in d.children(&mut cur) {
                            if child.kind() == "identifier" || child.kind() == "field_identifier" {
                                return Some(Self::node_text(&child, source).to_string());
                            }
                            if child.kind() == "qualified_identifier" {
                                return Some(Self::node_text(&child, source).to_string());
                            }
                        }
                        None
                    });
                    if let Some(name) = name {
                        let id =
                            Self::make_id("fn", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: "https://agent-os.org/ontology/code/Function".to_string(),
                            description: Some(format!("function {}", name)),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                        Self::extract_generic_calls(
                            &node,
                            source,
                            file_path,
                            &id,
                            relations,
                            "call_expression",
                        );
                    }
                }
                "class_specifier" | "struct_specifier" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let name = Self::node_text(&name_node, source).to_string();
                        let entity_type = if node.kind() == "class_specifier" {
                            "https://agent-os.org/ontology/code/Class"
                        } else {
                            "https://agent-os.org/ontology/code/Struct"
                        };
                        let id =
                            Self::make_id("class", file_path, &name, node.start_position().row + 1);
                        entities.push(AstEntity {
                            id: id.clone(),
                            label: name.clone(),
                            entity_type: entity_type.to_string(),
                            description: Some(format!(
                                "{} {}",
                                node.kind().replace("_specifier", ""),
                                name
                            )),
                            properties: HashMap::new(),
                            source_location: Some(format!(
                                "{}:{}",
                                file_path,
                                node.start_position().row + 1
                            )),
                        });
                    }
                }
                "preproc_include" => {
                    let include_text = Self::node_text(&node, source).to_string();
                    let id = Self::make_id(
                        "include",
                        file_path,
                        &include_text.replace(' ', ""),
                        node.start_position().row + 1,
                    );
                    entities.push(AstEntity {
                        id: id.clone(),
                        label: include_text.clone(),
                        entity_type: "https://agent-os.org/ontology/code/Import".to_string(),
                        description: Some(include_text),
                        properties: HashMap::new(),
                        source_location: Some(format!(
                            "{}:{}",
                            file_path,
                            node.start_position().row + 1
                        )),
                    });
                }
                _ => {}
            }

            if cursor.goto_first_child() {
                Self::walk_c_cpp(cursor, source, file_path, entities, relations);
                cursor.goto_parent();
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    fn extract_generic_calls(
        node: &Node,
        source: &str,
        file_path: &str,
        caller_id: &str,
        relations: &mut Vec<AstRelation>,
        call_kind: &str,
    ) {
        let mut call_cursor = node.walk();
        Self::find_generic_calls_recursive(
            &mut call_cursor,
            source,
            file_path,
            caller_id,
            relations,
            call_kind,
        );
    }

    fn find_generic_calls_recursive(
        cursor: &mut tree_sitter::TreeCursor,
        source: &str,
        file_path: &str,
        caller_id: &str,
        relations: &mut Vec<AstRelation>,
        call_kind: &str,
    ) {
        let node = cursor.node();
        if node.kind() == call_kind {
            if let Some(func_node) = node.child_by_field_name("function") {
                let call_name = Self::node_text(&func_node, source).to_string();
                let callee_id = Self::make_id("fn", file_path, &call_name, 0);
                relations.push(AstRelation {
                    source: caller_id.to_string(),
                    target: callee_id,
                    relation: "https://agent-os.org/ontology/code/calls".to_string(),
                });
            }
        }

        if cursor.goto_first_child() {
            loop {
                Self::find_generic_calls_recursive(
                    cursor, source, file_path, caller_id, relations, call_kind,
                );
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_language_from_extension() {
        assert_eq!(CodeLanguage::from_extension("rs"), CodeLanguage::Rust);
        assert_eq!(CodeLanguage::from_extension("py"), CodeLanguage::Python);
        assert_eq!(CodeLanguage::from_extension("js"), CodeLanguage::JavaScript);
        assert_eq!(CodeLanguage::from_extension("ts"), CodeLanguage::TypeScript);
        assert_eq!(CodeLanguage::from_extension("tsx"), CodeLanguage::Tsx);
        assert_eq!(CodeLanguage::from_extension("go"), CodeLanguage::Go);
        assert_eq!(CodeLanguage::from_extension("java"), CodeLanguage::Java);
        assert_eq!(CodeLanguage::from_extension("c"), CodeLanguage::C);
        assert_eq!(CodeLanguage::from_extension("cpp"), CodeLanguage::Cpp);
        assert_eq!(CodeLanguage::from_extension("xyz"), CodeLanguage::Unknown);
    }

    #[test]
    fn test_code_language_from_path() {
        assert_eq!(
            CodeLanguage::from_path(Path::new("src/main.rs")),
            CodeLanguage::Rust
        );
        assert_eq!(
            CodeLanguage::from_path(Path::new("app.py")),
            CodeLanguage::Python
        );
        assert_eq!(
            CodeLanguage::from_path(Path::new("Makefile")),
            CodeLanguage::Unknown
        );
    }

    #[test]
    fn test_extract_rust_source() {
        let source = r#"
use std::collections::HashMap;

struct MyStruct {
    field: i32,
}

impl MyStruct {
    fn new() -> Self {
        Self { field: 0 }
    }
}

fn main() {
    let s = MyStruct::new();
    println!("hello");
}
"#;
        let result = CodeAstExtractor::extract_from_source(
            source,
            &CodeLanguage::Rust,
            "test.rs",
            "graph:test",
        )
        .unwrap();

        assert!(
            result.entity_count >= 3,
            "should extract at least 3 entities (use/struct/fn), actual: {}",
            result.entity_count
        );
        assert!(result.quads.len() > 0, "should generate RDF quads");
    }

    #[test]
    fn test_extract_python_source() {
        let source = r#"
import os
from typing import List

class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof"

def helper():
    print("help")
"#;
        let result = CodeAstExtractor::extract_from_source(
            source,
            &CodeLanguage::Python,
            "test.py",
            "graph:test",
        )
        .unwrap();

        assert!(
            result.entity_count >= 3,
            "should extract at least 3 entities (import/class/def), actual: {}",
            result.entity_count
        );
        assert!(result.quads.len() > 0, "should generate RDF quads");
    }

    #[test]
    fn test_extract_javascript_source() {
        let source = r#"
import React from 'react';

class Button extends React.Component {
    render() {
        return <button>Click</button>;
    }
}

function handleClick() {
    console.log('clicked');
}

const App = () => {
    return <div />;
};
"#;
        let result = CodeAstExtractor::extract_from_source(
            source,
            &CodeLanguage::JavaScript,
            "test.js",
            "graph:test",
        )
        .unwrap();

        assert!(
            result.entity_count >= 2,
            "should extract at least 2 entities, actual: {}",
            result.entity_count
        );
    }

    #[test]
    fn test_extract_go_source() {
        let source = r#"
package main

import "fmt"

type Server struct {
    Addr string
}

func (s *Server) Start() {
    fmt.Println("starting")
}

func main() {
    s := Server{Addr: ":8080"}
    s.Start()
}
"#;
        let result = CodeAstExtractor::extract_from_source(
            source,
            &CodeLanguage::Go,
            "test.go",
            "graph:test",
        )
        .unwrap();

        assert!(
            result.entity_count >= 3,
            "should extract at least 3 entities (import/type/func), actual: {}",
            result.entity_count
        );
    }

    #[test]
    fn test_extract_java_source() {
        let source = r#"
import java.util.List;

public class Main extends Base implements Runnable {
    @Override
    public void run() {
        System.out.println("running");
    }

    private void helper() {
        doWork();
    }
}
"#;
        let result = CodeAstExtractor::extract_from_source(
            source,
            &CodeLanguage::Java,
            "Main.java",
            "graph:test",
        )
        .unwrap();

        assert!(
            result.entity_count >= 3,
            "should extract at least 3 entities (import/class/method), actual: {}",
            result.entity_count
        );
    }

    #[test]
    fn test_extract_c_source() {
        let source = r#"
#include <stdio.h>

typedef struct {
    int x;
} Point;

int main() {
    printf("hello");
    return 0;
}
"#;
        let result =
            CodeAstExtractor::extract_from_source(source, &CodeLanguage::C, "test.c", "graph:test")
                .unwrap();

        assert!(
            result.entity_count >= 2,
            "should extract at least 2 entities (include/function), actual: {}",
            result.entity_count
        );
    }

    #[test]
    fn test_unknown_language_returns_error() {
        let result = CodeAstExtractor::extract_from_source(
            "hello",
            &CodeLanguage::Unknown,
            "test.xyz",
            "graph:test",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_rust_impl_trait_relation() {
        let source = r#"
trait Draw {
    fn draw(&self);
}

struct Circle {}

impl Draw for Circle {
    fn draw(&self) {}
}
"#;
        let result = CodeAstExtractor::extract_from_source(
            source,
            &CodeLanguage::Rust,
            "test.rs",
            "graph:test",
        )
        .unwrap();

        let has_implements = result
            .quads
            .iter()
            .any(|q| q.predicate.contains("implements"));
        assert!(has_implements, "should extract implements relation");
    }

    #[test]
    fn test_python_inheritance_relation() {
        let source = r#"
class Base:
    pass

class Child(Base):
    pass
"#;
        let result = CodeAstExtractor::extract_from_source(
            source,
            &CodeLanguage::Python,
            "test.py",
            "graph:test",
        )
        .unwrap();

        let has_inherits = result
            .quads
            .iter()
            .any(|q| q.predicate.contains("inherits"));
        assert!(has_inherits, "should extract inherits relation");
    }

    #[test]
    fn test_sha256_hash_deterministic() {
        let source = "fn main() {}";
        let hash1 = compute_sha256(source);
        let hash2 = compute_sha256(source);
        assert_eq!(hash1, hash2, "hashes of same content should be equal");
        assert_eq!(hash1.len(), 64, "SHA256 hash should be 64 hex characters");
    }

    #[test]
    fn test_sha256_hash_different_content() {
        let hash1 = compute_sha256("fn main() {}");
        let hash2 = compute_sha256("fn other() {}");
        assert_ne!(hash1, hash2, "hashes of different content should differ");
    }

    #[test]
    fn test_extract_includes_content_hash() {
        let source = "fn hello() {}";
        let result = CodeAstExtractor::extract_from_source(
            source,
            &CodeLanguage::Rust,
            "test.rs",
            "graph:test",
        )
        .unwrap();

        let has_hash = result
            .quads
            .iter()
            .any(|q| q.predicate.contains("contentHash"));
        assert!(
            has_hash,
            "extraction result should include contentHash property"
        );

        let has_source_file = result
            .quads
            .iter()
            .any(|q| q.predicate.contains("sourceFile"));
        assert!(
            has_source_file,
            "extraction result should include sourceFile property"
        );
    }

    #[test]
    fn test_incremental_create_then_unchanged() {
        let store = KnowledgeGraphStore::new().unwrap();
        let graph = "graph:test_incremental";
        let source = "fn hello() {}\nfn world() {}\n";

        let tmp_dir = tempfile::tempdir().unwrap();
        let file_path = tmp_dir.path().join("test_incr.rs");
        std::fs::write(&file_path, source).unwrap();
        let path_str = file_path.to_str().unwrap();

        let result1 = CodeAstExtractor::extract_incremental(path_str, graph, &store).unwrap();
        match result1 {
            IncrementalResult::Created {
                entity_count,
                quad_count,
                ..
            } => {
                assert!(entity_count > 0, "first extraction should have entities");
                assert!(quad_count > 0, "first extraction should have quads");
            }
            _ => panic!("first extraction should be Created, actual: {:?}", result1),
        }

        let result2 = CodeAstExtractor::extract_incremental(path_str, graph, &store).unwrap();
        assert_eq!(
            result2,
            IncrementalResult::Unchanged,
            "unchanged file should be Unchanged"
        );
    }

    #[test]
    fn test_incremental_update_detects_changes() {
        let store = KnowledgeGraphStore::new().unwrap();
        let graph = "graph:test_update";
        let source_v1 = "fn hello() {}\n";
        let source_v2 = "fn hello() {}\nfn world() {}\n";

        let tmp_dir = tempfile::tempdir().unwrap();
        let file_path = tmp_dir.path().join("test_update.rs");
        std::fs::write(&file_path, source_v1).unwrap();
        let path_str = file_path.to_str().unwrap();

        let result1 = CodeAstExtractor::extract_incremental(path_str, graph, &store).unwrap();
        match result1 {
            IncrementalResult::Created { .. } => {}
            _ => panic!("first should be Created"),
        }

        std::fs::write(&file_path, source_v2).unwrap();

        let result2 = CodeAstExtractor::extract_incremental(path_str, graph, &store).unwrap();
        match result2 {
            IncrementalResult::Updated {
                entity_count,
                deleted_quads,
                ..
            } => {
                assert!(entity_count > 0, "updated result should have entities");
                assert!(deleted_quads > 0, "update should delete old quads");
            }
            _ => panic!("changed file should be Updated, actual: {:?}", result2),
        }

        let result3 = CodeAstExtractor::extract_incremental(path_str, graph, &store).unwrap();
        assert_eq!(
            result3,
            IncrementalResult::Unchanged,
            "unchanged after update should be Unchanged"
        );
    }

    #[test]
    fn test_incremental_update_removes_declarations_no_longer_in_file() {
        let store = KnowledgeGraphStore::new().unwrap();
        let graph = "graph:code";
        let tmp_dir = tempfile::tempdir().unwrap();
        let file_path = tmp_dir.path().join("stale.rs");
        std::fs::write(&file_path, "fn removed_symbol() {}\n").unwrap();
        let path_str = file_path.to_str().unwrap();
        CodeAstExtractor::extract_incremental(path_str, graph, &store).unwrap();

        std::fs::write(&file_path, "fn replacement_symbol() {}\n").unwrap();
        CodeAstExtractor::extract_incremental(path_str, graph, &store).unwrap();

        let sparql = format!(
            "SELECT ?s WHERE {{ GRAPH <{}> {{ ?s <http://www.w3.org/2000/01/rdf-schema#label> \"removed_symbol\" . }} }}",
            graph
        );
        assert!(store.query_sparql(&sparql, None).unwrap().is_empty());
    }

    #[test]
    fn test_incremental_no_duplicate_entities() {
        let store = KnowledgeGraphStore::new().unwrap();
        let graph = "graph:test_dedup";
        let source = "fn hello() {}\n";

        let tmp_dir = tempfile::tempdir().unwrap();
        let file_path = tmp_dir.path().join("test_dedup.rs");
        std::fs::write(&file_path, source).unwrap();
        let path_str = file_path.to_str().unwrap();

        CodeAstExtractor::extract_incremental(path_str, graph, &store).unwrap();
        CodeAstExtractor::extract_incremental(path_str, graph, &store).unwrap();
        CodeAstExtractor::extract_incremental(path_str, graph, &store).unwrap();

        let count_sparql = format!(
            "SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE {{ GRAPH <{}> {{ ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://agent-os.org/ontology/code/Function> . }} }}",
            graph
        );
        let results = store.query_sparql(&count_sparql, None).unwrap();
        let count = results
            .first()
            .and_then(|r| r.get("?count"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        assert_eq!(
            count, 1,
            "repeated incremental extraction should not produce duplicate entities, actual: {}",
            count
        );
    }
}
