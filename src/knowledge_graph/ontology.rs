use super::types::{OntologyTerm, OntologyTermType};
use std::sync::OnceLock;

static BUILT_IN_ONTOLOGY: OnceLock<Vec<OntologyTerm>> = OnceLock::new();

fn core(s: &str) -> String {
    format!("https://agentos.ontology/core/{}", s)
}
fn eng(s: &str) -> String {
    format!("https://agentos.ontology/eng/{}", s)
}
fn code(s: &str) -> String {
    format!("https://agentos.ontology/code/{}", s)
}
fn biz(s: &str) -> String {
    format!("https://agentos.ontology/biz/{}", s)
}

pub struct OntologyManager {
    domain_terms: Vec<OntologyTerm>,
}

impl OntologyManager {
    pub fn new() -> Self {
        Self {
            domain_terms: Vec::new(),
        }
    }

    pub fn add_domain_term(&mut self, term: OntologyTerm) {
        self.domain_terms.push(term);
    }

    pub fn load_domains_json(&mut self, dir: &std::path::Path) -> Result<usize, crate::CoreError> {
        let mut count = 0;
        if !dir.exists() {
            return Ok(0);
        }
        for entry in std::fs::read_dir(dir).map_err(|e| crate::CoreError::Internal {
            message: format!("failed to read domain directory: {}", e),
        })? {
            let path = entry
                .map_err(|e| crate::CoreError::Internal {
                    message: format!("failed to read directory entry: {}", e),
                })?
                .path();
            if path.extension().map_or(false, |e| e == "json") {
                let content =
                    std::fs::read_to_string(&path).map_err(|e| crate::CoreError::Internal {
                        message: format!("failed to read domain file {}: {}", path.display(), e),
                    })?;
                let domain: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
                    crate::CoreError::InvalidJsonLd {
                        message: format!("domain file {} JSON parse failed: {}", path.display(), e),
                    }
                })?;
                let ns = domain["namespace"].as_str().unwrap_or("");
                if let Some(terms) = domain["terms"].as_array() {
                    for t in terms {
                        let iri = format!("{}{}", ns, t["id"].as_str().unwrap_or(""));
                        let label = t["label"].as_str().unwrap_or("").to_string();
                        let desc = t["description"].as_str().unwrap_or("").to_string();
                        let tt = match t["type"].as_str().unwrap_or("Class") {
                            "Relation" => OntologyTermType::Relation,
                            "Property" => OntologyTermType::Property,
                            _ => OntologyTermType::Class,
                        };
                        self.domain_terms.push(OntologyTerm {
                            iri,
                            label,
                            description: desc,
                            term_type: tt,
                        });
                        count += 1;
                    }
                }
            }
        }
        Ok(count)
    }

    pub fn get_vocabulary(&self, domain: Option<&str>) -> Vec<OntologyTerm> {
        let mut all = Self::built_in_terms().clone();
        all.extend(self.domain_terms.clone());
        match domain {
            Some(d) => all.into_iter().filter(|t| t.iri.contains(d)).collect(),
            None => all,
        }
    }

    pub fn format_vocabulary_for_prompt(&self, terms: &[OntologyTerm]) -> String {
        let mut classes: Vec<&OntologyTerm> = Vec::new();
        let mut properties: Vec<&OntologyTerm> = Vec::new();
        let mut relations: Vec<&OntologyTerm> = Vec::new();
        for t in terms {
            match t.term_type {
                OntologyTermType::Class => classes.push(t),
                OntologyTermType::Property => properties.push(t),
                OntologyTermType::Relation => relations.push(t),
            }
        }
        let mut result = String::new();
        if !classes.is_empty() {
            result.push_str("## Available entity types\n");
            for c in &classes {
                result.push_str(&format!(
                    "- IRI: {} | name: {} | {}\n",
                    c.iri, c.label, c.description
                ));
            }
        }
        if !properties.is_empty() {
            result.push_str("## Available properties\n");
            for p in &properties {
                result.push_str(&format!(
                    "- IRI: {} | name: {} | {}\n",
                    p.iri, p.label, p.description
                ));
            }
        }
        if !relations.is_empty() {
            result.push_str("## Available relations\n");
            for r in &relations {
                result.push_str(&format!(
                    "- IRI: {} | name: {} | {}\n",
                    r.iri, r.label, r.description
                ));
            }
        }
        result
    }

    fn built_in_terms() -> &'static Vec<OntologyTerm> {
        BUILT_IN_ONTOLOGY.get_or_init(|| {
            let mut terms = vec![
                // ═══════ Core: Agent OS engine (10 classes) ═══════
                OntologyTerm::class(core("Agent"), "Agent", "PA/DA/CA/AA"),
                OntologyTerm::class(
                    core("Task"),
                    "Task",
                    "Execution unit triggered by user input",
                ),
                OntologyTerm::class(core("Plan"), "Plan", "Step sequence generated by PA"),
                OntologyTerm::class(core("Action"), "Action", "Tool call record"),
                OntologyTerm::class(core("File"), "File", "File entity (path/SHA256)"),
                OntologyTerm::class(core("Decision"), "Decision", "CA/AA audit conclusion"),
                OntologyTerm::class(core("Session"), "Session", "L1 session summary"),
                OntologyTerm::class(core("Error"), "Error", "Execution failure details"),
                OntologyTerm::class(core("Metric"), "Metric", "token/duration/rounds"),
                OntologyTerm::class(core("Goal"), "Goal", "Expected outcome"),
                // ═══════ Engineering: task/project (14 classes) ═══════
                OntologyTerm::class(
                    eng("Requirement"),
                    "Requirement",
                    "Formal requirement specification",
                ),
                OntologyTerm::class(
                    eng("Deliverable"),
                    "Deliverable",
                    "Output with acceptance criteria",
                ),
                OntologyTerm::class(eng("Issue"), "Issue", "Obstacle/defect during execution"),
                OntologyTerm::class(eng("Risk"), "Risk", "Potential negative outcome"),
                OntologyTerm::class(eng("Resource"), "Resource", "Consumable item (token/quota)"),
                OntologyTerm::class(eng("Milestone"), "Milestone", "Progress checkpoint"),
                OntologyTerm::class(eng("Change"), "Change", "Modification to existing output"),
                OntologyTerm::class(
                    eng("Constraint"),
                    "Constraint",
                    "Conditions that restrict execution",
                ),
                OntologyTerm::class(eng("Review"), "Review", "Formal review of output"),
                OntologyTerm::class(eng("Test"), "Test", "Correctness verification"),
                OntologyTerm::class(eng("Pattern"), "Pattern", "Reusable experience/lesson"),
                OntologyTerm::class(
                    eng("Artifact"),
                    "Artifact",
                    "General output (parent of File)",
                ),
                OntologyTerm::class(eng("Project"), "Project", "Task container"),
                OntologyTerm::class(eng("Role"), "Role", "Plan/Do/Check/Act"),
                // ═══════ Code: code understanding (10 classes) ═══════
                OntologyTerm::class(code("Function"), "Function", ""),
                OntologyTerm::class(code("Struct"), "Struct", ""),
                OntologyTerm::class(code("Enum"), "Enum", ""),
                OntologyTerm::class(code("Trait"), "Trait", ""),
                OntologyTerm::class(code("Class"), "Class", ""),
                OntologyTerm::class(code("Interface"), "Interface", ""),
                OntologyTerm::class(code("Impl"), "Impl block", ""),
                OntologyTerm::class(code("Module"), "Module", ""),
                OntologyTerm::class(code("Calls"), "Call relationship", ""),
                OntologyTerm::class(code("DependsOn"), "Dependency relationship", ""),
                // ═══════ Business: domain compatibility (7 classes) ═══════
                OntologyTerm::class(biz("Person"), "Person", ""),
                OntologyTerm::class(biz("Organization"), "Organization", ""),
                OntologyTerm::class(biz("Product"), "Product", ""),
                OntologyTerm::class(biz("Project"), "Business project", ""),
                OntologyTerm::class(core("Concept"), "Abstract concept", ""),
                OntologyTerm::class(core("Event"), "Event", ""),
                OntologyTerm::class(core("Knowledge"), "Knowledge fragment", ""),
                // ═══════ Relations: 20 core + engineering relations ═══════
                OntologyTerm::relation(core("generatedBy"), "generated by", "Action→Agent"),
                OntologyTerm::relation(core("hasSubTask"), "has subtask", "Task decomposition"),
                OntologyTerm::relation(core("produces"), "produces", "→Artifact/File"),
                OntologyTerm::relation(core("refersTo"), "refers to", "→KnowledgeRef"),
                OntologyTerm::relation(core("followsPlan"), "follows plan", "Action→Plan"),
                OntologyTerm::relation(core("auditedBy"), "audited by", "Decision→Agent"),
                OntologyTerm::relation(core("dependsOn"), "depends on", "Cross-task"),
                OntologyTerm::relation(core("assignedTo"), "assigned to", "Task→Agent"),
                OntologyTerm::relation(eng("addresses"), "addresses", "Action→Requirement"),
                OntologyTerm::relation(eng("resolves"), "resolves", "Action→Issue"),
                OntologyTerm::relation(eng("blocks"), "blocks", "Issue→Task"),
                OntologyTerm::relation(eng("validates"), "validates", "Review→Deliverable"),
                OntologyTerm::relation(eng("constrains"), "constrains", "Constraint→Task"),
                OntologyTerm::relation(eng("risks"), "risks", "Risk→Task"),
                OntologyTerm::relation(eng("consumes"), "consumes", "Action→Resource"),
                OntologyTerm::relation(eng("marks"), "marks milestone", "Milestone→Task"),
                OntologyTerm::relation(eng("captures"), "captures", "→Knowledge"),
                OntologyTerm::relation(eng("generalizes"), "generalizes", "Pattern→Knowledge"),
                OntologyTerm::relation(eng("prioritizes"), "prioritizes", "Plan→Requirement"),
                OntologyTerm::relation(eng("reports"), "reports", "Metric→Goal"),
                // Legacy compatibility
                OntologyTerm::relation(biz("worksFor"), "works for", "Person→Org"),
                OntologyTerm::relation(biz("manages"), "manages", "Person→Project"),
                OntologyTerm::relation(core("hasSkill"), "has skill", ""),
                OntologyTerm::relation(core("applicableIn"), "applicable in", ""),
                // ═══════ Properties (10) ═══════
                OntologyTerm::property(core("hasStatus"), "status"),
                OntologyTerm::property(core("filePath"), "file path"),
                OntologyTerm::property(core("fileHash"), "file SHA256"),
                OntologyTerm::property(core("tokenCost"), "token cost"),
                OntologyTerm::property(core("duration"), "duration (seconds)"),
                OntologyTerm::property(core("createdAt"), "created at"),
                OntologyTerm::property(core("completedAt"), "completed at"),
                OntologyTerm::property(eng("priority"), "priority"),
                OntologyTerm::property(eng("severity"), "severity"),
                OntologyTerm::property(core("confidence"), "confidence"),
            ];
            terms.sort_by_key(|t| t.term_type.clone() as u8);
            terms
        })
    }
}
