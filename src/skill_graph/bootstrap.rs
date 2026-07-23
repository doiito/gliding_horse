use crate::core::CoreError;
use crate::memory::l0_store::L0Store;
use crate::memory::l2_blackboard::Blackboard;
use crate::skill_graph::types::*;
use crate::CoreConfig;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnRequest {
    pub task_iri: String,
    pub task_description: String,
    pub execution_steps: Vec<ExecutionStep>,
    pub outcome: ExecutionOutcome,
    pub agent_id: String,
    pub quality_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub step_id: String,
    pub action: String,
    pub tool_used: Option<String>,
    pub input: Option<serde_json::Value>,
    pub output: Option<serde_json::Value>,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionOutcome {
    pub success: bool,
    pub result_summary: String,
    pub tokens_used: u32,
    pub duration_ms: u64,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceRequest {
    pub source_skill_iri: String,
    pub context: String,
    pub focus_areas: Vec<String>,
    pub agent_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    pub skill_iri: String,
    pub skill_name: String,
    pub operation: BootstrapOperation,
    pub quality_score: f32,
    pub created_at: DateTime<Utc>,
    pub storage_tier: StorageTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapOperation {
    Learn,
    Reduce,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    pub min_quality_score: f32,
    pub min_success_rate: f32,
    pub max_skill_complexity: u32,
    pub enable_auto_reduce: bool,
    pub reduce_threshold_uses: u32,
    pub storage_tier_default: StorageTier,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            min_quality_score: 0.7,
            min_success_rate: 0.8,
            max_skill_complexity: 10,
            enable_auto_reduce: true,
            reduce_threshold_uses: 100,
            storage_tier_default: StorageTier::L0Permanent,
        }
    }
}

pub struct BootstrapEngine {
    l0_store: Arc<RwLock<L0Store>>,
    l2_blackboard: Arc<RwLock<Blackboard>>,
    core_config: CoreConfig,
    config: BootstrapConfig,
    bootstrap_meta: RwLock<HashMap<String, SkillBootstrapMeta>>,
}

impl BootstrapEngine {
    pub fn new(
        l0_store: Arc<RwLock<L0Store>>,
        l2_blackboard: Arc<RwLock<Blackboard>>,
        core_config: CoreConfig,
        config: BootstrapConfig,
    ) -> Self {
        Self {
            l0_store,
            l2_blackboard,
            core_config,
            config,
            bootstrap_meta: RwLock::new(HashMap::new()),
        }
    }

    pub async fn learn_from_task(
        &self,
        request: LearnRequest,
    ) -> Result<BootstrapResult, CoreError> {
        if request.quality_score < self.config.min_quality_score {
            return Err(CoreError::ValidationFailed {
                message: format!(
                    "Quality score {} below minimum {}",
                    request.quality_score, self.config.min_quality_score
                ),
            });
        }

        if !request.outcome.success {
            return Err(CoreError::ValidationFailed {
                message: "Cannot learn from failed task execution".to_string(),
            });
        }

        if request.execution_steps.len() > self.config.max_skill_complexity as usize {
            return Err(CoreError::ValidationFailed {
                message: format!(
                    "Execution has {} steps, exceeding maximum skill complexity {}",
                    request.execution_steps.len(),
                    self.config.max_skill_complexity,
                ),
            });
        }

        let skill_iri = self.generate_skill_iri(&request.task_iri);
        let skill_name = self.extract_skill_name(&request.task_description);

        let w2h = self.build_5w2h_from_execution(&request);
        let content = self.build_content_from_steps(&request.execution_steps);

        let security_info = SkillSecurityInfo::new(SkillSource::BootstrapLearn)
            .with_trust_level(TrustLevel::from_success_rate(request.quality_score));

        let mut skill = SkillGraphNode::new(&skill_iri, &skill_name, &request.task_description)
            .with_node_type(SkillNodeType::Bootstrap)
            .with_5w2h(w2h)
            .with_content(content)
            .with_security_info(security_info)
            .with_storage_tier(self.config.storage_tier_default);

        skill.tags = self.extract_tags(&request);
        skill.graph_meta.avg_token_consumption = request.outcome.tokens_used;

        let bootstrap_source =
            BootstrapSource::new(BootstrapSourceType::TaskExecution, &request.agent_id)
                .with_task(&request.task_iri)
                .with_quality_score(request.quality_score);

        self.store_skill(&skill, StorageTier::L0Permanent).await?;

        {
            let mut meta = self.bootstrap_meta.write().await;
            let entry = meta
                .entry(skill_iri.clone())
                .or_insert_with(SkillBootstrapMeta::new);
            entry.record_learn(bootstrap_source);
        }

        self.index_to_l2(&skill).await?;

        Ok(BootstrapResult {
            skill_iri,
            skill_name,
            operation: BootstrapOperation::Learn,
            quality_score: request.quality_score,
            created_at: Utc::now(),
            storage_tier: StorageTier::L0Permanent,
        })
    }

    pub async fn reduce_skill(&self, request: ReduceRequest) -> Result<BootstrapResult, CoreError> {
        let source_skill = self.retrieve_skill(&request.source_skill_iri).await?;

        let reduced_iri = format!("{}:reduced", request.source_skill_iri);
        let reduced_name = format!("{} (Reduced)", source_skill.name);

        let reduced_content = self.reduce_content(
            &source_skill.content,
            &request.context,
            &request.focus_areas,
        );

        let reduced_w2h = self.reduce_5w2h(&source_skill.w2h, &request.context);

        let security_info = SkillSecurityInfo::new(SkillSource::BootstrapReduce)
            .with_trust_level(source_skill.get_trust_level());

        let mut reduced_skill =
            SkillGraphNode::new(&reduced_iri, &reduced_name, &source_skill.description)
                .with_node_type(SkillNodeType::Bootstrap)
                .with_5w2h(reduced_w2h)
                .with_security_info(security_info)
                .with_storage_tier(StorageTier::L2Blackboard);

        reduced_skill.content = reduced_content;
        reduced_skill.tags = self.filter_tags(&source_skill.tags, &request.focus_areas);

        {
            let mut meta = self.bootstrap_meta.write().await;
            let entry = meta
                .entry(reduced_iri.clone())
                .or_insert_with(SkillBootstrapMeta::new);
            entry.set_parent(&request.source_skill_iri);
            entry.record_reduce(&reduced_iri);
        }

        {
            let mut parent_meta = self.bootstrap_meta.write().await;
            if let Some(parent) = parent_meta.get_mut(&request.source_skill_iri) {
                parent.record_reduce(&reduced_iri);
            }
        }

        self.store_skill(&reduced_skill, StorageTier::L2Blackboard)
            .await?;

        let quality_score = self.calculate_reduce_quality(&source_skill, &reduced_skill);

        Ok(BootstrapResult {
            skill_iri: reduced_iri,
            skill_name: reduced_name,
            operation: BootstrapOperation::Reduce,
            quality_score,
            created_at: Utc::now(),
            storage_tier: StorageTier::L2Blackboard,
        })
    }

    pub async fn update_skill_quality(
        &self,
        skill_iri: &str,
        success: bool,
        tokens_used: u32,
    ) -> Result<(), CoreError> {
        let mut skill = self.retrieve_skill(skill_iri).await?;

        skill.graph_meta.record_usage(success);
        skill.graph_meta.avg_token_consumption =
            (skill.graph_meta.avg_token_consumption + tokens_used) / 2;

        if let Some(ref mut security_info) = skill.security_info {
            let new_trust = TrustLevel::from_success_rate(skill.graph_meta.success_rate);
            if new_trust != security_info.trust_level {
                security_info.trust_level = new_trust;
                security_info.update_risk_score();
            }
        }

        self.store_skill(&skill, skill.storage_tier).await?;

        Ok(())
    }

    pub async fn get_bootstrap_meta(&self, skill_iri: &str) -> Option<SkillBootstrapMeta> {
        let meta = self.bootstrap_meta.read().await;
        meta.get(skill_iri).cloned()
    }

    pub async fn list_bootstrap_skills(&self) -> Result<Vec<String>, CoreError> {
        let meta = self.bootstrap_meta.read().await;
        Ok(meta.keys().cloned().collect())
    }

    pub async fn check_auto_reduce(
        &self,
        skill_iri: &str,
    ) -> Result<Option<ReduceRequest>, CoreError> {
        if !self.config.enable_auto_reduce {
            return Ok(None);
        }

        // `learn_count` measures how many bootstrap requests created a node,
        // not how often this particular skill was exercised. Reduction is a
        // usage-driven suggestion, so its evidence must come from the
        // persisted SkillGraphMeta updated by `update_skill_quality`.
        let skill = self.retrieve_skill(skill_iri).await?;
        if skill.graph_meta.usage_count < self.config.reduce_threshold_uses
            || skill.graph_meta.success_rate < self.config.min_success_rate
        {
            return Ok(None);
        }

        Ok(Some(ReduceRequest {
            source_skill_iri: skill_iri.to_string(),
            context: "Auto-reduce based on observed successful usage patterns".to_string(),
            focus_areas: skill.tags.clone(),
            agent_id: "system:auto-reduce".to_string(),
        }))
    }

    fn generate_skill_iri(&self, task_iri: &str) -> String {
        format!(
            "iri://skills/bootstrap/{}",
            task_iri.replace("iri://task/", "").replace("/", "-")
        )
    }

    fn extract_skill_name(&self, description: &str) -> String {
        let words: Vec<&str> = description.split_whitespace().take(5).collect();
        words.join(" ")
    }

    fn build_5w2h_from_execution(&self, request: &LearnRequest) -> Skill5W2H {
        Skill5W2H {
            what: request.task_description.clone(),
            why: "Learned from successful task execution".to_string(),
            who: SkillRole {
                role_name: "Bootstrap Skill".to_string(),
                required_agent_role: Some("DA".to_string()),
            },
            when: SkillTrigger {
                applicable_phases: vec!["Do".to_string()],
                trigger_condition: None,
                deadline_constraint: None,
            },
            where_: SkillContext {
                target_stack: vec![],
                repo_pattern: None,
            },
            how: SkillApproach {
                approach: "Execute learned steps".to_string(),
                plan_iri: Some(request.task_iri.clone()),
            },
            how_much: SkillCost {
                avg_token_cost: request.outcome.tokens_used,
                avg_duration_seconds: (request.outcome.duration_ms / 1000) as u32,
                max_sub_agents: 1,
            },
        }
    }

    fn build_content_from_steps(&self, steps: &[ExecutionStep]) -> SkillContent {
        let skill_steps: Vec<SkillStep> = steps
            .iter()
            .enumerate()
            .map(|(i, step)| {
                let mut skill_step = SkillStep::new(&format!("step-{}", i), i as u32, &step.action);
                if let Some(ref tool) = step.tool_used {
                    skill_step = skill_step.with_reference(&format!("tool:{}", tool));
                }
                skill_step
            })
            .collect();

        SkillContent {
            summary: format!("Learned skill with {} steps", steps.len()),
            steps: skill_steps,
            validation: Some(SkillValidation {
                method: "Execute all steps in sequence".to_string(),
                success_condition: "All steps complete successfully".to_string(),
            }),
        }
    }

    fn extract_tags(&self, request: &LearnRequest) -> Vec<String> {
        let mut tags = vec!["bootstrap".to_string(), "learned".to_string()];

        for step in &request.execution_steps {
            if let Some(ref tool) = step.tool_used {
                tags.push(format!("tool:{}", tool));
            }
        }

        tags.sort();
        tags.dedup();
        tags
    }

    fn reduce_content(
        &self,
        content: &Option<SkillContent>,
        context: &str,
        focus_areas: &[String],
    ) -> Option<SkillContent> {
        content.as_ref().map(|c| {
            let filtered_steps: Vec<SkillStep> = c
                .steps
                .iter()
                .filter(|step| {
                    focus_areas
                        .iter()
                        .any(|focus| step.action.to_lowercase().contains(&focus.to_lowercase()))
                })
                .cloned()
                .collect();

            SkillContent {
                summary: format!("Reduced: {} (Context: {})", c.summary, context),
                steps: filtered_steps,
                validation: c.validation.clone(),
            }
        })
    }

    fn reduce_5w2h(&self, w2h: &Skill5W2H, context: &str) -> Skill5W2H {
        Skill5W2H {
            what: format!("{} (Reduced)", w2h.what),
            why: format!("{} - Context: {}", w2h.why, context),
            who: w2h.who.clone(),
            when: w2h.when.clone(),
            where_: w2h.where_.clone(),
            how: SkillApproach {
                approach: format!("Reduced approach: {}", w2h.how.approach),
                plan_iri: w2h.how.plan_iri.clone(),
            },
            how_much: SkillCost {
                avg_token_cost: (w2h.how_much.avg_token_cost as f32 * 0.7) as u32,
                avg_duration_seconds: (w2h.how_much.avg_duration_seconds as f32 * 0.7) as u32,
                max_sub_agents: w2h.how_much.max_sub_agents,
            },
        }
    }

    fn filter_tags(&self, tags: &[String], focus_areas: &[String]) -> Vec<String> {
        let mut filtered: Vec<String> = tags
            .iter()
            .filter(|tag| {
                focus_areas
                    .iter()
                    .any(|focus| tag.to_lowercase().contains(&focus.to_lowercase()))
            })
            .cloned()
            .collect();
        filtered.push("reduced".to_string());
        filtered
    }

    fn calculate_reduce_quality(&self, original: &SkillGraphNode, reduced: &SkillGraphNode) -> f32 {
        let original_steps = original
            .content
            .as_ref()
            .map(|c| c.steps.len())
            .unwrap_or(0);
        let reduced_steps = reduced.content.as_ref().map(|c| c.steps.len()).unwrap_or(0);

        if original_steps == 0 {
            return 0.5;
        }

        let efficiency_gain = 1.0 - (reduced_steps as f32 / original_steps as f32);
        let trust_factor = reduced.get_trust_level() as u8 as f32 / 4.0;

        (0.5 + efficiency_gain * 0.3 + trust_factor * 0.2).min(1.0)
    }

    async fn store_skill(
        &self,
        skill: &SkillGraphNode,
        tier: StorageTier,
    ) -> Result<(), CoreError> {
        match tier {
            StorageTier::L0Permanent => {
                let store = self.l0_store.read().await;
                let json =
                    serde_json::to_string(skill).map_err(|e| CoreError::ValidationFailed {
                        message: format!("JSON error: {}", e),
                    })?;
                store.store(&skill.skill_iri, &json)?;
            }
            StorageTier::L2Blackboard => {
                let blackboard = self.l2_blackboard.read().await;
                let json_ld = skill.to_json_ld();
                let json_str =
                    serde_json::to_string(&json_ld).map_err(|e| CoreError::ValidationFailed {
                        message: format!("JSON error: {}", e),
                    })?;
                blackboard.write_node(&skill.skill_iri, &json_str, &self.core_config)?;
            }
            _ => {}
        }
        Ok(())
    }

    async fn retrieve_skill(&self, skill_iri: &str) -> Result<SkillGraphNode, CoreError> {
        {
            let store = self.l0_store.read().await;
            if let Some(entry) = store.retrieve(skill_iri)? {
                let skill: SkillGraphNode = serde_json::from_str(&entry.content).map_err(|e| {
                    CoreError::ValidationFailed {
                        message: format!("JSON parse error: {}", e),
                    }
                })?;
                return Ok(skill);
            }
        }

        {
            let blackboard = self.l2_blackboard.read().await;
            if let Some(node) = blackboard.read_node(skill_iri)? {
                let skill = self.json_ld_to_skill(&node.json_ld)?;
                return Ok(skill);
            }
        }

        Err(CoreError::SkillNotFound {
            iri: skill_iri.to_string(),
        })
    }

    async fn index_to_l2(&self, skill: &SkillGraphNode) -> Result<(), CoreError> {
        let blackboard = self.l2_blackboard.read().await;
        let json_ld = skill.to_json_ld();
        let json_str =
            serde_json::to_string(&json_ld).map_err(|e| CoreError::ValidationFailed {
                message: format!("JSON error: {}", e),
            })?;
        blackboard.write_node(&skill.skill_iri, &json_str, &self.core_config)?;
        Ok(())
    }

    fn json_ld_to_skill(&self, json_ld: &str) -> Result<SkillGraphNode, CoreError> {
        let parsed: serde_json::Value =
            serde_json::from_str(json_ld).map_err(|e| CoreError::ValidationFailed {
                message: format!("JSON parse error: {}", e),
            })?;

        let skill_iri = parsed
            .get("@id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CoreError::ValidationFailed {
                message: "Missing @id".to_string(),
            })?
            .to_string();

        let name = parsed
            .get("schema:name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        let description = parsed
            .get("schema:description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(SkillGraphNode::new(&skill_iri, &name, &description))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine(config: BootstrapConfig) -> (BootstrapEngine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(RwLock::new(
            L0Store::new(dir.path().to_str().unwrap()).unwrap(),
        ));
        let l2 = Arc::new(RwLock::new(Blackboard::new().unwrap()));
        (
            BootstrapEngine::new(l0, l2, CoreConfig::default(), config),
            dir,
        )
    }

    fn create_test_learn_request() -> LearnRequest {
        LearnRequest {
            task_iri: "iri://task/test-001".to_string(),
            task_description: "Implement JWT authentication".to_string(),
            execution_steps: vec![
                ExecutionStep {
                    step_id: "step-1".to_string(),
                    action: "Install jsonwebtoken crate".to_string(),
                    tool_used: Some("bash".to_string()),
                    input: None,
                    output: None,
                    success: true,
                },
                ExecutionStep {
                    step_id: "step-2".to_string(),
                    action: "Create JWT middleware".to_string(),
                    tool_used: Some("file_write".to_string()),
                    input: None,
                    output: None,
                    success: true,
                },
            ],
            outcome: ExecutionOutcome {
                success: true,
                result_summary: "JWT authentication implemented".to_string(),
                tokens_used: 1500,
                duration_ms: 5000,
                error_message: None,
            },
            agent_id: "agent:da/inst-001".to_string(),
            quality_score: 0.85,
        }
    }

    #[test]
    fn test_bootstrap_config_default() {
        let config = BootstrapConfig::default();
        assert_eq!(config.min_quality_score, 0.7);
        assert_eq!(config.min_success_rate, 0.8);
        assert!(config.enable_auto_reduce);
    }

    #[test]
    fn test_learn_request_validation() {
        let request = create_test_learn_request();
        assert!(request.outcome.success);
        assert!(request.quality_score >= 0.7);
        assert_eq!(request.execution_steps.len(), 2);
    }

    #[test]
    fn test_execution_step() {
        let step = ExecutionStep {
            step_id: "test".to_string(),
            action: "Test action".to_string(),
            tool_used: Some("test_tool".to_string()),
            input: Some(serde_json::json!({"key": "value"})),
            output: None,
            success: true,
        };

        assert!(step.success);
        assert!(step.tool_used.is_some());
    }

    #[test]
    fn test_bootstrap_result() {
        let result = BootstrapResult {
            skill_iri: "iri://skills/test".to_string(),
            skill_name: "Test Skill".to_string(),
            operation: BootstrapOperation::Learn,
            quality_score: 0.9,
            created_at: Utc::now(),
            storage_tier: StorageTier::L0Permanent,
        };

        assert_eq!(result.operation, BootstrapOperation::Learn);
        assert!(result.quality_score > 0.8);
    }

    #[test]
    fn test_reduce_request() {
        let request = ReduceRequest {
            source_skill_iri: "iri://skills/source".to_string(),
            context: "Focus on authentication".to_string(),
            focus_areas: vec!["jwt".to_string(), "auth".to_string()],
            agent_id: "agent:sa/inst-001".to_string(),
        };

        assert_eq!(request.focus_areas.len(), 2);
    }

    #[tokio::test]
    async fn auto_reduce_requires_observed_usage_and_success_rate() {
        let config = BootstrapConfig {
            reduce_threshold_uses: 2,
            min_success_rate: 0.8,
            ..BootstrapConfig::default()
        };
        let (engine, _dir) = test_engine(config);
        let learned = engine
            .learn_from_task(create_test_learn_request())
            .await
            .unwrap();

        // One bootstrap creation is not two uses of the created skill.
        assert!(engine
            .check_auto_reduce(&learned.skill_iri)
            .await
            .unwrap()
            .is_none());

        engine
            .update_skill_quality(&learned.skill_iri, true, 100)
            .await
            .unwrap();
        engine
            .update_skill_quality(&learned.skill_iri, true, 100)
            .await
            .unwrap();
        assert!(engine
            .check_auto_reduce(&learned.skill_iri)
            .await
            .unwrap()
            .is_some());

        // A later failure lowers the empirical success rate below the policy
        // threshold and suppresses the suggestion again.
        engine
            .update_skill_quality(&learned.skill_iri, false, 100)
            .await
            .unwrap();
        assert!(engine
            .check_auto_reduce(&learned.skill_iri)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn learn_rejects_overly_complex_skill() {
        let (engine, _dir) = test_engine(BootstrapConfig {
            max_skill_complexity: 1,
            ..BootstrapConfig::default()
        });
        let error = engine
            .learn_from_task(create_test_learn_request())
            .await
            .unwrap_err();
        assert!(matches!(error, CoreError::ValidationFailed { .. }));
    }
}
