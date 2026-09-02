use std::collections::{HashMap, VecDeque};
use std::io;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Task queue error
#[derive(Debug, Error)]
pub enum QueueError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Serialization error: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("Queue error: {0}")]
    Queue(String),

    #[error("Timeout")]
    Timeout,

    #[error("Queue closed")]
    Closed,
}

/// Task context data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContextData {
    /// Stage ID
    pub stage_id: String,
    /// Stage type
    pub stage_type: String,
    /// Project ID
    pub project_id: String,
    /// Project directory
    pub project_dir: String,
    /// User requirement
    pub user_requirement: String,
    /// Previous stage outputs
    pub prev_outputs: HashMap<String, serde_json::Value>,
    /// LLM configuration
    pub llm_config: LlmConfig,
}

/// LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Legacy in-memory field. Never serialized to the durable queue.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub api_key: String,
    /// Reference resolved by the worker, e.g. `env:DEEPSEEK_API_KEY`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_ref: Option<String>,
    pub base_url: String,
    pub model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            credential_ref: None,
            base_url: String::new(),
            model: String::new(),
        }
    }
}

/// Agent OS task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOsTask {
    /// Task ID
    pub task_id: String,
    /// Task IRI
    pub task_iri: String,
    /// Prompt
    pub prompt: String,
    /// Task context
    pub context: TaskContextData,
    /// Creation timestamp
    pub created_at: u64,
}

impl AgentOsTask {
    pub fn new(task_iri: String, prompt: String, context: TaskContextData) -> Self {
        Self {
            task_id: uuid::Uuid::new_v4().to_string(),
            task_iri,
            prompt,
            context,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Agent OS execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOsResult {
    /// Corresponding task ID
    pub task_id: String,
    /// Execution status
    pub status: String,
    /// Summary
    pub summary: String,
    /// Output data
    pub output: Option<serde_json::Value>,
    /// JSON-LD output
    pub jsonld_output: Option<serde_json::Value>,
    /// Artifacts list
    pub artifacts: Vec<String>,
    /// Errors list
    pub errors: Vec<String>,
    /// Execution duration (milliseconds)
    pub duration_ms: u64,
    /// Tool call count
    pub tool_call_count: u32,
    /// Turn count
    pub turn_count: u32,
}

impl AgentOsResult {
    pub fn success(task_id: String, summary: String) -> Self {
        Self {
            task_id,
            status: "success".to_string(),
            summary,
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: Vec::new(),
            duration_ms: 0,
            tool_call_count: 0,
            turn_count: 0,
        }
    }

    pub fn failure(task_id: String, error: String) -> Self {
        Self {
            task_id,
            status: "failed".to_string(),
            summary: error.clone(),
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: vec![error],
            duration_ms: 0,
            tool_call_count: 0,
            turn_count: 0,
        }
    }
}

impl From<crate::core::agent_runner::TaskResult> for AgentOsResult {
    fn from(result: crate::core::agent_runner::TaskResult) -> Self {
        Self {
            task_id: result
                .task_iri
                .split('/')
                .last()
                .unwrap_or(&result.task_iri)
                .to_string(),
            status: result.status,
            summary: result.summary,
            output: result.output,
            jsonld_output: result.jsonld_output,
            artifacts: result
                .artifacts
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            errors: result.errors,
            duration_ms: 0,
            tool_call_count: result.tool_call_count,
            turn_count: result.turn_count,
        }
    }
}

// ============================================================
// yaque queue implementation
// ============================================================

use yaque::queue::{Receiver, Sender};

/// Activity-side task queue
pub struct TaskQueue {
    base_path: String,
    last_task_id: Option<String>,
}

impl TaskQueue {
    /// Create a new task queue (full version, holds both sender and receiver)
    pub fn new(base_path: &str) -> Result<Self, QueueError> {
        Ok(Self {
            base_path: base_path.to_string(),
            last_task_id: None,
        })
    }

    /// Create a client queue (only sends tasks, receives results)
    pub fn new_client(base_path: &str) -> Result<Self, QueueError> {
        Ok(Self {
            base_path: base_path.to_string(),
            last_task_id: None,
        })
    }

    /// Send task
    pub async fn send_task(&mut self, task: &AgentOsTask) -> Result<(), QueueError> {
        let data = serde_json::to_vec(task)?;
        let task_path = format!("{}/tasks", self.base_path);
        let mut task_sender = None;
        for attempt in 0..100 {
            match Sender::open(&task_path) {
                Ok(sender) => {
                    task_sender = Some(sender);
                    break;
                }
                Err(error) if attempt < 99 => {
                    tracing::debug!(attempt, error = %error, "Task queue sender busy; retrying");
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(error) => return Err(QueueError::Io(error)),
            }
        }
        tracing::info!(task_id = %task.task_id, task_iri = %task.task_iri, data_len = data.len(), "Sending task to queue");
        task_sender
            .ok_or_else(|| QueueError::Queue("Unable to acquire task sender".to_string()))?
            .send(data)
            .await?;
        self.last_task_id = Some(task.task_id.clone());
        tracing::info!(task_id = %task.task_id, "Task sent successfully");
        Ok(())
    }

    /// Receive result for a specific task (with timeout, matched by task_id)
    pub async fn recv_result_for_task(
        &mut self,
        task_id: &str,
        timeout: Duration,
    ) -> Result<Option<AgentOsResult>, QueueError> {
        tracing::info!(expected_task_id = %task_id, "Starting to wait for result");

        let result_path = format!("{}/results/by-task/{}", self.base_path, task_id);
        let mut receiver = Receiver::open(&result_path)?;
        let guard = match tokio::time::timeout(timeout, receiver.recv()).await {
            Ok(result) => result.map_err(|error| QueueError::Queue(error.to_string()))?,
            Err(_) => return Ok(None),
        };
        let result: AgentOsResult = serde_json::from_slice(&*guard)?;
        guard.commit()?;
        if result.task_id != task_id {
            return Err(QueueError::Queue(format!(
                "Result queue routing mismatch: expected {}, got {}",
                task_id, result.task_id
            )));
        }
        Ok(Some(result))
    }

    /// Receive result (with timeout, no task_id matching)
    pub async fn recv_result_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<AgentOsResult>, QueueError> {
        let task_id = self.last_task_id.clone().ok_or_else(|| {
            QueueError::Queue("No task has been sent by this queue client".to_string())
        })?;
        self.recv_result_for_task(&task_id, timeout).await
    }

    /// Get queue base path
    pub fn base_path(&self) -> &str {
        &self.base_path
    }
}

/// Worker-side task queue
pub struct WorkerQueue {
    task_receiver: Receiver,
    base_path: String,
    recovered_claims: VecDeque<ClaimedTask>,
}

#[derive(Debug, Clone)]
pub(crate) struct ClaimedTask {
    pub task: AgentOsTask,
    claim_path: std::path::PathBuf,
}

impl WorkerQueue {
    /// Create a new Worker queue
    pub fn new(base_path: &str) -> Result<Self, QueueError> {
        let task_path = format!("{}/tasks", base_path);
        let task_receiver = Receiver::open(&task_path)?;
        let recovered_claims = Self::load_recovered_claims(base_path)?;

        Ok(Self {
            task_receiver,
            base_path: base_path.to_string(),
            recovered_claims,
        })
    }

    fn load_recovered_claims(base_path: &str) -> Result<VecDeque<ClaimedTask>, QueueError> {
        let inflight = std::path::Path::new(base_path).join("inflight");
        std::fs::create_dir_all(&inflight)?;
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&inflight)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
            .collect();
        paths.sort();
        let mut recovered = VecDeque::new();
        for claim_path in paths {
            let data = std::fs::read(&claim_path)?;
            let task: AgentOsTask = serde_json::from_slice(&data)?;
            recovered.push_back(ClaimedTask { task, claim_path });
        }
        Ok(recovered)
    }

    pub(crate) async fn claim_next(&mut self) -> Result<ClaimedTask, QueueError> {
        if let Some(claim) = self.recovered_claims.pop_front() {
            tracing::info!(task_id = %claim.task.task_id, "Recovered inflight task claim");
            return Ok(claim);
        }
        let base_path = self.base_path.clone();
        let guard = self
            .task_receiver
            .recv()
            .await
            .map_err(|e| QueueError::Queue(e.to_string()))?;
        let task: AgentOsTask = serde_json::from_slice(&*guard)?;
        let inflight = std::path::Path::new(&base_path).join("inflight");
        tokio::fs::create_dir_all(&inflight).await?;
        let claim_path = inflight.join(format!("{}.json", task.task_id));
        let temp_path = inflight.join(format!("{}.{}.tmp", task.task_id, uuid::Uuid::new_v4()));
        let data = serde_json::to_vec(&task)?;
        let mut file = tokio::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp_path)
            .await?;
        use tokio::io::AsyncWriteExt;
        file.write_all(&data).await?;
        file.sync_all().await?;
        drop(file);
        tokio::fs::rename(&temp_path, &claim_path).await?;
        guard.commit()?;
        tracing::debug!(task_id = %task.task_id, path = %claim_path.display(), "Task durably claimed");
        Ok(ClaimedTask { task, claim_path })
    }

    pub(crate) async fn persist_result(
        base_path: &str,
        result: &AgentOsResult,
    ) -> Result<(), QueueError> {
        let data = serde_json::to_vec(result)?;
        let result_path = format!("{}/results/by-task/{}", base_path, result.task_id);
        let mut result_sender = Sender::open(&result_path)?;
        result_sender.send(data).await?;
        Ok(())
    }

    pub(crate) async fn complete_claim(claim: &ClaimedTask) -> Result<(), QueueError> {
        match tokio::fs::remove_file(&claim.claim_path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(QueueError::Io(error)),
        }
    }

    /// Execute one task with at-least-once delivery. The queue message is
    /// committed only after its routed result has been durably enqueued.
    pub async fn process_next<F, Fut>(&mut self, execute: F) -> Result<AgentOsResult, QueueError>
    where
        F: FnOnce(AgentOsTask) -> Fut,
        Fut: std::future::Future<Output = AgentOsResult>,
    {
        let claim = self.claim_next().await?;
        let result = execute(claim.task.clone()).await;
        Self::persist_result(&self.base_path, &result).await?;
        Self::complete_claim(&claim).await?;
        Ok(result)
    }

    /// Send result
    pub async fn send_result(&mut self, result: &AgentOsResult) -> Result<(), QueueError> {
        let data = serde_json::to_vec(result)?;
        let result_path = format!("{}/results/by-task/{}", self.base_path, result.task_id);
        let mut result_sender = Sender::open(&result_path)?;
        tracing::info!(task_id = %result.task_id, status = %result.status, data_len = data.len(), "Sending result to queue");
        result_sender.send(data).await?;
        tracing::info!(task_id = %result.task_id, "Result sent successfully");
        Ok(())
    }

    /// Get queue base path
    pub fn base_path(&self) -> &str {
        &self.base_path
    }
}

// ============================================================
// Unix Domain Socket implementation (alternative)
// ============================================================

#[cfg(unix)]
pub mod uds {
    use super::*;
    use futures::{SinkExt, StreamExt};
    use tokio::net::{UnixListener, UnixStream};
    use tokio_util::codec::{Framed, LengthDelimitedCodec};

    /// UDS task client
    pub struct UdsTaskClient {
        stream: Framed<UnixStream, LengthDelimitedCodec>,
    }

    impl UdsTaskClient {
        pub async fn connect(path: &str) -> Result<Self, QueueError> {
            let stream = UnixStream::connect(path).await?;
            Ok(Self {
                stream: Framed::new(stream, LengthDelimitedCodec::new()),
            })
        }

        pub async fn send(&mut self, task: &AgentOsTask) -> Result<(), QueueError> {
            let data = serde_json::to_vec(task)?;
            self.stream
                .send(data.into())
                .await
                .map_err(|e| QueueError::Queue(e.to_string()))?;
            Ok(())
        }

        pub async fn recv(&mut self) -> Result<AgentOsResult, QueueError> {
            let data = self
                .stream
                .next()
                .await
                .ok_or(QueueError::Closed)?
                .map_err(|e| QueueError::Queue(e.to_string()))?;
            let result = serde_json::from_slice(&data)?;
            Ok(result)
        }

        pub async fn recv_timeout(
            &mut self,
            timeout: Duration,
        ) -> Result<Option<AgentOsResult>, QueueError> {
            match tokio::time::timeout(timeout, self.recv()).await {
                Ok(result) => result.map(Some),
                Err(_) => Ok(None),
            }
        }
    }

    /// UDS task server
    pub struct UdsTaskServer {
        listener: UnixListener,
    }

    impl UdsTaskServer {
        pub async fn bind(path: &str) -> Result<Self, QueueError> {
            if std::path::Path::new(path).exists() {
                std::fs::remove_file(path)?;
            }
            let listener = UnixListener::bind(path)?;
            Ok(Self { listener })
        }

        pub async fn accept(&self) -> Result<UdsTaskConnection, QueueError> {
            let (stream, _) = self.listener.accept().await?;
            Ok(UdsTaskConnection {
                stream: Framed::new(stream, LengthDelimitedCodec::new()),
            })
        }
    }

    /// UDS connection
    pub struct UdsTaskConnection {
        stream: Framed<UnixStream, LengthDelimitedCodec>,
    }

    impl UdsTaskConnection {
        pub async fn recv_task(&mut self) -> Result<AgentOsTask, QueueError> {
            let data = self
                .stream
                .next()
                .await
                .ok_or(QueueError::Closed)?
                .map_err(|e| QueueError::Queue(e.to_string()))?;
            let task = serde_json::from_slice(&data)?;
            Ok(task)
        }

        pub async fn send_result(&mut self, result: &AgentOsResult) -> Result<(), QueueError> {
            let data = serde_json::to_vec(result)?;
            self.stream
                .send(data.into())
                .await
                .map_err(|e| QueueError::Queue(e.to_string()))?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_task_queue_basic() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        let mut task_queue = TaskQueue::new(base_path).unwrap();
        let mut worker_queue = WorkerQueue::new(base_path).unwrap();

        let task = AgentOsTask::new(
            "iri://task/test".to_string(),
            "Test task".to_string(),
            TaskContextData {
                stage_id: "test".to_string(),
                stage_type: "requirement".to_string(),
                project_id: "proj_1".to_string(),
                project_dir: "/tmp".to_string(),
                user_requirement: "test".to_string(),
                prev_outputs: HashMap::new(),
                llm_config: LlmConfig::default(),
            },
        );

        task_queue.send_task(&task).await.unwrap();
        let expected_task_id = task.task_id.clone();
        worker_queue
            .process_next(|received| async move {
                assert_eq!(received.task_iri, "iri://task/test");
                AgentOsResult::success(received.task_id, "completed".to_string())
            })
            .await
            .unwrap();

        let received_result = task_queue
            .recv_result_timeout(Duration::from_secs(5))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(received_result.status, "success");
        assert_eq!(received_result.task_id, expected_task_id);
    }

    #[test]
    fn task_serialization_never_persists_raw_api_key() {
        let mut llm = LlmConfig::default();
        llm.api_key = "raw-secret-must-not-persist".to_string();
        llm.credential_ref = Some("env:WORKER_MODEL_KEY".to_string());
        let task = AgentOsTask::new(
            "iri://task/secret".to_string(),
            "test".to_string(),
            TaskContextData {
                stage_id: String::new(),
                stage_type: String::new(),
                project_id: String::new(),
                project_dir: String::new(),
                user_requirement: String::new(),
                prev_outputs: HashMap::new(),
                llm_config: llm,
            },
        );
        let serialized = serde_json::to_string(&task).unwrap();
        assert!(!serialized.contains("raw-secret-must-not-persist"));
        assert!(serialized.contains("env:WORKER_MODEL_KEY"));
    }

    #[tokio::test]
    async fn uncommitted_task_is_delivered_again_after_worker_panic() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();
        let mut client = TaskQueue::new_client(base_path).unwrap();
        let task = AgentOsTask::new(
            "iri://task/retry".to_string(),
            "retry".to_string(),
            TaskContextData {
                stage_id: String::new(),
                stage_type: String::new(),
                project_id: String::new(),
                project_dir: String::new(),
                user_requirement: String::new(),
                prev_outputs: HashMap::new(),
                llm_config: LlmConfig::default(),
            },
        );
        client.send_task(&task).await.unwrap();

        let mut first_worker = WorkerQueue::new(base_path).unwrap();
        let crashed = tokio::spawn(async move {
            first_worker
                .process_next(|_| async move { panic!("simulated worker crash") })
                .await
        })
        .await;
        assert!(crashed.is_err());

        let mut recovered_worker = WorkerQueue::new(base_path).unwrap();
        recovered_worker
            .process_next(|received| async move {
                AgentOsResult::success(received.task_id, "recovered".to_string())
            })
            .await
            .unwrap();
        let result = client
            .recv_result_for_task(&task.task_id, Duration::from_secs(2))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.summary, "recovered");
    }

    #[tokio::test]
    async fn results_are_isolated_by_task_queue() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();
        let context = || TaskContextData {
            stage_id: String::new(),
            stage_type: String::new(),
            project_id: String::new(),
            project_dir: String::new(),
            user_requirement: String::new(),
            prev_outputs: HashMap::new(),
            llm_config: LlmConfig::default(),
        };
        let first = AgentOsTask::new("iri://task/one".into(), "one".into(), context());
        let second = AgentOsTask::new("iri://task/two".into(), "two".into(), context());
        let mut first_client = TaskQueue::new_client(base_path).unwrap();
        let mut second_client = TaskQueue::new_client(base_path).unwrap();
        first_client.send_task(&first).await.unwrap();
        second_client.send_task(&second).await.unwrap();
        let mut worker = WorkerQueue::new(base_path).unwrap();
        for _ in 0..2 {
            worker
                .process_next(|task| async move {
                    let summary = task.prompt.clone();
                    AgentOsResult::success(task.task_id, summary)
                })
                .await
                .unwrap();
        }
        let second_result = second_client
            .recv_result_for_task(&second.task_id, Duration::from_secs(2))
            .await
            .unwrap()
            .unwrap();
        let first_result = first_client
            .recv_result_for_task(&first.task_id, Duration::from_secs(2))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second_result.summary, "two");
        assert_eq!(first_result.summary, "one");
    }

    #[tokio::test]
    async fn durable_claims_allow_parallel_processing() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();
        let context = || TaskContextData {
            stage_id: String::new(),
            stage_type: String::new(),
            project_id: String::new(),
            project_dir: String::new(),
            user_requirement: String::new(),
            prev_outputs: HashMap::new(),
            llm_config: LlmConfig::default(),
        };
        let first = AgentOsTask::new("iri://task/parallel-one".into(), "one".into(), context());
        let second = AgentOsTask::new("iri://task/parallel-two".into(), "two".into(), context());
        let mut client = TaskQueue::new_client(base_path).unwrap();
        client.send_task(&first).await.unwrap();
        client.send_task(&second).await.unwrap();
        let mut dispatcher = WorkerQueue::new(base_path).unwrap();
        let first_claim = dispatcher.claim_next().await.unwrap();
        let second_claim = dispatcher.claim_next().await.unwrap();
        assert!(first_claim.claim_path.exists());
        assert!(second_claim.claim_path.exists());

        let rendezvous = std::sync::Arc::new(tokio::sync::Barrier::new(3));
        let first_rendezvous = rendezvous.clone();
        let first_job = async {
            first_rendezvous.wait().await;
            WorkerQueue::complete_claim(&first_claim).await.unwrap();
        };
        let second_rendezvous = rendezvous.clone();
        let second_job = async {
            second_rendezvous.wait().await;
            WorkerQueue::complete_claim(&second_claim).await.unwrap();
        };
        let joined = async { tokio::join!(first_job, second_job, rendezvous.wait()) };
        tokio::time::timeout(Duration::from_secs(2), joined)
            .await
            .expect("both durable claims must reach the rendezvous concurrently");
    }
}
