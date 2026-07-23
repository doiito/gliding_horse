use std::collections::HashMap;
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
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.deepseek.com".to_string(),
            model: "deepseek-v4-flash".to_string(),
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
    task_sender: Sender,
    result_receiver: Receiver,
    base_path: String,
    pending_results: HashMap<String, AgentOsResult>,
}

impl TaskQueue {
    /// Create a new task queue (full version, holds both sender and receiver)
    pub fn new(base_path: &str) -> Result<Self, QueueError> {
        let task_path = format!("{}/tasks", base_path);
        let result_path = format!("{}/results", base_path);

        let task_sender = Sender::open(&task_path)?;
        let result_receiver = Receiver::open(&result_path)?;

        Ok(Self {
            task_sender,
            result_receiver,
            base_path: base_path.to_string(),
            pending_results: HashMap::new(),
        })
    }

    /// Create a client queue (only sends tasks, receives results)
    pub fn new_client(base_path: &str) -> Result<Self, QueueError> {
        let task_path = format!("{}/tasks", base_path);
        let result_path = format!("{}/results", base_path);

        let task_sender = Sender::open(&task_path)?;
        let result_receiver = Receiver::open(&result_path)?;

        Ok(Self {
            task_sender,
            result_receiver,
            base_path: base_path.to_string(),
            pending_results: HashMap::new(),
        })
    }

    /// Send task
    pub async fn send_task(&mut self, task: &AgentOsTask) -> Result<(), QueueError> {
        let data = serde_json::to_vec(task)?;
        tracing::info!(task_id = %task.task_id, task_iri = %task.task_iri, data_len = data.len(), "Sending task to queue");
        self.task_sender.send(data).await?;
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

        if let Some(result) = self.pending_results.remove(task_id) {
            tracing::info!(task_id = %task_id, "Found matching result from cache");
            return Ok(Some(result));
        }

        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            let remaining = deadline.duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                tracing::warn!(expected_task_id = %task_id, "Timed out waiting for result");
                return Ok(None);
            }

            match tokio::time::timeout(remaining, self.result_receiver.recv()).await {
                Ok(guard_result) => {
                    let guard = guard_result.map_err(|e| QueueError::Queue(e.to_string()))?;
                    let result: AgentOsResult = serde_json::from_slice(&*guard)?;
                    let _ = guard.commit();

                    tracing::info!(
                        expected_task_id = %task_id,
                        got_task_id = %result.task_id,
                        status = %result.status,
                        "Result received"
                    );

                    if result.task_id == task_id {
                        tracing::info!(task_id = %task_id, "Result matched, returning");
                        return Ok(Some(result));
                    } else {
                        tracing::warn!(expected_task_id = %task_id, got_task_id = %result.task_id, "Result does not match, caching");
                        self.pending_results.insert(result.task_id.clone(), result);
                    }
                }
                Err(_) => {
                    tracing::warn!(expected_task_id = %task_id, "Timed out receiving result");
                    return Ok(None);
                }
            }
        }
    }

    /// Receive result (with timeout, no task_id matching)
    pub async fn recv_result_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<AgentOsResult>, QueueError> {
        let deadline = tokio::time::Instant::now() + timeout;

        let guard = tokio::time::timeout_at(deadline, self.result_receiver.recv())
            .await
            .map_err(|_| QueueError::Timeout)?
            .map_err(|e| QueueError::Queue(e.to_string()))?;

        let result: AgentOsResult = serde_json::from_slice(&*guard)?;
        let _ = guard.commit();
        tracing::debug!(task_id = %result.task_id, status = %result.status, "Result received");
        Ok(Some(result))
    }

    /// Get queue base path
    pub fn base_path(&self) -> &str {
        &self.base_path
    }
}

/// Worker-side task queue
pub struct WorkerQueue {
    task_receiver: Receiver,
    result_sender: Sender,
    base_path: String,
}

impl WorkerQueue {
    /// Create a new Worker queue
    pub fn new(base_path: &str) -> Result<Self, QueueError> {
        let task_path = format!("{}/tasks", base_path);
        let result_path = format!("{}/results", base_path);

        let task_receiver = Receiver::open(&task_path)?;
        let result_sender = Sender::open(&result_path)?;

        Ok(Self {
            task_receiver,
            result_sender,
            base_path: base_path.to_string(),
        })
    }

    /// Receive task
    pub async fn recv_task(&mut self) -> Result<AgentOsTask, QueueError> {
        let guard = self
            .task_receiver
            .recv()
            .await
            .map_err(|e| QueueError::Queue(e.to_string()))?;
        let task: AgentOsTask = serde_json::from_slice(&*guard)?;
        let _ = guard.commit();
        tracing::debug!(task_id = %task.task_id, "Task received");
        Ok(task)
    }

    /// Send result
    pub async fn send_result(&mut self, result: &AgentOsResult) -> Result<(), QueueError> {
        let data = serde_json::to_vec(result)?;
        tracing::info!(task_id = %result.task_id, status = %result.status, data_len = data.len(), "Sending result to queue");
        self.result_sender.send(data).await?;
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

        let received = worker_queue.recv_task().await.unwrap();
        assert_eq!(received.task_iri, "iri://task/test");

        let result = AgentOsResult::success(task.task_id.clone(), "completed".to_string());
        worker_queue.send_result(&result).await.unwrap();

        let received_result = task_queue
            .recv_result_timeout(Duration::from_secs(5))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(received_result.status, "success");
    }
}
