use glidinghorse::config::GatewaySettings;
use glidinghorse::gateway::UnifiedGateway;
use std::sync::Arc;

fn get_gateway() -> Arc<UnifiedGateway> {
    let api_key = std::env::var("AGENT_OS_GATEWAY_API_KEY")
        .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
        .expect("AGENT_OS_GATEWAY_API_KEY or DEEPSEEK_API_KEY must be set");

    let base_url = std::env::var("AGENT_OS_GATEWAY_BASE_URL")
        .or_else(|_| std::env::var("DEEPSEEK_API_URL"))
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());

    let settings = GatewaySettings {
        base_url,
        api_key,
        default_model: "deepseek-v4-flash".to_string(),
        timeout_seconds: 60,
        max_retries: 2,
        retry_base_ms: 500,
        model_mapping: Default::default(),
    };

    Arc::new(UnifiedGateway::new(&settings).expect("Failed to create gateway"))
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_deepseek_chat_completion() {
    let gateway = get_gateway();

    let msg = glidinghorse::gateway::unified_gateway::ChatMessage {
        role: "user".to_string(),
        content: "Say 'Hello from Agent OS' and nothing else.".to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };

    let response = gateway
        .chat_with_model("deepseek-v4-flash", vec![msg])
        .await
        .expect("DeepSeek API call failed");

    let choice = response.choices.first().expect("No choices in response");
    let content = choice.message.content.as_deref().unwrap_or("");

    assert!(!content.is_empty(), "Response content should not be empty");
    eprintln!("DeepSeek response: {}", content);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_deepseek_sa_workflow() {
    let gateway = get_gateway();

    let msg = glidinghorse::gateway::unified_gateway::ChatMessage {
        role: "user".to_string(),
        content: "Classify this task as simple, standard, or emergency: \
                  'Build a web application with user authentication and database'. \
                  Respond with just one word."
            .to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };

    let response = gateway
        .chat_with_model("deepseek-v4-flash", vec![msg])
        .await
        .expect("DeepSeek classification failed");
    let content = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_deref())
        .unwrap_or("");

    assert!(!content.is_empty(), "Classification should not be empty");
    eprintln!("Classification: {}", content);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_deepseek_with_tools() {
    let gateway = get_gateway();

    let msg = glidinghorse::gateway::unified_gateway::ChatMessage {
        role: "user".to_string(),
        content: "What is 2+2?".to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };

    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    })];

    let response = gateway
        .chat_with_params(
            "deepseek-v4-flash",
            vec![msg],
            Some(0.7),
            Some(512),
            Some(tools),
            None,
        )
        .await;

    match response {
        Ok(r) => eprintln!(
            "Tool response: {:?}",
            r.choices.first().and_then(|c| c.message.content.as_deref())
        ),
        Err(e) => eprintln!("Tool call failed: {}", e),
    }
}
