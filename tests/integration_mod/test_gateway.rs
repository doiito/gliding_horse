use glidinghorse::config::GatewaySettings;
use glidinghorse::gateway::UnifiedGateway;
use std::collections::HashMap;

fn test_gateway_settings() -> GatewaySettings {
    GatewaySettings {
        base_url: "http://localhost:3000".to_string(),
        api_key: "sk-test-key".to_string(),
        default_model: "deepseek-v4-flash".to_string(),
        timeout_seconds: 30,
        max_retries: 3,
        retry_base_ms: 500,
        model_mapping: HashMap::from([
            ("planning".to_string(), "deepseek-v4-pro".to_string()),
            ("execution".to_string(), "deepseek-v4-pro".to_string()),
            ("analysis".to_string(), "deepseek-v4-flash".to_string()),
            ("default".to_string(), "deepseek-v4-flash".to_string()),
        ]),
    }
}

#[test]
fn test_gateway_model_mapping() {
    let gateway = UnifiedGateway::new(&test_gateway_settings()).unwrap();
    assert_eq!(gateway.get_model("planning"), "deepseek-v4-pro");
    assert_eq!(gateway.get_model("execution"), "deepseek-v4-pro");
    assert_eq!(gateway.get_model("analysis"), "deepseek-v4-flash");
    assert_eq!(gateway.get_model("unknown"), "deepseek-v4-flash");
    assert_eq!(gateway.get_model("default"), "deepseek-v4-flash");
}

#[test]
fn test_gateway_default_model() {
    let gateway = UnifiedGateway::new(&test_gateway_settings()).unwrap();
    assert_eq!(gateway.default_model(), "deepseek-v4-flash");
}

#[test]
fn test_gateway_set_model_mapping() {
    let gateway = UnifiedGateway::new(&test_gateway_settings()).unwrap();
    gateway.set_model_mapping("custom_task".to_string(), "deepseek-v4-pro".to_string());
    assert_eq!(gateway.get_model("custom_task"), "deepseek-v4-pro");
}

#[test]
fn test_gateway_construction() {
    let gateway = UnifiedGateway::new(&test_gateway_settings());
    assert!(gateway.is_ok());
}
