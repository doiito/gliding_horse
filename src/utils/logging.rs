use tracing_appender::{non_blocking, rolling};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

use crate::config::settings::LoggingSettings;

pub struct LoggingGuard {
    _file_guard: Option<tracing_appender::non_blocking::WorkerGuard>,
}

impl LoggingGuard {
    pub fn new() -> Self {
        Self { _file_guard: None }
    }
}

pub fn init_logging(settings: &LoggingSettings) -> LoggingGuard {
    let mut guard = LoggingGuard::new();

    let level = parse_level(&settings.level);
    let mut env_filter = EnvFilter::from_default_env()
        .add_directive(level.into())
        .add_directive("redb=warn".parse().unwrap_or_default());

    for filter in &settings.filters {
        let directive_str = format!("{}={}", filter.module, filter.level.to_lowercase());
        if let Ok(directive) = directive_str.parse() {
            env_filter = env_filter.add_directive(directive);
        }
    }

    let mut layers = Vec::new();

    if settings.console_output {
        let console_layer = if settings.format == "json" {
            fmt::layer()
                .with_target(true)
                .with_ansi(true)
                .json()
                .with_filter(env_filter.clone())
                .boxed()
        } else {
            fmt::layer()
                .with_target(true)
                .with_ansi(true)
                .with_filter(env_filter.clone())
                .boxed()
        };
        layers.push(console_layer);
    }

    if settings.file_output.enabled {
        if let Ok(()) = std::fs::create_dir_all(&settings.file_output.path) {
            let file_appender = match settings.file_output.rotation.as_str() {
                "daily" => rolling::RollingFileAppender::new(
                    rolling::Rotation::DAILY,
                    &settings.file_output.path,
                    &settings.file_output.prefix,
                ),
                "hourly" => rolling::RollingFileAppender::new(
                    rolling::Rotation::HOURLY,
                    &settings.file_output.path,
                    &settings.file_output.prefix,
                ),
                "minutely" => rolling::RollingFileAppender::new(
                    rolling::Rotation::MINUTELY,
                    &settings.file_output.path,
                    &settings.file_output.prefix,
                ),
                _ => rolling::RollingFileAppender::new(
                    rolling::Rotation::NEVER,
                    &settings.file_output.path,
                    &settings.file_output.prefix,
                ),
            };

            let (non_blocking_file, file_guard) = non_blocking(file_appender);
            guard._file_guard = Some(file_guard);

            let file_layer = if settings.format == "json" {
                fmt::layer()
                    .with_target(true)
                    .with_ansi(false)
                    .json()
                    .with_writer(non_blocking_file)
                    .with_filter(env_filter)
                    .boxed()
            } else {
                fmt::layer()
                    .with_target(true)
                    .with_ansi(false)
                    .with_writer(non_blocking_file)
                    .with_filter(env_filter)
                    .boxed()
            };
            layers.push(file_layer);
        }
    }

    if !layers.is_empty() {
        let _ = tracing_subscriber::registry().with(layers).try_init();
    }

    guard
}

fn parse_level(level: &str) -> tracing::Level {
    match level.to_lowercase().as_str() {
        "trace" => tracing::Level::TRACE,
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "warn" | "warning" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::INFO,
    }
}

pub fn sanitize_sensitive_fields(value: &str, sensitive_fields: &[String]) -> String {
    let mut result = value.to_string();
    for field in sensitive_fields {
        let patterns = vec![
            format!(r#""{}":\s*"[^"]*""#, field),
            format!(r#"{field}=[^,\s\]]+"#),
            format!(r#"{field}:\s*[^,\s\}}]+"#),
        ];

        for pattern in patterns {
            if let Ok(re) = regex::Regex::new(&pattern) {
                let replacement = format!(r#""{}": "[REDACTED]""#, field);
                result = re.replace_all(&result, &replacement).to_string();
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_level() {
        assert_eq!(parse_level("trace"), tracing::Level::TRACE);
        assert_eq!(parse_level("DEBUG"), tracing::Level::DEBUG);
        assert_eq!(parse_level("Info"), tracing::Level::INFO);
        assert_eq!(parse_level("warn"), tracing::Level::WARN);
        assert_eq!(parse_level("ERROR"), tracing::Level::ERROR);
        assert_eq!(parse_level("unknown"), tracing::Level::INFO);
    }

    #[test]
    fn test_sanitize_sensitive_fields() {
        let sensitive = vec!["api_key".to_string(), "password".to_string()];

        let input = r#"{"api_key": "sk-secret123", "name": "test"}"#;
        let sanitized = sanitize_sensitive_fields(input, &sensitive);
        assert!(sanitized.contains("[REDACTED]"));
        assert!(!sanitized.contains("sk-secret123"));

        let input2 = r#"api_key=secret123, name=test"#;
        let sanitized2 = sanitize_sensitive_fields(input2, &sensitive);
        assert!(sanitized2.contains("[REDACTED]"));
    }
}
