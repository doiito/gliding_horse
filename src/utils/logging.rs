use std::io::{self, Write};
use std::sync::Arc;

use regex::Regex;
use tracing::Metadata;
use tracing_appender::{non_blocking, rolling};
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

use crate::config::settings::LoggingSettings;

/// Maximum size of one formatted tracing record retained before redaction.
///
/// A tracing formatter normally emits one short newline-terminated record. A
/// bounded buffer is nevertheless required because the writer is also a
/// security boundary: emitting a prefix early would allow a sensitive field
/// name and its value to be split across writes and evade redaction.
const MAX_BUFFERED_LOG_LINE_BYTES: usize = 1024 * 1024;
const OVERSIZED_LOG_LINE_PLACEHOLDER: &[u8] = b"[REDACTED_OVERSIZED_LOG_LINE]";

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

    let sanitizer = SensitiveFieldSanitizer::new(&settings.sensitive_fields);
    let mut layers = Vec::new();

    if settings.console_output {
        // Redaction belongs at the final byte-writer boundary. This covers
        // message text, formatted fields, JSON logs, and fragmented formatter
        // writes without requiring every tracing call site to remember it.
        let console_writer =
            RedactingMakeWriter::with_sanitizer(std::io::stdout, sanitizer.clone());
        let console_layer = if settings.format == "json" {
            fmt::layer()
                .with_target(true)
                .with_ansi(true)
                .json()
                .with_writer(console_writer)
                .with_filter(env_filter.clone())
                .boxed()
        } else {
            fmt::layer()
                .with_target(true)
                .with_ansi(true)
                .with_writer(console_writer)
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
            let file_writer =
                RedactingMakeWriter::with_sanitizer(non_blocking_file, sanitizer.clone());

            let file_layer = if settings.format == "json" {
                fmt::layer()
                    .with_target(true)
                    .with_ansi(false)
                    .json()
                    .with_writer(file_writer)
                    .with_filter(env_filter)
                    .boxed()
            } else {
                fmt::layer()
                    .with_target(true)
                    .with_ansi(false)
                    .with_writer(file_writer)
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

#[derive(Debug)]
struct SensitiveFieldPattern {
    escaped_json: Regex,
    json: Regex,
    assignment: Regex,
}

/// Precompiled sensitive-field policy shared by all writers in one subscriber.
///
/// Field names are treated as literals, matched case-insensitively, and cover
/// JSON as well as the common `key=value` / `key: value` text forms. The value
/// grammar accepts quoted values (including escaped quotes) and conservatively
/// consumes malformed or unquoted values to the next structural delimiter.
#[derive(Clone, Debug)]
pub struct SensitiveFieldSanitizer {
    patterns: Arc<[SensitiveFieldPattern]>,
}

impl SensitiveFieldSanitizer {
    pub fn new(sensitive_fields: &[String]) -> Self {
        let mut normalized_fields = sensitive_fields
            .iter()
            .map(|field| field.trim())
            .filter(|field| !field.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        normalized_fields.sort_by_key(|field| field.to_lowercase());
        normalized_fields.dedup_by(|left, right| left.eq_ignore_ascii_case(right));

        let patterns = normalized_fields
            .into_iter()
            .filter_map(|field| {
                let escaped = regex::escape(&field);
                let quoted_value = r#""(?:\\.|[^"\\])*""#;
                let single_quoted_value = r#"'(?:\\.|[^'\\])*'"#;
                let credential_scheme_value =
                    r#"(?i:(?:Bearer|Basic|Token)\s+[^,\s\r\n}\]]+)"#;
                // JSON embedded in a formatted string is escaped again by a
                // JSON tracing formatter (for example `\"token\":\"...\"`).
                let escaped_json = Regex::new(&format!(
                    r#"(?i)(?P<prefix>\\"{escaped}\\"\s*:\s*)\\"(?:\\.|[^"\\])*\\""#
                ))
                .ok()?;
                let json = Regex::new(&format!(
                    r#"(?i)(?P<prefix>"{escaped}"\s*:\s*)(?:{quoted_value}|[^,\r\n}}\]]*)"#
                ))
                .ok()?;
                let assignment = Regex::new(&format!(
                    r#"(?i)(?P<prefix>\b{escaped}\s*[=:]\s*)(?:{quoted_value}|{single_quoted_value}|{credential_scheme_value}|[^,\s\r\n}}\]]*)"#
                ))
                .ok()?;
                Some(SensitiveFieldPattern {
                    escaped_json,
                    json,
                    assignment,
                })
            })
            .collect::<Vec<_>>();

        Self {
            patterns: patterns.into(),
        }
    }

    pub fn sanitize(&self, value: &str) -> String {
        let mut result = value.to_owned();
        for pattern in self.patterns.iter() {
            result = pattern
                .escaped_json
                .replace_all(&result, "${prefix}\\\"[REDACTED]\\\"")
                .into_owned();
            result = pattern
                .json
                .replace_all(&result, "${prefix}\"[REDACTED]\"")
                .into_owned();
            result = pattern
                .assignment
                .replace_all(&result, "${prefix}[REDACTED]")
                .into_owned();
        }
        result
    }
}

pub fn sanitize_sensitive_fields(value: &str, sensitive_fields: &[String]) -> String {
    SensitiveFieldSanitizer::new(sensitive_fields).sanitize(value)
}

/// A [`MakeWriter`] adapter that installs line-buffered redaction immediately
/// before bytes reach an output sink.
#[derive(Clone, Debug)]
pub struct RedactingMakeWriter<M> {
    inner: M,
    sanitizer: SensitiveFieldSanitizer,
}

impl<M> RedactingMakeWriter<M> {
    pub fn new(inner: M, sensitive_fields: &[String]) -> Self {
        Self::with_sanitizer(inner, SensitiveFieldSanitizer::new(sensitive_fields))
    }

    pub fn with_sanitizer(inner: M, sanitizer: SensitiveFieldSanitizer) -> Self {
        Self { inner, sanitizer }
    }
}

impl<'a, M> MakeWriter<'a> for RedactingMakeWriter<M>
where
    M: MakeWriter<'a>,
{
    type Writer = RedactingWriter<M::Writer>;

    fn make_writer(&'a self) -> Self::Writer {
        RedactingWriter::with_sanitizer(self.inner.make_writer(), self.sanitizer.clone())
    }

    fn make_writer_for(&'a self, metadata: &Metadata<'_>) -> Self::Writer {
        RedactingWriter::with_sanitizer(
            self.inner.make_writer_for(metadata),
            self.sanitizer.clone(),
        )
    }
}

/// A bounded, line-buffered redacting writer.
///
/// Bytes are never forwarded until a complete line, explicit flush, or drop.
/// This is important because formatters are allowed to split a field name and
/// value across arbitrary `write` calls. Oversized records are discarded and
/// replaced with a fixed marker instead of releasing an unsafe prefix.
pub struct RedactingWriter<W: Write> {
    inner: W,
    sanitizer: SensitiveFieldSanitizer,
    pending: Vec<u8>,
    discarding_oversized_line: bool,
}

impl<W: Write> RedactingWriter<W> {
    pub fn new(inner: W, sensitive_fields: &[String]) -> Self {
        Self::with_sanitizer(inner, SensitiveFieldSanitizer::new(sensitive_fields))
    }

    pub fn with_sanitizer(inner: W, sanitizer: SensitiveFieldSanitizer) -> Self {
        Self {
            inner,
            sanitizer,
            pending: Vec::with_capacity(1024),
            discarding_oversized_line: false,
        }
    }

    fn flush_pending_record(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let sanitized = self
            .sanitizer
            .sanitize(&String::from_utf8_lossy(&self.pending));
        self.inner.write_all(sanitized.as_bytes())?;
        self.pending.clear();
        Ok(())
    }

    fn accept_segment(&mut self, segment: &[u8], ends_line: bool) -> io::Result<()> {
        if self.discarding_oversized_line {
            if ends_line {
                self.inner.write_all(b"\n")?;
                self.discarding_oversized_line = false;
            }
            return Ok(());
        }

        if self.pending.len().saturating_add(segment.len()) > MAX_BUFFERED_LOG_LINE_BYTES {
            self.inner.write_all(OVERSIZED_LOG_LINE_PLACEHOLDER)?;
            self.pending.clear();
            if ends_line {
                self.inner.write_all(b"\n")?;
            } else {
                self.discarding_oversized_line = true;
            }
            return Ok(());
        }

        self.pending.extend_from_slice(segment);
        if ends_line {
            self.flush_pending_record()?;
        }
        Ok(())
    }
}

impl<W: Write> Write for RedactingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        for segment in buf.split_inclusive(|byte| *byte == b'\n') {
            self.accept_segment(segment, segment.last() == Some(&b'\n'))?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        // Do not expose an incomplete record: a formatter may flush between
        // fragments such as `api_` and `key=...`. Complete lines have already
        // reached the sink; drop handles a final non-newline-terminated event.
        self.inner.flush()
    }
}

impl<W: Write> Drop for RedactingWriter<W> {
    fn drop(&mut self) {
        let _ = self.flush_pending_record();
        let _ = self.inner.flush();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Clone, Default)]
    struct SharedBytes(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBytes {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("test sink lock poisoned").extend(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl SharedBytes {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().expect("test sink lock poisoned").clone())
                .expect("redacted log should remain UTF-8")
        }
    }

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
        assert!(!sanitized2.contains("secret123"));
    }

    #[test]
    fn sanitizer_handles_case_quotes_escapes_and_literal_field_names() {
        let sensitive = vec![
            "token".to_string(),
            "client.secret".to_string(),
            "TOKEN".to_string(),
        ];
        let input = concat!(
            r#"{"ToKeN":"a\\\"b", "client.secret":  dangerous value } "#,
            r#"token='quoted value', client.secret=plain-secret"#,
        );

        let sanitized = sanitize_sensitive_fields(input, &sensitive);

        assert_eq!(sanitized.matches("[REDACTED]").count(), 4);
        assert!(!sanitized.contains("dangerous"));
        assert!(!sanitized.contains("quoted value"));
        assert!(!sanitized.contains("plain-secret"));
        assert!(!sanitized.contains(r#"a\\\"b"#));
    }

    #[test]
    fn fragmented_writes_cannot_bypass_redaction() {
        let sink = SharedBytes::default();
        {
            let sensitive = vec!["api_key".to_string(), "password".to_string()];
            let mut writer = RedactingWriter::new(sink.clone(), &sensitive);
            writer.write_all(br#"request api_"#).unwrap();
            writer.flush().unwrap();
            assert!(sink.text().is_empty());
            writer.write_all(br#"key=sk-frag"#).unwrap();
            writer.flush().unwrap();
            writer.write_all(br#"mented-secret, pass"#).unwrap();
            writer.write_all(br#"word: "also secret""#).unwrap();
            writer.write_all(b"\nnext=safe\n").unwrap();
        }

        let output = sink.text();
        assert_eq!(output.matches("[REDACTED]").count(), 2);
        assert!(!output.contains("sk-fragmented-secret"));
        assert!(!output.contains("also secret"));
        assert!(output.contains("next=safe"));
    }

    #[test]
    fn unterminated_record_is_redacted_on_drop() {
        let sink = SharedBytes::default();
        {
            let sensitive = vec!["authorization".to_string()];
            let mut writer = RedactingWriter::new(sink.clone(), &sensitive);
            writer.write_all(b"authorization=Bearer never-log").unwrap();
        }

        let output = sink.text();
        assert!(output.contains("authorization=[REDACTED]"));
        assert!(!output.contains("never-log"));
    }

    #[test]
    fn oversized_fragmented_record_is_replaced_without_leaking_prefix() {
        let sink = SharedBytes::default();
        {
            let sensitive = vec!["api_key".to_string()];
            let mut writer = RedactingWriter::new(sink.clone(), &sensitive);
            writer.write_all(b"api_key=must-not-leak ").unwrap();
            writer
                .write_all(&vec![b'x'; MAX_BUFFERED_LOG_LINE_BYTES])
                .unwrap();
            writer.write_all(b" ignored-tail\nvisible=safe\n").unwrap();
        }

        let output = sink.text();
        assert_eq!(output, "[REDACTED_OVERSIZED_LOG_LINE]\nvisible=safe\n");
        assert!(!output.contains("must-not-leak"));
    }

    #[test]
    fn actual_tracing_formatter_redacts_structured_and_escaped_json_fields() {
        let sink = SharedBytes::default();
        let sink_for_writer = sink.clone();
        let sensitive = vec!["api_key".to_string(), "token".to_string()];
        let writer = RedactingMakeWriter::new(move || sink_for_writer.clone(), &sensitive);
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_ansi(false)
            .with_writer(writer)
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(
                api_key = "structured-never-log",
                payload = r#"{"token":"nested-never-log"}"#,
                "redaction integration test"
            );
        });

        let output = sink.text();
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("structured-never-log"));
        assert!(!output.contains("nested-never-log"));
    }
}
