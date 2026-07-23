#[cfg(test)]
mod tests {
    use crate::root_cause::RootCauseConfig;

    #[test]
    fn test_default_config() {
        let config = RootCauseConfig::default();
        assert_eq!(config.max_trace_depth, 5);
        assert!((config.min_confidence - 0.7).abs() < 0.01);
        assert!(config.enable_auto_trace);
        assert_eq!(config.trace_timeout_ms, 30_000);
    }
}
