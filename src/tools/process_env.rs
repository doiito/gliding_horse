use std::ffi::{OsStr, OsString};

const SAFE_EXACT: &[&str] = &[
    "PATH",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "TERM",
    "COLORTERM",
    "LANG",
    "TZ",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "GOPATH",
    "GOROOT",
    "JAVA_HOME",
    "NODE_PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "PKG_CONFIG_PATH",
    "CC",
    "CXX",
    "AR",
    "LD",
];

const SAFE_PREFIXES: &[&str] = &[
    "LC_",
    "XDG_",
    "CARGO_",
    "RUST_",
    "RUSTUP_",
    "GO",
    "JAVA_",
    "NODE_",
    "NPM_",
    "PNPM_",
    "YARN_",
    "PYTHON",
    "PIP_",
    "PKG_CONFIG_",
    "CMAKE_",
];

const SENSITIVE_FRAGMENTS: &[&str] = &[
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "CREDENTIAL",
    "AUTHORIZATION",
    "COOKIE",
];

fn is_sensitive_key(key: &str) -> bool {
    let upper = key.to_ascii_uppercase();
    SENSITIVE_FRAGMENTS
        .iter()
        .any(|fragment| upper.contains(fragment))
}

fn is_safe_key(key: &str, include_home: bool) -> bool {
    if is_sensitive_key(key) {
        return false;
    }
    (include_home && key == "HOME")
        || SAFE_EXACT.contains(&key)
        || SAFE_PREFIXES.iter().any(|prefix| key.starts_with(prefix))
}

fn sanitize_environment<I>(vars: I, include_home: bool) -> Vec<(OsString, OsString)>
where
    I: IntoIterator<Item = (OsString, OsString)>,
{
    vars.into_iter()
        .filter(|(key, _)| {
            key.to_str()
                .is_some_and(|key| is_safe_key(key, include_home))
        })
        .collect()
}

pub(crate) fn sanitized_child_environment(include_home: bool) -> Vec<(OsString, OsString)> {
    sanitize_environment(std::env::vars_os(), include_home)
}

pub(crate) fn overlay_explicit_environment<K, V, I>(command: &mut tokio::process::Command, vars: I)
where
    K: AsRef<OsStr>,
    V: AsRef<OsStr>,
    I: IntoIterator<Item = (K, V)>,
{
    command.envs(vars);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(entries: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
        entries
            .iter()
            .map(|(key, value)| (OsString::from(key), OsString::from(value)))
            .collect()
    }

    #[test]
    fn child_environment_keeps_toolchain_but_drops_secrets() {
        let sanitized = sanitize_environment(
            vars(&[
                ("PATH", "/bin"),
                ("HOME", "/home/user"),
                ("CARGO_HOME", "/cargo"),
                ("CARGO_REGISTRIES_PRIVATE_TOKEN", "secret"),
                ("DEEPSEEK_API_KEY", "secret"),
                ("GITHUB_TOKEN", "secret"),
                ("RANDOM_UNSAFE_VALUE", "value"),
            ]),
            true,
        );
        let keys: Vec<&str> = sanitized
            .iter()
            .filter_map(|(key, _)| key.to_str())
            .collect();

        assert!(keys.contains(&"PATH"));
        assert!(keys.contains(&"HOME"));
        assert!(keys.contains(&"CARGO_HOME"));
        assert!(!keys.contains(&"CARGO_REGISTRIES_PRIVATE_TOKEN"));
        assert!(!keys.contains(&"DEEPSEEK_API_KEY"));
        assert!(!keys.contains(&"GITHUB_TOKEN"));
        assert!(!keys.contains(&"RANDOM_UNSAFE_VALUE"));
    }

    #[test]
    fn mcp_baseline_does_not_inherit_home() {
        let sanitized = sanitize_environment(vars(&[("PATH", "/bin"), ("HOME", "/secret")]), false);
        let keys: Vec<&str> = sanitized
            .iter()
            .filter_map(|(key, _)| key.to_str())
            .collect();
        assert_eq!(keys, vec!["PATH"]);
    }
}
