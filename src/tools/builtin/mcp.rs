use crate::config::{McpServerConfig, ScopedMcpServerConfig};

const COLLAPSE_NAME_PREFIX: &str = "example.com ";
const CCR_PROXY_PATH_MARKERS: [&str; 2] = ["/v2/session_ingress/shttp/mcp/", "/v2/ccr-sessions/"];

#[must_use]
pub fn normalize_name_for_mcp(name: &str) -> String {
    let mut normalized = name
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => ch,
            _ => '_',
        })
        .collect::<String>();

    if name.starts_with(COLLAPSE_NAME_PREFIX) {
        normalized = collapse_underscores(&normalized)
            .trim_matches('_')
            .to_string();
    }

    normalized
}

#[must_use]
pub fn mcp_tool_prefix(server_name: &str) -> String {
    format!("mcp__{}__", normalize_name_for_mcp(server_name))
}

#[must_use]
pub fn mcp_tool_name(server_name: &str, tool_name: &str) -> String {
    format!(
        "{}{}",
        mcp_tool_prefix(server_name),
        normalize_name_for_mcp(tool_name)
    )
}

#[must_use]
pub fn unwrap_ccr_proxy_url(url: &str) -> String {
    if !CCR_PROXY_PATH_MARKERS
        .iter()
        .any(|marker| url.contains(marker))
    {
        return url.to_string();
    }

    let Some(query_start) = url.find('?') else {
        return url.to_string();
    };
    let query = &url[query_start + 1..];
    for pair in query.split('&') {
        let mut parts = pair.splitn(2, '=');
        if matches!(parts.next(), Some("mcp_url")) {
            if let Some(value) = parts.next() {
                return percent_decode(value);
            }
        }
    }

    url.to_string()
}

#[must_use]
pub fn mcp_server_signature(config: &McpServerConfig) -> Option<String> {
    match config {
        McpServerConfig::Stdio(config) => {
            let mut command = vec![config.command.clone()];
            command.extend(config.args.clone());
            Some(format!("stdio:{}", render_command_signature(&command)))
        }
        McpServerConfig::Http(config) => Some(format!("url:{}", unwrap_ccr_proxy_url(&config.url))),
    }
}

#[must_use]
pub fn scoped_mcp_config_hash(config: &ScopedMcpServerConfig) -> String {
    let rendered = match &config.config {
        McpServerConfig::Stdio(stdio) => format!(
            "stdio|{}|{}|{}",
            stdio.command,
            render_command_signature(&stdio.args),
            render_env_signature(&stdio.env),
        ),
        McpServerConfig::Http(remote) => format!(
            "http|{}|{}",
            remote.url,
            render_env_signature(&remote.headers),
        ),
    };
    stable_hex_hash(&rendered)
}

fn render_command_signature(command: &[String]) -> String {
    let escaped = command
        .iter()
        .map(|part| part.replace('\\', "\\\\").replace('|', "\\|"))
        .collect::<Vec<_>>();
    format!("[{}]", escaped.join("|"))
}

fn render_env_signature(map: &std::collections::BTreeMap<String, String>) -> String {
    map.iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn stable_hex_hash(value: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn collapse_underscores(value: &str) -> String {
    let mut collapsed = String::with_capacity(value.len());
    let mut last_was_underscore = false;
    for ch in value.chars() {
        if ch == '_' {
            if !last_was_underscore {
                collapsed.push(ch);
            }
            last_was_underscore = true;
        } else {
            collapsed.push(ch);
            last_was_underscore = false;
        }
    }
    collapsed
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                let hex = &value[index + 1..index + 3];
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    decoded.push(byte);
                    index += 3;
                    continue;
                }
                decoded.push(bytes[index]);
                index += 1;
            }
            b'+' => {
                decoded.push(b' ');
                index += 1;
            }
            byte => {
                decoded.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::config::{
        McpRemoteServerConfig, McpServerConfig, McpStdioServerConfig, ScopedMcpServerConfig,
    };

    use super::{
        mcp_server_signature, mcp_tool_name, normalize_name_for_mcp, scoped_mcp_config_hash,
        unwrap_ccr_proxy_url,
    };

    #[test]
    fn normalizes_server_names_for_mcp_tooling() {
        assert_eq!(normalize_name_for_mcp("github.com"), "github_com");
        assert_eq!(normalize_name_for_mcp("tool name!"), "tool_name_");
        assert_eq!(
            normalize_name_for_mcp("example.com Example   Server!!"),
            "example_com_Example_Server"
        );
        assert_eq!(
            mcp_tool_name("example.com Example Server", "weather tool"),
            "mcp__example_com_Example_Server__weather_tool"
        );
    }

    #[test]
    fn unwraps_ccr_proxy_urls_for_signature_matching() {
        let wrapped = "https://api.example.com/v2/session_ingress/shttp/mcp/123?mcp_url=https%3A%2F%2Fvendor.example%2Fmcp&other=1";
        assert_eq!(unwrap_ccr_proxy_url(wrapped), "https://vendor.example/mcp");
        assert_eq!(
            unwrap_ccr_proxy_url("https://vendor.example/mcp"),
            "https://vendor.example/mcp"
        );
    }

    #[test]
    fn computes_signatures_for_stdio_and_remote_servers() {
        let stdio = McpServerConfig::Stdio(McpStdioServerConfig {
            command: "uvx".to_string(),
            args: vec!["mcp-server".to_string()],
            env: BTreeMap::from([("TOKEN".to_string(), "secret".to_string())]),
            tool_call_timeout_ms: None,
        });
        assert_eq!(
            mcp_server_signature(&stdio),
            Some("stdio:[uvx|mcp-server]".to_string())
        );

        let remote = McpServerConfig::Http(McpRemoteServerConfig {
            url: "https://vendor.example/mcp".to_string(),
            headers: BTreeMap::new(),
        });
        assert!(mcp_server_signature(&remote).is_some());
    }

    #[test]
    fn scoped_hash_tracks_config_content() {
        let base = McpServerConfig::Http(McpRemoteServerConfig {
            url: "https://vendor.example/mcp".to_string(),
            headers: BTreeMap::from([("Authorization".to_string(), "Bearer token".to_string())]),
        });
        let hash_a = scoped_mcp_config_hash(&ScopedMcpServerConfig { config: base });

        let changed = McpServerConfig::Http(McpRemoteServerConfig {
            url: "https://vendor.example/v2/mcp".to_string(),
            headers: BTreeMap::new(),
        });
        let hash_b = scoped_mcp_config_hash(&ScopedMcpServerConfig { config: changed });
        assert_ne!(hash_a, hash_b);
    }
}
