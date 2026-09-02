#!/usr/bin/env bash
set -euo pipefail

workspace_version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -n 1)"
if [[ -z "${workspace_version}" ]]; then
  echo "workspace.package.version is missing" >&2
  exit 1
fi

for manifest in Cargo.toml apps/gliding_code/Cargo.toml crates/hyperspace-engine/Cargo.toml crates/ontologies/Cargo.toml; do
  if [[ "${manifest}" != "Cargo.toml" ]] && ! grep -q '^version.workspace = true$' "${manifest}"; then
    echo "${manifest} does not inherit the workspace version" >&2
    exit 1
  fi
done

for readme in README.md README.zh.md; do
  if ! grep -q "release-v${workspace_version}-blue" "${readme}"; then
    echo "${readme} release badge does not match ${workspace_version}" >&2
    exit 1
  fi
done

echo "workspace version ${workspace_version} is consistent"
