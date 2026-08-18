#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCAN_PATHS=(src apps crates tests docs skills workflow.jsonld)
LEGACY='pdca-agent\.org|agent-harness\.os|wild-agent-os\.org|agentos\.ontology|http://agent-os\.org'

# The migration guide intentionally names historical hosts as part of the
# documented upgrade path, so exclude that one reference-only document.
if rg -n -i "$LEGACY" "${SCAN_PATHS[@]}" \
  --glob '!docs/16-ONTOLOGY_NAMESPACE_MIGRATION.md'; then
  echo "legacy ontology namespace detected"
  exit 1
fi

if ! rg -n 'https://agent-os\.org/ontology/' "${SCAN_PATHS[@]}" >/dev/null; then
  echo "canonical ontology namespace not found"
  exit 1
fi

echo "ontology namespace check passed"
