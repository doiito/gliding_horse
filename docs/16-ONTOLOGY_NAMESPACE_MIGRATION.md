# Ontology Namespace Migration

## Summary

Gliding Horse uses JSON-LD and RDF across the memory, skill, task, and code
knowledge subsystems. Those subsystems must use the same IRI for the same
semantic concept. RDF treats different hosts, schemes, and paths as different
resources, even when their local names are identical.

The canonical ontology base is now:

```text
https://agent-os.org/ontology/
```

Domain namespaces remain intentionally separated below that base:

```text
https://agent-os.org/ontology/core/
https://agent-os.org/ontology/agent#
https://agent-os.org/ontology/task#
https://agent-os.org/ontology/skill#
https://agent-os.org/ontology/memory#
https://agent-os.org/ontology/security#
https://agent-os.org/ontology/monitoring#
https://agent-os.org/ontology/template#
https://agent-os.org/ontology/experience#
https://agent-os.org/ontology/advisory#
https://agent-os.org/ontology/node#
https://agent-os.org/ontology/eng/
https://agent-os.org/ontology/code/
https://agent-os.org/ontology/biz/
```

The domain split is intentional. The previous host split was not: it caused
the same concepts to be emitted under unrelated IRIs by different modules.

## What changed

The following layers now use the canonical base:

- JSON-LD constants and the embedded `context.json`
- JSON-LD framing templates
- L2 triple materialization and L3 SPARQL projections
- ontology vocabulary and code-AST extraction
- skill registry and syscall access-control predicates
- memory, batch, sharing, bridge, and result-router metadata
- workflow and skill fixtures
- source examples and regression tests

Generic core terms use the `core` namespace. For example:

```text
core:Task       → https://agent-os.org/ontology/core/Task
core:summary    → https://agent-os.org/ontology/core/summary
task:what      → https://agent-os.org/ontology/task#what
skill:AtomicSkill → https://agent-os.org/ontology/skill#AtomicSkill
```

This keeps domain separation while making all modules resolve through one
canonical ontology authority.

## Historical namespaces

The following namespaces are no longer valid write targets:

```text
https://pdca-agent.org/...
https://agent-harness.os/...
https://agentos.ontology/...
http://agent-os.org/...
```

This change intentionally does not silently rewrite arbitrary persisted RDF.
Existing installations with data written under a historical namespace should
be migrated explicitly before relying on queries using the canonical IRIs.

For a rolling deployment, the safe order is:

1. Export or snapshot the existing RDF store.
2. Rewrite known legacy ontology IRIs to their canonical equivalents.
3. Deploy the canonical writer and query code.
4. Verify cross-graph joins for Task, Skill, Agent, and core properties.
5. Keep the snapshot until the migration has been validated in production.

The application should not introduce a permanent `UNION` over every historical
namespace as a substitute for migration. Such a fallback hides future drift
and makes query behavior ambiguous.

## Regression protection

`scripts/check_namespace.sh` fails when a historical host appears in source,
tests, documentation, skills, or workflow files. It runs in the
`Namespace Consistency` GitHub Actions workflow for pushes and pull requests
to `main`.

The JSON-LD context unit tests also verify that every registered ontology
prefix resolves below `https://agent-os.org/ontology/`.
