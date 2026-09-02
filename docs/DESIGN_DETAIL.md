# Gliding Horse Agent OS — Design Detail

> *This document describes the core architectural design and key innovations of Gliding Horse Agent OS. For a concise project overview, see [README.md](../README.md).*

---

## 1. Generalized PDCA Orchestration: Beyond Traditional Management

### 1.1 What Makes It Different?

Traditional PDCA (Plan-Do-Check-Act) is a **management methodology** for process improvement. Gliding Horse Agent OS implements a **generalized computational PDCA** that transcends management and becomes a **universal task execution model** adaptable to any complexity level.

```mermaid
graph TB
    subgraph "Traditional PDCA"
        T1["Linear cycle<br/>Human-driven"]
        T2["Fixed phases"]
        T3["Manual handoffs"]
    end
    
    subgraph "Gliding Horse Generalized PDCA"
        G1["Recursive cycles<br/>Agent-autonomous"]
        G2["Adaptive phases<br/>Emergency/Exploratory modes"]
        G3["Event-driven transitions<br/>Perception-triggered"]
        G4["7 complexity levels<br/>L0 instant → L3 recursive"]
    end
    
    T1 -->|Evolution| G1
    T2 -->|Evolution| G2
    T3 -->|Evolution| G3
```

### 1.2 Seven Task Complexity Levels

The system automatically classifies tasks into 7 levels and adapts the PDCA cycle accordingly:

| Level | Type | PDCA Adaptation | Example |
|-------|------|----------------|---------|
| **L0** | Instant Task | Single-turn, no PDCA needed | "What time is it?" |
| **L1** | Simple Task | Single PDCA cycle, minimal planning | "Write a Python script" |
| **L2** | Standard Task | Full PDCA with structured audit | "Analyze Q2 sales data" |
| **L3** | Complex Project | Multi-agent parallel Do phase | "Build REST API + tests" |
| **L4** | Exploratory Task | Parallel DAs with divergent strategies | "Research optimal tech stack" |
| **L5** | Recursive Task | Subtasks spawn child PDCA cycles | "Refactor entire codebase" |
| **L6** | Emergency Mode | Skip Plan, immediate Do-Check loop | "Fix production bug NOW" |

**Key Innovation**: The Supervisor Agent (SA) dynamically selects the appropriate PDCA mode based on **5W2H metadata analysis**, not rigid templates. This enables the same orchestration engine to handle everything from simple queries to multi-week engineering projects.

### 1.3 Adaptive Cycle Modes

```mermaid
stateDiagram-v2
    [*] --> EMERGENCY: Critical failure detected
    [*] --> EXPLORATORY: High uncertainty
    [*] --> STANDARD: Normal complexity
    
    EMERGENCY --> DO: Skip planning
    DO --> CHECK: Immediate validation
    CHECK --> ACT: Decision
    ACT --> [*]: Resolved or escalate
    
    EXPLORATORY --> PARALLEL_DO: Spawn multiple DAs
    PARALLEL_DO --> CONVERGE: Compare results
    CONVERGE --> CHECK: Validate best outcome
    CHECK --> ACT: Select strategy
    ACT --> [*]
    
    STANDARD --> PLAN: Full planning
    PLAN --> DO: Sequential execution
    DO --> CHECK: Structured audit
    CHECK --> ACT: Archive or iterate
    ACT --> [*]
```

---

## 2. Four-Layer Memory Architecture: CPU Cache Philosophy Applied to AI

### 2.1 Revolutionary Design Inspired by Computer Architecture

Unlike conventional agent frameworks with flat context windows, Gliding Horse implements a **four-layer hierarchical memory system** directly inspired by CPU cache hierarchies (L1/L2/L3 caches + disk storage).

```mermaid
graph TB
    subgraph L1["L1: Context Window<br/>(~8KB Token Budget)<br/>━━━━━━━━━━━━━━━<br/>Speed: Instant<br/>Capacity: ~20 summaries + IRI pointers"]
        L1A["Compressed summaries"]
        L1B["IRI references to L2"]
        L1C["Active skill fragments"]
    end
    
    subgraph L2["L2: Working Blackboard<br/>(Oxigraph In-Memory RDF)<br/>━━━━━━━━━━━━━━━<br/>Speed: ~2ms query<br/>Capacity: Task graph + hypotheses"]
        L2A["Task tree structure"]
        L2B["Intermediate results"]
        L2C["Permission matrix"]
        L2D["MESI coherence states"]
    end
    
    subgraph L3["L3: Projection Engine<br/>(SPARQL CONSTRUCT + Frame)<br/>━━━━━━━━━━━━━━━<br/>Speed: ~15ms projection<br/>Capacity: On-demand subgraphs"]
        L3A["8 predefined frame templates"]
        L3B["Dynamic SPARQL queries"]
        L3C["Materialized view cache"]
    end
    
    subgraph L0["L0: Persistent Store<br/>(redb KV + HyperspaceEngine)<br/>━━━━━━━━━━━━━━━<br/>Speed: ~1ms read<br/>Capacity: Unlimited (disk-backed)"]
        L0A["Full conversation history"]
        L0B["Vector embeddings"]
        L0C["Experience archives"]
        L0D["Skill graph (full)"]
    end
    
    L1 <-->|"Token-aware eviction<br/>Semantic LRU"| L2
    L2 <-->|"Frame-driven projection<br/>SPARQL CONSTRUCT"| L3
    L3 <-->|"Write-back policy<br/>Batch persistence"| L0
```

### 2.2 MESI Cache Coherence Protocol for Distributed Agents

**Innovation**: First application of CPU cache coherence protocols (MESI: Modified, Exclusive, Shared, Invalid) to multi-agent memory systems.

| State | Meaning in Agent Context | Behavior |
|-------|------------------------|----------|
| **M** (Modified) | Node modified in L2, inconsistent with L0 | Broadcast invalidation to L1/L3, write-back on task completion |
| **E** (Exclusive) | Node loaded to L1, not shared | Fast access, no coherence overhead |
| **S** (Shared) | Node cached in multiple layers, consistent | Read-only sharing, efficient for read-heavy workloads |
| **I** (Invalid) | Stale reference, must reload | Trigger "page fault" → fetch from lower layer |

**Consistency Engine Workflow**:
1. DA modifies a node in L2 blackboard → state becomes **M**
2. Consistency Engine sends `Invalidate(IRI)` to L1 → summary marked **I**
3. L3 receives invalidation → materialized view removed
4. Next access triggers reload from L0 with updated data

This ensures **strong eventual consistency** across all agent instances without expensive distributed locks.

### 2.3 Intelligent Prefetching: Diffusion Activation Algorithm

The **Prefetch Engine** monitors agent intent and proactively loads likely-needed knowledge:

```mermaid
sequenceDiagram
    participant Agent
    participant Prefetch as Prefetch Engine
    participant L3 as L3 Manager
    participant L0 as L0 Store
    
    Agent->>Prefetch: Intent detected: "plan travel"
    Prefetch->>Prefetch: Diffusion activation:<br/>travel → flights, hotels, attractions
    Prefetch->>L3: Async SPARQL query for related subgraph
    L3->>L0: Fetch memories tagged "flights", "hotels"
    L0-->>L3: Return relevant blocks + embeddings
    L3-->>Prefetch: Pre-fetched subgraph ready
    Prefetch->>L2: Load to prefetch buffer
    Note over L2: Agent hits cache on next access<br/>Latency: ~2ms vs ~50ms cold load
```

**Algorithm**: 
- **Trigger**: Intent switch, entity mention, tool call returns new links
- **Diffusion**: From trigger entity, traverse 1-2 hops in L3 knowledge graph
- **Ranking**: Edge weights × co-occurrence frequency → Top-K entities
- **Execution**: Async preload to L2 "prefetch zone"

Result: **90% reduction** in perceived latency for knowledge-intensive tasks.

---

## 3. JSON-LD Semantic Data Bus: Universal Interoperability Layer

### 3.1 Why JSON-LD, Not Just JSON?

Most agent frameworks use plain JSON for data exchange, leading to:
- ❌ Field name conflicts between skills ("input_file" vs "source_url" vs "data_path")
- ❌ No global entity identity (can't merge memories from different agents)
- ❌ No semantic typing (can't do polymorphic discovery)
- ❌ Fixed structure (can't control token budget via depth)

Gliding Horse uses **JSON-LD 1.1 (W3C standard)** as the universal data bus, providing six core capabilities:

```mermaid
graph LR
    subgraph "JSON-LD Six Core Features"
        F1["@context<br/>Field→IRI mapping"]
        F2["@id<br/>Global entity ID"]
        F3["@type<br/>Multi-type inheritance"]
        F4["Nesting vs IRI<br/>Depth control"]
        F5["@graph<br/>Named graphs"]
        F6["Frame<br/>Shape projection"]
    end
    
    subgraph "Architectural Value"
        V1["Duck-typing<br/>Zero-cost integration"]
        V2["Auto-merge<br/>Cross-agent alignment"]
        V3["Polymorphic discovery<br/>SPARQL matching"]
        V4["Token budget<br/>Physical control"]
        V5["Conflict-free parallel<br/>Precise provenance"]
        V6["On-demand projection<br/>Context economy"]
    end
    
    F1 --> V1
    F2 --> V2
    F3 --> V3
    F4 --> V4
    F5 --> V5
    F6 --> V6
```

### 3.2 @context: Duck-Typing for Skills

Different developers write skills with different parameter names. JSON-LD `@context` maps all variants to unified IRIs:

```json
{
  "@context": {
    "skill": "https://agent-os.org/ontology/skill#",
    "skill:inputMapping": {
      "file_path": { "@id": "skill:sourceDataURI" },
      "source_url": { "@id": "skill:sourceDataURI" },
      "data_path": { "@id": "skill:sourceDataURI" }
    }
  }
}
```

Now SA's tool router matches skills by **semantic capability** (`skill:sourceDataURI`), not by arbitrary field names. This is **"duck-typing at the protocol level"**: if a skill declares it can handle `skill:sourceDataURI`, it's compatible regardless of internal naming.

### 3.3 @id: Cross-Agent Entity Alignment

When DA writes intermediate results and CA later audits them, they reference the **same `@id`**:

```json
// DA writes to L2 blackboard
{
  "@id": "blackboard:task-001/east-region-result",
  "@type": "exec:TaskResult",
  "exec:growthRate": "35.2",
  "exec:producedBy": { "@id": "agent:da/inst-003" }
}

// CA queries by same @id (no explicit passing needed)
SELECT ?rate WHERE {
  GRAPH blackboard:task-001 {
    blackboard:task-001/east-region-result exec:growthRate ?rate .
  }
}
```

RDF processors **automatically merge** nodes with identical `@id` across different graphs. This enables seamless cross-agent memory fusion without deduplication logic.

### 3.4 @type: Polymorphic Discovery

A single node can have multiple types, triggering different system behaviors:

```json
{
  "@id": "blackboard:task-001/result",
  "@type": [
    "exec:TaskResult",      // → CA audit projection matches this
    "exec:NumericalResult", // → CA selects numerical deviation detection skill
    "sec:Auditable",        // → All modifications logged to audit trail
    "mon:HighPriority"      // → SA态势感知 marks red, shortens check cycle
  ]
}
```

**SPARQL polymorphic query**:
```sparql
SELECT ?skill WHERE {
  ?skill a ?skillType .
  FILTER(?skillType IN (skill:NumericalProcessor, skill:TabularProcessor))
}
```

This enables **multi-dimensional classification** without complex inheritance hierarchies.

### 3.5 Nesting vs IRI Reference: Physical Token Budget Control

Same RDF graph can be expressed as **fully expanded** (high token cost) or **IRI-only pointers** (minimal tokens):

```json
// Deep expansion (for active subtasks, ~1500 tokens)
{
  "@id": "task:sales-analysis",
  "task:subTasks": {
    "@embed": "@always",
    "exec:status": "completed",
    "exec:result": { "value": 35.2 }
  }
}

// Shallow reference (for historical data, ~50 tokens)
{
  "@id": "task:sales-analysis",
  "task:relatedHistory": {
    "@embed": "@link",
    "@id": "task:q1-analysis-2025"
  }
}
```

**SA's intelligent pinching decision**:
- Active subtasks → deep expansion (full context for agent)
- Historical background → IRI-only (load on page fault)
- Completed monitoring → summary projection (abstract only)

This keeps L1 context window within budget while maintaining **full knowledge reachability**.

### 3.6 @graph Named Graphs: Conflict-Free Parallel Writes

Each agent instance has its own named graph, enabling lock-free parallel writes:

```mermaid
graph TB
    subgraph "L0 Persistent Named Graphs"
        G1["agent:da/inst-001<br/>DA private memory"]
        G2["agent:ca/inst-001<br/>CA private memory"]
        G3["system:knowledge<br/>Global KB"]
        G4["system:experience<br/>Pattern library"]
    end
    
    subgraph "L2 Blackboard Named Graphs"
        G5["blackboard:shared<br/>Public zone<br/>Multi-agent R/W"]
        G6["blackboard:task-001<br/>Task 1 private"]
        G7["blackboard:prefetch<br/>Prefetch buffer"]
    end
```

**Access permission matrix**:

| Graph Name | SA | PA | DA | CA | AA |
|-----------|-----|-----|-----|-----|-----|
| `blackboard:shared` | RW | R | RW | RW | R |
| `blackboard:task-{id}` | RW | R | RW | R | R |
| `agent:{id}` | R | — | — | — | — |
| `system:audit-log` | R | — | — | — | — |

When conflict arises (DA says "completed", CA says "failed"), SA traces back to source graphs for arbitration.

### 3.7 JSON-LD Framing: On-Demand Projection

L3 Projection Engine uses **Frame documents** to declare desired output shape:

```json
{
  "@context": { "exec": "https://agent-os.org/ontology/exec#" },
  "@type": "task:AnalysisTask",
  "task:subTasks": {
    "@embed": "@always",           // Expand fully
    "exec:assignedTo": { "@embed": "@link" }  // IRI only
  },
  "task:relatedHistory": {
    "@embed": "@link"              // History as pointers
  }
}
```

**Five-level progressive disclosure**:

| Level | Content | Tokens | User |
|-------|---------|--------|------|
| **L1** | MOC index scan (name + count) | ~200 | SA initial analysis |
| **L2** | Skill 5W2H summary (what/why/when) | ~500 | SA skill matching |
| **L3** | Link relationships (prerequisites) | ~800 | SA/PA chain discovery |
| **L4** | Schema + steps list | ~1500 | DA tool invocation |
| **L5** | Full content (code + validation) | On-demand | DA execution / CA audit |

This ensures each agent sees **exactly what it needs, nothing more**.

### 3.8 Simplified JSON-LD Usage: Bridging LLM and Knowledge Graph

**The Challenge**: LLMs are not proficient at generating complex JSON-LD structures. They excel at producing natural language and simple JSON objects.

**Our Solution**: A hybrid approach that leverages the strengths of both paradigms:

```mermaid
flowchart TB
    subgraph "LLM Output (Simple JSON)"
        LLM_OUT["{<br/>  'think': 'Planning...',<br/>  'content': 'CREATE TABLE...',<br/>  'summary': 'Schema designed'<br/>}"]
    end
    
    subgraph "L2 Blackboard Processing"
        L2B["AgentRunner / L2 Blackboard<br/>━━━━━━━━<br/>1. Validate against JSON Schema<br/>2. Convert to JSON-LD node<br/>3. Assign @id<br/>4. Write to L2 blackboard"]
    end
    
    subgraph "Storage Layers"
        L2_MEM["L2 Oxigraph Memory<br/>━━━━━━━━<br/>In-memory RDF<br/>Fast queries ~2ms"]
        L0_STORE["L0 Persistent Store<br/>━━━━━━━━<br/>redb KV + HyperspaceEngine<br/>Unlimited capacity"]
    end
    
    LLM_OUT --> HARNESS
    HARNESS --> L2_MEM
    L2_MEM -->|Batch write-back| L0_STORE
```

**LLM Response Structure** (Optimized for Multi-Turn Conversations):

```json
{
  "think": "Analyzing user request for database schema design...",
  "content": "CREATE TABLE users (id UUID PRIMARY KEY, email VARCHAR(255) UNIQUE NOT NULL);",
  "summary": "Database schema for user table with UUID primary key and unique email constraint"
}
```

**Why This Three-Field Structure?**

| Field | Purpose | Token Efficiency |
|-------|---------|-----------------|
| **think** | Chain-of-thought reasoning (discarded after turn) | Temporary, not archived |
| **content** | Full detailed output (archived to L0 for traceability) | Complete fidelity |
| **summary** | Concise abstract (kept in L1 context window) | ~90% token savings vs full content |

**Multi-Turn Conversation Optimization**:

```
Turn 1: User asks for schema design
  → LLM produces think/content/summary
  → summary appended to L1 context (~50 tokens)
  → content archived to L0 with @id: "memory:session-001/block-042"

Turn 2: User asks "What tables did we create?"
  → L1 context contains summary: "Database schema for user table..."
  → If details needed, Harness resolves IRI "memory:session-001/block-042" from L0
  → Result: L1 stays small, no information loss
```

**AgentRunner & L2 Blackboard Role**:

The AgentRunner (via L2 Blackboard) acts as the **translation layer** between:
- **LLM's comfort zone**: Simple JSON with think/content/summary
- **System's requirements**: JSON-LD with @id, @type, @context for interoperability

Processing pipeline:
```rust
// Pseudo-code illustrating the transformation
let llm_output = llm_client.generate(prompt).await?; // Returns simple JSON

// Step 1: Validate against JSON Schema
validation_engine.validate(&llm_output.content, &skill.input_schema)?;

// Step 2: Convert to JSON-LD node
let jsonld_node = json!({
    "@id": format!("memory:{}/block-{}", session_id, block_counter),
    "@type": ["mem:MemoryBlock", "exec:TaskResult"],
    "mem:content": llm_output.content,
    "mem:summary": llm_output.summary,
    "mem:embedding": embedding_service.index(&llm_output.content).await?
});

// Step 3: Write to L2 blackboard (Oxigraph in-memory)
l2_manager.insert_node(&jsonld_node)?;

// Step 4: Schedule batch write-back to L0
scheduler.schedule_writeback(session_id, block_counter);
```

This design achieves:
- ✅ **Performance**: L2 in-memory queries at ~2ms latency
- ✅ **Scalability**: L0 disk-backed storage with unlimited capacity
- ✅ **Token Economy**: Summary-based L1 context keeps token usage minimal
- ✅ **Traceability**: Full content preserved in L0 with IRI references
- ✅ **Interoperability**: JSON-LD enables cross-agent data sharing

---

## 4. 5W2H Task Ontology: Structured Intent Modeling

### 4.1 Why 5W2H: The Universal Task Ontology

**The Foundation of All Structured Thinking**

Gliding Horse Agent OS is built on **two universal frameworks** that are essential for handling any task:

1. **5W2H (What, Why, Who, When, Where, How, How Much)** - The **Task Ontology**
   - Answers: "What exactly needs to be done?"
   - Purpose: Clarifies intent, constraints, and success criteria
   - Timing: Applied at **task initialization** phase

2. **PDCA Cycle (Plan-Do-Check-Act)** - The **Execution Model**
   - Answers: "How do we systematically execute and improve?"
   - Purpose: Provides iterative execution with continuous feedback
   - Timing: Applied throughout **task lifecycle**

```mermaid
graph TB
    subgraph "Universal Frameworks (Always Required)"
        U1["5W2H<br/>━━━━━━━━<br/>Task Ontology<br/>Clarifies WHAT to do"]
        U2["PDCA Cycle<br/>━━━━━━━━<br/>Execution Model<br/>Defines HOW to execute"]
    end
    
    subgraph "Specialized Models (Skill Extensions)"
        S1["SWOT Analysis<br/>Strategic positioning"]
        S2["5 Whys<br/>Root cause analysis"]
        S3["SMART Goals<br/>Goal refinement"]
        S4["Kanban<br/>Workflow visualization"]
    end
    
    U1 -->|Foundation| TASK[Executable Task]
    U2 -->|Process| TASK
    S1 -.->|Optional Skill| TASK
    S2 -.->|Optional Skill| TASK
    S3 -.->|Optional Skill| TASK
    S4 -.->|Optional Skill| TASK
    
    style U1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style U2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style S1 fill:#fff3e0,stroke:#f57c00,stroke-dasharray: 5 5
    style S2 fill:#fff3e0,stroke:#f57c00,stroke-dasharray: 5 5
    style S3 fill:#fff3e0,stroke:#f57c00,stroke-dasharray: 5 5
    style S4 fill:#fff3e0,stroke:#f57c00,stroke-dasharray: 5 5
```

**Why Both Are Irreplaceable:**

```
Any Executable Task = 5W2H (Intent Clarity) + PDCA (Systematic Execution)
```

| Framework | Role | Without It... |
|-----------|------|---------------|
| **5W2H** | Defines **WHAT** needs to be done | Ambiguous goals → misaligned expectations |
| **PDCA** | Defines **HOW** to execute iteratively | Chaotic implementation → no quality control |

**The Complete Workflow:**

```mermaid
sequenceDiagram
    participant User
    participant SA as SupervisorAgent
    participant PA as PlanAgent
    participant DA as DoAgent
    participant CA as CheckAgent
    participant AA as ActAgent
    
    User->>SA: Submit task request
    Note over SA: Step 1: Extract 5W2H<br/>(What/Why/Who/When/Where/How/HowMuch)
    
    SA->>PA: Execute PLAN phase
    PA->>PA: Generate micro-flow DAG
    PA-->>SA: Return execution plan
    
    SA->>DA: Execute DO phase
    DA->>DA: Call tools, write artifacts
    DA-->>SA: Return implementation
    
    SA->>CA: Execute CHECK phase
    CA->>CA: Audit by 5W2H dimensions
    CA-->>SA: Return audit verdict
    
    SA->>AA: Execute ACT phase
    AA->>AA: Decide: Pass/Rollback/Terminate
    AA-->>SA: Final decision
    
    SA-->>User: Deliver result + archive
```

### 4.2 Beyond Free-Text Prompts

Traditional agents accept unstructured prompts, leading to ambiguous goals and unauditable execution. Gliding Horse introduces **5W2H task ontology** as the standardized metadata framework for all non-trivial tasks.

```mermaid
graph TB
    WHAT["What: Core objective<br/>━━━━━━━━<br/>Required at creation"] --> WHY
    WHY["Why: Intent & success criteria<br/>━━━━━━━━<br/>Required at creation<br/>Sub: priority, criteria"] --> WHO
    WHY --> WHEN
    WHAT --> HOW
    HOW["How: Methods & steps<br/>━━━━━━━━<br/>Filled by PA<br/>Sub: planIRI, skills, dependencies"] --> WHERE
    HOW --> HOWMUCH
    WHO["Who: Stakeholders & roles<br/>━━━━━━━━<br/>Filled by SA/PA<br/>Sub: requestor, assignees, requiredRole"] --> HOWMUCH
    WHEN["When: Time constraints<br/>━━━━━━━━<br/>Filled by user/SA/PA<br/>Sub: deadline, duration, timezone"] --> HOWMUCH
    WHERE["Where: Data sources & environment<br/>━━━━━━━━<br/>Filled by PA/DA<br/>Sub: repos, branches, env"] --> HOWMUCH
    HOWMUCH["How Much: Resource budget<br/>━━━━━━━━<br/>Filled by SA/PA/CA<br/>Sub: tokenBudget, maxCycles, quality"]
    
    HOWMUCH -.-> CA_AUDIT[CA audit basis]
    WHEN -.-> CA_AUDIT
    WHAT -.-> CA_AUDIT
    WHY -.-> CA_AUDIT
```

### 4.3 Progressive Filling Lifecycle

Each dimension has a `fillStage` attribute marking when it should be populated:

```mermaid
stateDiagram-v2
    CREATE: Task Creation
    PLAN: PA Planning
    DO: DA Execution
    CHECK: CA Audit
    ACT: AA Decision
    DONE: Archive to L0
    
    CREATE --> PLAN: SA scheduling
    PLAN --> DO: PA outputs plan
    DO --> CHECK: DA completes
    CHECK --> ACT: CA outputs audit
    ACT --> PLAN: Rollback replan
    ACT --> DONE: Pass
    
    note right of CREATE: Fill What / Why / partial Who & When
    note right of PLAN: Fill How / Where / complete When & Who
    note right of DO: Fill Where details / preliminary HowMuch
    note right of CHECK: Fill HowMuch actual values / verify all dimensions
    note right of DONE: Freeze full 5W2H archive to L0
```

**Example lifecycle**:

```json
// Stage 1: Creation (SA extracts minimal set)
{
  "@id": "task:sales-q2-analysis",
  "task:5W2H": {
    "what": "Analyze Q2 regional sales data and generate forecast report",
    "why": {
      "description": "Provide basis for inventory planning",
      "successCriteria": ["Output visualization with regional growth comparison and forecast"],
      "priority": "high"
    },
    "who": { "requestor": "user:vp-sales", "requiredRole": "agent:Do" },
    "when": { "deadline": "2026-05-20T18:00:00+08:00" }
  }
}

// Stage 2: Planning (PA completes How/Where)
{
  "task:5W2H": {
    "where": {
      "dataSources": ["file://data/sales_q2.csv", "db://crm/deals"],
      "executionEnvironment": "sandbox"
    },
    "how": {
      "planIRI": "plan:task-tree/sales-q2",
      "preferredSkills": ["skill:python-analysis", "skill:forecasting"],
      "requiredSteps": "1. Data cleaning → 2. Regional grouping → 3. Forecast modeling → 4. Report generation"
    }
  }
}

// Stage 3: Audit (CA fills actual HowMuch)
{
  "task:5W2H": {
    "howMuch": {
      "tokenBudget": 5000,
      "actualCost": 5600,
      "maxPDCACycles": 3,
      "actualCycles": 2
    }
  }
}
```

### 4.4 Dimension-Level Structured Audit

CA doesn't just say "PASS/FAIL". It audits **each 5W2H dimension independently**:

```json
{
  "auditBy5W2H": {
    "what": { "verdict": "PASS", "evidence": "Report generated with regional comparison and forecast" },
    "why": { "verdict": "PASS", "evidence": "Conclusions directly usable for inventory planning" },
    "when": { "verdict": "PASS", "evidence": "Delivered at 5/19 14:00, before deadline" },
    "where": { "verdict": "PASS", "evidence": "Data sources matched, sandbox environment secure" },
    "how": { "verdict": "PASS", "evidence": "All four steps completed as planned" },
    "howMuch": { "verdict": "WARNING", "evidence": "Token exceeded by 12%, but result quality high" }
  },
  "overallVerdict": "CONDITIONAL_PASS"
}
```

AA then makes dimension-aware decisions:
- What/Why FAIL → Rollback to SA for re-analysis
- How/Where FAIL → Rollback to PA for plan correction
- When/HowMuch FAIL → If justified, pass; otherwise degrade or terminate

### 4.5 Pattern Recognition: 5W2H-Driven Experience Reuse

L0 stores all completed tasks as frozen `task:CompletedTaskSnapshot`. SA's pattern recognition organ queries for similar experiences:

```sparql
PREFIX task: <https://agent-os.org/ontology/task#>

SELECT ?pastTask ?whySimilarity ?howSimilarity
WHERE {
  GRAPH system:experience {
    ?pastTask a task:CompletedTaskSnapshot .
    ?pastTask task:5W2H/task:why ?pastWhy .
    ?pastTask task:5W2H/task:how/task:planIRI ?pastPlan .
    BIND(external:cosineSimilarity(?currentWhyVec, ?pastWhyVec) AS ?whySimilarity)
  }
  FILTER(?whySimilarity > 0.85)
}
ORDER BY DESC(?whySimilarity)
LIMIT 5
```

Matched historical 5W2H subgraphs are injected into SA decision context:
- Recommend same `task:how/preferredSkills`
- Warn about historical `task:where` pitfalls (e.g., unstable branch)
- Provide historical `task:howMuch/actualCost` as budget reference

---

## 5. Skill Graph: Cognitive Knowledge Network with Automatic Evolution

### 5.1 Beyond Static Skill Libraries

Traditional agent frameworks treat skills as static function libraries. Gliding Horse implements a **dynamic cognitive knowledge network** where skills evolve through usage, gain experience fragments, and self-organize via semantic links.

```mermaid
graph TB
    BASE["skill:Skill<br/>━━━━━━━━<br/>Base class with 5W2H metadata<br/>+ Schema + Signature"]

    BASE --> ATOMIC["skill:AtomicSkill<br/>━━━━━━━━<br/>Indivisible atomic skill<br/>With clear entrypoint"]
    BASE --> COMPOSITE["skill:CompositeSkill<br/>━━━━━━━━<br/>Composite skill<br/>Links to sub-skills"]
    BASE --> MOC["skill:MOC<br/>━━━━━━━━<br/>Map of Content navigation node<br/>Pure navigation, no entrypoint"]
    BASE --> FRAGMENT["skill:KnowledgeFragment<br/>━━━━━━━━<br/>Experience knowledge fragment<br/>Attached to specific Skill"]
    BASE --> MCP_WRAPPER["skill:MCPTool<br/>━━━━━━━━<br/>MCP tool wrapper<br/>Bridges external MCP ecosystem"]

    style BASE fill:#e3f2fd
    style ATOMIC fill:#e8f5e9
    style COMPOSITE fill:#fff3e0
    style MOC fill:#f3e5f5
    style FRAGMENT fill:#ffebee
    style MCP_WRAPPER fill:#e0f2f1
```

### 5.2 Six Semantic Link Types

Skills connect via six relationship types, each triggering different SA reasoning behaviors:

| Link Type | SA Reasoning Behavior | Example |
|-----------|----------------------|---------|
| `PrerequisiteLink` | Auto-include Skill B when selecting A | JWT auth → auto-load Rust basics |
| `CompositionLink` | Recursively expand sub-skills / MOC navigation | MOC auth domain → expand JWT/OAuth2/Token |
| `RelatedLink` | Recommend B after completing A | Complete JWT implementation → suggest middleware integration |
| `AlternativeLink` | Auto-switch to B if A unavailable | Rust env unavailable → switch to Node.js version |
| `ExtendsLink` | Choose A for basic, B for advanced | Basic JWT → OAuth2 full authorization |
| `GeneralizationLink` | Map specific tasks to general templates | Sales forecast → time series forecasting |

**SPARQL property path recursion** discovers dependency chains up to 3 levels deep:

```sparql
?target (skill:links/skill:target){0,3} ?chainNode .
```

### 5.3 Automatic Evolution Driven by AA

After each task completion, AA analyzes execution trajectory and evolves the Skill Graph:

```mermaid
flowchart TD
    TASK_DONE[Task Completed] --> AA_ANALYZE[AA Analyzes Execution Trace]

    AA_ANALYZE --> NEW_FAIL{New<br/>Failure Mode?}
    NEW_FAIL -->|Yes| WRITE_FRAG[Create KnowledgeFragment<br/>Attach to corresponding Skill]

    AA_ANALYZE --> NEW_LINK{New<br/>Skill Association?}
    NEW_LINK -->|Yes| WRITE_LINK[Create RelatedLink<br/>Connect two skills]

    AA_ANALYZE --> MATURITY{Maturity<br/>Needs Adjustment?}
    MATURITY -->|Yes| UPDATE_MAT[Update skill:maturity]

    AA_ANALYZE --> STATS{Update Statistics}
    STATS --> UPDATE_STATS[Update graphMeta:<br/>usageCount / successRate]

    WRITE_FRAG --> UPDATE_STATS
    WRITE_LINK --> UPDATE_STATS
    UPDATE_MAT --> UPDATE_STATS
    UPDATE_STATS --> DONE[Archive to L0]
```

**Example**: CA discovers that JWT key rotation causes mass user logout. AA creates a KnowledgeFragment:

```json
{
  "@id": "skill:fragment/jwt-key-rotation-pitfall",
  "@type": "skill:KnowledgeFragment",
  "schema:name": "JWT Key Rotation Pitfall",
  "skill:attachedTo": "skill:rust-jwt-auth",
  "skill:content": {
    "problem": "Directly replacing old key during rotation invalidates all issued tokens",
    "recommendation": "Use JWKS endpoint to publish multiple public keys simultaneously for graceful transition",
    "alternativeSkill": "skill:jwks-implementation"
  }
}
```

Future SA executions encountering JWT tasks will see this fragment and recommend JWKS approach.

### 5.4 Self-Bootstrapping: /learn and /reduce Mechanisms

When DA encounters a problem with no available skill:

```mermaid
sequenceDiagram
    participant DA
    participant SA
    participant L0
    participant Harness

    DA->>Harness: Report: No skill available for current problem
    Harness->>SA: Notify: New skill needed

    rect rgb(230,240,255)
        Note over SA: /learn Phase
        SA->>SA: Analyze problem characteristics
        SA->>SA: Generate 5W2H draft
        SA->>L0: Create Skill node (status: draft)
        SA->>L0: Establish link to relevant MOC
    end

    DA->>DA: Continue solving (without skill guidance)
    DA->>Harness: Return solution
    Harness->>L0: Record to temporary experience node

    rect rgb(255,240,230)
        Note over SA: /reduce Phase
        SA->>L0: Extract solution
        SA->>L0: Populate Skill content/steps
        SA->>L0: Update status: active
        SA->>L0: Compute Ed25519 signature
    end

    SA-->>DA: New skill ready
```

This enables **autonomous skill acquisition** without human intervention.

### 5.5 Advanced Skill Graph Features

The Skill Graph has evolved beyond the six link types and automatic evolution described above, adding four advanced subsystems:

**Hypergraph Composition** (`src/skill_graph/types.rs`): First-class `Hyperedge` type with `CompositionType` enum supporting Sequential, Parallel, Conditional, Optional, and Fallback compositions — enabling complex multi-skill workflows as first-class graph nodes.

**Poincaré Structural Embeddings** (`src/skill_graph/embedding.rs`): Computes geometric embeddings from graph topology (prerequisite depth + tag fingerprinting), enabling Poincaré ball-based similarity search and structural clustering of skills.

**Graph Algorithms** (`src/skill_graph/graph_algorithms.rs`): PageRank for skill importance ranking, betweenness centrality for bottleneck detection, label-propagation community detection for automatic skill clustering, DFS prerequisite chain discovery, and Tarjan SCC for cycle detection in the skill dependency graph.

**Formal Invariant Verification** (`src/skill_graph/verification.rs`): Six invariant checks enforcing graph integrity — acyclicity (no circular dependencies), link existence (no dangling references), composite reachability (all sub-skills accessible), no deprecated prerequisites, valid 5W2H metadata, and valid security levels. Violations are flagged before graph mutations are committed.

**Temporal Versioning**: Snapshot and rollback support, allowing the Skill Graph to be checkpointed before mutations and reverted if verification fails.

**Oxigraph SPARQL Bridge**: Real-time bidirectional sync between `SkillGraphStore` and the unified Oxigraph RDF store via SPARQL INSERT/DELETE operations and named graph isolation (`system:skills`).

---

## 6. Proactive Perception Engine: Anomaly Detection & Intelligent Intervention

### 6.1 Ten Perception Triggers

The ProactiveEngine monitors execution through ten distinct triggers, each mapped to specific intervention plans:

```mermaid
graph TB
    subgraph "10 Perception Triggers"
        T1["TaskStart: Complexity analysis"]
        T2["PlanCompleted: Subtask limit check"]
        T3["ProgressAnomaly: Deduplication window"]
        T4["CheckCompleted: Verdict-based alerts"]
        T5["TaskEnd: Experience extraction"]
        T6["CycleTimeout: Elapsed time monitoring"]
        T7["AgentBlocked: Health detection"]
        T8["ResourceConflict: Queue/delay analysis"]
        T9["QualityDegradation: Rollback signals"]
        T10["UserFeedback: Feedback log"]
    end
    
    subgraph "Intervention Plans"
        A1["Reassess current plan"]
        A2["Restart blocked agent"]
        A3["Adjust resource allocation"]
        A4["Notify human supervisor"]
    end
    
    T1 & T2 & T3 & T4 & T5 & T6 & T7 & T8 & T9 & T10 -->|"Detect Anomaly"| A1 & A2 & A3 & A4
```

### 6.2 Anomaly Deduplication

Time-window based filtering prevents alert storms:

```yaml
perception:
  anomaly_dedup_window_seconds: 60  # Suppress duplicate alerts within 60s
  simple_input_threshold: 50         # Input < 50 chars → simple task
  medium_input_threshold: 200        # Input < 200 chars → medium complexity
  cycle_timeout_secs: 300            # Alert if cycle exceeds 5 minutes
  max_iterations_before_alert: 10    # Alert after 10 iterations without progress
  error_rate_threshold: 0.5          # Alert if > 50% tool calls fail
```

### 6.3 5W2H Constraint Checking

ProactiveEngine validates execution against 5W2H constraints:

- **Deadline violation**: Current time > `task:when/deadline` → escalate to human
- **Budget overrun**: Token consumption > `task:howMuch/tokenBudget × 0.8` → warn SA
- **Role mismatch**: Assigned agent role ≠ `task:who/requiredRole` → reassign
- **Environment conflict**: Two tasks modifying same repo/branch → serialize execution

---

## 7. Advanced Tool Execution Framework

### 7.1 Built-in Tools (20+) with Micro-Tool System

| Category | Tools | Innovation |
|----------|-------|-----------|
| **File Operations** | `file_read`, `file_write`, `file_edit`, `file_list`, `glob_search`, `grep_search` | Symlink detection, path traversal prevention |
| **Network** | `WebFetch`, `WebSearch` (DuckDuckGo fallback chain) | TLS enforcement, proxy support |
| **Execution** | `Bash`, `PowerShell` (timeout; optional namespace sandbox) | Bounded output and configurable isolation |
| **Knowledge Import** | `knowledge_import_json`, `knowledge_import_url`, `knowledge_import_directory` | Auto-graphification to RDF |
| **Knowledge Graph** | `knowledge_extract`, `knowledge_query`, `kg_search`, `kg_neighbors`, `knowledge_extract_code` | SPARQL queries, AST parsing |
| **Skill Management** | `create_skill`, `convert_skill`, `list_skills` | LLM-powered skill generation |
| **Ontology** | `ontology_register`, `knowledge_bridge` | Cross-domain semantic alignment |

> Security boundary: shell namespace/filesystem isolation is opt-in and may be unavailable under host or container permission constraints. Every shell result reports `sandbox_status`; execution without an active sandbox must not be treated as safe for untrusted code. The deterministic hash embedding backend is likewise a degraded availability fallback, not a semantic embedding equivalent.

**Micro-Tool Innovation**: For large tool results (>8KB), system automatically generates conversational micro-tools:

```rust
// After file_read returns 50KB content
Micro-Tool: "search_in_results" 
Description: "Search within the previously read file content"
Parameters: { "query": "string", "context_lines": "number" }
```

This transforms unwieldy outputs into **interactive queryable artifacts**.

### 7.2 Model Context Protocol (MCP) Integration

External tool server integration via MCP standard:
- Connect to remote tool providers (GitHub, Slack, Jira, etc.)
- Dynamic tool discovery at runtime
- Secure authentication with API key rotation

---

## 8. Checkpoint & Recovery: Fault-Tolerant Execution

Session state persistence enables recovery from crashes:

```rust
// Create checkpoint at critical points
let checkpoint_id = checkpoint_manager.create(
    &task_iri,
    &format!("cycle:{}", cycle_id),
    &state_json,
    &metadata_json,
    &context_json,
    &artifacts
)?;

// Restore after crash
let restored_state = checkpoint_manager.restore(&task_iri)?;
```

**Use Cases**:
- Long-running task recovery (hours/days)
- Agent restart without losing context
- Debugging and replay for post-mortem analysis

---

## 9. Worker Task Queue: Background Job Processing

Persistent queue for asynchronous operations:

- **Technology**: yaque (Yet Another Queue) + bincode serialization
- **Features**: Disk-backed persistence, acknowledgment, peek operations
- **Use Cases**: 
  - Batch knowledge import (thousands of documents)
  - Scheduled skill evolution (nightly optimization)
  - Periodic cleanup (expired cache entries)
  - Asynchronous embedding generation

---

## 10. Template Engine & JSON Schema Validation

### 10.1 Markdown-Based Prompt Templates

```
## Role: {{agent_role}}
## Task: {{task_description}}

### Context
{{l3_projection}}

### Available Skills
{{skill_list}}

### 5W2H Constraints
- What: {{what}}
- Why: {{why}}
- When: {{deadline}}
- How Much: {{token_budget}}

### Instructions
...
```

**Features**:
- Recursive directory scanning
- Variable interpolation (`{placeholder}` syntax)
- Template inheritance (via includes)
- Version-controlled in Git

### 10.2 One Roundtrip, Double Harvest

Advanced validation mode that extracts both metadata and converts to JSON-LD in a single LLM call:

```
// LLM output
{
  "thought": "Planning database schema...",
  "content": "CREATE TABLE users...",
  "summary": "Database schema designed",
  "metadata": {
    "tables": ["users", "orders"],
    "relationships": ["one-to-many"]
  }
}

// System processes:
// 1. Validate metadata against JSON Schema
// 2. Convert validated metadata to JSON-LD node
// 3. Write to L2 blackboard with @id
// Result: Single LLM call → validated structured data + natural language
```

This doubles information extraction efficiency compared to traditional single-purpose prompts.

---

## 11. Architecture

### 11.1 System Components

```mermaid
graph TB
    subgraph Client["Client Applications"]
        PY["Python Orchestrator"]
        TS["TypeScript Frontend"]
        GO["Go Services"]
    end
    
    subgraph API["API Layer"]
        GRPC["gRPC Server<br/>tonic [::1]:50051"]
        HTTP["HTTP Edge Daemon<br/>axum :8080"]
    end
    
    subgraph Core["Core Coordination"]
        SA["SupervisorAgent<br/>PDCA orchestration"]
        AR["AgentRunner<br/>ReAct loop"]
        BA["BizAgent<br/>Isolated execution"]
        EB["EventBus<br/>Async dispatch"]
        CP["Checkpoint<br/>State persistence"]
        SG["SyscallGate<br/>Permission control"]
    end
    
    subgraph Memory["Memory System"]
        L0["L0: redb + HyperspaceEngine<br/>Persistent KV + Vectors"]
        L1["L1: Session<br/>Per-agent conversation"]
        L2["L2: Oxigraph<br/>Shared blackboard + RDF"]
        L3["L3: SPARQL CONSTRUCT<br/>Projection engine"]
        MM["MemoryManager<br/>Cross-layer coordination"]
    end
    
    subgraph Tools["Tool System"]
        TE["ToolExecutor<br/>25+ built-in tools"]
        SR["SkillRegistry<br/>Skill catalog"]
        MCP["MCP Client<br/>External tools"]
        KG["Knowledge Graph Tools<br/>Code AST, RDF, bridge"]
    end
    
    subgraph Infra["Infrastructure"]
        TMPL["TemplateEngine<br/>Prompt templates"]
        LLM["LLMClient<br/>OpenAI-compatible"]
        PE["ProactiveEngine<br/>Anomaly detection"]
        JL["JSON-LD Framing<br/>Context projection"]
        SKG["Skill Graph<br/>15 modules"]
        WQ["Worker TaskQueue<br/>yaque + bincode"]
    end
    
    Client --> GRPC
    Client --> HTTP
    GRPC --> Core
    HTTP --> Core
    Core --> Memory
    Core --> Tools
    Core --> Infra
    Tools --> Memory
```

### 11.2 Data Flow: The Modern Gliding Horse in Action

```mermaid
sequenceDiagram
    participant User
    participant SA as SupervisorAgent<br/>(The Strategist)
    participant PA as PlanAgent<br/>(The Navigator)
    participant DA as DoAgent<br/>(The Carrier)
    participant L2 as L2 Blackboard<br/>(The Supply Route)
    participant L0 as L0 Store<br/>(The Archive)
    participant TE as ToolExecutor<br/>(The Mechanism)
    
    User->>SA: Submit task ("Build a REST API")
    Note over SA: Like Zhuge Liang planning<br/>the Northern Expedition
    SA->>SA: Analyze complexity (5W2H)
    SA->>PA: Execute planning phase
    PA->>L2: Write execution plan (RDF node)
    PA-->>SA: Return strategic roadmap
    SA->>DA: Execute with plan
    Note over DA: Like the Gliding Horse<br/>carrying supplies uphill
    DA->>TE: Call tools (file_write, bash)
    TE->>L2: Auto-graphify results → RDF
    DA->>L2: Write code artifacts
    DA-->>SA: Return implementation
    SA->>L0: Archive task summary
    Note over L0: Preserving wisdom for<br/>future expeditions
    SA-->>User: Deliver final result
```

---

> *This document focuses on the architectural design and system innovations of Gliding Horse Agent OS. For a quick-start guide, application showcase, and project overview, see the [README.md](../README.md).*
