# Knowledge Graph Ingestion Pipeline — Architecture & Plan

## Why We're Building This

The current agent uses a chain of hard-coded fast-paths (`_OVERALL_INTENT_PATTERNS`, `_NIQ_METRIC_TERMS`, `_NIQ_PANEL_LEVEL_METRICS`, etc.) to handle common NIQ query patterns before they reach the LLM planner. This works for the queries it was written for, but it doesn't scale:

- Every new query type requires new Python code and regex
- Business knowledge (e.g. HERISTO = ANIMONDA + MJAMJAM, penetration uses MAX not AVG) is buried in constants
- Cross-file queries (MAT ↔ panel) are structurally impossible
- There is no way to ask the system what it knows or audit its reasoning

The fix is to make the system's knowledge **explicit, inspectable, and traversable** — a semantic knowledge graph that the decomposer queries deterministically, before the LLM is ever called.

---

## What the Graph Is

A **two-layer knowledge graph** stored in DuckDB adjacency tables (migratable to Kuzu/Neo4j later).

### Layer 1 — Schema layer (auto-generated at ingestion)
Encodes what exists in each file: columns, types, periods, distinct values, and structural relationships between them.

### Layer 2 — Concept layer (human-authored, versioned in git)
Encodes what things *mean* and how they relate across files: business concepts, math relationships, corporate hierarchy, entity groups, cross-file bridges.

---

## Node Types (7)

| Node | Description |
|---|---|
| `Concept` | Abstract business idea: Revenue, BuyerReach, MarketShare. Defined once, never changes when files change. |
| `Metric` | A concrete queryable column in a specific file. One node per column per file. |
| `Dimension` | A concrete grouping/filter column in a specific file. |
| `Entity` | A real-world value inside a dimension: ANIMONDA, WET, Germany. Not tied to files. |
| `EntityGroup` | A named, human-declared aggregation of entities: HERISTO, LEH+DM+Pet. |
| `File` | A loaded dataset: cat_mat, dog_mat, panel. Carries grain, period structure, format. |
| `Period` | A specific time window: `52 W bis 29/03/26`. Shared across files where comparable. |

---

## Edge Types (14)

| Edge | Direction | Auto/Human | Purpose |
|---|---|---|---|
| `MEASURES` | Metric → Concept | LLM-assisted + human review | Maps columns to business concepts |
| `AVAILABLE_IN` | Metric → File | Auto | Column lives in this file |
| `SLICES` | Dimension → File | Auto | Column used for grouping in this file |
| `PRIOR_PERIOD_OF` | Metric → Metric | Auto (VJ suffix) | Prior-year companion column |
| `DELTA_OF` | Metric → Metric | Auto (% Ver. suffix) | Change column for a metric |
| `DERIVED_FROM` | Metric → [Metric, Metric] | Human | Math relationships: SpendPerBuyer = Ausgaben / Käuferhaushalte |
| `CONTAINS` | Dimension → Entity | Auto | Distinct values in a dimension |
| `MEMBER_OF` | Entity → EntityGroup | Human | Corporate/competitive groupings |
| `SCOPED_TO` | EntityGroup → File | Human | Where an entity group is valid |
| `CHILD_OF` | Entity → Entity | Human | Corporate hierarchy |
| `LEVEL_IN` | Dimension → Hierarchy | Human | Category > AnimalType > Subcategory > ... |
| `BRIDGES` | Concept → Concept | Human | Cross-file economic equivalence (MAT Revenue ↔ panel Ausgaben) |
| `COVERS` | Period → File | Auto | Period is present in this file, with CY/PY role |
| `JOINABLE_ON` | File → File | Auto-proposed + human confirm | Join keys between files |

**Edge metadata:** Every edge carries `source` (ingest / declared) and `confidence` (float). The decomposer traverses `confidence ≥ 0.85` automatically; below that it triggers a confirmation step.

---

## The Six Query Patterns the Graph Must Answer

1. *"What columns measure this business concept?"* — `MEASURES` traversal
2. *"What does this entity group expand to in this file?"* — `MEMBER_OF` + `SCOPED_TO`
3. *"What is the prior-period companion of this metric?"* — `PRIOR_PERIOD_OF`
4. *"Can these two files be compared, and on what?"* — `JOINABLE_ON` + `BRIDGES`
5. *"What is the mathematical relationship between these columns?"* — `DERIVED_FROM`
6. *"What is the correct grain and scope for this query?"* — `COVERS` + `SLICES` + `SCOPED_TO`

---

## How the Decomposer Uses the Graph

For a query like *"compare HERISTO Umsatz vs Nestle in the latest period"*:

1. **Concept resolution** — `"Umsatz"` → `MEASURES` lookup → `Revenue` Concept node
2. **File resolution** — `Revenue` → reverse `MEASURES` → all Metric nodes → `AVAILABLE_IN` → candidate files → disambiguate by session context
3. **Entity resolution** — `"HERISTO"` → `EntityGroup` → `SCOPED_TO` check → `MEMBER_OF` reverse → [ANIMONDA, MJAMJAM]
4. **Period resolution** — `"latest period"` → `COVERS` edges on resolved file with `role=CY`
5. **Join resolution** — single file query; no join needed
6. **SQL generation** — handler receives fully resolved spec, builds SQL from node attributes

**The LLM is only called if any step returns ambiguous results.** Steps 1–6 are deterministic graph traversals.

---

## What Is Man-Made vs Automated

### Auto-generated (zero ongoing effort)
- All Metric, Dimension, Entity, Period, File nodes
- `AVAILABLE_IN`, `SLICES`, `CONTAINS`, `COVERS` edges
- `PRIOR_PERIOD_OF`, `DELTA_OF` edges (VJ / % Ver. suffix detection)
- Draft `MEASURES` edges (LLM proposes, human approves)

### Human-authored (once per domain, versioned in git)
- Concept vocabulary (`kg/concepts/registry.yaml`)
- `BRIDGES` edges (cross-file economic equivalence)
- `DERIVED_FROM` edges (math relationships)
- `MEMBER_OF` + `SCOPED_TO` (entity groups and scope)
- `CHILD_OF` (corporate hierarchy)
- `LEVEL_IN` (petfood category hierarchy)

**Estimated one-time authoring effort: ~2–3 hours for the full petfood domain.**  
Ongoing: new NIQ file = run ingestion + 20-minute concept mapping review.

---

## Phased Build Plan

### Phase 1 — Schema layer (fully automated)
**Goal:** All auto-generated nodes and edges from schema JSONs.

```
kg/
├── models.py           # Node + Edge dataclasses, NodeType/EdgeType enums
├── store.py            # DuckDB adjacency tables + read/write helpers
└── ingest/
    ├── schema_reader.py    # profiler JSON → raw column metadata
    ├── classifier.py       # dtype + stats → Metric vs Dimension
    ├── suffix_detector.py  # VJ / % Ver. → PRIOR_PERIOD_OF, DELTA_OF
    ├── period_parser.py    # period string → Period node + COVERS edge
    └── pipeline.py         # end-to-end orchestrator
```

**Exit criterion:** `python -m kg.ingest.pipeline --schema <schema.json>` produces a populated graph. Query patterns 1–3 answerable.

---

### Phase 2 — Concept mapping with HITL
**Goal:** `MEASURES` edges via LLM proposal + human approval CLI.

```
kg/
├── concepts/
│   ├── registry.yaml       # versioned concept vocabulary
│   └── validator.py
└── ingest/
    └── concept_mapper.py   # LLM → proposed MEASURES edges
hitl/
└── review_cli.py           # terminal UI: accept / reject / remap
```

**HITL:** Items with `confidence < 0.85` flagged for review. Rest auto-accepted.  
**Exit criterion:** Query pattern 1 works across all three files.

---

### Phase 3 — Cross-file edges + join detection
**Goal:** `JOINABLE_ON` edges and `BRIDGES` authoring.

```
kg/ingest/
└── join_detector.py        # overlapping dimension names → proposed join keys
hitl/
└── bridge_author.py        # guided CLI for BRIDGES + JOINABLE_ON confirmation
```

**Exit criterion:** Query pattern 4 works. Cross-file queries structurally expressible.

---

### Phase 4 — Concept-layer authoring
**Goal:** `DERIVED_FROM`, `MEMBER_OF`, `SCOPED_TO`, `CHILD_OF`, `LEVEL_IN` — all human-authored via guided wizards.

```
hitl/
├── derived_author.py
├── group_author.py
└── hierarchy_author.py
kg/authored/
├── derived_metrics.yaml
├── entity_groups.yaml
└── hierarchies.yaml
```

**Exit criterion:** Query patterns 5 + 6 work. HERISTO expands correctly per file.

---

### Phase 5 — Decomposer + orchestrator integration
**Goal:** Replace fast-paths with graph traversal. LLM only handles genuine ambiguity.

```
kg/
└── decomposer.py           # six traversal steps → resolved query spec
agent/
└── nodes.py                # orchestrator calls decomposer before LLM
tests/
└── test_decomposer.py      # canonical query suite
```

**Exit criterion:** All six query patterns resolve via traversal. Fast-paths 1–8 removed. Regression suite green.

---

## Branch

`feat/knowledge-graph-ingestion` — all KG work happens here, merged to `main` at Phase 5 completion.

## Files Authored Per Phase

| Phase | Artifacts committed |
|---|---|
| 1 | `kg/` package, DuckDB store, ingest pipeline |
| 2 | `kg/concepts/registry.yaml`, `hitl/review_cli.py` |
| 3 | `hitl/bridge_author.py`, auto-proposed join edges |
| 4 | `kg/authored/*.yaml`, authoring wizards |
| 5 | `kg/decomposer.py`, updated `agent/nodes.py`, test suite |
