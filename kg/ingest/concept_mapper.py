"""Phase 2: Propose MEASURES edges from Metric nodes → Concept nodes.

Two-stage pipeline:
  1. Alias fast-path: if a column name contains a known alias, propose with
     confidence=0.95 (skips LLM call).
  2. LLM proposal: for columns not matched by alias, call the LLM once per
     batch and parse the JSON response.

All proposals are written to the KG store.  Items with confidence < 0.85 are
marked source='proposed' so the HITL CLI can surface them for review.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from kg.concepts.validator import aliases_for, concept_ids, concept_by_id
from kg.models import Edge, EdgeType, Node, NodeType
from kg.store import all_nodes, init_store, upsert_edge, upsert_node

_HITL_THRESHOLD = 0.85


@dataclass
class MeasuresProposal:
    metric_node_id: str
    col_name: str
    concept_id: str
    confidence: float
    source: str  # "alias" | "llm"


# ---------------------------------------------------------------------------
# Stage 1 — alias fast-path
# ---------------------------------------------------------------------------

def _alias_match(col_name: str) -> str | None:
    """Return concept_id if col_name contains any registered alias."""
    col_lower = col_name.lower()
    for cid in concept_ids():
        for alias in aliases_for(cid):
            if alias.lower() in col_lower:
                return cid
    return None


# ---------------------------------------------------------------------------
# Stage 2 — LLM batch proposal
# ---------------------------------------------------------------------------

def _llm():
    """Re-use the same LLM factory pattern as agent/nodes.py."""
    backend = os.getenv("LLM_BACKEND", "openai").lower().strip()
    if backend == "ollama":
        from langchain_ollama import ChatOllama  # type: ignore[import]
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0,
        )
    from langchain_openai import ChatOpenAI  # type: ignore[import]
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _llm_propose(col_names: list[str]) -> dict[str, tuple[str, float]]:
    """
    Ask the LLM to map each col_name to a concept_id.
    Returns { col_name: (concept_id, confidence) } for matched columns.
    """
    if not col_names:
        return {}

    cids = concept_ids()
    concept_descriptions = "\n".join(
        f"  {c['id']}: {c['description']}"
        for c in [
            concept_by_id(cid) for cid in cids
            if concept_by_id(cid)
        ]
    )

    prompt = f"""You are mapping NIQ / retail analytics column names to abstract business concepts.

Available concepts:
{concept_descriptions}

Column names to classify:
{json.dumps(col_names, ensure_ascii=False)}

For each column, return the best matching concept_id and a confidence score (0.0-1.0).
If no concept fits, omit the column from the output.

Output ONLY valid JSON in this exact format:
{{"mappings": [
  {{"col": "<col_name>", "concept_id": "<id>", "confidence": <float>}},
  ...
]}}"""

    from langchain_core.messages import HumanMessage
    try:
        response = _llm().invoke([HumanMessage(content=prompt)])
        raw = response.content if isinstance(response.content, str) else str(response.content)
        # Extract JSON robustly
        start = raw.find("{")
        if start == -1:
            return {}
        parsed = json.loads(raw[start:])
        result: dict[str, tuple[str, float]] = {}
        for item in parsed.get("mappings", []):
            col = item.get("col", "")
            cid = item.get("concept_id", "")
            conf = float(item.get("confidence", 0.0))
            if col and cid in cids:
                result[col] = (cid, conf)
        return result
    except Exception as exc:
        print(f"[concept_mapper] LLM call failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def propose_measures_edges(
    db_path: str | None = None,
    file_label: str | None = None,
) -> list[MeasuresProposal]:
    """
    Scan all Metric nodes (optionally filtered by file_label) in the KG store,
    propose MEASURES edges, write them to the store, and return the proposals.

    Edges with confidence >= _HITL_THRESHOLD are written with source='ingest'.
    Edges below threshold are written with source='proposed' (pending HITL review).
    """
    db_path = db_path or os.environ.get("KG_DB_PATH", "kg/kg.duckdb")
    init_store(db_path)

    # Ensure Concept nodes exist for all registered concepts
    for cid in concept_ids():
        c = concept_by_id(cid)
        upsert_node(Node(
            id=f"concept::{cid}",
            node_type=NodeType.CONCEPT,
            label=c["label"],
            props={"description": c.get("description", "")},
        ), db_path)

    metrics = all_nodes(NodeType.METRIC, db_path)
    if file_label:
        metrics = [m for m in metrics if f"::{file_label}::" in m["id"]]

    proposals: list[MeasuresProposal] = []
    unmatched_cols: list[str] = []
    unmatched_ids: dict[str, str] = {}  # col_name -> node_id

    # Stage 1: alias fast-path
    for m in metrics:
        col_name = m["label"]
        node_id = m["id"]
        cid = _alias_match(col_name)
        if cid:
            proposals.append(MeasuresProposal(
                metric_node_id=node_id,
                col_name=col_name,
                concept_id=cid,
                confidence=0.95,
                source="alias",
            ))
        else:
            unmatched_cols.append(col_name)
            unmatched_ids[col_name] = node_id

    # Stage 2: LLM for unmatched
    if unmatched_cols:
        llm_results = _llm_propose(unmatched_cols)
        for col_name, (cid, conf) in llm_results.items():
            node_id = unmatched_ids.get(col_name)
            if not node_id:
                continue
            proposals.append(MeasuresProposal(
                metric_node_id=node_id,
                col_name=col_name,
                concept_id=cid,
                confidence=conf,
                source="llm",
            ))

    # Write all proposals to the store
    for p in proposals:
        source = "ingest" if p.confidence >= _HITL_THRESHOLD else "proposed"
        upsert_edge(Edge(
            src_id=p.metric_node_id,
            dst_id=f"concept::{p.concept_id}",
            edge_type=EdgeType.MEASURES,
            source=source,
            confidence=p.confidence,
        ), db_path)

    auto = sum(1 for p in proposals if p.confidence >= _HITL_THRESHOLD)
    pending = len(proposals) - auto
    print(f"[concept_mapper] {len(proposals)} proposals: {auto} auto-accepted, {pending} pending HITL review")
    return proposals
