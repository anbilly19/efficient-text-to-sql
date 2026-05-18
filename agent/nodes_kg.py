"""KG ingestion node — runs as part of the LangGraph agent in ingestion mode.

Triggered when the orchestrator detects an ingestion intent
(e.g. user uploads a file or says 'ingest this file into the knowledge graph').

This node is SELF-CONTAINED: it calls kg.ingest.pipeline.ingest_excel and
returns a human-readable summary back into the agent state.
No other agent nodes are modified.
"""
from __future__ import annotations

import os

from langchain_core.messages import AIMessage

from agent.state import AnalyticsState


def kg_ingest_node(state: AnalyticsState) -> dict:
    """LangGraph node: ingest an uploaded Excel file into the KG store."""
    # Import here to keep agent startup fast when KG is not used
    from kg.ingest.pipeline import ingest_excel

    filepath = state.kg_ingest_path
    if not filepath or not os.path.isfile(filepath):
        msg = f"[kg_ingest] file not found or not specified: {filepath!r}"
        return {
            "kg_result": msg,
            "ingestion_mode": False,
            "final_answer": msg,
        }

    try:
        summary = ingest_excel(filepath)
        lines = [f"✅ Knowledge graph updated from `{os.path.basename(filepath)}`:", ""]
        for sheet, counts in summary.items():
            lines.append(f"  • Sheet **{sheet}**: {counts['nodes']} nodes, {counts['edges']} edges added")
        result = "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        result = f"❌ Ingestion failed: {exc}"

    return {
        "kg_result": result,
        "ingestion_mode": False,
        "kg_ingest_path": None,
        "messages": [AIMessage(content=result)],
        "final_answer": result,
    }
