"""AnalyticsState – the single shared state object flowing through the LangGraph."""
from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in the orchestrator's execution plan."""

    id: int
    type: str  # "profile" | "sql" | "verify"
    description: str
    status: str = "pending"  # pending | running | done | failed
    result: str | None = None


class AnalyticsState(BaseModel):
    """Complete state object for the analytics agent graph."""

    # ── Conversation ─────────────────────────────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    # ── User intent ──────────────────────────────────────────────────────────────────────
    user_query: str = ""

    # ── File loading ─────────────────────────────────────────────────────────────────────
    load_file_path: str = ""      # set by orchestrator when user asks to load a file
    load_file_dataset: str = ""   # table name to register the file as

    # ── Orchestrator plan ───────────────────────────────────────────────────────────────
    plan: list[PlanStep] = Field(default_factory=list)
    current_step: PlanStep | None = None

    # ── SQL pipeline ───────────────────────────────────────────────────────────────────
    last_sql: str = ""
    last_query_result: str = ""   # JSON string of first 50 rows
    last_query_metadata: dict[str, Any] = Field(default_factory=dict)

    # ── Verification ────────────────────────────────────────────────────────────────────
    verification_verdict: str = ""   # "pass" | "fail" | "warning"
    verification_feedback: str = ""

    # ── Final answer ────────────────────────────────────────────────────────────────────
    final_answer: str = ""
    error: str = ""
