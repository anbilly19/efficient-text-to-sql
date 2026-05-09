"""AnalyticsState – the single shared state object flowing through the LangGraph."""
from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, model_validator


class PlanStep(BaseModel):
    """A single step in the orchestrator's execution plan."""

    id: int
    type: str  # "profile" | "sql" | "verify"
    description: str
    status: str = "pending"  # pending | running | done | failed
    result: str | None = None


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from a message content field.

    Handles three formats emitted by LangGraph Studio / API:
      - str                          → returned as-is
      - [{"type": "text", "text": ...}]  → joined plain text blocks
      - [{"type": "human", "content": ...}]  → recursed
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "content" in block:
                    parts.append(_extract_text_from_content(block["content"]))
        return " ".join(p for p in parts if p).strip()
    return str(content) if content is not None else ""


class AnalyticsState(BaseModel):
    """Complete state object for the analytics agent graph."""

    # ── Conversation ───────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    # ── User intent ──────────────────────────────────────────────
    user_query: str = ""

    # ── File loading ─────────────────────────────────────────────
    load_file_path: str | None = None
    load_file_dataset: str | None = None

    # ── Orchestrator plan ──────────────────────────────────────────
    plan: list[PlanStep] = Field(default_factory=list)
    current_step: PlanStep | None = None

    # ── SQL pipeline ──────────────────────────────────────────────
    last_sql: str = ""
    last_query_result: str = ""   # JSON string of first 50 rows
    last_query_metadata: dict[str, Any] = Field(default_factory=dict)

    # ── Verification ──────────────────────────────────────────────
    verification_verdict: str = ""   # "pass" | "fail" | "warning"
    verification_feedback: str = ""
    retry_count: int = 0

    # ── Final answer ──────────────────────────────────────────────
    final_answer: str = ""
    error: str = ""

    @model_validator(mode="after")
    def _populate_user_query_from_messages(self) -> "AnalyticsState":
        """If user_query is empty, fill it from the last HumanMessage.

        This handles the LangGraph Studio / API input format where the test
        sends messages with structured content blocks like::

            {"type": "human", "content": [{"type": "text", "text": "..."}]}

        rather than a plain ``user_query`` string.
        """
        if self.user_query:
            return self
        for msg in reversed(self.messages):
            if isinstance(msg, HumanMessage):
                text = _extract_text_from_content(msg.content)
                if text:
                    self.user_query = text
                    break
        return self
