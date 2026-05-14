"""
test_langgraph_niq_query_rounds.py

Round-trip LangGraph tests for NIQ panel data.
Mirrors test_langgraph_query_rounds.py in structure.

Run a specific round:
    pytest tests/test_langgraph_niq_query_rounds.py -m niq_round_01 -v

Run all NIQ rounds:
    pytest tests/test_langgraph_niq_query_rounds.py -v

Skip if no NIQ file is available:
    Tests that require NIQ_SYNTHETIC_PATH are auto-skipped when the env var
    is not set, so CI passes without a real Excel fixture.

Note on adversarial queries
---------------------------
Some queries intentionally reference columns that do NOT exist in NIQ data
(e.g. 'customer_id', 'invoice_number'). The agent must report gracefully
that the column is absent rather than hallucinating a result.
"""
import os
import time
from typing import Any

import pytest
import requests


LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024").rstrip("/")
ASSISTANT_ID      = os.getenv("ASSISTANT_ID",      "fe096781-5601-53d2-b2f6-0d3403f7e9ca")
NIQ_PATH          = os.getenv("NIQ_SYNTHETIC_PATH", "")
NIQ_DATASET       = os.getenv("NIQ_DATASET_NAME",   "niq_panel")

LOAD_NIQ_CMD = f"load {NIQ_PATH} as {NIQ_DATASET}" if NIQ_PATH else ""

REQUEST_TIMEOUT = float(os.getenv("TEST_REQUEST_TIMEOUT", "120"))
POLL_SECONDS    = float(os.getenv("TEST_POLL_SECONDS",    "0.5"))

NIQ_SKIP = pytest.mark.skipif(
    not NIQ_PATH,
    reason="NIQ_SYNTHETIC_PATH not set — skipping NIQ round-trip tests",
)


# ---------------------------------------------------------------------------
# Query rounds
# Each entry: (round_id, mark_label, [queries])
# ---------------------------------------------------------------------------

QUERY_ROUNDS = [
    (
        "niq_round_01_basic_aggregations",
        "niq_round_01",
        [
            f"What is the overall penetration rate in the {NIQ_DATASET} table?",
            f"What is the average spend per buyer across all brands in {NIQ_DATASET}?",
            f"How many distinct brands are in {NIQ_DATASET}?",
        ],
    ),
    (
        "niq_round_02_german_aliases",
        "niq_round_02",
        [
            f"Was ist die Käuferreichweite in {NIQ_DATASET}?",
            f"Zeig mir die Ausgaben je Käufer nach Marke in {NIQ_DATASET}.",
            f"Veränderung zum Vorjahr der Penetration in {NIQ_DATASET}?",
        ],
    ),
    (
        "niq_round_03_yoy_comparison",
        "niq_round_03",
        [
            f"Which brands had a positive YoY change in penetration in {NIQ_DATASET}?",
            f"Show me CY vs PY spend per buyer for every brand in {NIQ_DATASET}.",
            f"What is the average year-on-year change in buyer frequency in {NIQ_DATASET}?",
        ],
    ),
    (
        "niq_round_04_ranking_topn",
        "niq_round_04",
        [
            f"What are the top 5 brands by penetration in {NIQ_DATASET}?",
            f"Rank all categories by spend per buyer in {NIQ_DATASET}.",
            f"Which brand has the lowest buyer reach in {NIQ_DATASET}?",
        ],
    ),
    (
        "niq_round_05_filtering",
        "niq_round_05",
        [
            f"Show brands in {NIQ_DATASET} with penetration above 20%.",
            f"Filter {NIQ_DATASET} to brands where YoY spend per buyer growth is negative.",
            f"Which products in {NIQ_DATASET} have both penetration above 10% and spend per buyer above 50?",
        ],
    ),
    (
        "niq_round_06_grain_period",
        "niq_round_06",
        [
            f"What periods are available in {NIQ_DATASET}?",
            f"Show me all annual figures for penetration in {NIQ_DATASET}.",
            f"What is the grain of {NIQ_DATASET} — annual, monthly, or quarterly?",
        ],
    ),
    (
        "niq_round_07_adversarial_missing_column",
        "niq_round_07",
        [
            # Valid query
            f"What is the total volume by brand in {NIQ_DATASET}?",
            # Adversarial: 'customer_id' does not exist in NIQ data
            f"Show me the top customers by invoice number in {NIQ_DATASET}.",
            # Adversarial: 'store_id' does not exist in NIQ data
            f"Which store_id had the highest revenue in {NIQ_DATASET}?",
        ],
    ),
    (
        "niq_round_08_semantic_map_lookup",
        "niq_round_08",
        [
            f"What does 'Käuferreichweite' map to in {NIQ_DATASET}?",
            f"Look up 'yoy' in the semantic map for {NIQ_DATASET}.",
            f"What column corresponds to 'Haushaltsdurchdringung' in {NIQ_DATASET}?",
        ],
    ),
]

_ROUND_MARKS = {
    mark: pytest.mark.__getattr__(mark)
    for _, mark, _ in QUERY_ROUNDS
}

_ALL_QUERIES = [
    pytest.param(
        round_name,
        idx,
        query,
        id=f"{round_name}[q{idx}]",
        marks=[_ROUND_MARKS[mark_label], NIQ_SKIP],
    )
    for round_name, mark_label, queries in QUERY_ROUNDS
    for idx, query in enumerate(queries, start=1)
]

TOTAL_QUERIES = len(_ALL_QUERIES)


# ---------------------------------------------------------------------------
# Helpers  (identical to test_langgraph_query_rounds.py)
# ---------------------------------------------------------------------------

class LangGraphTestError(AssertionError):
    pass


def _request(method: str, path: str, **kwargs) -> requests.Response:
    resp = requests.request(
        method,
        f"{LANGGRAPH_API_URL}{path}",
        timeout=REQUEST_TIMEOUT,
        **kwargs,
    )
    resp.raise_for_status()
    return resp


def _create_thread() -> str:
    resp = _request("POST", "/threads", json={})
    data = resp.json()
    thread_id = data.get("thread_id") or data.get("id")
    if not thread_id:
        raise LangGraphTestError(f"Could not create thread. Response: {data}")
    return thread_id


def _extract_outputs(payload: dict[str, Any]) -> dict[str, Any]:
    if "outputs" in payload and isinstance(payload["outputs"], dict):
        return payload["outputs"]
    if "values" in payload and isinstance(payload["values"], dict):
        return payload["values"]
    return payload


def _extract_last_ai_text(outputs: dict[str, Any]) -> str:
    final_answer = outputs.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        return final_answer.strip()

    messages = outputs.get("messages") or []
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("type") != "ai":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            chunks = [
                block.get("text") or block.get("content") or ""
                for block in content
                if isinstance(block, dict)
            ]
            joined = " ".join(c for c in chunks if c).strip()
            if joined:
                return joined
    return ""


def _run_wait(thread_id: str, user_text: str) -> dict[str, Any]:
    payload = {
        "assistant_id": ASSISTANT_ID,
        "input": {
            "messages": [
                {
                    "type": "human",
                    "content": [{"type": "text", "text": user_text}],
                }
            ]
        },
    }

    paths_to_try = [
        f"/threads/{thread_id}/runs/wait",
        "/runs/wait",
    ]

    last_error = None
    for path in paths_to_try:
        body = payload if path.startswith(f"/threads/{thread_id}") else {**payload, "thread_id": thread_id}
        try:
            resp = _request("POST", path, json=body)
            data = resp.json()
            outputs = _extract_outputs(data)
            if isinstance(outputs, dict) and outputs:
                return outputs
            return data
        except Exception as exc:  # pragma: no cover
            last_error = exc
            continue

    raise LangGraphTestError(f"Run request failed for thread {thread_id}: {last_error}")


def _assert_niq_loaded(outputs: dict[str, Any], cmd: str) -> None:
    answer = _extract_last_ai_text(outputs)
    assert answer, f"No response for NIQ load command '{cmd}'. Outputs: {outputs}"
    ok = (
        "successfully loaded" in answer.lower()
        or "niq panel loaded" in answer.lower()
        or "you can now ask" in answer.lower()
        or "already loaded" in answer.lower()
        or "table is ready" in answer.lower()
    )
    assert ok, f"NIQ load did not succeed for '{cmd}'.\nAnswer: {answer}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def niq_thread_id() -> str:
    return _create_thread()


@pytest.fixture(scope="session", autouse=True)
def loaded_niq(niq_thread_id: str) -> None:
    """Load the NIQ panel file once for the whole test session."""
    if not NIQ_PATH:
        pytest.skip("NIQ_SYNTHETIC_PATH not set — skipping NIQ fixture setup")
    outputs = _run_wait(niq_thread_id, LOAD_NIQ_CMD)
    _assert_niq_loaded(outputs, LOAD_NIQ_CMD)


_query_counter: dict[str, int] = {"n": 0}


# ---------------------------------------------------------------------------
# Per-query parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("round_name,query_index,query", _ALL_QUERIES)
def test_niq_query(
    round_name: str,
    query_index: int,
    query: str,
    niq_thread_id: str,
    loaded_niq: None,
) -> None:
    _query_counter["n"] += 1
    n = _query_counter["n"]
    print(f"\n[{n:>2}/{TOTAL_QUERIES}] {round_name}[q{query_index}]  \u27a4  {query}")

    outputs = _run_wait(niq_thread_id, query)
    answer = _extract_last_ai_text(outputs)
    error = str(outputs.get("error", "") or "")
    result_text = str(outputs.get("last_query_result", "") or "")

    print(f"         answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")

    assert answer, f"No final answer returned.\nOutputs: {outputs}"
    assert "empty sql" not in answer.lower(), f"Empty SQL surfaced in answer: {answer}"
    assert "sql error" not in answer.lower(), f"SQL error surfaced in answer: {answer}"
    assert "wasn't able to answer" not in answer.lower(), f"Failure answer surfaced: {answer}"
    assert "empty sql" not in error.lower(), f"Empty SQL in error field: {error}"
    assert not result_text.startswith("ERROR:"), f"Query execution error: {result_text}"

    time.sleep(POLL_SECONDS)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def test_niq_query_catalog_is_complete() -> None:
    round_count = len(QUERY_ROUNDS)
    query_count = sum(len(queries) for _, _, queries in QUERY_ROUNDS)
    assert round_count == 8,  f"Expected 8 rounds, found {round_count}"
    assert query_count == 24, f"Expected 24 total queries, found {query_count}"
