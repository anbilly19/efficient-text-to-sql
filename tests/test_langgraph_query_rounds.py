"""
test_langgraph_query_rounds.py

Run a specific round:
    pytest tests/test_langgraph_query_rounds.py -m round_01 -v

Run from round_05 onward:
    pytest tests/test_langgraph_query_rounds.py -m "round_05 or round_06 or round_07 or round_08 or round_09 or round_10 or round_11 or round_12" -v

Skip multi-table rounds:
    pytest tests/test_langgraph_query_rounds.py -m "not round_12" -v

All rounds:
    pytest tests/test_langgraph_query_rounds.py -v

Note on customer queries
------------------------
Some queries intentionally reference 'customer', which is NOT a column in
any loaded table. These are adversarial checks: the LLM should gracefully
report that no customer column exists rather than hallucinating a result.
"""
import os
import time
from typing import Any

import pytest
import requests


LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024").rstrip("/")
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "fe096781-5601-53d2-b2f6-0d3403f7e9ca")

LOAD_SALES        = os.getenv("LOAD_SALES",        "load data/sample_sales_1000.xlsx as sales1000")
LOAD_REP_TARGETS  = os.getenv("LOAD_REP_TARGETS",  "load data/sales_rep_targets.xlsx as sales_rep_targets")
LOAD_PROD_METRICS = os.getenv("LOAD_PROD_METRICS", "load data/product_metrics.xlsx as product_metrics")

REQUEST_TIMEOUT = float(os.getenv("TEST_REQUEST_TIMEOUT", "120"))
POLL_SECONDS    = float(os.getenv("TEST_POLL_SECONDS", "0.5"))


# ---------------------------------------------------------------------------
# Query rounds
# Each entry: (round_id, mark_label, [queries])
# Queries marked [ADVERSARIAL] intentionally use a non-existent column
# (customer) to test graceful error handling in the LLM.
# ---------------------------------------------------------------------------

QUERY_ROUNDS = [
    (
        "round_01_basic_aggregations",
        "round_01",
        [
            "What is the total revenue in the sales1000 table?",
            "What is the average order value?",
            "How many unique sales reps are there?",
        ],
    ),
    (
        "round_02_date_filtering",
        "round_02",
        [
            "What is the total revenue for 2023?",
            "What is the month-by-month revenue trend for 2024?",
            "Which month had the highest sales in 2023?",
        ],
    ),
    (
        "round_03_grouping_ranking",
        "round_03",
        [
            "Which product category had the most orders?",
            "What are the top 5 regions by total revenue?",
            "What percentage of orders were above average order value?",
        ],
    ),
    (
        "round_04_multi_step_complex",
        "round_04",
        [
            "What is the 2023-2024 revenue growth?",
            "Show me the revenue breakdown by category and year",
            "Which region has the highest average order value?",
        ],
    ),
    (
        "round_05_window_functions",
        "round_05",
        [
            "Rank regions by total revenue and show their percentile",
            "What is the running total of revenue by order date?",
            "Show month-over-month revenue growth rate for 2023",
        ],
    ),
    (
        "round_06_conditional_logic",
        "round_06",
        [
            "What percentage of orders were placed on weekends?",
            "How many orders had a revenue above the 90th percentile?",
            "Classify orders as high/medium/low value and count each tier",
        ],
    ),
    (
        "round_07_multi_step_reasoning",
        "round_07",
        [
            "Which product category had the fastest revenue growth from 2022 to 2023?",
            "Which regions had orders in both 2022 and 2023?",
            "Show the top category per region by total revenue",
        ],
    ),
    (
        "round_08_adversarial_duckdb",
        "round_08",
        [
            "What is the correlation between unit price and revenue?",
            "List the bottom 10% of sales reps by order frequency",
            "Which day of the week generates the most revenue on average?",
        ],
    ),
    (
        "round_09_adversarial_missing_column",
        "round_09",
        [
            # Valid queries
            "Which sales reps have more than one order and what is their order count?",
            "Show the top category per sales rep by total revenue",
            # Adversarial: 'customer' does not exist -- LLM should report gracefully
            "What are the top 5 customers by total spend?",
        ],
    ),
    (
        "round_10_subqueries_ctes",
        "round_10",
        [
            "Find orders where the revenue is above the average revenue for that product category",
            "Show regions whose total revenue is above the average region revenue",
            # Adversarial: 'customer' does not exist
            "Which customers placed their first order in 2023 and are still active in 2024?",
        ],
    ),
    (
        "round_11_aggregation_on_aggregation",
        "round_11",
        [
            "What is the average number of orders per sales rep?",
            "What is the average order value per sales rep, and what is the average of those averages?",
            # Adversarial: 'customer' does not exist
            "Show the distribution of order counts per customer (how many customers placed 1, 2, 3... orders)",
        ],
    ),
    (
        "round_12_multi_table",
        "round_12",
        [
            "Which sales reps are below their annual quota? Show actual revenue vs quota and the gap.",
            "Rank sales reps by quota attainment (actual revenue / annual quota) within each region.",
            "What is the total quota gap across all regions combined?",
            "Which products have a higher average unit price than the historical average in product_metrics?",
            "Show total revenue per product category and compare it to the baseline revenue in product_metrics.",
            "For each sales rep, break down their revenue by product category and show what share of their quota each category contributes.",
            "Which sales reps below quota are selling the top-5 revenue products? Show the rep, region, quota gap, and the top product they sell.",
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
        marks=[_ROUND_MARKS[mark_label]],
    )
    for round_name, mark_label, queries in QUERY_ROUNDS
    for idx, query in enumerate(queries, start=1)
]

TOTAL_QUERIES = len(_ALL_QUERIES)


# ---------------------------------------------------------------------------
# Helpers
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


def _assert_loaded(outputs: dict[str, Any], cmd: str) -> None:
    answer = _extract_last_ai_text(outputs)
    assert answer, f"No response for load command '{cmd}'. Outputs: {outputs}"
    ok = (
        "successfully loaded" in answer.lower()
        or "you can now ask questions" in answer.lower()
        or "already loaded" in answer.lower()
        or "table is ready" in answer.lower()
    )
    assert ok, f"Load did not succeed for '{cmd}'.\nAnswer: {answer}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def thread_id() -> str:
    return _create_thread()


@pytest.fixture(scope="session", autouse=True)
def loaded_datasets(thread_id: str) -> None:
    outputs = _run_wait(thread_id, LOAD_SALES)
    _assert_loaded(outputs, LOAD_SALES)

    for cmd in (LOAD_REP_TARGETS, LOAD_PROD_METRICS):
        try:
            outputs = _run_wait(thread_id, cmd)
            _assert_loaded(outputs, cmd)
        except AssertionError as exc:
            import warnings
            warnings.warn(
                f"Optional table load skipped (round_12 will likely fail):\n{exc}",
                stacklevel=1,
            )


_query_counter: dict[str, int] = {"n": 0}


# ---------------------------------------------------------------------------
# Per-query parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("round_name,query_index,query", _ALL_QUERIES)
def test_query(
    round_name: str,
    query_index: int,
    query: str,
    thread_id: str,
    loaded_datasets: None,
) -> None:
    _query_counter["n"] += 1
    n = _query_counter["n"]
    print(f"\n[{n:>2}/{TOTAL_QUERIES}] {round_name}[q{query_index}]  ➤  {query}")

    outputs = _run_wait(thread_id, query)
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

def test_query_catalog_is_complete() -> None:
    round_count = len(QUERY_ROUNDS)
    query_count = sum(len(queries) for _, _, queries in QUERY_ROUNDS)
    assert round_count == 12, f"Expected 12 rounds, found {round_count}"
    assert query_count == 40, f"Expected 40 total queries, found {query_count}"
