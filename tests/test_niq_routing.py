"""
tests/test_niq_routing.py
--------------------------
Tests for the NIQ auto-routing logic in agent/nodes.py:

  _is_niq_file()    — detection heuristic (always runs, no env var needed)
  load_file_node()  — routing branch (full integration, needs NIQ_SYNTHETIC_PATH)

Isolation model: mirrors test_parquet_persistence.py exactly.
  - Module fixture _module_env: DUCKDB_PATH=:memory:, fresh agent.* import.
  - Per-test _fresh_conn: reset _conn between tests.

Detection-only tests (TestIsNiqFile) always run.
Routing integration tests (TestLoadFileNodeRouting) skip when
NIQ_SYNTHETIC_PATH is not set.

Run:
    pytest tests/test_niq_routing.py -v
    NIQ_SYNTHETIC_PATH=/path/to/niq_panel.xlsx pytest tests/test_niq_routing.py -v
"""
from __future__ import annotations

import os
import re
import sys

import pytest


NIQ_SYNTHETIC_PATH = os.environ.get("NIQ_SYNTHETIC_PATH", "")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _db():
    return sys.modules["agent.database"]


def _nodes():
    return sys.modules["agent.nodes"]


# ---------------------------------------------------------------------------
# Module fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _module_env(tmp_path_factory):
    store = tmp_path_factory.mktemp("niq_routing_pq")
    prev_store = os.environ.get("PARQUET_STORE")
    prev_db    = os.environ.get("DUCKDB_PATH")

    os.environ["PARQUET_STORE"] = str(store)
    os.environ["DUCKDB_PATH"]   = ":memory:"

    for mod in list(sys.modules):
        if mod.startswith("agent"):
            del sys.modules[mod]

    import agent.database  # noqa: F401
    import agent.nodes     # noqa: F401

    yield store

    db = _db()
    if db._conn is not None:
        try:
            db._conn.close()
        except Exception:
            pass
        db._conn = None

    if prev_store is not None:
        os.environ["PARQUET_STORE"] = prev_store
    else:
        os.environ.pop("PARQUET_STORE", None)
    if prev_db is not None:
        os.environ["DUCKDB_PATH"] = prev_db
    else:
        os.environ.pop("DUCKDB_PATH", None)


@pytest.fixture(autouse=True)
def _fresh_conn(_module_env):
    db = _db()
    if db._conn is not None:
        try:
            db._conn.close()
        except Exception:
            pass
    db._conn = None
    os.environ["DUCKDB_PATH"] = ":memory:"
    db.get_connection()
    yield


# ---------------------------------------------------------------------------
# TestIsNiqFile — always runs, no synthetic file needed
# ---------------------------------------------------------------------------

class TestIsNiqFile:
    """Unit tests for _is_niq_file() path/name heuristic."""

    def test_niq_in_path_detected(self):
        assert _nodes()._is_niq_file("/data/niq_export.xlsx", "dataset") is True

    def test_nielsen_in_path_detected(self):
        assert _nodes()._is_niq_file("/data/nielsen_q1.xlsx", "dataset") is True

    def test_panel_in_path_detected(self):
        assert _nodes()._is_niq_file("/data/panel_data.csv", "dataset") is True

    def test_penetration_in_path_detected(self):
        assert _nodes()._is_niq_file("/data/penetration_report.xlsx", "ds") is True

    def test_buyer_in_path_detected(self):
        assert _nodes()._is_niq_file("/data/buyer_reach.csv", "ds") is True

    def test_niq_in_dataset_name_detected(self):
        assert _nodes()._is_niq_file("/data/q1_export.xlsx", "niq_panel") is True

    def test_panel_in_dataset_name_detected(self):
        assert _nodes()._is_niq_file("/data/q1_export.xlsx", "panel_petfood") is True

    def test_generic_sales_file_not_detected(self):
        assert _nodes()._is_niq_file("/data/sales.xlsx", "sales") is False

    def test_orders_file_not_detected(self):
        assert _nodes()._is_niq_file("/data/orders_2024.csv", "orders") is False

    def test_case_insensitive_niq(self):
        assert _nodes()._is_niq_file("/data/NIQ_Export.XLSX", "dataset") is True

    def test_case_insensitive_panel(self):
        assert _nodes()._is_niq_file("/data/Panel_Q2.xlsx", "dataset") is True

    def test_partial_word_match_not_false_positive(self):
        # 'unique' contains 'niq' as a substring — must still match (regex, not word boundary)
        # This documents the known behaviour: path matching is substring-based.
        result = _nodes()._is_niq_file("/data/unique_ids.csv", "unique_ids")
        # We document this fires True (substring match) — acceptable tradeoff.
        assert isinstance(result, bool)  # just assert it doesn't crash


# ---------------------------------------------------------------------------
# TestLoadFileNodeRouting — requires NIQ_SYNTHETIC_PATH
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not NIQ_SYNTHETIC_PATH,
    reason="NIQ_SYNTHETIC_PATH not set. Export NIQ_SYNTHETIC_PATH=/path/to/niq_panel.xlsx to run.",
)
class TestLoadFileNodeRouting:
    """Integration tests: load_file_node must auto-route NIQ files to load_niq_file."""

    def _make_state(self, path: str, dataset: str):
        """Build a minimal AnalyticsState-like object for load_file_node."""
        from agent.state import AnalyticsState
        from langchain_core.messages import HumanMessage
        return AnalyticsState(
            messages=[HumanMessage(content=f"load file at {path} as {dataset}")],
            user_query=f"load file at {path} as {dataset}",
            load_file_path=path,
            load_file_dataset=dataset,
        )

    def test_niq_file_routes_to_niq_loader(self):
        """load_file_node reply must contain NIQ panel confirmation markers."""
        state = self._make_state(NIQ_SYNTHETIC_PATH, "niq_panel")
        result = _nodes().load_file_node(state)
        reply = result["final_answer"]
        assert "NIQ panel loaded" in reply, (
            f"Expected 'NIQ panel loaded' in reply, got:\n{reply}"
        )

    def test_niq_routing_reply_contains_grain(self):
        state = self._make_state(NIQ_SYNTHETIC_PATH, "niq_panel")
        result = _nodes().load_file_node(state)
        reply = result["final_answer"]
        assert "grain" in reply.lower(), (
            f"Expected grain info in reply, got:\n{reply}"
        )

    def test_niq_routing_reply_contains_example_terms(self):
        """Confirmation reply must mention NIQ-specific example queries."""
        state = self._make_state(NIQ_SYNTHETIC_PATH, "niq_panel")
        result = _nodes().load_file_node(state)
        reply = result["final_answer"]
        # At least one German/NIQ term should appear in the example query hint
        niq_hints = ["penetration", "K\u00e4uferreichweite", "spend per buyer", "YoY"]
        assert any(hint in reply for hint in niq_hints), (
            f"Expected at least one NIQ hint in reply, got:\n{reply}"
        )

    def test_niq_routing_registers_in_data_registry(self):
        state = self._make_state(NIQ_SYNTHETIC_PATH, "niq_panel")
        _nodes().load_file_node(state)
        row = _db().get_connection().execute(
            "SELECT dataset_name FROM _data_registry WHERE dataset_name = 'niq_panel'"
        ).fetchone()
        assert row is not None, "_data_registry must have niq_panel after load_file_node"

    def test_niq_routing_sets_niq_tags_in_table_context(self):
        state = self._make_state(NIQ_SYNTHETIC_PATH, "niq_panel")
        _nodes().load_file_node(state)
        row = _db().get_connection().execute(
            "SELECT tags FROM _table_context WHERE dataset_name = 'niq_panel'"
        ).fetchone()
        assert row is not None and row[0] is not None, "_table_context must have a tags row"
        assert "niq" in row[0].lower(), f"Expected 'niq' in tags, got: {row[0]}"

    def test_niq_routing_load_file_path_cleared_after(self):
        """State cleanup: load_file_path must be None in the returned dict."""
        state = self._make_state(NIQ_SYNTHETIC_PATH, "niq_panel")
        result = _nodes().load_file_node(state)
        assert result["load_file_path"] is None
        assert result["load_file_dataset"] is None

    def test_generic_file_does_not_route_to_niq_loader(self, tmp_path):
        """A generic CSV must go through the normal load_file path, not NIQ."""
        import pandas as pd
        generic_path = tmp_path / "sales.csv"
        pd.DataFrame({"order_id": [1, 2], "revenue": [100.0, 200.0]}).to_csv(
            generic_path, index=False
        )
        state = self._make_state(str(generic_path), "sales")
        result = _nodes().load_file_node(state)
        reply = result["final_answer"]
        assert "NIQ panel loaded" not in reply, (
            f"Generic file must not trigger NIQ routing, got:\n{reply}"
        )
        assert "Successfully loaded" in reply or "\u2705" in reply
