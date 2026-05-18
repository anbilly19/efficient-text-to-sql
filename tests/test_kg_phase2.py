"""Phase 2 KG tests — concept registry, validator, alias mapper, HITL store ops.

No LLM calls are made in this suite (LLM path is tested via mock).
Run: uv run pytest tests/test_kg_phase2.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_excel(path: str) -> None:
    df = pd.DataFrame({
        "Marke":                                ["ANIMONDA", "MJAMJAM"] * 15,
        "Umsatz 52 W bis 29/03/26":             [100.0, 200.0] * 15,
        "Menge 52 W bis 29/03/26":              [10, 20] * 15,
        "Penetration (%) 52 W bis 29/03/26":    [30.0, 40.0] * 15,
        "K\u00e4uferhaushalte 52 W bis 29/03/26": [500, 600] * 15,
        "Unknown Metric XYZ 52 W bis 29/03/26": [1.0, 2.0] * 15,
    })
    df.to_excel(path, index=False)


# ---------------------------------------------------------------------------
# 1. Registry loading + validator
# ---------------------------------------------------------------------------

def test_registry_loads():
    from kg.concepts.validator import load_registry, concept_ids
    reg = load_registry()
    assert "concepts" in reg
    assert len(concept_ids()) >= 5


def test_validate_known_concept():
    from kg.concepts.validator import validate_concept_id
    assert validate_concept_id("revenue") is True
    assert validate_concept_id("buyer_reach") is True


def test_validate_unknown_concept():
    from kg.concepts.validator import validate_concept_id
    assert validate_concept_id("nonexistent_concept_xyz") is False


def test_aliases_for_concept():
    from kg.concepts.validator import aliases_for
    aliases = aliases_for("revenue")
    assert "umsatz" in aliases
    assert len(aliases) >= 3


# ---------------------------------------------------------------------------
# 2. Alias fast-path (no LLM)
# ---------------------------------------------------------------------------

def test_alias_match_umsatz():
    from kg.ingest.concept_mapper import _alias_match
    assert _alias_match("Umsatz 52 W bis 29/03/26") == "revenue"


def test_alias_match_menge():
    from kg.ingest.concept_mapper import _alias_match
    assert _alias_match("Menge 52 W bis 29/03/26") == "volume"


def test_alias_match_penetration():
    from kg.ingest.concept_mapper import _alias_match
    assert _alias_match("Penetration (%) 52 W bis 29/03/26") == "buyer_reach"


def test_alias_match_no_match():
    from kg.ingest.concept_mapper import _alias_match
    assert _alias_match("Unknown Metric XYZ") is None


# ---------------------------------------------------------------------------
# 3. propose_measures_edges — alias path only (mock LLM to avoid API calls)
# ---------------------------------------------------------------------------

def test_propose_measures_alias_only(tmp_path, monkeypatch):
    """Alias-matched columns get proposals; LLM path is mocked to return empty."""
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from kg.models import EdgeType, NodeType
    from kg.store import get_neighbors, all_nodes

    # Mock LLM so no API call is made
    monkeypatch.setattr(
        "kg.ingest.concept_mapper._llm_propose",
        lambda cols: {},
    )

    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    _make_excel(xlsx)
    ingest_excel(xlsx, db_path=db)

    proposals = propose_measures_edges(db_path=db)

    # Should have proposals for Umsatz, Menge, Penetration, Käuferhaushalte
    assert len(proposals) >= 3

    # All alias proposals should have confidence=0.95 and source='alias'
    alias_props = [p for p in proposals if p.source == "alias"]
    assert all(p.confidence == 0.95 for p in alias_props)

    # Concept nodes must exist in the store
    concepts = all_nodes(NodeType.CONCEPT, db)
    assert len(concepts) >= 5

    # MEASURES edge: Umsatz metric → revenue concept
    umsatz_nodes = [
        n for n in all_nodes(NodeType.METRIC, db)
        if "Umsatz" in n["label"] and "VJ" not in n["label"] and "Ver" not in n["label"]
    ]
    assert len(umsatz_nodes) >= 1
    measures_targets = get_neighbors(
        umsatz_nodes[0]["id"], EdgeType.MEASURES, "out", db_path=db
    )
    assert any("revenue" in t["id"] for t in measures_targets)


# ---------------------------------------------------------------------------
# 4. HITL store operations (no terminal interaction)
# ---------------------------------------------------------------------------

def test_hitl_accept(tmp_path, monkeypatch):
    """Accepting a pending proposal sets source='ingest', confidence=1.0."""
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges, _HITL_THRESHOLD
    from hitl.review_cli import _fetch_pending, _accept
    import duckdb

    monkeypatch.setattr("kg.ingest.concept_mapper._llm_propose",
                        lambda cols: {
                            col: ("distribution", 0.60)   # below threshold → pending
                            for col in cols
                        })

    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    _make_excel(xlsx)
    ingest_excel(xlsx, db_path=db)
    propose_measures_edges(db_path=db)

    pending = _fetch_pending(db)
    if not pending:
        pytest.skip("No pending proposals generated (all matched by alias)")

    item = pending[0]
    _accept(db, item)

    con = duckdb.connect(db)
    row = con.execute(
        "SELECT source, confidence FROM edges WHERE src_id=? AND dst_id=? AND edge_type='MEASURES'",
        [item["src_id"], item["dst_id"]],
    ).fetchone()
    con.close()
    assert row is not None
    assert row[0] == "ingest"
    assert row[1] == 1.0


def test_hitl_reject(tmp_path, monkeypatch):
    """Rejecting a pending proposal removes the edge."""
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from hitl.review_cli import _fetch_pending, _reject
    import duckdb

    monkeypatch.setattr("kg.ingest.concept_mapper._llm_propose",
                        lambda cols: {col: ("distribution", 0.50) for col in cols})

    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    _make_excel(xlsx)
    ingest_excel(xlsx, db_path=db)
    propose_measures_edges(db_path=db)

    pending = _fetch_pending(db)
    if not pending:
        pytest.skip("No pending proposals generated")

    item = pending[0]
    _reject(db, item)

    con = duckdb.connect(db)
    row = con.execute(
        "SELECT 1 FROM edges WHERE src_id=? AND dst_id=? AND edge_type='MEASURES'",
        [item["src_id"], item["dst_id"]],
    ).fetchone()
    con.close()
    assert row is None


# ---------------------------------------------------------------------------
# 5. Query pattern 1: concept resolution via MEASURES traversal
# ---------------------------------------------------------------------------

def test_query_pattern_1_measures_traversal(tmp_path, monkeypatch):
    """After ingestion + mapping, 'what columns measure Revenue?' works via
    reverse MEASURES traversal from the concept node."""
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from kg.models import EdgeType, NodeType
    from kg.store import get_neighbors, all_nodes

    monkeypatch.setattr("kg.ingest.concept_mapper._llm_propose", lambda cols: {})

    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    _make_excel(xlsx)
    ingest_excel(xlsx, db_path=db)
    propose_measures_edges(db_path=db)

    # Find the 'revenue' concept node
    concepts = [n for n in all_nodes(NodeType.CONCEPT, db) if "revenue" in n["id"]]
    assert len(concepts) == 1

    # Reverse MEASURES: concept → all metrics that measure it
    metrics_measuring_revenue = get_neighbors(
        concepts[0]["id"], EdgeType.MEASURES, direction="in", db_path=db
    )
    assert len(metrics_measuring_revenue) >= 1
    labels = [m["label"] for m in metrics_measuring_revenue]
    assert any("Umsatz" in lbl for lbl in labels)
