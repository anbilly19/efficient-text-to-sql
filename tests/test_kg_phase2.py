"""Phase 2 KG tests — concept registry, validator, alias mapper, HITL store ops.

No LLM calls are made in this suite (LLM path is tested via mock).
Run: uv run pytest tests/test_kg_phase2.py -v
"""
from __future__ import annotations

import pytest
from tests.helpers.kg_fixtures import make_excel, noop_llm_propose


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
# 3. propose_measures_edges — alias path only (LLM mocked)
# ---------------------------------------------------------------------------

def test_propose_measures_alias_only(tmp_path, monkeypatch):
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from kg.models import EdgeType, NodeType
    from kg.store import get_neighbors, all_nodes

    noop_llm_propose(monkeypatch)

    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    make_excel(xlsx)
    ingest_excel(xlsx, db_path=db)

    proposals = propose_measures_edges(db_path=db)
    assert len(proposals) >= 3

    alias_props = [p for p in proposals if p.source == "alias"]
    assert all(p.confidence == 0.95 for p in alias_props)

    concepts = all_nodes(NodeType.CONCEPT, db)
    assert len(concepts) >= 5

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
# 4. HITL store operations
# ---------------------------------------------------------------------------

def test_hitl_accept(tmp_path, monkeypatch):
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from hitl.review_cli import _fetch_pending, _accept
    import duckdb

    monkeypatch.setattr(
        "kg.ingest.concept_mapper._llm_propose",
        lambda cols: {col: ("distribution", 0.60) for col in cols},
    )

    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    make_excel(xlsx)
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
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from hitl.review_cli import _fetch_pending, _reject
    import duckdb

    monkeypatch.setattr(
        "kg.ingest.concept_mapper._llm_propose",
        lambda cols: {col: ("distribution", 0.50) for col in cols},
    )

    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    make_excel(xlsx)
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
# 5. Query pattern 1: reverse MEASURES traversal
# ---------------------------------------------------------------------------

def test_query_pattern_1_measures_traversal(tmp_path, monkeypatch):
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from kg.models import EdgeType, NodeType
    from kg.store import get_neighbors, all_nodes

    noop_llm_propose(monkeypatch)

    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    make_excel(xlsx)
    ingest_excel(xlsx, db_path=db)
    propose_measures_edges(db_path=db)

    concepts = [n for n in all_nodes(NodeType.CONCEPT, db) if "revenue" in n["id"]]
    assert len(concepts) == 1

    metrics_measuring_revenue = get_neighbors(
        concepts[0]["id"], EdgeType.MEASURES, direction="in", db_path=db
    )
    assert len(metrics_measuring_revenue) >= 1
    labels = [m["label"] for m in metrics_measuring_revenue]
    assert any("Umsatz" in lbl for lbl in labels)
