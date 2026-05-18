"""Phase 3 KG tests — join detection and BRIDGES authoring.

Run: uv run pytest tests/test_kg_phase3.py -v
"""
from __future__ import annotations

import pytest
from tests.helpers.kg_fixtures import make_excel, noop_llm_propose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ingest_two_files(tmp_path, monkeypatch):
    """Ingest cat_mat.xlsx and dog_mat.xlsx (same schema, different names).
    Returns (db_path, cat_file_id, dog_file_id).
    """
    import pandas as pd
    from kg.ingest.pipeline import ingest_excel
    from kg.models import NodeType
    from kg.store import all_nodes

    noop_llm_propose(monkeypatch)

    cat_xlsx = str(tmp_path / "cat_mat.xlsx")
    dog_xlsx = str(tmp_path / "dog_mat.xlsx")
    db        = str(tmp_path / "kg.duckdb")

    make_excel(cat_xlsx)
    # dog_mat: same structure, different brand values — shares 'Marke' dimension
    make_excel(dog_xlsx, extra_columns={})

    ingest_excel(cat_xlsx, db_path=db)
    ingest_excel(dog_xlsx, db_path=db)

    files = all_nodes(NodeType.FILE, db)
    assert len(files) == 2
    cat_id = next(f["id"] for f in files if "cat_mat" in f["id"])
    dog_id = next(f["id"] for f in files if "dog_mat" in f["id"])
    return db, cat_id, dog_id


# ---------------------------------------------------------------------------
# 1. Join detector — dimension overlap detection
# ---------------------------------------------------------------------------

def test_join_detector_finds_shared_dimensions(tmp_path, monkeypatch):
    """Two files with the same 'Marke' column → at least one exact join proposal."""
    from kg.ingest.join_detector import detect_join_candidates

    db, cat_id, dog_id = _ingest_two_files(tmp_path, monkeypatch)
    proposals = detect_join_candidates(db_path=db)

    assert len(proposals) >= 1
    exact = [p for p in proposals if p.match_type == "exact"]
    assert len(exact) >= 1
    assert any(p.col_a == p.col_b for p in exact)


def test_join_detector_confidence_exact():
    """Exact match proposals always have confidence == 1.0."""
    from kg.ingest.join_detector import JoinProposal
    p = JoinProposal(
        file_a="file::a", file_b="file::b",
        col_a="Marke", col_b="Marke",
        match_type="exact", confidence=1.0,
    )
    assert p.confidence == 1.0
    assert p.match_type == "exact"


def test_join_detector_confidence_alias():
    """Alias match proposals have confidence == 0.75."""
    from kg.ingest.join_detector import JoinProposal
    p = JoinProposal(
        file_a="file::a", file_b="file::b",
        col_a="Marke", col_b="Brand",
        match_type="alias", confidence=0.75,
    )
    assert p.confidence == 0.75


def test_join_detector_writes_to_store(tmp_path, monkeypatch):
    """After detection, JOINABLE_ON edges exist in the store."""
    import duckdb
    from kg.ingest.join_detector import detect_join_candidates

    db, _, _ = _ingest_two_files(tmp_path, monkeypatch)
    detect_join_candidates(db_path=db)

    con = duckdb.connect(db)
    rows = con.execute(
        "SELECT COUNT(*) FROM edges WHERE edge_type='JOINABLE_ON'"
    ).fetchone()
    con.close()
    assert rows[0] >= 1


def test_join_detector_single_file_noop(tmp_path, monkeypatch):
    """Single file in KG → no proposals."""
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.join_detector import detect_join_candidates

    noop_llm_propose(monkeypatch)
    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    make_excel(xlsx)
    ingest_excel(xlsx, db_path=db)

    proposals = detect_join_candidates(db_path=db)
    assert len(proposals) == 0


# ---------------------------------------------------------------------------
# 2. BRIDGES authoring (programmatic API, no terminal interaction)
# ---------------------------------------------------------------------------

def test_write_bridge(tmp_path, monkeypatch):
    """write_bridge() creates a BRIDGES edge between two concept nodes."""
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from hitl.bridge_author import write_bridge
    from kg.models import EdgeType
    from kg.store import get_neighbors
    import duckdb

    noop_llm_propose(monkeypatch)
    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    make_excel(xlsx)
    ingest_excel(xlsx, db_path=db)
    propose_measures_edges(db_path=db)

    # Declare: MAT 'revenue' bridges panel 'revenue' (same concept here, but
    # in practice would be revenue <-> spend or similar across different files)
    write_bridge(db, "revenue", "volume",
                 file_a_id="file::cat_mat",
                 file_b_id="file::cat_mat")  # same file is fine for unit test

    con = duckdb.connect(db)
    row = con.execute(
        "SELECT source, confidence FROM edges "
        "WHERE src_id='concept::revenue' AND dst_id='concept::volume' AND edge_type='BRIDGES'"
    ).fetchone()
    con.close()
    assert row is not None
    assert row[0] == "declared"
    assert row[1] == 1.0


def test_confirm_join(tmp_path, monkeypatch):
    """confirm_join() promotes a proposed JOINABLE_ON to declared."""
    from kg.ingest.join_detector import detect_join_candidates
    from hitl.bridge_author import confirm_join, _fetch_confirmed_joins, _fetch_pending_joins

    db, cat_id, dog_id = _ingest_two_files(tmp_path, monkeypatch)
    proposals = detect_join_candidates(db_path=db)
    assert len(proposals) >= 1

    p = proposals[0]
    confirm_join(db, p.file_a, p.file_b, {
        "join_col_a": p.col_a,
        "join_col_b": p.col_b,
        "match_type": p.match_type,
    })

    confirmed = _fetch_confirmed_joins(db)
    assert len(confirmed) >= 1


# ---------------------------------------------------------------------------
# 3. Query pattern 4: cross-file join expressibility
# ---------------------------------------------------------------------------

def test_query_pattern_4_join_traversal(tmp_path, monkeypatch):
    """After confirming a join, can traverse JOINABLE_ON from either file."""
    from kg.ingest.join_detector import detect_join_candidates
    from hitl.bridge_author import confirm_join
    from kg.models import EdgeType
    from kg.store import get_neighbors

    db, cat_id, dog_id = _ingest_two_files(tmp_path, monkeypatch)
    proposals = detect_join_candidates(db_path=db)
    p = proposals[0]
    confirm_join(db, p.file_a, p.file_b, {
        "join_col_a": p.col_a,
        "join_col_b": p.col_b,
        "match_type": p.match_type,
    })

    # Forward traversal: cat → dog
    neighbours = get_neighbors(cat_id, EdgeType.JOINABLE_ON, "out", db_path=db)
    assert any(n["id"] == dog_id for n in neighbours)


def test_query_pattern_4_bridges_traversal(tmp_path, monkeypatch):
    """After writing a BRIDGE, reverse traversal finds the bridged concept."""
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from hitl.bridge_author import write_bridge
    from kg.models import EdgeType
    from kg.store import get_neighbors

    noop_llm_propose(monkeypatch)
    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    make_excel(xlsx)
    ingest_excel(xlsx, db_path=db)
    propose_measures_edges(db_path=db)

    write_bridge(db, "revenue", "spend_per_buyer",
                 file_a_id="file::cat_mat", file_b_id="file::cat_mat")

    bridged = get_neighbors("concept::revenue", EdgeType.BRIDGES, "out", db_path=db)
    assert any("spend_per_buyer" in n["id"] for n in bridged)
