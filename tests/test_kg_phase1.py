"""Phase 1 KG tests — no real Excel files required.

All tests are offline / no LLM calls.
Run: uv run pytest tests/test_kg_phase1.py -v
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_excel(path: str) -> None:
    """Write a minimal synthetic NIQ-style Excel file."""
    df = pd.DataFrame({
        "Marke":          ["ANIMONDA", "MJAMJAM", "NESTLE"] * 10,
        "Tierart":        ["Katze"] * 30,
        "Umsatz 52 W bis 29/03/26":       [100.0, 200.0, 300.0] * 10,
        "Umsatz VJ 52 W bis 29/03/26":    [90.0,  180.0, 270.0] * 10,
        "Umsatz % Ver. 52 W bis 29/03/26": [11.1,  11.1,  11.1] * 10,
        "Menge 52 W bis 29/03/26":        [10, 20, 30] * 10,
    })
    df.to_excel(path, index=False)


# ---------------------------------------------------------------------------
# 1. Models
# ---------------------------------------------------------------------------

def test_node_and_edge_dataclasses():
    from kg.models import Edge, EdgeType, Node, NodeType
    n = Node(id="metric::test::Umsatz", node_type=NodeType.METRIC, label="Umsatz")
    assert n.node_type == NodeType.METRIC

    e = Edge(src_id=n.id, dst_id="file::test", edge_type=EdgeType.AVAILABLE_IN)
    assert e.confidence == 1.0
    assert e.source == "ingest"


# ---------------------------------------------------------------------------
# 2. Store (DuckDB)
# ---------------------------------------------------------------------------

def test_store_init_and_upsert(tmp_path):
    from kg.models import Edge, EdgeType, Node, NodeType
    from kg.store import all_nodes, init_store, upsert_edge, upsert_node

    db = str(tmp_path / "test.duckdb")
    init_store(db)

    n = Node(id="file::cat_mat", node_type=NodeType.FILE, label="cat_mat")
    upsert_node(n, db)

    nodes = all_nodes(NodeType.FILE, db)
    assert len(nodes) == 1
    assert nodes[0]["id"] == "file::cat_mat"

    # Upsert again — should not duplicate
    upsert_node(n, db)
    assert len(all_nodes(NodeType.FILE, db)) == 1


def test_store_get_neighbors(tmp_path):
    from kg.models import Edge, EdgeType, Node, NodeType
    from kg.store import get_neighbors, init_store, upsert_edge, upsert_node

    db = str(tmp_path / "test.duckdb")
    init_store(db)

    file_node = Node(id="file::cat_mat", node_type=NodeType.FILE, label="cat_mat")
    metric_node = Node(id="metric::cat_mat::Umsatz", node_type=NodeType.METRIC, label="Umsatz")
    upsert_node(file_node, db)
    upsert_node(metric_node, db)
    upsert_edge(Edge(
        src_id=metric_node.id,
        dst_id=file_node.id,
        edge_type=EdgeType.AVAILABLE_IN,
    ), db)

    # Outgoing from metric -> should reach file
    neighbours = get_neighbors(metric_node.id, EdgeType.AVAILABLE_IN, "out", db_path=db)
    assert len(neighbours) == 1
    assert neighbours[0]["id"] == file_node.id

    # Incoming to file -> should reach metric
    neighbours_in = get_neighbors(file_node.id, EdgeType.AVAILABLE_IN, "in", db_path=db)
    assert len(neighbours_in) == 1
    assert neighbours_in[0]["id"] == metric_node.id


# ---------------------------------------------------------------------------
# 3. Excel reader
# ---------------------------------------------------------------------------

def test_excel_reader(tmp_path):
    from kg.ingest.excel_reader import read_excel_schema

    xlsx = str(tmp_path / "test.xlsx")
    _make_excel(xlsx)
    schema = read_excel_schema(xlsx)

    assert len(schema) == 1  # one sheet
    sheet = list(schema.values())[0]
    assert sheet["row_count"] == 30
    assert "Marke" in sheet["columns"]
    assert "Umsatz 52 W bis 29/03/26" in sheet["columns"]


def test_excel_reader_dtypes(tmp_path):
    from kg.ingest.excel_reader import read_excel_schema

    xlsx = str(tmp_path / "test.xlsx")
    _make_excel(xlsx)
    schema = read_excel_schema(xlsx)
    cols = list(schema.values())[0]["columns"]

    umsatz = cols["Umsatz 52 W bis 29/03/26"]
    assert umsatz["dtype"] in ("float64", "Float64", "int64")
    assert "min" in umsatz
    assert "max" in umsatz


# ---------------------------------------------------------------------------
# 4. Classifier
# ---------------------------------------------------------------------------

def test_classifier_metric_vs_dimension():
    from kg.ingest.classifier import classify_column
    from kg.models import NodeType

    metric_meta = {"dtype": "float64", "n_unique": 1000, "_row_count": 1000}
    assert classify_column("Umsatz", metric_meta) == NodeType.METRIC

    dim_meta = {"dtype": "object", "n_unique": 5, "_row_count": 1000}
    assert classify_column("Marke", dim_meta) == NodeType.DIMENSION

    # Low-cardinality numeric -> Dimension
    low_card = {"dtype": "int64", "n_unique": 3, "_row_count": 1000}
    assert classify_column("Tierart_code", low_card) == NodeType.DIMENSION

    # Datetime -> always Dimension
    dt_meta = {"dtype": "object", "likely_datetime": True, "n_unique": 50, "_row_count": 1000}
    assert classify_column("Datum", dt_meta) == NodeType.DIMENSION


# ---------------------------------------------------------------------------
# 5. Suffix detector
# ---------------------------------------------------------------------------

def test_suffix_detector_vj():
    from kg.ingest.suffix_detector import detect_suffix_pairs

    cols = [
        "Umsatz 52 W bis 29/03/26",
        "Umsatz VJ 52 W bis 29/03/26",
        "Menge 52 W bis 29/03/26",
    ]
    pairs = detect_suffix_pairs(cols)
    assert "Umsatz VJ 52 W bis 29/03/26" in pairs
    base, etype = pairs["Umsatz VJ 52 W bis 29/03/26"]
    assert base == "Umsatz 52 W bis 29/03/26"
    assert etype == "PRIOR_PERIOD_OF"


def test_suffix_detector_delta():
    from kg.ingest.suffix_detector import detect_suffix_pairs

    cols = [
        "Umsatz 52 W bis 29/03/26",
        "Umsatz % Ver. 52 W bis 29/03/26",
    ]
    pairs = detect_suffix_pairs(cols)
    assert "Umsatz % Ver. 52 W bis 29/03/26" in pairs
    _, etype = pairs["Umsatz % Ver. 52 W bis 29/03/26"]
    assert etype == "DELTA_OF"


def test_suffix_detector_no_false_positives():
    from kg.ingest.suffix_detector import detect_suffix_pairs

    cols = ["Marke", "Tierart", "Umsatz"]
    assert detect_suffix_pairs(cols) == {}


# ---------------------------------------------------------------------------
# 6. Period parser
# ---------------------------------------------------------------------------

def test_period_parser_rolling():
    from kg.ingest.period_parser import parse_period

    p = parse_period("Umsatz 52 W bis 29/03/26")
    assert p is not None
    assert p.window_weeks == 52
    assert p.end_date == "29/03/26"
    assert p.period_type == "rolling"
    assert "52W" in p.label


def test_period_parser_named():
    from kg.ingest.period_parser import parse_period

    p = parse_period("MAT 2025")
    assert p is not None
    assert p.period_type == "MAT"
    assert p.label == "MAT_2025"


def test_period_parser_no_match():
    from kg.ingest.period_parser import parse_period

    assert parse_period("Marke") is None
    assert parse_period("Tierart") is None


def test_extract_periods_deduplication():
    from kg.ingest.period_parser import extract_periods_from_columns

    cols = [
        "Umsatz 52 W bis 29/03/26",
        "Umsatz VJ 52 W bis 29/03/26",       # same period, different col
        "Umsatz % Ver. 52 W bis 29/03/26",   # same period again
        "Menge 4 W bis 29/03/26",            # different window
    ]
    periods = extract_periods_from_columns(cols)
    labels = {p.label for p in periods}
    assert len(labels) == 2  # 52W and 4W, deduplicated


# ---------------------------------------------------------------------------
# 7. Full pipeline (end-to-end, synthetic Excel, tmp DuckDB)
# ---------------------------------------------------------------------------

def test_pipeline_end_to_end(tmp_path):
    from kg.ingest.pipeline import ingest_excel
    from kg.models import EdgeType, NodeType
    from kg.store import all_nodes, get_neighbors

    xlsx = str(tmp_path / "cat_mat.xlsx")
    db   = str(tmp_path / "kg.duckdb")
    _make_excel(xlsx)

    summary = ingest_excel(xlsx, db_path=db)
    assert len(summary) == 1  # one sheet
    counts = list(summary.values())[0]
    assert counts["nodes"] > 0
    assert counts["edges"] > 0

    # File node exists
    files = all_nodes(NodeType.FILE, db)
    assert any("cat_mat" in f["id"] for f in files)

    # Metric nodes exist
    metrics = all_nodes(NodeType.METRIC, db)
    assert len(metrics) > 0

    # AVAILABLE_IN edges: a metric should reach the file
    metric_id = metrics[0]["id"]
    neighbours = get_neighbors(metric_id, EdgeType.AVAILABLE_IN, "out", db_path=db)
    assert len(neighbours) == 1
    assert neighbours[0]["node_type"] == NodeType.FILE.value

    # Period nodes exist
    periods = all_nodes(NodeType.PERIOD, db)
    assert len(periods) > 0

    # PRIOR_PERIOD_OF edge: VJ col -> base col
    vj_col = "Umsatz VJ 52 W bis 29/03/26"
    vj_nodes = [n for n in all_nodes(NodeType.METRIC, db) if vj_col in n["label"]]
    assert len(vj_nodes) == 1
    ppo_edges = get_neighbors(vj_nodes[0]["id"], EdgeType.PRIOR_PERIOD_OF, "out", db_path=db)
    assert len(ppo_edges) == 1


# ---------------------------------------------------------------------------
# 8. Agent graph routing — ingestion_mode=True must route to kg_ingest_node
# ---------------------------------------------------------------------------

def test_graph_routes_to_kg_ingest_when_mode_set():
    from agent.graph import route_after_orchestrator
    from agent.state import AnalyticsState

    state = AnalyticsState(
        user_query="ingest cat_mat.xlsx into the knowledge graph",
        ingestion_mode=True,
        kg_ingest_path="/tmp/cat_mat.xlsx",
    )
    assert route_after_orchestrator(state) == "kg_ingest_node"


def test_graph_does_not_route_to_kg_ingest_normally():
    from agent.graph import route_after_orchestrator
    from agent.state import AnalyticsState

    state = AnalyticsState(
        user_query="what is the revenue of ANIMONDA?",
        ingestion_mode=False,
    )
    # No plan, no load_file_path -> should go to END, not kg_ingest_node
    result = route_after_orchestrator(state)
    assert result != "kg_ingest_node"
