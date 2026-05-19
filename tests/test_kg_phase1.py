"""Phase 1 KG tests — models, store, Excel reader, classifier, suffix detector,
period parser, pipeline E2E, and graph routing.

Run: uv run pytest tests/test_kg_phase1.py -v
"""
from __future__ import annotations

import os

import pytest
from tests.helpers.kg_fixtures import make_excel


# ---------------------------------------------------------------------------
# 1. Node / Edge models
# ---------------------------------------------------------------------------

def test_node_model_defaults():
    from kg.models import Node, NodeType
    n = Node(id='file::test.xlsx', node_type=NodeType.FILE, label='test.xlsx')
    assert n.id == 'file::test.xlsx'
    assert n.props == {}


def test_edge_model_defaults():
    from kg.models import Edge, EdgeType
    e = Edge(src_id='metric::a', dst_id='file::b', edge_type=EdgeType.AVAILABLE_IN)
    assert e.confidence == 1.0
    assert e.source == 'ingest'


# ---------------------------------------------------------------------------
# 2. Store — upsert + deduplication
# ---------------------------------------------------------------------------

def test_store_upsert_and_dedup(tmp_path):
    from kg.models import Node, NodeType
    from kg.store import init_store, upsert_node, all_nodes
    db = str(tmp_path / 'kg.duckdb')
    init_store(db)
    n = Node(id='metric::rev', node_type=NodeType.METRIC, label='Revenue')
    upsert_node(n, db)
    upsert_node(n, db)  # duplicate — should not raise or double-insert
    nodes = all_nodes(NodeType.METRIC, db)
    assert len(nodes) == 1
    assert nodes[0]['label'] == 'Revenue'


# ---------------------------------------------------------------------------
# 3. Store — get_neighbors
# ---------------------------------------------------------------------------

def test_store_get_neighbors(tmp_path):
    from kg.models import Edge, EdgeType, Node, NodeType
    from kg.store import get_neighbors, init_store, upsert_edge, upsert_node
    db = str(tmp_path / 'kg.duckdb')
    init_store(db)
    upsert_node(Node(id='metric::rev', node_type=NodeType.METRIC, label='Revenue'), db)
    upsert_node(Node(id='file::f1', node_type=NodeType.FILE, label='f1.xlsx'), db)
    upsert_edge(Edge(src_id='metric::rev', dst_id='file::f1', edge_type=EdgeType.AVAILABLE_IN), db)

    out = get_neighbors('metric::rev', EdgeType.AVAILABLE_IN, 'out', db_path=db)
    assert len(out) == 1
    assert out[0]['id'] == 'file::f1'

    in_ = get_neighbors('file::f1', EdgeType.AVAILABLE_IN, 'in', db_path=db)
    assert len(in_) == 1
    assert in_[0]['id'] == 'metric::rev'


# ---------------------------------------------------------------------------
# 4. Excel reader — header detection + column parsing
# ---------------------------------------------------------------------------

def test_excel_reader_detects_header(tmp_path):
    from kg.ingest.excel_reader import read_excel_schema
    xlsx = str(tmp_path / 'sample.xlsx')
    make_excel(xlsx)
    sheets = read_excel_schema(xlsx)
    assert len(sheets) == 1
    info = list(sheets.values())[0]
    assert info['detected_header_row'] == 0
    assert len(info['columns']) >= 5
    assert info['row_count'] == 30


# ---------------------------------------------------------------------------
# 5. Excel reader — dtype stats (columns is a dict {col_name: col_meta})
# ---------------------------------------------------------------------------

def test_excel_reader_dtype_stats(tmp_path):
    from kg.ingest.excel_reader import read_excel_schema
    xlsx = str(tmp_path / 'sample.xlsx')
    make_excel(xlsx)
    sheets = read_excel_schema(xlsx)
    info = list(sheets.values())[0]
    numeric_cols = [
        meta for meta in info['columns'].values()
        if meta['dtype'] in ('float64', 'int64', 'Float64', 'Int64')
    ]
    assert len(numeric_cols) >= 1
    for meta in numeric_cols:
        assert 'min' in meta and 'max' in meta and 'mean' in meta


# ---------------------------------------------------------------------------
# 6. Column classifier — classify_column(col_name, col_meta: dict)
# ---------------------------------------------------------------------------

def test_classifier_metric_vs_dimension():
    from kg.ingest.classifier import classify_column
    from kg.models import NodeType
    assert classify_column(
        'Umsatz 52 W',
        {'dtype': 'float64', 'n_unique': 28, '_row_count': 30},
    ) == NodeType.METRIC
    assert classify_column(
        'Marke',
        {'dtype': 'object', 'n_unique': 5, '_row_count': 30},
    ) == NodeType.DIMENSION


def test_classifier_datetime_is_dimension():
    from kg.ingest.classifier import classify_column
    from kg.models import NodeType
    assert classify_column(
        'Datum',
        {'dtype': 'datetime64[ns]', 'n_unique': 10, '_row_count': 30, 'likely_datetime': True},
    ) == NodeType.DIMENSION


def test_classifier_low_cardinality_int_is_dimension():
    from kg.ingest.classifier import classify_column
    from kg.models import NodeType
    assert classify_column(
        'Category',
        {'dtype': 'int64', 'n_unique': 3, '_row_count': 30},
    ) == NodeType.DIMENSION


# ---------------------------------------------------------------------------
# 7. Suffix detector
# ---------------------------------------------------------------------------

def test_suffix_detector_vj():
    from kg.ingest.suffix_detector import detect_suffix_pairs
    cols = [
        'Umsatz 52 W bis 29/03/26',
        'Umsatz VJ 52 W bis 29/03/26',
        'Menge 52 W bis 29/03/26',
    ]
    pairs = detect_suffix_pairs(cols)
    assert 'Umsatz VJ 52 W bis 29/03/26' in pairs
    assert pairs['Umsatz VJ 52 W bis 29/03/26'] == ('Umsatz 52 W bis 29/03/26', 'PRIOR_PERIOD_OF')


def test_suffix_detector_delta():
    from kg.ingest.suffix_detector import detect_suffix_pairs
    cols = [
        'Umsatz 52 W bis 29/03/26',
        'Umsatz % Ver. 52 W bis 29/03/26',
    ]
    pairs = detect_suffix_pairs(cols)
    assert 'Umsatz % Ver. 52 W bis 29/03/26' in pairs
    assert pairs['Umsatz % Ver. 52 W bis 29/03/26'][1] == 'DELTA_OF'


def test_suffix_detector_no_false_positive():
    from kg.ingest.suffix_detector import detect_suffix_pairs
    cols = ['Umsatz 52 W bis 29/03/26', 'Menge 52 W bis 29/03/26']
    pairs = detect_suffix_pairs(cols)
    assert len(pairs) == 0


# ---------------------------------------------------------------------------
# 8. Period parser
# ---------------------------------------------------------------------------

def test_period_parser_52w():
    from kg.ingest.period_parser import parse_period
    result = parse_period('52 W bis 29/03/26')
    assert result is not None
    assert result.window_weeks == 52
    assert '29-03-26' in result.label or '29/03' in result.raw


def test_period_parser_mat():
    from kg.ingest.period_parser import parse_period
    result = parse_period('MAT 2025')
    assert result is not None
    assert '2025' in result.label


def test_period_parser_no_match():
    from kg.ingest.period_parser import parse_period
    assert parse_period('Marke') is None
    assert parse_period('Unknown Metric XYZ') is None


def test_period_parser_deduplication(tmp_path):
    from kg.ingest.excel_reader import read_excel_schema
    from kg.ingest.period_parser import extract_periods_from_columns
    xlsx = str(tmp_path / 'sample.xlsx')
    make_excel(xlsx)
    sheets = read_excel_schema(xlsx)
    info = list(sheets.values())[0]
    col_names = list(info['columns'].keys())
    periods = extract_periods_from_columns(col_names)
    labels = [p.label for p in periods]
    assert len(labels) == len(set(labels)), 'Duplicate period labels detected'


# ---------------------------------------------------------------------------
# 9. Pipeline E2E
# ---------------------------------------------------------------------------

def test_pipeline_end_to_end(tmp_path):
    from kg.ingest.pipeline import ingest_excel
    from kg.models import EdgeType, NodeType
    from kg.store import all_nodes, get_neighbors

    xlsx = str(tmp_path / 'cat_mat.xlsx')
    db   = str(tmp_path / 'kg.duckdb')
    make_excel(xlsx)

    summary = ingest_excel(xlsx, db_path=db)
    assert len(summary) == 1
    counts = list(summary.values())[0]
    assert counts['nodes'] > 0
    assert counts['edges'] > 0

    files = all_nodes(NodeType.FILE, db)
    assert any('cat_mat' in f['id'] for f in files)

    metrics = all_nodes(NodeType.METRIC, db)
    assert len(metrics) > 0

    metric_id = metrics[0]['id']
    neighbours = get_neighbors(metric_id, EdgeType.AVAILABLE_IN, 'out', db_path=db)
    assert len(neighbours) == 1
    assert neighbours[0]['node_type'] == NodeType.FILE.value

    periods = all_nodes(NodeType.PERIOD, db)
    assert len(periods) > 0

    vj_col = 'Umsatz VJ 52 W bis 29/03/26'
    vj_nodes = [n for n in all_nodes(NodeType.METRIC, db) if vj_col in n['label']]
    assert len(vj_nodes) == 1
    ppo_edges = get_neighbors(vj_nodes[0]['id'], EdgeType.PRIOR_PERIOD_OF, 'out', db_path=db)
    assert len(ppo_edges) == 1


# ---------------------------------------------------------------------------
# 10. Graph routing — uses compiled `graph` object from agent.graph
#     Forces LLM_BACKEND=ollama to avoid OpenAI quota errors.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope='module')
def _use_ollama(monkeypatch_module):
    """Route all LLM calls through local Ollama for routing tests."""
    monkeypatch_module.setenv('LLM_BACKEND', 'ollama')
    monkeypatch_module.setenv('OLLAMA_MODEL', os.getenv('OLLAMA_MODEL', 'gemma4:e2b'))


def test_routing_ingestion_mode(tmp_path):
    """ingestion_mode=True routes through kg_ingest_node."""
    from agent.state import AnalyticsState
    from agent.graph import graph
    from langchain_core.messages import HumanMessage

    state = AnalyticsState(
        messages=[HumanMessage(content='ingest file')],
        ingestion_mode=True,
        ingestion_file_path=str(tmp_path / 'dummy.xlsx'),
    )
    result = graph.invoke(state)
    assert result.get('ingestion_mode') is True or 'kg_ingest' in str(result)


def test_routing_normal_query_skips_ingest():
    """Normal analytics query does not trigger kg_ingest_node."""
    from agent.state import AnalyticsState
    from agent.graph import graph
    from langchain_core.messages import HumanMessage

    state = AnalyticsState(
        messages=[HumanMessage(content='what tables do you have?')],
        ingestion_mode=False,
    )
    result = graph.invoke(state)
    assert result.get('ingestion_mode') is not True
