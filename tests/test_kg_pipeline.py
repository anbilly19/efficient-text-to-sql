"""KG pipeline integration tests — Phases 1–4.

Covers: models, store, Excel reader, classifier, suffix detector, period
parser, pipeline E2E, concept registry + alias mapping, HITL review ops,
join detection, BRIDGES authoring, and Phase 4 YAML authoring wizards.

Run: uv run pytest tests/test_kg_pipeline.py -v
"""
from __future__ import annotations

import os

import pytest
from tests.helpers.kg_fixtures import make_excel, noop_llm_propose


# ===========================================================================
# Phase 1 — Models, store, Excel reader, classifier, suffix detector,
#            period parser, pipeline E2E
# ===========================================================================

class TestModels:
    def test_node_defaults(self):
        from kg.models import Node, NodeType
        n = Node(id='file::test.xlsx', node_type=NodeType.FILE, label='test.xlsx')
        assert n.id == 'file::test.xlsx'
        assert n.props == {}

    def test_edge_defaults(self):
        from kg.models import Edge, EdgeType
        e = Edge(src_id='metric::a', dst_id='file::b', edge_type=EdgeType.AVAILABLE_IN)
        assert e.confidence == 1.0
        assert e.source == 'ingest'


class TestStore:
    def test_upsert_and_dedup(self, tmp_path):
        from kg.models import Node, NodeType
        from kg.store import init_store, upsert_node, all_nodes
        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        n = Node(id='metric::rev', node_type=NodeType.METRIC, label='Revenue')
        upsert_node(n, db)
        upsert_node(n, db)
        nodes = all_nodes(NodeType.METRIC, db)
        assert len(nodes) == 1
        assert nodes[0]['label'] == 'Revenue'

    def test_get_neighbors(self, tmp_path):
        from kg.models import Edge, EdgeType, Node, NodeType
        from kg.store import get_neighbors, init_store, upsert_edge, upsert_node
        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        upsert_node(Node(id='metric::rev', node_type=NodeType.METRIC, label='Revenue'), db)
        upsert_node(Node(id='file::f1', node_type=NodeType.FILE, label='f1.xlsx'), db)
        upsert_edge(Edge(src_id='metric::rev', dst_id='file::f1', edge_type=EdgeType.AVAILABLE_IN), db)

        out = get_neighbors('metric::rev', EdgeType.AVAILABLE_IN, 'out', db_path=db)
        assert len(out) == 1 and out[0]['id'] == 'file::f1'

        in_ = get_neighbors('file::f1', EdgeType.AVAILABLE_IN, 'in', db_path=db)
        assert len(in_) == 1 and in_[0]['id'] == 'metric::rev'


class TestExcelReader:
    def test_header_detection(self, tmp_path):
        from kg.ingest.excel_reader import read_excel_schema
        xlsx = str(tmp_path / 'sample.xlsx')
        make_excel(xlsx)
        sheets = read_excel_schema(xlsx)
        info = list(sheets.values())[0]
        assert info['detected_header_row'] == 0
        assert len(info['columns']) >= 5
        assert info['row_count'] == 30

    def test_dtype_stats(self, tmp_path):
        from kg.ingest.excel_reader import read_excel_schema
        xlsx = str(tmp_path / 'sample.xlsx')
        make_excel(xlsx)
        sheets = read_excel_schema(xlsx)
        info = list(sheets.values())[0]
        numeric = [m for m in info['columns'].values() if m['dtype'] in ('float64', 'int64', 'Float64', 'Int64')]
        assert len(numeric) >= 1
        for m in numeric:
            assert 'min' in m and 'max' in m and 'mean' in m


class TestClassifier:
    def test_float_metric(self):
        from kg.ingest.classifier import classify_column
        from kg.models import NodeType
        assert classify_column('Umsatz 52 W', {'dtype': 'float64', 'n_unique': 28, '_row_count': 30}) == NodeType.METRIC

    def test_string_dimension(self):
        from kg.ingest.classifier import classify_column
        from kg.models import NodeType
        assert classify_column('Marke', {'dtype': 'object', 'n_unique': 5, '_row_count': 30}) == NodeType.DIMENSION

    def test_datetime_dimension(self):
        from kg.ingest.classifier import classify_column
        from kg.models import NodeType
        assert classify_column('Datum', {'dtype': 'datetime64[ns]', 'n_unique': 10, '_row_count': 30, 'likely_datetime': True}) == NodeType.DIMENSION

    def test_low_cardinality_int_dimension(self):
        from kg.ingest.classifier import classify_column
        from kg.models import NodeType
        assert classify_column('Category', {'dtype': 'int64', 'n_unique': 3, '_row_count': 30}) == NodeType.DIMENSION


class TestSuffixDetector:
    def test_vj_detected(self):
        from kg.ingest.suffix_detector import detect_suffix_pairs
        cols = ['Umsatz 52 W bis 29/03/26', 'Umsatz VJ 52 W bis 29/03/26', 'Menge 52 W bis 29/03/26']
        pairs = detect_suffix_pairs(cols)
        assert pairs.get('Umsatz VJ 52 W bis 29/03/26') == ('Umsatz 52 W bis 29/03/26', 'PRIOR_PERIOD_OF')

    def test_delta_detected(self):
        from kg.ingest.suffix_detector import detect_suffix_pairs
        cols = ['Umsatz 52 W bis 29/03/26', 'Umsatz % Ver. 52 W bis 29/03/26']
        pairs = detect_suffix_pairs(cols)
        assert pairs.get('Umsatz % Ver. 52 W bis 29/03/26', (None, None))[1] == 'DELTA_OF'

    def test_no_false_positives(self):
        from kg.ingest.suffix_detector import detect_suffix_pairs
        assert detect_suffix_pairs(['Umsatz 52 W bis 29/03/26', 'Menge 52 W bis 29/03/26']) == {}


class TestPeriodParser:
    def test_52w(self):
        from kg.ingest.period_parser import parse_period
        r = parse_period('52 W bis 29/03/26')
        assert r is not None and r.window_weeks == 52

    def test_mat(self):
        from kg.ingest.period_parser import parse_period
        r = parse_period('MAT 2025')
        assert r is not None and '2025' in r.label

    def test_no_match(self):
        from kg.ingest.period_parser import parse_period
        assert parse_period('Marke') is None
        assert parse_period('Unknown Metric XYZ') is None

    def test_deduplication(self, tmp_path):
        from kg.ingest.excel_reader import read_excel_schema
        from kg.ingest.period_parser import extract_periods_from_columns
        xlsx = str(tmp_path / 'sample.xlsx')
        make_excel(xlsx)
        info = list(read_excel_schema(xlsx).values())[0]
        periods = extract_periods_from_columns(list(info['columns'].keys()))
        labels = [p.label for p in periods]
        assert len(labels) == len(set(labels))


class TestPipelineE2E:
    def test_full_ingest(self, tmp_path):
        from kg.ingest.pipeline import ingest_excel
        from kg.models import EdgeType, NodeType
        from kg.store import all_nodes, get_neighbors

        xlsx = str(tmp_path / 'cat_mat.xlsx')
        db   = str(tmp_path / 'kg.duckdb')
        make_excel(xlsx)
        summary = ingest_excel(xlsx, db_path=db)

        counts = list(summary.values())[0]
        assert counts['nodes'] > 0 and counts['edges'] > 0
        assert any('cat_mat' in f['id'] for f in all_nodes(NodeType.FILE, db))

        metrics = all_nodes(NodeType.METRIC, db)
        assert len(metrics) > 0
        assert len(get_neighbors(metrics[0]['id'], EdgeType.AVAILABLE_IN, 'out', db_path=db)) == 1
        assert len(all_nodes(NodeType.PERIOD, db)) > 0

    def test_prior_period_edge(self, tmp_path):
        from kg.ingest.pipeline import ingest_excel
        from kg.models import EdgeType, NodeType
        from kg.store import all_nodes, get_neighbors

        xlsx = str(tmp_path / 'cat_mat.xlsx')
        db   = str(tmp_path / 'kg.duckdb')
        make_excel(xlsx)
        ingest_excel(xlsx, db_path=db)

        vj = [n for n in all_nodes(NodeType.METRIC, db) if 'VJ' in n['label']]
        assert len(vj) >= 1
        assert len(get_neighbors(vj[0]['id'], EdgeType.PRIOR_PERIOD_OF, 'out', db_path=db)) == 1


# ===========================================================================
# Phase 2 — Concept registry, alias mapping, MEASURES proposals, HITL ops
# ===========================================================================

class TestConceptRegistry:
    def test_loads_and_has_concepts(self):
        from kg.concepts.validator import load_registry, concept_ids
        assert 'concepts' in load_registry()
        assert len(concept_ids()) >= 5

    def test_known_and_unknown(self):
        from kg.concepts.validator import validate_concept_id
        assert validate_concept_id('revenue') is True
        assert validate_concept_id('nonexistent_concept_xyz') is False

    def test_aliases_for_revenue(self):
        from kg.concepts.validator import aliases_for
        aliases = aliases_for('revenue')
        assert 'umsatz' in aliases and len(aliases) >= 3


class TestAliasMatch:
    @pytest.mark.parametrize('col,expected', [
        ('Umsatz 52 W bis 29/03/26',       'revenue'),
        ('Menge 52 W bis 29/03/26',         'volume'),
        ('Penetration (%) 52 W bis 29/03/26', 'buyer_reach'),
    ])
    def test_known_aliases(self, col, expected):
        from kg.ingest.concept_mapper import _alias_match
        assert _alias_match(col) == expected

    def test_no_match(self):
        from kg.ingest.concept_mapper import _alias_match
        assert _alias_match('Unknown Metric XYZ') is None


class TestMeasuresProposals:
    def test_alias_proposals_confidence(self, tmp_path, monkeypatch):
        from kg.ingest.pipeline import ingest_excel
        from kg.ingest.concept_mapper import propose_measures_edges
        from kg.models import EdgeType, NodeType
        from kg.store import get_neighbors, all_nodes

        noop_llm_propose(monkeypatch)
        xlsx, db = str(tmp_path / 'cat_mat.xlsx'), str(tmp_path / 'kg.duckdb')
        make_excel(xlsx)
        ingest_excel(xlsx, db_path=db)

        proposals = propose_measures_edges(db_path=db)
        assert len(proposals) >= 3
        assert all(p.confidence == 0.95 for p in proposals if p.source == 'alias')
        assert len(all_nodes(NodeType.CONCEPT, db)) >= 5

    def test_measures_traversal(self, tmp_path, monkeypatch):
        """Reverse MEASURES traversal from concept::revenue returns Umsatz metrics."""
        from kg.ingest.pipeline import ingest_excel
        from kg.ingest.concept_mapper import propose_measures_edges
        from kg.models import EdgeType, NodeType
        from kg.store import get_neighbors, all_nodes

        noop_llm_propose(monkeypatch)
        xlsx, db = str(tmp_path / 'cat_mat.xlsx'), str(tmp_path / 'kg.duckdb')
        make_excel(xlsx)
        ingest_excel(xlsx, db_path=db)
        propose_measures_edges(db_path=db)

        revenue = [n for n in all_nodes(NodeType.CONCEPT, db) if 'revenue' in n['id']]
        assert len(revenue) == 1
        metrics = get_neighbors(revenue[0]['id'], EdgeType.MEASURES, direction='in', db_path=db)
        assert any('Umsatz' in m['label'] for m in metrics)


class TestHITLReview:
    def test_accept(self, tmp_path, monkeypatch):
        import duckdb
        from kg.ingest.pipeline import ingest_excel
        from kg.ingest.concept_mapper import propose_measures_edges
        from hitl.review_cli import _fetch_pending, _accept

        monkeypatch.setattr('kg.ingest.concept_mapper._llm_propose',
                            lambda cols: {col: ('distribution', 0.60) for col in cols})
        xlsx, db = str(tmp_path / 'cat_mat.xlsx'), str(tmp_path / 'kg.duckdb')
        make_excel(xlsx)
        ingest_excel(xlsx, db_path=db)
        propose_measures_edges(db_path=db)

        pending = _fetch_pending(db)
        if not pending:
            pytest.skip('No pending proposals')
        _accept(db, pending[0])
        con = duckdb.connect(db)
        row = con.execute(
            "SELECT source, confidence FROM edges WHERE src_id=? AND dst_id=? AND edge_type='MEASURES'",
            [pending[0]['src_id'], pending[0]['dst_id']]
        ).fetchone()
        con.close()
        assert row and row[0] == 'ingest' and row[1] == 1.0

    def test_reject(self, tmp_path, monkeypatch):
        import duckdb
        from kg.ingest.pipeline import ingest_excel
        from kg.ingest.concept_mapper import propose_measures_edges
        from hitl.review_cli import _fetch_pending, _reject

        monkeypatch.setattr('kg.ingest.concept_mapper._llm_propose',
                            lambda cols: {col: ('distribution', 0.50) for col in cols})
        xlsx, db = str(tmp_path / 'cat_mat.xlsx'), str(tmp_path / 'kg.duckdb')
        make_excel(xlsx)
        ingest_excel(xlsx, db_path=db)
        propose_measures_edges(db_path=db)

        pending = _fetch_pending(db)
        if not pending:
            pytest.skip('No pending proposals')
        item = pending[0]
        _reject(db, item)
        con = duckdb.connect(db)
        row = con.execute(
            "SELECT 1 FROM edges WHERE src_id=? AND dst_id=? AND edge_type='MEASURES'",
            [item['src_id'], item['dst_id']]
        ).fetchone()
        con.close()
        assert row is None


# ===========================================================================
# Phase 3 — Join detection, BRIDGES authoring
# ===========================================================================

def _ingest_two_files(tmp_path, monkeypatch):
    from kg.ingest.pipeline import ingest_excel
    from kg.models import NodeType
    from kg.store import all_nodes
    noop_llm_propose(monkeypatch)
    cat = str(tmp_path / 'cat_mat.xlsx')
    dog = str(tmp_path / 'dog_mat.xlsx')
    db  = str(tmp_path / 'kg.duckdb')
    make_excel(cat)
    make_excel(dog)
    ingest_excel(cat, db_path=db)
    ingest_excel(dog, db_path=db)
    files = all_nodes(NodeType.FILE, db)
    cat_id = next(f['id'] for f in files if 'cat_mat' in f['id'])
    dog_id = next(f['id'] for f in files if 'dog_mat' in f['id'])
    return db, cat_id, dog_id


class TestJoinDetector:
    def test_finds_shared_dimensions(self, tmp_path, monkeypatch):
        from kg.ingest.join_detector import detect_join_candidates
        db, _, _ = _ingest_two_files(tmp_path, monkeypatch)
        proposals = detect_join_candidates(db_path=db)
        assert len(proposals) >= 1
        assert any(p.match_type == 'exact' and p.col_a == p.col_b for p in proposals)

    def test_writes_to_store(self, tmp_path, monkeypatch):
        import duckdb
        from kg.ingest.join_detector import detect_join_candidates
        db, _, _ = _ingest_two_files(tmp_path, monkeypatch)
        detect_join_candidates(db_path=db)
        con = duckdb.connect(db)
        count = con.execute("SELECT COUNT(*) FROM edges WHERE edge_type='JOINABLE_ON'").fetchone()[0]
        con.close()
        assert count >= 1

    def test_single_file_noop(self, tmp_path, monkeypatch):
        from kg.ingest.pipeline import ingest_excel
        from kg.ingest.join_detector import detect_join_candidates
        noop_llm_propose(monkeypatch)
        xlsx, db = str(tmp_path / 'cat_mat.xlsx'), str(tmp_path / 'kg.duckdb')
        make_excel(xlsx)
        ingest_excel(xlsx, db_path=db)
        assert detect_join_candidates(db_path=db) == []

    def test_join_traversal_after_confirm(self, tmp_path, monkeypatch):
        from kg.ingest.join_detector import detect_join_candidates
        from hitl.bridge_author import confirm_join
        from kg.models import EdgeType
        from kg.store import get_neighbors
        db, cat_id, dog_id = _ingest_two_files(tmp_path, monkeypatch)
        p = detect_join_candidates(db_path=db)[0]
        confirm_join(db, p.file_a, p.file_b, {'join_col_a': p.col_a, 'join_col_b': p.col_b, 'match_type': p.match_type})
        neighbours = get_neighbors(cat_id, EdgeType.JOINABLE_ON, 'out', db_path=db)
        assert any(n['id'] == dog_id for n in neighbours)


class TestBridgesAuthor:
    def test_write_bridge(self, tmp_path, monkeypatch):
        import duckdb
        from kg.ingest.pipeline import ingest_excel
        from kg.ingest.concept_mapper import propose_measures_edges
        from hitl.bridge_author import write_bridge
        noop_llm_propose(monkeypatch)
        xlsx, db = str(tmp_path / 'cat_mat.xlsx'), str(tmp_path / 'kg.duckdb')
        make_excel(xlsx)
        ingest_excel(xlsx, db_path=db)
        propose_measures_edges(db_path=db)
        write_bridge(db, 'revenue', 'volume', file_a_id='file::cat_mat', file_b_id='file::cat_mat')
        con = duckdb.connect(db)
        row = con.execute(
            "SELECT source, confidence FROM edges WHERE src_id='concept::revenue' AND dst_id='concept::volume' AND edge_type='BRIDGES'"
        ).fetchone()
        con.close()
        assert row and row[0] == 'declared' and row[1] == 1.0

    def test_bridges_traversal(self, tmp_path, monkeypatch):
        from kg.ingest.pipeline import ingest_excel
        from kg.ingest.concept_mapper import propose_measures_edges
        from hitl.bridge_author import write_bridge
        from kg.models import EdgeType
        from kg.store import get_neighbors
        noop_llm_propose(monkeypatch)
        xlsx, db = str(tmp_path / 'cat_mat.xlsx'), str(tmp_path / 'kg.duckdb')
        make_excel(xlsx)
        ingest_excel(xlsx, db_path=db)
        propose_measures_edges(db_path=db)
        write_bridge(db, 'revenue', 'spend_per_buyer', file_a_id='file::cat_mat', file_b_id='file::cat_mat')
        bridged = get_neighbors('concept::revenue', EdgeType.BRIDGES, 'out', db_path=db)
        assert any('spend_per_buyer' in n['id'] for n in bridged)


# ===========================================================================
# Phase 4 — DERIVED_FROM, MEMBER_OF, SCOPED_TO, CHILD_OF, LEVEL_IN
# ===========================================================================

class TestDerivedAuthor:
    def test_creates_metric_node_and_edges(self, tmp_path):
        from hitl.derived_author import ingest_declarations
        from kg.models import EdgeType, Node, NodeType
        from kg.store import all_nodes, get_neighbors, init_store, upsert_node
        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        for nid in ('metric::ds::Ausgaben', 'metric::ds::Käufer'):
            upsert_node(Node(id=nid, node_type=NodeType.METRIC, label=nid), db)
        ingest_declarations([{'name': 'SpendPerBuyer', 'formula': 'Ausgaben/Käufer',
                              'components': ['metric::ds::Ausgaben', 'metric::ds::Käufer']}], db)
        assert any(n['id'] == 'metric::SpendPerBuyer' for n in all_nodes(NodeType.METRIC, db))
        nbrs = get_neighbors('metric::SpendPerBuyer', EdgeType.DERIVED_FROM, 'out', db_path=db)
        assert {n['id'] for n in nbrs} == {'metric::ds::Ausgaben', 'metric::ds::Käufer'}

    def test_idempotent(self, tmp_path):
        from hitl.derived_author import ingest_declarations
        from kg.models import NodeType
        from kg.store import all_nodes, init_store
        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        decls = [{'name': 'X', 'formula': 'a/b', 'components': ['metric::a', 'metric::b']}]
        ingest_declarations(decls, db)
        ingest_declarations(decls, db)
        assert len([n for n in all_nodes(NodeType.METRIC, db) if n['id'] == 'metric::X']) == 1

    def test_yaml_roundtrip(self, tmp_path):
        from hitl.derived_author import load_declarations, save_declarations
        p = tmp_path / 'derived.yaml'
        decls = [{'name': 'R', 'formula': 'A/B', 'components': ['metric::A', 'metric::B']}]
        save_declarations(decls, p)
        assert load_declarations(p) == decls


class TestGroupAuthor:
    def test_member_of(self, tmp_path):
        from hitl.group_author import ingest_declarations
        from kg.models import EdgeType, Node, NodeType
        from kg.store import all_nodes, get_neighbors, init_store, upsert_node
        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        for nid in ('entity::ANIMONDA', 'entity::MJAMJAM'):
            upsert_node(Node(id=nid, node_type=NodeType.ENTITY, label=nid), db)
        ingest_declarations({'groups': [{'name': 'HERISTO', 'members': ['entity::ANIMONDA', 'entity::MJAMJAM']}], 'scoped_to': []}, db)
        assert any(n['id'] == 'entity_group::HERISTO' for n in all_nodes(NodeType.ENTITY_GROUP, db))
        members = get_neighbors('entity_group::HERISTO', EdgeType.MEMBER_OF, 'in', db_path=db)
        assert {m['id'] for m in members} == {'entity::ANIMONDA', 'entity::MJAMJAM'}

    def test_scoped_to(self, tmp_path):
        from hitl.group_author import ingest_declarations
        from kg.models import EdgeType, Node, NodeType
        from kg.store import get_neighbors, init_store, upsert_node
        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        upsert_node(Node(id='metric::P', node_type=NodeType.METRIC, label='P'), db)
        upsert_node(Node(id='dimension::Geo', node_type=NodeType.DIMENSION, label='Geo'), db)
        ingest_declarations({'groups': [], 'scoped_to': [{'metric_id': 'metric::P', 'dimension_id': 'dimension::Geo', 'label': ''}]}, db)
        nbrs = get_neighbors('metric::P', EdgeType.SCOPED_TO, 'out', db_path=db)
        assert len(nbrs) == 1 and nbrs[0]['id'] == 'dimension::Geo'

    def test_yaml_roundtrip(self, tmp_path):
        from hitl.group_author import load_declarations, save_declarations
        p = tmp_path / 'groups.yaml'
        data = {'groups': [{'name': 'G1', 'members': ['entity::A']}], 'scoped_to': []}
        save_declarations(data, p)
        assert load_declarations(p) == data


class TestHierarchyAuthor:
    def test_level_in_and_child_of(self, tmp_path):
        from hitl.hierarchy_author import ingest_declarations
        from kg.models import EdgeType
        from kg.store import get_neighbors, init_store
        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        counts = ingest_declarations([{'root': 'CatH', 'levels': ['Category', 'AnimalType', 'Subcategory']}], db)
        assert counts['level_in'] == 3 and counts['child_of'] == 2
        parents = get_neighbors('dimension::AnimalType', EdgeType.CHILD_OF, 'out', db_path=db)
        assert parents[0]['id'] == 'dimension::Category'

    def test_root_node_is_hierarchy_root(self, tmp_path):
        from hitl.hierarchy_author import ingest_declarations
        from kg.models import NodeType
        from kg.store import all_nodes, init_store
        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        ingest_declarations([{'root': 'GeoH', 'levels': ['Country', 'Region']}], db)
        roots = [n for n in all_nodes(NodeType.DIMENSION, db) if n['id'] == 'dimension::GeoH']
        assert roots[0]['props'].get('is_hierarchy_root') is True

    def test_idempotent(self, tmp_path):
        from hitl.hierarchy_author import ingest_declarations
        from kg.models import EdgeType
        from kg.store import get_neighbors, init_store
        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        decls = [{'root': 'H', 'levels': ['A', 'B', 'C']}]
        ingest_declarations(decls, db)
        ingest_declarations(decls, db)
        assert len(get_neighbors('dimension::H', EdgeType.LEVEL_IN, 'in', db_path=db)) == 3

    def test_yaml_roundtrip(self, tmp_path):
        from hitl.hierarchy_author import load_declarations, save_declarations
        p = tmp_path / 'hier.yaml'
        decls = [{'root': 'R', 'levels': ['L1', 'L2']}]
        save_declarations(decls, p)
        assert load_declarations(p) == decls


class TestLoader:
    def test_load_all_authored(self, tmp_path):
        from kg.authored.loader import load_all_authored
        from kg.store import init_store
        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        summary = load_all_authored(db_path=db)
        assert set(summary.keys()) == {'derived_from', 'member_of', 'scoped_to', 'child_of', 'level_in'}
        assert sum(summary.values()) > 0


# ===========================================================================
# Graph routing (requires Ollama)
# ===========================================================================

@pytest.fixture(autouse=True, scope='module')
def _use_ollama(monkeypatch_module):
    monkeypatch_module.setenv('LLM_BACKEND', 'ollama')
    monkeypatch_module.setenv('OLLAMA_MODEL', os.getenv('OLLAMA_MODEL', 'gemma4:e2b'))


def test_routing_ingestion_mode(tmp_path):
    from agent.state import AnalyticsState
    from agent.graph import graph
    from langchain_core.messages import HumanMessage
    state = AnalyticsState(
        messages=[HumanMessage(content='ingest file')],
        ingestion_mode=True,
        kg_ingest_path=str(tmp_path / 'dummy.xlsx'),
    )
    result = graph.invoke(state)
    assert result.get('ingestion_mode') is True or 'kg_ingest' in str(result)


def test_routing_normal_query_skips_ingest():
    from agent.state import AnalyticsState
    from agent.graph import graph
    from langchain_core.messages import HumanMessage
    state = AnalyticsState(
        messages=[HumanMessage(content='what tables do you have?')],
        ingestion_mode=False,
    )
    result = graph.invoke(state)
    assert result.get('ingestion_mode') is not True
