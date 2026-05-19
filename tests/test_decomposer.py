"""Phase 5: canonical decomposer test suite.

All six query patterns from the knowledge-graph-plan are covered.
No LLM calls are made — LLM path is patched to no-op where needed.

Run: uv run pytest tests/test_decomposer.py -v
"""
from __future__ import annotations

import pytest
from tests.helpers.kg_fixtures import make_excel, noop_llm_propose


# ---------------------------------------------------------------------------
# Shared fixture: populated KG store
# ---------------------------------------------------------------------------

@pytest.fixture()
def populated_db(tmp_path, monkeypatch):
    """Ingest synthetic Excel + concept mapping + Phase 4 authored data."""
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from hitl.derived_author import ingest_declarations as ingest_derived
    from hitl.group_author import ingest_declarations as ingest_groups
    from kg.store import init_store, upsert_node
    from kg.models import Node, NodeType

    noop_llm_propose(monkeypatch)

    xlsx = str(tmp_path / 'cat_mat.xlsx')
    db   = str(tmp_path / 'kg.duckdb')
    make_excel(xlsx)
    init_store(db)
    ingest_excel(xlsx, db_path=db)
    propose_measures_edges(db_path=db)

    # Seed entity nodes so MEMBER_OF edges have valid source nodes
    for brand in ('ANIMONDA', 'MJAMJAM'):
        upsert_node(
            Node(id=f'entity::{brand}', node_type=NodeType.ENTITY, label=brand),
            db,
        )

    # Phase 4a: entity groups
    ingest_groups({
        'groups': [{
            'name': 'HERISTO',
            'members': ['entity::ANIMONDA', 'entity::MJAMJAM'],
        }],
        'scoped_to': [],
    }, db)

    # Phase 4b: derived metrics
    # SpendPerBuyer = Umsatz / Käuferhaushalte — use real col node ids from the fixture
    ingest_derived([
        {
            'name': 'SpendPerBuyer',
            'formula': 'Umsatz / Käuferhaushalte',
            'components': [
                'metric::cat_mat::Umsatz 52 W bis 29/03/26',
                'metric::cat_mat::Käuferhaushalte 52 W bis 29/03/26',
            ],
        },
    ], db)

    return db


# ---------------------------------------------------------------------------
# Pattern 1 — concept resolution + MEASURES traversal
# ---------------------------------------------------------------------------

class TestPattern1ConceptResolution:
    def test_umsatz_resolves_to_revenue(self, populated_db):
        from kg.decomposer import resolve_concepts
        concepts = resolve_concepts('Umsatz by brand', populated_db)
        assert 'revenue' in concepts

    def test_penetration_resolves_to_buyer_reach(self, populated_db):
        from kg.decomposer import resolve_concepts
        concepts = resolve_concepts('show penetration by retailer', populated_db)
        assert 'buyer_reach' in concepts

    def test_unknown_term_returns_empty(self, populated_db):
        from kg.decomposer import resolve_concepts
        concepts = resolve_concepts('show me the weather in Berlin', populated_db)
        assert concepts == []

    def test_metric_columns_resolved_for_revenue(self, populated_db):
        from kg.decomposer import resolve_files
        _files, metric_cols = resolve_files(['revenue'], populated_db)
        assert len(metric_cols.get('revenue', [])) >= 1
        assert any('Umsatz' in c for c in metric_cols['revenue'])


# ---------------------------------------------------------------------------
# Pattern 2 — entity group expansion
# ---------------------------------------------------------------------------

class TestPattern2EntityGroups:
    def test_heristo_expands_to_members(self, populated_db):
        from kg.decomposer import resolve_entity_groups
        groups = resolve_entity_groups('HERISTO Umsatz latest', populated_db)
        assert 'HERISTO' in groups, \
            f'HERISTO not found. All groups resolved: {groups}'
        members = groups['HERISTO']
        assert len(members) == 2, \
            f'Expected 2 members, got {len(members)}: {members}'
        assert 'entity::ANIMONDA' in members
        assert 'entity::MJAMJAM' in members

    def test_no_group_match_returns_empty(self, populated_db):
        from kg.decomposer import resolve_entity_groups
        groups = resolve_entity_groups('total market revenue', populated_db)
        assert groups == {}


# ---------------------------------------------------------------------------
# Pattern 3 — period resolution
# ---------------------------------------------------------------------------

class TestPattern3Periods:
    def test_latest_resolves_cy_period(self, populated_db):
        from kg.decomposer import resolve_periods
        periods = resolve_periods('latest period Umsatz', ['cat_mat'], populated_db)
        assert len(periods) >= 1
        assert all('label' in p for p in periods)
        assert all('file' in p for p in periods)

    def test_period_tied_to_file(self, populated_db):
        from kg.decomposer import resolve_periods
        periods = resolve_periods('current Umsatz', ['cat_mat'], populated_db)
        assert all(p['file'] == 'cat_mat' for p in periods)


# ---------------------------------------------------------------------------
# Pattern 4 — join resolution (single file = no joins)
# ---------------------------------------------------------------------------

class TestPattern4Joins:
    def test_single_file_no_joins(self, populated_db):
        from kg.decomposer import resolve_joins
        joins, bridges = resolve_joins(['cat_mat'], populated_db)
        assert joins == []

    def test_unknown_files_no_joins(self, populated_db):
        from kg.decomposer import resolve_joins
        joins, bridges = resolve_joins(['file_a', 'file_b'], populated_db)
        assert joins == []


# ---------------------------------------------------------------------------
# Pattern 5 — derived metric (DERIVED_FROM)
# ---------------------------------------------------------------------------

class TestPattern5DerivedFrom:
    def test_spend_per_buyer_has_components(self, populated_db):
        from kg.models import EdgeType
        from kg.store import get_neighbors
        neighbours = get_neighbors(
            'metric::SpendPerBuyer', EdgeType.DERIVED_FROM, 'out', db_path=populated_db
        )
        assert len(neighbours) == 2, \
            f'Expected 2 DERIVED_FROM components, got {len(neighbours)}: {neighbours}'
        ids = [n['id'] for n in neighbours]
        assert 'metric::cat_mat::Umsatz 52 W bis 29/03/26' in ids
        assert 'metric::cat_mat::K\u00e4uferhaushalte 52 W bis 29/03/26' in ids


# ---------------------------------------------------------------------------
# Pattern 6 — grain + scope
# ---------------------------------------------------------------------------

class TestPattern6GrainAndScope:
    def test_grain_does_not_crash(self, populated_db):
        from kg.decomposer import resolve_grain_and_scope
        grain, scoped = resolve_grain_and_scope(['cat_mat'], populated_db)
        assert isinstance(grain, dict)
        assert isinstance(scoped, list)


# ---------------------------------------------------------------------------
# Full decompose() integration
# ---------------------------------------------------------------------------

class TestDecomposeIntegration:
    def test_full_decompose_revenue_query(self, populated_db):
        from kg.decomposer import decompose
        spec = decompose('Umsatz by brand latest period', db_path=populated_db)
        assert 'revenue' in spec.concepts
        assert 'cat_mat' in spec.files
        assert len(spec.periods) >= 1
        assert isinstance(spec.ambiguities, list)

    def test_full_decompose_heristo(self, populated_db):
        from kg.decomposer import decompose
        spec = decompose('HERISTO penetration current period', db_path=populated_db)
        assert 'buyer_reach' in spec.concepts
        assert 'HERISTO' in spec.entity_groups

    def test_ambiguity_flagged_for_unknown_query(self, populated_db):
        from kg.decomposer import decompose
        spec = decompose('what is the capital of France', db_path=populated_db)
        assert len(spec.ambiguities) >= 1

    def test_kg_context_for_query_returns_string(self, populated_db):
        from agent.kg_resolver import kg_context_for_query
        ctx, spec = kg_context_for_query(
            'Umsatz latest period', db_path=populated_db
        )
        assert isinstance(ctx, str)
