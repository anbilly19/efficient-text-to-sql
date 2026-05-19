"""Phase 5: canonical decomposer test suite covering all 6 query patterns.

No LLM calls are made — alias fast-path + noop mock covers all proposals.
The `populated_db` fixture is defined in conftest.py.

Run: uv run pytest tests/test_decomposer.py -v
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Pattern 1 — concept resolution + MEASURES traversal
# ---------------------------------------------------------------------------

class TestPattern1ConceptResolution:
    def test_umsatz_resolves_to_revenue(self, populated_db):
        from kg.decomposer import resolve_concepts
        assert 'revenue' in resolve_concepts('Umsatz by brand', populated_db)

    def test_penetration_resolves_to_buyer_reach(self, populated_db):
        from kg.decomposer import resolve_concepts
        assert 'buyer_reach' in resolve_concepts('show penetration by retailer', populated_db)

    def test_unknown_term_returns_empty(self, populated_db):
        from kg.decomposer import resolve_concepts
        assert resolve_concepts('show me the weather in Berlin', populated_db) == []


# ---------------------------------------------------------------------------
# Pattern 2 — entity group expansion
# ---------------------------------------------------------------------------

class TestPattern2EntityGroups:
    def test_heristo_expands_to_members(self, populated_db):
        from kg.decomposer import resolve_entity_groups
        groups = resolve_entity_groups('HERISTO Umsatz latest', populated_db)
        assert 'HERISTO' in groups
        assert set(groups['HERISTO']) == {'entity::ANIMONDA', 'entity::MJAMJAM'}

    def test_no_group_match_returns_empty(self, populated_db):
        from kg.decomposer import resolve_entity_groups
        assert resolve_entity_groups('total market revenue', populated_db) == {}


# ---------------------------------------------------------------------------
# Pattern 3 — period resolution
# ---------------------------------------------------------------------------

class TestPattern3Periods:
    def test_latest_resolves_period(self, populated_db):
        from kg.decomposer import resolve_periods
        periods = resolve_periods('latest period Umsatz', ['cat_mat'], populated_db)
        assert len(periods) >= 1
        assert all('label' in p and 'file' in p for p in periods)

    def test_period_tied_to_file(self, populated_db):
        from kg.decomposer import resolve_periods
        periods = resolve_periods('current Umsatz', ['cat_mat'], populated_db)
        assert all(p['file'] == 'cat_mat' for p in periods)


# ---------------------------------------------------------------------------
# Pattern 4 — join resolution
# ---------------------------------------------------------------------------

class TestPattern4Joins:
    def test_single_file_no_joins(self, populated_db):
        from kg.decomposer import resolve_joins
        joins, _ = resolve_joins(['cat_mat'], populated_db)
        assert joins == []

    def test_unknown_files_no_joins(self, populated_db):
        from kg.decomposer import resolve_joins
        joins, _ = resolve_joins(['file_a', 'file_b'], populated_db)
        assert joins == []


# ---------------------------------------------------------------------------
# Pattern 5 — derived metric (DERIVED_FROM)
# ---------------------------------------------------------------------------

class TestPattern5DerivedFrom:
    def test_spend_per_buyer_has_two_components(self, populated_db):
        from kg.models import EdgeType
        from kg.store import get_neighbors
        nbrs = get_neighbors('metric::SpendPerBuyer', EdgeType.DERIVED_FROM, 'out', db_path=populated_db)
        assert len(nbrs) == 2
        ids = {n['id'] for n in nbrs}
        assert 'metric::cat_mat::Umsatz 52 W bis 29/03/26' in ids
        assert 'metric::cat_mat::K\u00e4uferhaushalte 52 W bis 29/03/26' in ids


# ---------------------------------------------------------------------------
# Pattern 6 — grain + scope
# ---------------------------------------------------------------------------

class TestPattern6GrainAndScope:
    def test_does_not_crash(self, populated_db):
        from kg.decomposer import resolve_grain_and_scope
        grain, scoped = resolve_grain_and_scope(['cat_mat'], populated_db)
        assert isinstance(grain, dict) and isinstance(scoped, list)


# ---------------------------------------------------------------------------
# Full decompose() integration
# ---------------------------------------------------------------------------

class TestDecomposeIntegration:
    def test_revenue_query(self, populated_db):
        from kg.decomposer import decompose
        spec = decompose('Umsatz by brand latest period', db_path=populated_db)
        assert 'revenue' in spec.concepts
        assert 'cat_mat' in spec.files
        assert len(spec.periods) >= 1

    def test_heristo_query(self, populated_db):
        from kg.decomposer import decompose
        spec = decompose('HERISTO penetration current period', db_path=populated_db)
        assert 'buyer_reach' in spec.concepts
        assert 'HERISTO' in spec.entity_groups

    def test_unknown_query_flags_ambiguity(self, populated_db):
        from kg.decomposer import decompose
        spec = decompose('what is the capital of France', db_path=populated_db)
        assert len(spec.ambiguities) >= 1

    def test_kg_context_returns_string(self, populated_db):
        from agent.kg_resolver import kg_context_for_query
        ctx, _ = kg_context_for_query('Umsatz latest period', db_path=populated_db)
        assert isinstance(ctx, str)
