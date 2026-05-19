"""Phase 4 KG tests — DERIVED_FROM, MEMBER_OF, SCOPED_TO, CHILD_OF, LEVEL_IN authoring.

Run: uv run pytest tests/test_kg_phase4.py -v
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. DERIVED_FROM
# ---------------------------------------------------------------------------

class TestDerivedAuthor:
    def test_ingest_creates_metric_node(self, tmp_path):
        from hitl.derived_author import ingest_declarations
        from kg.models import NodeType
        from kg.store import all_nodes, init_store

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        decls = [{
            'name': 'SpendPerBuyer',
            'formula': 'Ausgaben / Käufer',
            'components': ['metric::ds::Ausgaben', 'metric::ds::Käufer'],
        }]
        ingest_declarations(decls, db)
        nodes = all_nodes(NodeType.METRIC, db)
        ids = [n['id'] for n in nodes]
        assert 'metric::SpendPerBuyer' in ids

    def test_ingest_creates_derived_from_edges(self, tmp_path):
        from hitl.derived_author import ingest_declarations
        from kg.models import EdgeType, NodeType
        from kg.store import get_neighbors, init_store, upsert_node
        from kg.models import Node

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        # pre-create component nodes so FK is valid
        for nid in ('metric::ds::Ausgaben', 'metric::ds::Käufer'):
            upsert_node(Node(id=nid, node_type=NodeType.METRIC, label=nid), db)

        decls = [{
            'name': 'SpendPerBuyer',
            'formula': 'Ausgaben / Käufer',
            'components': ['metric::ds::Ausgaben', 'metric::ds::Käufer'],
        }]
        ingest_declarations(decls, db)
        neighbours = get_neighbors('metric::SpendPerBuyer', EdgeType.DERIVED_FROM, 'out', db_path=db)
        assert len(neighbours) == 2
        assert {n['id'] for n in neighbours} == {'metric::ds::Ausgaben', 'metric::ds::Käufer'}

    def test_idempotent(self, tmp_path):
        from hitl.derived_author import ingest_declarations
        from kg.models import NodeType
        from kg.store import all_nodes, init_store

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        decls = [{'name': 'X', 'formula': 'a/b', 'components': ['metric::a', 'metric::b']}]
        ingest_declarations(decls, db)
        ingest_declarations(decls, db)  # second call — should not duplicate
        nodes = [n for n in all_nodes(NodeType.METRIC, db) if n['id'] == 'metric::X']
        assert len(nodes) == 1

    def test_yaml_roundtrip(self, tmp_path):
        from hitl.derived_author import load_declarations, save_declarations

        yaml_path = tmp_path / 'derived.yaml'
        decls = [{'name': 'Ratio', 'formula': 'A/B', 'components': ['metric::A', 'metric::B']}]
        save_declarations(decls, yaml_path)
        loaded = load_declarations(yaml_path)
        assert loaded == decls


# ---------------------------------------------------------------------------
# 2. MEMBER_OF / SCOPED_TO
# ---------------------------------------------------------------------------

class TestGroupAuthor:
    def test_member_of_creates_group_node(self, tmp_path):
        from hitl.group_author import ingest_declarations
        from kg.models import NodeType
        from kg.store import all_nodes, init_store

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        data = {
            'groups': [{'name': 'HERISTO', 'members': ['entity::ANIMONDA', 'entity::MJAMJAM']}],
            'scoped_to': [],
        }
        ingest_declarations(data, db)
        nodes = all_nodes(NodeType.ENTITY_GROUP, db)
        assert any(n['id'] == 'entity_group::HERISTO' for n in nodes)

    def test_member_of_edges(self, tmp_path):
        from hitl.group_author import ingest_declarations
        from kg.models import EdgeType, Node, NodeType
        from kg.store import get_neighbors, init_store, upsert_node

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        for nid in ('entity::ANIMONDA', 'entity::MJAMJAM'):
            upsert_node(Node(id=nid, node_type=NodeType.ENTITY, label=nid), db)

        data = {
            'groups': [{'name': 'HERISTO', 'members': ['entity::ANIMONDA', 'entity::MJAMJAM']}],
            'scoped_to': [],
        }
        ingest_declarations(data, db)
        # inbound edges to the group node
        members = get_neighbors('entity_group::HERISTO', EdgeType.MEMBER_OF, 'in', db_path=db)
        assert {m['id'] for m in members} == {'entity::ANIMONDA', 'entity::MJAMJAM'}

    def test_scoped_to_edge(self, tmp_path):
        from hitl.group_author import ingest_declarations
        from kg.models import EdgeType, Node, NodeType
        from kg.store import get_neighbors, init_store, upsert_node

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        for nid, nt in (('metric::Penetration', NodeType.METRIC), ('dimension::Geo', NodeType.DIMENSION)):
            upsert_node(Node(id=nid, node_type=nt, label=nid), db)

        data = {
            'groups': [],
            'scoped_to': [{'metric_id': 'metric::Penetration', 'dimension_id': 'dimension::Geo', 'label': 'geo scope'}],
        }
        ingest_declarations(data, db)
        neighbours = get_neighbors('metric::Penetration', EdgeType.SCOPED_TO, 'out', db_path=db)
        assert len(neighbours) == 1
        assert neighbours[0]['id'] == 'dimension::Geo'

    def test_yaml_roundtrip(self, tmp_path):
        from hitl.group_author import load_declarations, save_declarations

        yaml_path = tmp_path / 'groups.yaml'
        data = {
            'groups': [{'name': 'G1', 'members': ['entity::A']}],
            'scoped_to': [{'metric_id': 'm::x', 'dimension_id': 'd::y', 'label': ''}],
        }
        save_declarations(data, yaml_path)
        loaded = load_declarations(yaml_path)
        assert loaded == data


# ---------------------------------------------------------------------------
# 3. CHILD_OF / LEVEL_IN
# ---------------------------------------------------------------------------

class TestHierarchyAuthor:
    def test_level_in_edges_created(self, tmp_path):
        from hitl.hierarchy_author import ingest_declarations
        from kg.models import EdgeType
        from kg.store import get_neighbors, init_store

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        decls = [{'root': 'CatH', 'levels': ['Category', 'AnimalType', 'Subcategory']}]
        counts = ingest_declarations(decls, db)
        assert counts['level_in'] == 3  # one per level

        # all three levels are reachable from root via LEVEL_IN (inbound)
        members = get_neighbors('dimension::CatH', EdgeType.LEVEL_IN, 'in', db_path=db)
        assert len(members) == 3

    def test_child_of_edges_created(self, tmp_path):
        from hitl.hierarchy_author import ingest_declarations
        from kg.models import EdgeType
        from kg.store import get_neighbors, init_store

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        decls = [{'root': 'CatH', 'levels': ['Category', 'AnimalType', 'Subcategory']}]
        counts = ingest_declarations(decls, db)
        assert counts['child_of'] == 2  # AnimalType→Category, Subcategory→AnimalType

        # AnimalType's parent is Category
        parents = get_neighbors('dimension::AnimalType', EdgeType.CHILD_OF, 'out', db_path=db)
        assert len(parents) == 1
        assert parents[0]['id'] == 'dimension::Category'

    def test_root_node_created(self, tmp_path):
        from hitl.hierarchy_author import ingest_declarations
        from kg.models import NodeType
        from kg.store import all_nodes, init_store

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        decls = [{'root': 'GeoH', 'levels': ['Country', 'Region']}]
        ingest_declarations(decls, db)
        dims = all_nodes(NodeType.DIMENSION, db)
        roots = [n for n in dims if n['id'] == 'dimension::GeoH']
        assert len(roots) == 1
        assert roots[0]['props'].get('is_hierarchy_root') is True

    def test_idempotent(self, tmp_path):
        from hitl.hierarchy_author import ingest_declarations
        from kg.models import EdgeType
        from kg.store import get_neighbors, init_store

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        decls = [{'root': 'H', 'levels': ['A', 'B', 'C']}]
        ingest_declarations(decls, db)
        ingest_declarations(decls, db)  # second call
        members = get_neighbors('dimension::H', EdgeType.LEVEL_IN, 'in', db_path=db)
        assert len(members) == 3  # no duplicates

    def test_yaml_roundtrip(self, tmp_path):
        from hitl.hierarchy_author import load_declarations, save_declarations

        yaml_path = tmp_path / 'hier.yaml'
        decls = [{'root': 'R', 'levels': ['L1', 'L2']}]
        save_declarations(decls, yaml_path)
        loaded = load_declarations(yaml_path)
        assert loaded == decls


# ---------------------------------------------------------------------------
# 4. loader.load_all_authored
# ---------------------------------------------------------------------------

class TestLoader:
    def test_load_all_authored_returns_counts(self, tmp_path):
        from kg.authored.loader import load_all_authored
        from kg.store import init_store

        db = str(tmp_path / 'kg.duckdb')
        init_store(db)
        summary = load_all_authored(db_path=db)
        # All keys present
        assert set(summary.keys()) == {'derived_from', 'member_of', 'scoped_to', 'child_of', 'level_in'}
        # At least some edges were ingested from the seed YAMLs
        assert sum(summary.values()) > 0
