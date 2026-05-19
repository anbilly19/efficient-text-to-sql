"""Session-level test configuration and shared KG fixtures.

Fixture execution order
-----------------------
_session_env  (session, autouse)
    Sets PARQUET_STORE to a throwaway temp dir for the whole session.

monkeypatch_module  (module)
    Module-scoped monkeypatch for suites that need env vars set per-module.

kg_excel  (function)
    Fresh synthetic NIQ-style workbook + blank DuckDB path.

populated_db  (function)
    Fully ingested KG store with Phase 1–4 data: metrics, concepts, entity
    groups, derived metrics, and hierarchy edges.
"""
from __future__ import annotations

import os
import shutil

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture(scope='session', autouse=True)
def _session_env(tmp_path_factory):
    parquet_dir = tmp_path_factory.mktemp('parquet_store_session')
    os.environ['PARQUET_STORE'] = str(parquet_dir)
    yield parquet_dir
    if parquet_dir.exists():
        shutil.rmtree(parquet_dir, ignore_errors=True)
    os.environ.pop('PARQUET_STORE', None)


@pytest.fixture(scope='module')
def monkeypatch_module():
    """Module-scoped monkeypatch (pytest built-in is function-scoped only)."""
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture()
def kg_excel(tmp_path):
    """Return (xlsx_path, db_path) for a fresh synthetic NIQ-style workbook."""
    from tests.helpers.kg_fixtures import make_excel
    xlsx = str(tmp_path / 'cat_mat.xlsx')
    db   = str(tmp_path / 'kg.duckdb')
    make_excel(xlsx)
    return xlsx, db


@pytest.fixture()
def populated_db(tmp_path, monkeypatch):
    """Fully populated KG store (Phases 1–4) for decomposer and traversal tests.

    Contains:
    - Ingested cat_mat.xlsx (metrics, dimensions, periods)
    - Auto-proposed MEASURES edges (alias fast-path, LLM mocked)
    - Entity nodes ANIMONDA, MJAMJAM + HERISTO group (MEMBER_OF)
    - SpendPerBuyer derived metric (DERIVED_FROM)
    """
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from hitl.derived_author import ingest_declarations as ingest_derived
    from hitl.group_author import ingest_declarations as ingest_groups
    from kg.store import init_store, upsert_node
    from kg.models import Node, NodeType
    from tests.helpers.kg_fixtures import make_excel, noop_llm_propose

    noop_llm_propose(monkeypatch)

    xlsx = str(tmp_path / 'cat_mat.xlsx')
    db   = str(tmp_path / 'kg.duckdb')
    make_excel(xlsx)
    init_store(db)
    ingest_excel(xlsx, db_path=db)
    propose_measures_edges(db_path=db)

    for brand in ('ANIMONDA', 'MJAMJAM'):
        upsert_node(
            Node(id=f'entity::{brand}', node_type=NodeType.ENTITY, label=brand), db
        )

    ingest_groups({
        'groups': [{'name': 'HERISTO', 'members': ['entity::ANIMONDA', 'entity::MJAMJAM']}],
        'scoped_to': [],
    }, db)

    ingest_derived([{
        'name': 'SpendPerBuyer',
        'formula': 'Umsatz / Käuferhaushalte',
        'components': [
            'metric::cat_mat::Umsatz 52 W bis 29/03/26',
            'metric::cat_mat::Käuferhaushalte 52 W bis 29/03/26',
        ],
    }], db)

    return db
