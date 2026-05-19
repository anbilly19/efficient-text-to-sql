"""Debug harness for the two failing Phase 2 tests.

Run with: uv run pytest tests/test_kg_phase2_debug.py -v -s
"""
from __future__ import annotations

from tests.helpers.kg_fixtures import make_excel, noop_llm_propose


def test_debug_classifier_raw(tmp_path):
    """Print raw schema stats + classify_column decision for every column."""
    from kg.ingest.excel_reader import read_excel_schema
    from kg.ingest.classifier import classify_column

    xlsx = str(tmp_path / 'cat_mat.xlsx')
    make_excel(xlsx)
    schema = read_excel_schema(xlsx)
    sheet_meta = list(schema.values())[0]
    row_count = sheet_meta['row_count']
    print(f'\n[schema] row_count={row_count}')
    for col, meta in sheet_meta['columns'].items():
        meta_with_rows = {**meta, '_row_count': row_count}
        decision = classify_column(col, meta_with_rows)
        print(
            f'  {decision.value:10s} | dtype={meta.get("dtype"):12s} '
            f'n_unique={meta.get("n_unique")} | {col}'
        )


def test_debug_propose_measures_alias_only(tmp_path, monkeypatch):
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges, _alias_match
    from kg.models import EdgeType, NodeType
    from kg.store import get_neighbors, all_nodes

    noop_llm_propose(monkeypatch)

    xlsx = str(tmp_path / 'cat_mat.xlsx')
    db   = str(tmp_path / 'kg.duckdb')
    make_excel(xlsx)

    summary = ingest_excel(xlsx, db_path=db)
    print(f'\n[ingest] summary: {summary}')

    all_metrics = all_nodes(NodeType.METRIC, db)
    all_dims = all_nodes(NodeType.DIMENSION, db)
    print(f'[store] METRIC nodes ({len(all_metrics)}): {[m["label"] for m in all_metrics]}')
    print(f'[store] DIMENSION nodes ({len(all_dims)}): {[d["label"] for d in all_dims]}')

    for m in all_metrics:
        hit = _alias_match(m['label'])
        print(f'  alias_match({m["label"]!r}) -> {hit}')

    proposals = propose_measures_edges(db_path=db)
    print(f'\n[proposals] total={len(proposals)}')
    for p in proposals:
        print(f'  {p.source:6s} conf={p.confidence:.2f}  {p.col_name!r} -> {p.concept_id}')

    concepts = all_nodes(NodeType.CONCEPT, db)
    print(f'\n[store] concept nodes ({len(concepts)}): {[c["id"] for c in concepts]}')

    umsatz_nodes = [
        n for n in all_metrics
        if 'Umsatz' in n['label'] and 'VJ' not in n['label'] and 'Ver' not in n['label']
    ]
    print(f'[umsatz] candidate nodes: {[n["id"] for n in umsatz_nodes]}')
    if umsatz_nodes:
        neighbours = get_neighbors(umsatz_nodes[0]['id'], EdgeType.MEASURES, 'out', db_path=db)
        print(f'[umsatz] MEASURES out-neighbours: {neighbours}')

    assert len(proposals) >= 3, \
        f'Expected >= 3 proposals, got {len(proposals)}. METRIC nodes: {[m["label"] for m in all_metrics]}'

    alias_props = [p for p in proposals if p.source == 'alias']
    assert all(p.confidence == 0.95 for p in alias_props)
    assert len(concepts) >= 5
    assert len(umsatz_nodes) >= 1, \
        f'No Umsatz metric node. All METRIC labels: {[m["label"] for m in all_metrics]}'
    measures_targets = get_neighbors(
        umsatz_nodes[0]['id'], EdgeType.MEASURES, 'out', db_path=db
    )
    assert any('revenue' in t['id'] for t in measures_targets), \
        f'No revenue in MEASURES targets: {[t["id"] for t in measures_targets]}'


def test_debug_query_pattern_1(tmp_path, monkeypatch):
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges
    from kg.models import EdgeType, NodeType
    from kg.store import get_neighbors, all_nodes

    noop_llm_propose(monkeypatch)

    xlsx = str(tmp_path / 'cat_mat.xlsx')
    db   = str(tmp_path / 'kg.duckdb')
    make_excel(xlsx)
    ingest_excel(xlsx, db_path=db)
    propose_measures_edges(db_path=db)

    concepts = all_nodes(NodeType.CONCEPT, db)
    revenue_concepts = [n for n in concepts if 'revenue' in n['id']]
    print(f'\n[store] revenue concept nodes: {revenue_concepts}')

    assert len(revenue_concepts) == 1

    metrics_measuring_revenue = get_neighbors(
        revenue_concepts[0]['id'], EdgeType.MEASURES, direction='in', db_path=db
    )
    print(f'[store] metrics measuring revenue: {[m["label"] for m in metrics_measuring_revenue]}')

    assert len(metrics_measuring_revenue) >= 1
    assert any('Umsatz' in m['label'] for m in metrics_measuring_revenue)
