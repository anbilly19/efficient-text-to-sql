"""Debug harness for the two failing Phase 2 tests.

Run with: uv run pytest tests/test_kg_phase2_debug.py -v -s
The -s flag captures print() output so you can see exactly what's in the store.
"""
from __future__ import annotations

from tests.helpers.kg_fixtures import make_excel, noop_llm_propose


def test_debug_propose_measures_alias_only(tmp_path, monkeypatch):
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges, _alias_match
    from kg.models import EdgeType, NodeType
    from kg.store import get_neighbors, all_nodes

    noop_llm_propose(monkeypatch)

    xlsx = str(tmp_path / 'cat_mat.xlsx')
    db   = str(tmp_path / 'kg.duckdb')
    make_excel(xlsx)

    # --- ingest ---
    summary = ingest_excel(xlsx, db_path=db)
    print(f'\n[ingest] summary: {summary}')

    # --- alias match sanity check ---
    all_metrics = all_nodes(NodeType.METRIC, db)
    print(f'[store] metric nodes ({len(all_metrics)}):'
          + ''.join(f'\n  {m["id"]} | {m["label"]}' for m in all_metrics))
    for m in all_metrics:
        hit = _alias_match(m['label'])
        print(f'  alias_match({m["label"]!r}) -> {hit}')

    # --- propose ---
    proposals = propose_measures_edges(db_path=db)
    print(f'\n[proposals] total={len(proposals)}')
    for p in proposals:
        print(f'  {p.source:6s} conf={p.confidence:.2f}  {p.col_name!r} -> {p.concept_id}')

    # --- concept nodes in store ---
    concepts = all_nodes(NodeType.CONCEPT, db)
    print(f'\n[store] concept nodes ({len(concepts)}):'
          + ''.join(f'\n  {c["id"]}' for c in concepts))

    # --- MEASURES edges on first Umsatz node ---
    umsatz_nodes = [
        n for n in all_metrics
        if 'Umsatz' in n['label'] and 'VJ' not in n['label'] and 'Ver' not in n['label']
    ]
    print(f'\n[umsatz] candidate nodes: {[n["id"] for n in umsatz_nodes]}')
    if umsatz_nodes:
        neighbours = get_neighbors(umsatz_nodes[0]['id'], EdgeType.MEASURES, 'out', db_path=db)
        print(f'[umsatz] MEASURES out-neighbours: {neighbours}')

    # --- assertions with verbose failure messages ---
    assert len(proposals) >= 3, \
        f'Expected >= 3 proposals, got {len(proposals)}: {proposals}'

    alias_props = [p for p in proposals if p.source == 'alias']
    assert all(p.confidence == 0.95 for p in alias_props), \
        f'Non-0.95 alias confidence: {[(p.col_name, p.confidence) for p in alias_props]}'

    assert len(concepts) >= 5, \
        f'Expected >= 5 concept nodes, got {len(concepts)}: {[c["id"] for c in concepts]}'

    assert len(umsatz_nodes) >= 1, \
        f'No Umsatz (non-VJ, non-Ver) metric node found. All metrics: {[m["label"] for m in all_metrics]}'

    measures_targets = get_neighbors(
        umsatz_nodes[0]['id'], EdgeType.MEASURES, 'out', db_path=db
    )
    assert any('revenue' in t['id'] for t in measures_targets), \
        f'No revenue concept in MEASURES targets: {[t["id"] for t in measures_targets]}'


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
    print(f'\n[store] all concept node ids: {[c["id"] for c in concepts]}')

    revenue_concepts = [n for n in concepts if 'revenue' in n['id']]
    print(f'[store] revenue concept nodes: {revenue_concepts}')

    assert len(revenue_concepts) == 1, \
        f'Expected exactly 1 revenue concept, got {len(revenue_concepts)}: {revenue_concepts}'

    metrics_measuring_revenue = get_neighbors(
        revenue_concepts[0]['id'], EdgeType.MEASURES, direction='in', db_path=db
    )
    print(f'[store] metrics measuring revenue (in-edges): {[m["label"] for m in metrics_measuring_revenue]}')

    assert len(metrics_measuring_revenue) >= 1, \
        f'No metrics with MEASURES edge to revenue concept. Concepts: {[c["id"] for c in concepts]}'

    labels = [m['label'] for m in metrics_measuring_revenue]
    assert any('Umsatz' in lbl for lbl in labels), \
        f'No Umsatz metric measuring revenue. Got labels: {labels}'
