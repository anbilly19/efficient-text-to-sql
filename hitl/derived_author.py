"""Interactive wizard for authoring DERIVED_FROM edges between metrics.

Usage:
    uv run python -m hitl.derived_author --db kg.duckdb

For each candidate derived metric (e.g. SpendPerBuyer), prompts the user to
confirm the numerator and denominator metrics, then writes a declaration to
kg/authored/derived_metrics.yaml and upserts the edges into the KG store.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from kg.models import Edge, EdgeType, Node, NodeType
from kg.store import init_store, upsert_edge, upsert_node

_YAML_PATH = Path('kg/authored/derived_metrics.yaml')


# ---------------------------------------------------------------------------

def load_declarations(path: Path = _YAML_PATH) -> list[dict]:
    """Return existing declarations from the YAML file, or empty list."""
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}
    return data.get('derived_metrics', [])


def save_declarations(declarations: list[dict], path: Path = _YAML_PATH) -> None:
    """Persist declarations to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        yaml.dump({'derived_metrics': declarations}, fh, allow_unicode=True, sort_keys=False)


def ingest_declarations(declarations: list[dict], db_path: str) -> int:
    """Upsert all DERIVED_FROM edges from declarations into the KG store."""
    init_store(db_path)
    count = 0
    for decl in declarations:
        derived_id = f'metric::{decl["name"]}'
        upsert_node(
            Node(id=derived_id, node_type=NodeType.METRIC, label=decl['name'],
                 props={'formula': decl.get('formula', ''), 'source': 'authored'}),
            db_path,
        )
        for component_id in decl.get('components', []):
            upsert_edge(
                Edge(
                    src_id=derived_id,
                    dst_id=component_id,
                    edge_type=EdgeType.DERIVED_FROM,
                    source='declared',
                    props={'role': 'component'},
                ),
                db_path,
            )
            count += 1
    return count


def run_wizard(db_path: str) -> None:
    """Interactive CLI wizard — add derived metric declarations one at a time."""
    declarations = load_declarations()
    print('\n=== DERIVED_FROM Authoring Wizard ===')
    print('Type a blank line at any prompt to finish.\n')

    while True:
        name = input('Derived metric name (e.g. SpendPerBuyer): ').strip()
        if not name:
            break
        formula = input(f'  Formula for {name} (e.g. Ausgaben / Käuferhaushalte): ').strip()
        components_raw = input('  Component metric node IDs (comma-separated): ').strip()
        components = [c.strip() for c in components_raw.split(',') if c.strip()]
        decl: dict = {'name': name, 'formula': formula, 'components': components}
        declarations.append(decl)
        print(f'  ✅ Added: {decl}\n')

    save_declarations(declarations)
    ingested = ingest_declarations(declarations, db_path)
    print(f'\nSaved {len(declarations)} declaration(s), upserted {ingested} DERIVED_FROM edge(s).')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Author DERIVED_FROM metric edges.')
    parser.add_argument('--db', default='kg.duckdb', help='Path to KG DuckDB file')
    parser.add_argument('--ingest-only', action='store_true',
                        help='Skip wizard, just ingest existing YAML')
    args = parser.parse_args()

    if args.ingest_only:
        decls = load_declarations()
        n = ingest_declarations(decls, args.db)
        print(f'Ingested {n} edge(s) from {_YAML_PATH}')
        sys.exit(0)

    run_wizard(args.db)
