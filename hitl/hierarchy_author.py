"""Interactive wizard for authoring CHILD_OF and LEVEL_IN hierarchy edges.

Usage:
    uv run python -m hitl.hierarchy_author --db kg.duckdb

CHILD_OF  : dimension_node → parent_dimension_node
            (e.g. AnimalType → Category)
LEVEL_IN  : dimension_node → hierarchy_root
            (e.g. Subcategory → CategoryHierarchy, level=3)

Declarations are persisted to kg/authored/hierarchies.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from kg.models import Edge, EdgeType, Node, NodeType
from kg.store import init_store, upsert_edge, upsert_node

_YAML_PATH = Path('kg/authored/hierarchies.yaml')


# ---------------------------------------------------------------------------

def load_declarations(path: Path = _YAML_PATH) -> list[dict]:
    """Return existing hierarchy declarations."""
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}
    return data.get('hierarchies', [])


def save_declarations(declarations: list[dict], path: Path = _YAML_PATH) -> None:
    """Persist hierarchy declarations to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        yaml.dump({'hierarchies': declarations}, fh, allow_unicode=True, sort_keys=False)


def ingest_declarations(declarations: list[dict], db_path: str) -> dict[str, int]:
    """Upsert CHILD_OF and LEVEL_IN edges into the KG store."""
    init_store(db_path)
    counts: dict[str, int] = {'child_of': 0, 'level_in': 0}

    for hier in declarations:
        root_id = f'dimension::{hier["root"]}'
        upsert_node(
            Node(id=root_id, node_type=NodeType.DIMENSION, label=hier['root'],
                 props={'is_hierarchy_root': True, 'source': 'authored'}),
            db_path,
        )
        levels: list[str] = hier.get('levels', [])
        for i, level_name in enumerate(levels):
            level_id = f'dimension::{level_name}'
            upsert_node(
                Node(id=level_id, node_type=NodeType.DIMENSION, label=level_name,
                     props={'source': 'authored'}),
                db_path,
            )
            # LEVEL_IN → root
            upsert_edge(
                Edge(src_id=level_id, dst_id=root_id,
                     edge_type=EdgeType.LEVEL_IN, source='declared',
                     props={'level': i + 1}),
                db_path,
            )
            counts['level_in'] += 1
            # CHILD_OF → next coarser level
            if i > 0:
                parent_id = f'dimension::{levels[i - 1]}'
                upsert_edge(
                    Edge(src_id=level_id, dst_id=parent_id,
                         edge_type=EdgeType.CHILD_OF, source='declared'),
                    db_path,
                )
                counts['child_of'] += 1

    return counts


def run_wizard(db_path: str) -> None:
    """Interactive CLI wizard for hierarchy authoring."""
    declarations = load_declarations()
    print('\n=== CHILD_OF / LEVEL_IN Hierarchy Wizard ===')
    print('Levels are entered from coarsest (top) to finest (leaf).\n')

    while True:
        root = input('Hierarchy root name (blank to finish): ').strip()
        if not root:
            break
        levels_raw = input(
            f'  Levels for {root} from coarsest to finest (comma-separated): '
        ).strip()
        levels = [lv.strip() for lv in levels_raw.split(',') if lv.strip()]
        declarations.append({'root': root, 'levels': levels})
        print(f'  ✅ Hierarchy added: {root} → {levels}\n')

    save_declarations(declarations)
    counts = ingest_declarations(declarations, db_path)
    print(f'\nUpserted {counts["child_of"]} CHILD_OF + {counts["level_in"]} LEVEL_IN edge(s).')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Author CHILD_OF / LEVEL_IN hierarchy edges.')
    parser.add_argument('--db', default='kg.duckdb')
    parser.add_argument('--ingest-only', action='store_true')
    args = parser.parse_args()

    if args.ingest_only:
        decls = load_declarations()
        counts = ingest_declarations(decls, args.db)
        print(f'Ingested {counts} from {_YAML_PATH}')
        sys.exit(0)

    run_wizard(args.db)
