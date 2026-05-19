"""Interactive wizard for authoring MEMBER_OF and SCOPED_TO edges.

Usage:
    uv run python -m hitl.group_author --db kg.duckdb

MEMBER_OF : entity → entity_group   (e.g. ANIMONDA → HERISTO)
SCOPED_TO : metric  → dimension      (e.g. SalesInPetFood → PetFood category)

Declarations are persisted to kg/authored/entity_groups.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from kg.models import Edge, EdgeType, Node, NodeType
from kg.store import init_store, upsert_edge, upsert_node

_YAML_PATH = Path('kg/authored/entity_groups.yaml')


# ---------------------------------------------------------------------------

def load_declarations(path: Path = _YAML_PATH) -> dict:
    """Return {'groups': [...], 'scoped_to': [...]} from YAML."""
    if not path.exists():
        return {'groups': [], 'scoped_to': []}
    with path.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {'groups': [], 'scoped_to': []}


def save_declarations(data: dict, path: Path = _YAML_PATH) -> None:
    """Persist declarations to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        yaml.dump(data, fh, allow_unicode=True, sort_keys=False)


def ingest_declarations(data: dict, db_path: str) -> dict[str, int]:
    """Upsert MEMBER_OF and SCOPED_TO edges into the KG store."""
    init_store(db_path)
    counts: dict[str, int] = {'member_of': 0, 'scoped_to': 0}

    for grp in data.get('groups', []):
        group_id = f'entity_group::{grp["name"]}'
        upsert_node(
            Node(id=group_id, node_type=NodeType.ENTITY_GROUP, label=grp['name'],
                 props={'source': 'authored'}),
            db_path,
        )
        for member_id in grp.get('members', []):
            upsert_edge(
                Edge(src_id=member_id, dst_id=group_id,
                     edge_type=EdgeType.MEMBER_OF, source='declared'),
                db_path,
            )
            counts['member_of'] += 1

    for scoped in data.get('scoped_to', []):
        upsert_edge(
            Edge(
                src_id=scoped['metric_id'],
                dst_id=scoped['dimension_id'],
                edge_type=EdgeType.SCOPED_TO,
                source='declared',
                props={'scope_label': scoped.get('label', '')},
            ),
            db_path,
        )
        counts['scoped_to'] += 1

    return counts


def run_wizard(db_path: str) -> None:
    """Interactive CLI wizard for MEMBER_OF and SCOPED_TO edges."""
    data = load_declarations()
    print('\n=== MEMBER_OF / SCOPED_TO Authoring Wizard ===')
    print('Press Enter on a blank group name to move to SCOPED_TO, blank again to finish.\n')

    # --- MEMBER_OF groups ---
    print('-- Entity Groups (MEMBER_OF) --')
    while True:
        grp_name = input('Group name (e.g. HERISTO): ').strip()
        if not grp_name:
            break
        members_raw = input(f'  Member entity IDs for {grp_name} (comma-separated): ').strip()
        members = [m.strip() for m in members_raw.split(',') if m.strip()]
        data['groups'].append({'name': grp_name, 'members': members})
        print(f'  ✅ Group added\n')

    # --- SCOPED_TO ---
    print('\n-- Metric Scope (SCOPED_TO) --')
    while True:
        metric_id = input('Metric node ID (blank to finish): ').strip()
        if not metric_id:
            break
        dim_id = input('  Dimension node ID: ').strip()
        label = input('  Scope label (optional): ').strip()
        data['scoped_to'].append({'metric_id': metric_id, 'dimension_id': dim_id, 'label': label})
        print('  ✅ Scope added\n')

    save_declarations(data)
    counts = ingest_declarations(data, db_path)
    print(f'\nUpserted {counts["member_of"]} MEMBER_OF + {counts["scoped_to"]} SCOPED_TO edge(s).')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Author MEMBER_OF / SCOPED_TO edges.')
    parser.add_argument('--db', default='kg.duckdb')
    parser.add_argument('--ingest-only', action='store_true')
    args = parser.parse_args()

    if args.ingest_only:
        decls = load_declarations()
        counts = ingest_declarations(decls, args.db)
        print(f'Ingested {counts} from {_YAML_PATH}')
        sys.exit(0)

    run_wizard(args.db)
