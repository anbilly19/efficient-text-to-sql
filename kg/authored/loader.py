"""Load all authored YAML declarations into the KG store in one call.

Typically called once at agent startup or after editing any authored YAML.

Usage:
    from kg.authored.loader import load_all_authored
    load_all_authored(db_path='kg.duckdb')
"""
from __future__ import annotations

from pathlib import Path


_AUTHORED_DIR = Path(__file__).parent


def load_all_authored(db_path: str) -> dict[str, int]:
    """Ingest all authored YAML files and return edge counts per file."""
    from hitl.derived_author import ingest_declarations as ingest_derived, load_declarations as load_derived
    from hitl.group_author import ingest_declarations as ingest_groups, load_declarations as load_groups
    from hitl.hierarchy_author import ingest_declarations as ingest_hier, load_declarations as load_hier

    summary: dict[str, int] = {}

    derived = load_derived(_AUTHORED_DIR / 'derived_metrics.yaml')
    summary['derived_from'] = ingest_derived(derived, db_path)

    groups = load_groups(_AUTHORED_DIR / 'entity_groups.yaml')
    counts_g = ingest_groups(groups, db_path)
    summary['member_of'] = counts_g['member_of']
    summary['scoped_to'] = counts_g['scoped_to']

    hier = load_hier(_AUTHORED_DIR / 'hierarchies.yaml')
    counts_h = ingest_hier(hier, db_path)
    summary['child_of'] = counts_h['child_of']
    summary['level_in'] = counts_h['level_in']

    return summary
