"""Phase 5: Deterministic KG decomposer.

Six traversal steps that resolve a natural-language query into a fully
specified query spec before any LLM is invoked.

Usage::

    from kg.decomposer import decompose
    spec = decompose("HERISTO Umsatz vs latest period", db_path="kg/kg.duckdb")

The returned ``QuerySpec`` is a plain dataclass — picklable, serialisable,
passable through LangGraph state.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from kg.models import EdgeType, NodeType
from kg.store import all_nodes, get_neighbors

_DEFAULT_DB = os.environ.get("KG_DB_PATH", "kg/kg.duckdb")


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class QuerySpec:
    """Fully resolved query specification produced by the decomposer."""

    # Step 1 — concept resolution
    concepts: list[str] = field(default_factory=list)          # e.g. ['revenue', 'buyer_reach']

    # Step 2 — file resolution
    files: list[str] = field(default_factory=list)             # dataset names

    # Step 3 — entity group expansion
    entity_groups: dict[str, list[str]] = field(default_factory=dict)   # group -> [entity ids]

    # Step 4 — period resolution
    periods: list[dict[str, Any]] = field(default_factory=list)         # {label, role, file}

    # Step 5 — join resolution
    joins: list[dict[str, Any]] = field(default_factory=list)           # {file_a, file_b, key}
    bridges: list[dict[str, Any]] = field(default_factory=list)         # cross-file concept bridges

    # Step 6 — grain / scope
    grain: dict[str, str] = field(default_factory=dict)        # file -> grain string
    scoped_dimensions: list[dict[str, Any]] = field(default_factory=list)

    # Metric columns resolved per concept per file
    metric_columns: dict[str, list[str]] = field(default_factory=dict)  # concept -> [col names]

    # Ambiguities that the LLM must resolve (empty = fully deterministic)
    ambiguities: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1 — Concept resolution
# ---------------------------------------------------------------------------

def resolve_concepts(query: str, db_path: str = _DEFAULT_DB) -> list[str]:
    """Return concept_ids whose labels or aliases appear in *query*."""
    from kg.concepts.validator import concept_ids, aliases_for
    q_lower = query.lower()
    matched: list[str] = []
    for cid in concept_ids():
        # match on concept id itself
        if cid.replace('_', ' ') in q_lower or cid in q_lower:
            matched.append(cid)
            continue
        # match on any registered alias
        for alias in aliases_for(cid):
            if alias.lower() in q_lower:
                matched.append(cid)
                break
    # also check concept nodes in the store (covers authored concepts)
    try:
        concept_nodes = all_nodes(NodeType.CONCEPT, db_path)
        stored_ids = {n['id'].replace('concept::', '') for n in concept_nodes}
        for cid in stored_ids:
            if cid not in matched:
                cid_readable = cid.replace('_', ' ')
                if cid_readable in q_lower or cid in q_lower:
                    matched.append(cid)
    except Exception:
        pass
    return list(dict.fromkeys(matched))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Step 2 — File resolution
# ---------------------------------------------------------------------------

def resolve_files(
    concepts: list[str],
    db_path: str = _DEFAULT_DB,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    For each concept, find which files contain a Metric node that MEASURES it.

    Returns:
        files        — deduplicated list of dataset names
        metric_cols  — {concept_id: [col_name, ...]} mapping
    """
    files: list[str] = []
    metric_cols: dict[str, list[str]] = {}

    for cid in concepts:
        concept_node_id = f'concept::{cid}'
        try:
            metrics = get_neighbors(
                concept_node_id, EdgeType.MEASURES, direction='in', db_path=db_path
            )
        except Exception:
            metrics = []

        metric_cols[cid] = [m['label'] for m in metrics]

        for metric in metrics:
            try:
                file_nodes = get_neighbors(
                    metric['id'], EdgeType.AVAILABLE_IN, direction='out', db_path=db_path
                )
                for fn in file_nodes:
                    fname = fn['label']
                    if fname not in files:
                        files.append(fname)
            except Exception:
                pass

    return files, metric_cols


# ---------------------------------------------------------------------------
# Step 3 — Entity group expansion
# ---------------------------------------------------------------------------

def resolve_entity_groups(
    query: str,
    db_path: str = _DEFAULT_DB,
) -> dict[str, list[str]]:
    """
    Find any EntityGroup nodes whose label appears in *query* and expand them
    to their member entity ids via MEMBER_OF edges.

    Returns: {group_label: [entity_id, ...]}
    """
    result: dict[str, list[str]] = {}
    q_lower = query.lower()
    try:
        groups = all_nodes(NodeType.ENTITY_GROUP, db_path)
    except Exception:
        return result

    for grp in groups:
        if grp['label'].lower() in q_lower:
            try:
                members = get_neighbors(
                    grp['id'], EdgeType.MEMBER_OF, direction='in', db_path=db_path
                )
                result[grp['label']] = [m['id'] for m in members]
            except Exception:
                result[grp['label']] = []
    return result


# ---------------------------------------------------------------------------
# Step 4 — Period resolution
# ---------------------------------------------------------------------------

_LATEST_RE = re.compile(
    r'\b(latest|current|aktuell|neuest|recent|cy|current year)\b', re.IGNORECASE
)
_PRIOR_RE  = re.compile(
    r'\b(prior|previous|last year|vj|vorjahr|py)\b', re.IGNORECASE
)


def resolve_periods(
    query: str,
    files: list[str],
    db_path: str = _DEFAULT_DB,
) -> list[dict[str, Any]]:
    """
    Resolve period references in *query* against COVERS edges for *files*.

    Returns list of {label, role, file} dicts.
    """
    want_cy = bool(_LATEST_RE.search(query)) or not _PRIOR_RE.search(query)
    want_py = bool(_PRIOR_RE.search(query))

    resolved: list[dict[str, Any]] = []
    try:
        period_nodes = all_nodes(NodeType.PERIOD, db_path)
    except Exception:
        return resolved

    period_by_id = {p['id']: p for p in period_nodes}

    try:
        file_nodes = all_nodes(NodeType.FILE, db_path)
    except Exception:
        return resolved

    target_files = [fn for fn in file_nodes if not files or fn['label'] in files]

    for fn in target_files:
        try:
            covering = get_neighbors(
                fn['id'], EdgeType.COVERS, direction='in', db_path=db_path
            )
        except Exception:
            continue
        for edge_node in covering:
            pid = edge_node['id']
            period = period_by_id.get(pid)
            if not period:
                continue
            role = edge_node.get('props', {}).get('role', 'CY')
            if want_cy and role == 'CY':
                resolved.append({'label': period['label'], 'role': 'CY', 'file': fn['label']})
            elif want_py and role == 'PY':
                resolved.append({'label': period['label'], 'role': 'PY', 'file': fn['label']})

    # fallback: if nothing matched role filter, return all periods for those files
    if not resolved:
        for fn in target_files:
            try:
                covering = get_neighbors(
                    fn['id'], EdgeType.COVERS, direction='in', db_path=db_path
                )
                for edge_node in covering:
                    pid = edge_node['id']
                    period = period_by_id.get(pid)
                    if period:
                        resolved.append({
                            'label': period['label'],
                            'role': 'CY',
                            'file': fn['label'],
                        })
            except Exception:
                pass

    return resolved


# ---------------------------------------------------------------------------
# Step 5 — Join + bridge resolution
# ---------------------------------------------------------------------------

def resolve_joins(
    files: list[str],
    db_path: str = _DEFAULT_DB,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Find JOINABLE_ON edges between the resolved files and BRIDGES edges
    between their measured concepts.

    Returns: (joins, bridges)
    """
    joins: list[dict[str, Any]] = []
    bridges: list[dict[str, Any]] = []

    if len(files) < 2:
        return joins, bridges

    try:
        file_nodes = all_nodes(NodeType.FILE, db_path)
        file_map = {fn['label']: fn['id'] for fn in file_nodes}
    except Exception:
        return joins, bridges

    seen_pairs: set[frozenset] = set()
    for fname in files:
        fid = file_map.get(fname)
        if not fid:
            continue
        try:
            joinable = get_neighbors(fid, EdgeType.JOINABLE_ON, direction='both', db_path=db_path)
        except Exception:
            joinable = []
        for jn in joinable:
            pair = frozenset([fname, jn['label']])
            if pair not in seen_pairs and jn['label'] in files:
                seen_pairs.add(pair)
                joins.append({
                    'file_a': fname,
                    'file_b': jn['label'],
                    'key': jn.get('props', {}).get('join_key', ''),
                })

    # BRIDGES: concept-level cross-file equivalences
    try:
        concept_nodes = all_nodes(NodeType.CONCEPT, db_path)
        for cn in concept_nodes:
            bridged = get_neighbors(cn['id'], EdgeType.BRIDGES, direction='both', db_path=db_path)
            for bn in bridged:
                bridges.append({
                    'concept_a': cn['id'],
                    'concept_b': bn['id'],
                    'note': bn.get('props', {}).get('note', ''),
                })
    except Exception:
        pass

    return joins, bridges


# ---------------------------------------------------------------------------
# Step 6 — Grain + scope resolution
# ---------------------------------------------------------------------------

def resolve_grain_and_scope(
    files: list[str],
    db_path: str = _DEFAULT_DB,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """
    For each file, read grain from _table_context (via File node props or
    direct DuckDB query) and collect SCOPED_TO edges for metrics in those files.

    Returns: (grain_map, scoped_dimensions)
    """
    grain_map: dict[str, str] = {}
    scoped: list[dict[str, Any]] = []

    for fname in files:
        # Try _table_context first (same logic as agent/nodes.py)
        try:
            import duckdb
            db_conn = os.environ.get('DUCKDB_PATH', 'data/analytics.duckdb')
            con = duckdb.connect(db_conn)
            row = con.execute(
                "SELECT summary FROM _table_context WHERE dataset_name = ?", [fname]
            ).fetchone()
            con.close()
            if row and row[0]:
                import re as _re
                m = _re.search(r"grain='([^']+)'", row[0], _re.IGNORECASE)
                if m:
                    grain_map[fname] = m.group(1)
        except Exception:
            pass

    # SCOPED_TO edges
    try:
        metric_nodes = all_nodes(NodeType.METRIC, db_path)
        for mn in metric_nodes:
            scoped_dims = get_neighbors(mn['id'], EdgeType.SCOPED_TO, direction='out', db_path=db_path)
            for sd in scoped_dims:
                scoped.append({
                    'metric': mn['label'],
                    'metric_id': mn['id'],
                    'dimension_id': sd['id'],
                    'dimension_label': sd['label'],
                })
    except Exception:
        pass

    return grain_map, scoped


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def decompose(
    query: str,
    db_path: str = _DEFAULT_DB,
    file_hint: str | None = None,
) -> QuerySpec:
    """
    Run all six resolution steps and return a fully populated QuerySpec.

    Args:
        query:     Natural-language user query.
        db_path:   Path to the KG DuckDB store.
        file_hint: Optional dataset name to narrow file resolution.
    """
    spec = QuerySpec()

    # Step 1
    spec.concepts = resolve_concepts(query, db_path)

    # Step 2
    spec.files, spec.metric_columns = resolve_files(spec.concepts, db_path)
    if file_hint and file_hint not in spec.files:
        spec.files.insert(0, file_hint)
    if not spec.files and file_hint:
        spec.files = [file_hint]

    # Step 3
    spec.entity_groups = resolve_entity_groups(query, db_path)

    # Step 4
    spec.periods = resolve_periods(query, spec.files, db_path)

    # Step 5
    spec.joins, spec.bridges = resolve_joins(spec.files, db_path)

    # Step 6
    spec.grain, spec.scoped_dimensions = resolve_grain_and_scope(spec.files, db_path)

    # Flag ambiguities
    if not spec.concepts:
        spec.ambiguities.append('No concepts resolved from query — LLM required for concept mapping.')
    if not spec.files:
        spec.ambiguities.append('No files resolved — LLM required for file selection.')
    if len(spec.files) > 1 and not spec.joins:
        spec.ambiguities.append(
            f'Multiple files ({spec.files}) with no known join key — LLM must decide join strategy.'
        )

    return spec
