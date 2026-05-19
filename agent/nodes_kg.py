"""KG nodes — ingest, group authoring, derived metric authoring,
hierarchy authoring, and HITL review — all integrated into the main graph.

Every node follows the same contract:
  - Receives AnalyticsState
  - Returns a dict with at minimum `kg_result`, `final_answer`, `messages`
  - Resets `ingestion_mode`, `intent`, and any consumed payload fields
"""
from __future__ import annotations

import os

from langchain_core.messages import AIMessage

from agent.state import AnalyticsState

_KG_DB = os.getenv('KG_DB_PATH', 'kg.duckdb')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reply(msg: str) -> dict:
    return {
        'kg_result': msg,
        'final_answer': msg,
        'ingestion_mode': False,
        'intent': '',
        'kg_author_payload': {},
        'messages': [AIMessage(content=msg)],
    }


# ---------------------------------------------------------------------------
# 1. File ingestion
# ---------------------------------------------------------------------------

def kg_ingest_node(state: AnalyticsState) -> dict:
    """Ingest an Excel file into the KG store."""
    from kg.ingest.pipeline import ingest_excel
    from kg.ingest.concept_mapper import propose_measures_edges

    filepath = state.kg_ingest_path
    if not filepath or not os.path.isfile(filepath):
        return _reply(f'\u274c File not found or not specified: {filepath!r}')

    try:
        summary = ingest_excel(filepath, db_path=_KG_DB)
        lines = [f'\u2705 KG updated from `{os.path.basename(filepath)}`:', '']
        for sheet, counts in summary.items():
            lines.append(f'  \u2022 **{sheet}**: {counts["nodes"]} nodes, {counts["edges"]} edges')

        proposals = propose_measures_edges(db_path=_KG_DB)
        auto = [p for p in proposals if p.source == 'alias']
        pending = [p for p in proposals if p.source != 'alias']
        lines.append(f'\n  \u2022 Concept mapping: {len(auto)} auto-accepted, {len(pending)} pending review')
        if pending:
            lines.append('  Run \'review pending proposals\' to accept/reject remaining concepts.')

        result = '\n'.join(lines)
    except Exception as exc:  # noqa: BLE001
        result = f'\u274c Ingestion failed: {exc}'

    return {**_reply(result), 'kg_ingest_path': None}


# ---------------------------------------------------------------------------
# 2. Entity group authoring (MEMBER_OF / SCOPED_TO)
# ---------------------------------------------------------------------------

def kg_group_node(state: AnalyticsState) -> dict:
    """Author MEMBER_OF edges from a structured payload or natural language."""
    from hitl.group_author import ingest_declarations
    from kg.store import upsert_node, init_store
    from kg.models import Node, NodeType

    payload = state.kg_author_payload
    groups = payload.get('groups', [])
    scoped_to = payload.get('scoped_to', [])

    if not groups and not scoped_to:
        return _reply(
            '\u26a0\ufe0f  Could not parse group definition from your message.\n'
            'Try: \'HERISTO group: ANIMONDA, MJAMJAM\''
        )

    try:
        init_store(_KG_DB)
        # Auto-upsert member entity nodes if they don\'t exist yet
        for grp in groups:
            for member_id in grp.get('members', []):
                label = member_id.split('::')[-1]
                upsert_node(
                    Node(id=member_id, node_type=NodeType.ENTITY, label=label),
                    _KG_DB,
                )

        counts = ingest_declarations({'groups': groups, 'scoped_to': scoped_to}, _KG_DB)
        lines = [f'\u2705 Entity groups saved:']
        for grp in groups:
            lines.append(f'  \u2022 **{grp["name"]}** \u2192 {len(grp["members"])} member(s): {", ".join(grp["members"])}')
        lines.append(f'  Edges written: {counts["member_of"]} MEMBER_OF, {counts["scoped_to"]} SCOPED_TO')
        return _reply('\n'.join(lines))
    except Exception as exc:  # noqa: BLE001
        return _reply(f'\u274c Group authoring failed: {exc}')


# ---------------------------------------------------------------------------
# 3. Derived metric authoring (DERIVED_FROM)
# ---------------------------------------------------------------------------

def kg_derived_node(state: AnalyticsState) -> dict:
    """Author DERIVED_FROM edges for computed KPIs."""
    from hitl.derived_author import ingest_declarations, save_declarations

    declarations = state.kg_author_payload.get('declarations', [])
    if not declarations:
        return _reply(
            '\u26a0\ufe0f  Could not parse derived metric from your message.\n'
            'Try: \'derive SpendPerBuyer = Umsatz / K\u00e4uferhaushalte\''
        )

    try:
        save_declarations(declarations)  # persist to YAML
        count = ingest_declarations(declarations, _KG_DB)
        lines = [f'\u2705 Derived metric(s) saved:']
        for d in declarations:
            lines.append(f'  \u2022 **{d["name"]}** = {d["formula"]}')
            lines.append(f'    Components: {", ".join(d["components"])}')
        lines.append(f'  {count} DERIVED_FROM edge(s) written.')
        return _reply('\n'.join(lines))
    except Exception as exc:  # noqa: BLE001
        return _reply(f'\u274c Derived metric authoring failed: {exc}')


# ---------------------------------------------------------------------------
# 4. Hierarchy authoring (CHILD_OF / LEVEL_IN)
# ---------------------------------------------------------------------------

def kg_hierarchy_node(state: AnalyticsState) -> dict:
    """Author CHILD_OF / LEVEL_IN edges for a dimension hierarchy."""
    from hitl.hierarchy_author import ingest_declarations, save_declarations

    payload = state.kg_author_payload
    root = payload.get('root')
    levels = payload.get('levels', [])

    if not root or not levels:
        return _reply(
            '\u26a0\ufe0f  Could not parse hierarchy from your message.\n'
            'Try: \'hierarchy CatH: Category > AnimalType > Subcategory\''
        )

    try:
        declarations = [{'root': root, 'levels': levels}]
        save_declarations(declarations)
        counts = ingest_declarations(declarations, _KG_DB)
        lines = [
            f'\u2705 Hierarchy **{root}** saved:',
            f'  Levels: {", ".join(levels)}',
            f'  Edges: {counts["level_in"]} LEVEL_IN, {counts["child_of"]} CHILD_OF',
        ]
        return _reply('\n'.join(lines))
    except Exception as exc:  # noqa: BLE001
        return _reply(f'\u274c Hierarchy authoring failed: {exc}')


# ---------------------------------------------------------------------------
# 5. HITL review node (accept / reject pending MEASURES proposals)
# ---------------------------------------------------------------------------

def kg_review_node(state: AnalyticsState) -> dict:
    """Accept or reject pending concept-mapping proposals."""
    import re as _re
    from hitl.review_cli import _fetch_pending, _accept, _reject

    raw = state.kg_author_payload.get('raw', '')
    action = 'accept' if _re.search(r'\baccept\b|\bapprove\b|\bconfirm\b', raw, _re.I) else 'reject'

    try:
        pending = _fetch_pending(_KG_DB)
        if not pending:
            return _reply('\u2139\ufe0f  No pending MEASURES proposals to review.')

        # Accept/reject all pending, or try to match specific ones from the message
        acted = []
        for item in pending:
            if action == 'accept':
                _accept(_KG_DB, item)
            else:
                _reject(_KG_DB, item)
            acted.append(item['src_id'])

        verb = 'Accepted' if action == 'accept' else 'Rejected'
        lines = [f'\u2705 {verb} {len(acted)} proposal(s):']
        for src in acted:
            lines.append(f'  \u2022 {src}')
        return _reply('\n'.join(lines))
    except Exception as exc:  # noqa: BLE001
        return _reply(f'\u274c Review failed: {exc}')
