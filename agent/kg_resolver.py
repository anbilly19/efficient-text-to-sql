"""Thin adapter: call the KG decomposer from agent nodes and inject the
resolved spec into the LLM prompt context.

This keeps agent/nodes.py clean — it just calls `kg_context_for_query()`
and prepends the returned string to its system prompt.
"""
from __future__ import annotations

import os

from kg.decomposer import QuerySpec, decompose

_DEFAULT_KG_DB = os.environ.get('KG_DB_PATH', 'kg/kg.duckdb')


def kg_context_for_query(
    query: str,
    file_hint: str | None = None,
    db_path: str = _DEFAULT_KG_DB,
) -> tuple[str, QuerySpec]:
    """
    Run the KG decomposer and format the result as a prompt-injectable string.

    Returns:
        context_str  -- Markdown block to prepend to the LLM system prompt.
        spec         -- Raw QuerySpec for state storage / inspection.
    """
    try:
        spec = decompose(query, db_path=db_path, file_hint=file_hint)
    except Exception as exc:  # noqa: BLE001
        return f'<!-- KG decomposer error: {exc} -->', QuerySpec()

    if not spec.concepts and not spec.files:
        return '', spec

    lines: list[str] = ['<!-- KG-resolved context -->']

    if spec.concepts:
        lines.append(f'**Concepts detected:** {", ".join(spec.concepts)}')

    if spec.metric_columns:
        lines.append('**Metric columns per concept:**')
        for cid, cols in spec.metric_columns.items():
            if cols:
                lines.append(f'  - `{cid}`: ' + ', '.join(f'`{c}`' for c in cols[:5]))

    if spec.files:
        lines.append(f'**Relevant files:** {", ".join(f"`{f}`" for f in spec.files)}')

    if spec.entity_groups:
        lines.append('**Entity groups expanded:**')
        for grp, members in spec.entity_groups.items():
            member_labels = [m.replace('entity::', '') for m in members]
            lines.append(f'  - `{grp}` → {{", ".join(member_labels)}}')

    if spec.periods:
        seen = set()
        lines.append('**Periods resolved:**')
        for p in spec.periods:
            key = (p['label'], p['file'])
            if key not in seen:
                seen.add(key)
                lines.append(f'  - `{p["label"]}` ({p["role"]}) in `{p["file"]}`')

    if spec.joins:
        lines.append('**Join paths:**')
        for j in spec.joins:
            lines.append(f'  - `{j["file_a"]}` ↔ `{j["file_b"]}` on `{j["key"]}`')

    if spec.grain:
        lines.append('**Data grain:**')
        for fname, g in spec.grain.items():
            lines.append(f'  - `{fname}`: {g}')

    if spec.ambiguities:
        lines.append('**⚠ Ambiguities (LLM must resolve):**')
        for a in spec.ambiguities:
            lines.append(f'  - {a}')

    lines.append('<!-- end KG context -->')
    return '\n'.join(lines), spec
