"""LangGraph node functions – one per agent role."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from agent.prompts import (
    PROFILER_SYSTEM,
    SQL_WRITER_SYSTEM,
    VERIFIER_SYSTEM,
)
from agent.state import AnalyticsState, PlanStep
from agent.tools import (
    ORCHESTRATOR_TOOLS,
    PROFILER_TOOLS,
    SQL_WRITER_TOOLS,
    get_schema,
    load_file,
    load_niq_file,
    run_sql,
    search_semantic_lookup,
)
from agent.database import get_connection, get_schema_context, get_table_summaries
from agent.db.catalog import validate_join_in_sql, list_relationships


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _llm(temperature: float = 0.0) -> BaseChatModel:
    backend = os.getenv("LLM_BACKEND", "openai").lower().strip()

    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "langchain-ollama is not installed. Run: pip install langchain-ollama"
            ) from exc
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI  # type: ignore[import]
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(p for p in parts if p).strip()
    return str(content)


def _try_parse_json(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    best = None
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        best = parsed
                except json.JSONDecodeError:
                    pass
                start = None
    return best


def _all_table_names() -> list[str]:
    """Return every table registered in _data_registry."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT dataset_name FROM _data_registry ORDER BY ingested_at").fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _most_recent_table() -> str:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT dataset_name FROM _data_registry ORDER BY ingested_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else ""
    except Exception:
        return ""


# NIQ-domain terms that boost table relevance scoring
_NIQ_QUERY_TERMS = {
    "penetration", "spend", "buyer", "panel", "niq", "nielsen",
    "käuferreichweite", "kaeuferreichweite", "ausgaben", "haushalt",
    "yoy", "yearonyear", "prior", "volume", "frequency", "trips",
    "share", "manufacturer", "brand", "category", "market",
}

# Path/name patterns that indicate a NIQ source file
_NIQ_FILE_PATTERNS = re.compile(
    r"niq|nielsen|panel|penetration|buyer|haushalt|käufer",
    re.IGNORECASE,
)


def _is_niq_file(path: str, dataset_name: str) -> bool:
    """Return True when the file path or dataset name looks like a NIQ panel extract."""
    combined = f"{path} {dataset_name}".lower()
    return bool(_NIQ_FILE_PATTERNS.search(combined))


def _select_relevant_tables(user_query: str, all_tables: list[str]) -> list[str]:
    """Select tables needed for *user_query* using _table_context keyword scoring.

    NIQ-tagged tables receive a +2 bonus when the query contains NIQ-domain
    terminology so that panel datasets win over generic fact tables.
    """
    if len(all_tables) <= 1:
        return all_tables

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT dr.dataset_name, tc.summary, tc.tags "
            "FROM _data_registry dr "
            "LEFT JOIN _table_context tc ON tc.dataset_name = dr.dataset_name"
        ).fetchall()
    except Exception:
        return all_tables

    if not rows:
        return all_tables

    q_lower = user_query.lower()
    q_words = set(re.findall(r"[a-z]{3,}", q_lower))

    # Detect whether the query is NIQ-flavoured
    q_niq_terms = {w.replace("ä", "a").replace("ü", "u").replace("ö", "o") for w in q_words}
    query_is_niq = bool(q_niq_terms & _NIQ_QUERY_TERMS)

    scored: list[tuple[int, str]] = []
    for dataset_name, summary, tags in rows:
        if dataset_name not in all_tables:
            continue
        text = " ".join([
            dataset_name.lower(),
            (summary or "").lower(),
            (tags or "").lower(),
        ])
        score = sum(1 for w in q_words if w in text)

        # NIQ bonus: panel datasets get +2 when query is NIQ-flavoured
        if query_is_niq and tags and "niq" in tags.lower():
            score += 2

        scored.append((score, dataset_name))

    selected = [name for score, name in scored if score > 0]
    if not selected:
        return _select_relevant_tables_llm(user_query, all_tables)
    return selected


def _select_relevant_tables_llm(user_query: str, all_tables: list[str]) -> list[str]:
    """LLM fallback for table selection when keyword scoring yields no hits."""
    summaries = get_table_summaries()
    prompt = (
        f"The following tables are loaded in DuckDB:\n{summaries}\n\n"
        f"User question: {user_query}\n\n"
        "Which tables are needed to answer this question?\n"
        'Output ONLY JSON: {"tables": ["table1", "table2"]}\n'
        "Only include tables that are strictly necessary."
    )
    try:
        response = _llm().invoke([HumanMessage(content=prompt)])
        parsed = _try_parse_json(_extract_text(response.content)) or {}
        selected = parsed.get("tables", [])
        valid = [t for t in selected if t in all_tables]
        return valid if valid else all_tables
    except Exception:
        return all_tables


def _get_date_cast_warnings(table_names: list[str]) -> str:
    conn = get_connection()
    warnings: list[str] = []
    for table_name in table_names:
        try:
            rows = conn.execute(
                "SELECT column_name, column_type FROM _column_catalog WHERE dataset_name = ?",
                [table_name],
            ).fetchall()
        except Exception:
            continue
        for col, dtype in rows:
            if dtype.upper() == "VARCHAR":
                try:
                    sample = conn.execute(
                        f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 1'
                    ).fetchone()
                    if sample and re.match(r"\d{4}-\d{2}-\d{2}", str(sample[0])):
                        warnings.append(
                            f'  ⚠️  {table_name}."{col}" is VARCHAR storing dates (e.g. \'{sample[0]}\').'
                            f' Always wrap: TRY_CAST("{col}" AS DATE)'
                        )
                except Exception:
                    pass
            elif dtype.upper() in ("DATE", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE"):
                warnings.append(
                    f'  ✅  {table_name}."{col}" is {dtype} — use YEAR/MONTH/DATE_TRUNC directly.'
                )
    return "\n".join(warnings)


def _get_varchar_date_columns_multi(table_names: list[str]) -> dict[str, set[str]]:
    conn = get_connection()
    result: dict[str, set[str]] = {}
    for table_name in table_names:
        result[table_name] = set()
        try:
            rows = conn.execute(
                "SELECT column_name, column_type FROM _column_catalog WHERE dataset_name = ?",
                [table_name],
            ).fetchall()
        except Exception:
            continue
        for col, dtype in rows:
            if dtype.upper() != "VARCHAR":
                continue
            try:
                sample = conn.execute(
                    f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 1'
                ).fetchone()
                if sample and re.match(r"\d{4}-\d{2}-\d{2}", str(sample[0])):
                    result[table_name].add(col)
            except Exception:
                pass
    return result


def _sanitize_sql(sql: str, varchar_date_cols: dict[str, set[str]]) -> str:
    if not sql:
        return sql
    sql = re.sub(r'"(\w+)"\"', r'"\1"', sql)
    sql = re.sub(r'""\\b(\w+)"', r'"\1"', sql)
    for _table, cols in varchar_date_cols.items():
        for col in cols:
            col_pattern = rf'(?:"?\w+"?\.)?\"?{re.escape(col)}\"?'
            sql = re.sub(
                rf'\bYEAR\s*\(\s*({col_pattern})\s*\)',
                lambda m, c=col: f'YEAR(TRY_CAST("{c}" AS DATE))',
                sql,
                flags=re.IGNORECASE,
            )
            sql = re.sub(
                rf'\bMONTH\s*\(\s*({col_pattern})\s*\)',
                lambda m, c=col: f'MONTH(TRY_CAST("{c}" AS DATE))',
                sql,
                flags=re.IGNORECASE,
            )
            sql = re.sub(
                rf"DATE_TRUNC\s*\(\s*('[^']*')\s*,\s*({col_pattern})\s*\)",
                lambda m, c=col: f"DATE_TRUNC({m.group(1)}, TRY_CAST(\"{c}\" AS DATE))",
                sql,
                flags=re.IGNORECASE,
            )
    return sql


def _next_pending_step(plan: list[PlanStep]) -> PlanStep | None:
    return next((s for s in plan if s.status == "pending"), None)


_LOAD_PATTERNS = [
    re.compile(r"load\s+(?:file\s+)?at\s+(?P<path>\S+)\s+as\s+(?P<dataset>\w+)", re.IGNORECASE),
    re.compile(r"import\s+(?P<path>\S+)\s+as\s+(?P<dataset>\w+)", re.IGNORECASE),
    re.compile(r"load\s+(?P<path>\S+\.(?:xlsx|xls|csv|parquet))\s+as\s+(?P<dataset>\w+)", re.IGNORECASE),
    re.compile(r"load\s+(?:file\s+)?(?P<path>\S+\.(?:xlsx|xls|csv|parquet))", re.IGNORECASE),
]

_SCHEMA_PATTERNS = [
    re.compile(r"\bwhat\s+columns?\b", re.IGNORECASE),
    re.compile(r"\blist\s+columns?\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:available\s+)?(?:columns?|fields?|schema)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:fields?|attributes?|schema)\b", re.IGNORECASE),
    re.compile(r"\bdescribe\s+(?:the\s+)?table\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s|\s+is)\s+(?:in|inside)\s+(?:the\s+)?\w+\s+table\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(?:columns?|fields?)\b", re.IGNORECASE),
    re.compile(r"\btable\s+structure\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+tables?\s+(?:do\s+(?:i|we|you)\s+have|are\s+(?:available|loaded))\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(?:me\s+)?(?:all\s+)?(?:available\s+)?tables?\b", re.IGNORECASE),
]

_META_PATTERNS = [
    re.compile(r"\b(?:do\s+you\s+have|have\s+you\s+(?:got|loaded)|is\s+there|can\s+you\s+(?:see|access|use))\b", re.IGNORECASE),
    re.compile(r"\b(?:access\s+to|loaded|available|registered)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:data|tables?|files?)\s+(?:do\s+you\s+have|are\s+(?:loaded|available))\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+tables?\s+(?:are|do\s+you)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+can\s+you\s+(?:do|answer|query)\b", re.IGNORECASE),
    re.compile(r"\bare\s+(?:you|any\s+tables?)\s+(?:ready|set\s+up|configured)\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Error pattern matchers
# ---------------------------------------------------------------------------

# DuckDB "table not found" — covers quoted/unquoted identifiers and truncated form
_TABLE_NOT_FOUND_RE = re.compile(
    r'(?:Table with name|Table|Catalog Error.*?table)\s+["\']?([\w]+)["\']?\s+does not exist'
    r'|(?:relation|table)\s+["\']?([\w.]+)["\']?\s+does not exist'
    r'|([\w"]+)\s+not exist',
    re.IGNORECASE,
)

# DuckDB Binder Error — column referenced in SQL doesn't exist in the FROM clause.
_BINDER_ERROR_RE = re.compile(
    r'Binder Error[:\s]+Referenced column\s+["\']?([\w ]+)["\']?\s+not found'
    r'|Binder Error[:\s]+.*?Column[\s]+["\']?([\w ]+)["\']?\s+not found',
    re.IGNORECASE | re.DOTALL,
)

# Extract candidate bindings DuckDB helpfully lists after the error
_CANDIDATES_RE = re.compile(
    r'Candidate bindings:\s*([^\n]+)',
    re.IGNORECASE,
)


def _detect_load_intent(text: str) -> dict | None:
    for pattern in _LOAD_PATTERNS:
        m = pattern.search(text)
        if m:
            path = m.group("path")
            try:
                dataset = m.group("dataset")
            except IndexError:
                dataset = Path(path).stem.replace(" ", "_").replace("-", "_").lower()
            return {"path": path, "dataset": dataset}
    return None


def _detect_schema_intent(text: str) -> bool:
    return any(p.search(text) for p in _SCHEMA_PATTERNS)


def _detect_meta_intent(text: str) -> bool:
    return any(p.search(text) for p in _META_PATTERNS)


def _build_meta_reply(query: str, all_tables: list[str]) -> str:
    conn = get_connection()
    query_lower = query.lower()

    if not all_tables:
        return (
            "No tables are loaded yet. Upload a file with:\n"
            "`load file at <path> as <name>`"
        )

    mentioned = [t for t in all_tables if t.lower() in query_lower]

    if mentioned:
        lines = []
        for t in mentioned:
            row = conn.execute(
                "SELECT row_count, column_count, ingested_at, source_file "
                "FROM _data_registry WHERE dataset_name = ?",
                [t],
            ).fetchone()
            if row:
                lines.append(
                    f"✅ **{t}** is loaded — {row[0]:,} rows, {row[1]} columns "
                    f"(source: `{Path(row[3]).name if row[3] else 'unknown'}`, "
                    f"ingested: {str(row[2])[:19]})"
                )
            else:
                lines.append(f"❌ **{t}** is not currently loaded.")
        return "\n".join(lines)

    rows = conn.execute(
        "SELECT dataset_name, row_count, column_count, ingested_at "
        "FROM _data_registry ORDER BY ingested_at"
    ).fetchall()
    lines = [f"I have access to {len(rows)} table(s):\n"]
    for r in rows:
        lines.append(f"  • **{r[0]}** — {r[1]:,} rows, {r[2]} columns (loaded {str(r[3])[:10]})")
    lines.append(
        "\nAsk me anything about this data, or load more files with "
        "`load file at <path> as <name>`."
    )
    return "\n".join(lines)


def _rows_to_markdown(rows_json: str) -> str:
    if not rows_json or rows_json.startswith("ERROR"):
        return rows_json or "No result."
    try:
        rows = json.loads(rows_json)
        if not rows:
            return "No rows returned."
        headers = list(rows[0].keys())
        header_row = " | ".join(headers)
        sep_row = " | ".join(["---"] * len(headers))
        data_rows = "\n".join(
            " | ".join(str(row.get(h, "")) for h in headers) for row in rows
        )
        return f"{header_row}\n{sep_row}\n{data_rows}"
    except Exception:
        return rows_json


def _resolve_user_query(state: AnalyticsState) -> str:
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            text = _extract_text(msg.content)
            if text:
                return text
    if state.user_query and state.user_query.strip():
        return state.user_query.strip()
    return ""


# ---------------------------------------------------------------------------
# Error message builders
# ---------------------------------------------------------------------------

def _build_table_not_found_message(error: str, sql: str, all_tables: list[str]) -> str | None:
    """Rich user-facing message when DuckDB says a table does not exist.

    Returns None when the error is not a table-not-found error.
    """
    m = _TABLE_NOT_FOUND_RE.search(error)
    if not m:
        return None

    missing = next((g.strip('"\'\ ') for g in m.groups() if g), None)
    if not missing:
        return None

    conn = get_connection()
    try:
        reg_rows = conn.execute(
            "SELECT dataset_name, source_file FROM _data_registry ORDER BY ingested_at"
        ).fetchall()
    except Exception:
        reg_rows = []

    loaded_lines = [
        f"  • **{name}** (from `{Path(src).name if src else 'unknown'}`)"
        for name, src in reg_rows
    ] or ["  • *(no tables loaded)*"]

    suggestion = ""
    missing_lower = missing.lower().replace("_", "").replace("-", "")
    for loaded_name, _ in reg_rows:
        if (
            loaded_name.lower().replace("_", "").replace("-", "") == missing_lower
            or missing_lower in loaded_name.lower()
            or loaded_name.lower() in missing_lower
        ):
            suggestion = (
                f"\n💡 **Did you mean `{loaded_name}`?** "
                f"The file was ingested as `{loaded_name}`, not `{missing}`.\n"
                f"Edit your question to use `{loaded_name}` instead, or reload the file with:\n"
                f"`load <file> as {missing}`"
            )
            break

    lines = [
        f"❌ **Table `{missing}` does not exist** in the current session.",
        "",
        "**Currently loaded tables:**",
        *loaded_lines,
        suggestion,
        "",
        "**To fix this:**",
        f"- Replace `{missing}` in your question with one of the table names above, **or**",
        f"- Reload your file under the expected name: `load <file> as {missing}`",
    ]
    return "\n".join(lines)


def _build_binder_error_message(error: str, sql: str) -> str | None:
    """Rich user-facing message when DuckDB raises a Binder Error (column not found).

    Returns None when the error is not a Binder Error.
    """
    if "Binder Error" not in error:
        return None

    bm = _BINDER_ERROR_RE.search(error)
    missing_col = next((g.strip('"\'\ ') for g in (bm.groups() if bm else []) if g), None)

    candidates_match = _CANDIDATES_RE.search(error)
    duckdb_candidates: list[str] = []
    if candidates_match:
        raw = candidates_match.group(1)
        duckdb_candidates = [c.strip().strip('"') for c in raw.split(",") if c.strip()]

    catalog_cols: dict[str, list[str]] = {}
    all_tables = _all_table_names()
    conn = get_connection()
    for table in all_tables:
        if table.lower() not in sql.lower():
            continue
        try:
            rows = conn.execute(
                "SELECT column_name FROM _column_catalog WHERE dataset_name = ? ORDER BY column_name",
                [table],
            ).fetchall()
            catalog_cols[table] = [r[0] for r in rows]
        except Exception:
            pass

    suggestions: list[str] = []
    if missing_col:
        missing_norm = missing_col.lower().replace("_", "").replace(" ", "")
        for tbl, cols in catalog_cols.items():
            for col in cols:
                col_norm = col.lower().replace("_", "").replace(" ", "")
                if missing_norm in col_norm or col_norm in missing_norm:
                    suggestions.append(f'`{tbl}."{col}"`')

    lines: list[str] = []

    if missing_col:
        lines.append(f'❌ **Column `{missing_col}` does not exist** in the loaded table(s).')
    else:
        lines.append('❌ **A column referenced in the query does not exist** in the loaded table(s).')

    lines.append("")

    if duckdb_candidates:
        lines.append(f"**DuckDB reported these available columns:** {', '.join(f'`{c}`' for c in duckdb_candidates)}")
        lines.append("")

    if catalog_cols:
        lines.append("**Full column list for the queried table(s):**")
        for tbl, cols in catalog_cols.items():
            col_list = ", ".join(f'"{c}"' for c in cols)
            lines.append(f"  • **{tbl}**: {col_list}")
        lines.append("")

    if suggestions:
        lines.append(f"💡 **Closest match(es):** {', '.join(suggestions)}")
        lines.append("Rephrase your question using one of those column names.")
    elif missing_col:
        lines.append(
            f"💡 No column named `{missing_col}` exists in any loaded table. "
            "Please rephrase your question using one of the column names listed above."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verifier helper: semantic gap detection
# ---------------------------------------------------------------------------

_QUERY_STOPWORDS = {
    "what", "show", "list", "give", "find", "from", "that", "this", "with",
    "have", "does", "each", "many", "much", "more", "most", "last", "year",
    "month", "week", "date", "time", "when", "where", "which", "the", "per",
    "and", "for", "all", "total", "annual", "revenue", "sales", "data",
    "table", "column", "value", "number", "count", "average", "mean",
    "group", "order", "sort", "filter", "between", "above", "below",
}


def _extract_select_columns(sql: str) -> set[str]:
    m = re.search(r'\bSELECT\b(.+?)\bFROM\b', sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return set()
    select_clause = m.group(1)
    tokens: set[str] = set()
    for part in select_clause.split(","):
        part = part.strip()
        alias_m = re.search(r'\bAS\s+"?([\w\s]+)"?\s*$', part, re.IGNORECASE)
        if alias_m:
            tokens.add(alias_m.group(1).strip().lower().replace('"', ''))
        else:
            bare = re.sub(r'^.*\.', '', part)
            bare = re.sub(r'["\\s]', '', bare).lower()
            if bare:
                tokens.add(bare)
    return tokens


def _detect_semantic_gaps(
    user_query: str,
    sql: str,
    table_names: list[str],
) -> str:
    conn = get_connection()

    catalog_cols: set[str] = set()
    for tbl in table_names:
        try:
            rows = conn.execute(
                "SELECT column_name FROM _column_catalog WHERE dataset_name = ?",
                [tbl],
            ).fetchall()
            for (col,) in rows:
                catalog_cols.add(col.lower())
                catalog_cols.add(col.lower().replace(" ", "").replace("_", ""))
        except Exception:
            pass

    user_concepts = [
        w for w in re.findall(r'[a-zA-Z]{4,}', user_query.lower())
        if w not in _QUERY_STOPWORDS
    ]

    sql_cols = _extract_select_columns(sql)

    gaps: list[str] = []
    for concept in user_concepts:
        concept_norm = concept.replace(" ", "").replace("_", "")
        in_catalog = any(
            concept_norm in c.replace(" ", "").replace("_", "")
            or c.replace(" ", "").replace("_", "") in concept_norm
            for c in catalog_cols
        )
        if not in_catalog:
            gaps.append(
                f'  ⚠️  Concept "{concept}" was requested but no matching column exists '
                f'in the loaded table(s): {table_names}. '
                f'Available columns: {sorted(catalog_cols)[:10]}'
            )
        else:
            in_sql = any(
                concept_norm in c.replace(" ", "").replace("_", "")
                or c.replace(" ", "").replace("_", "") in concept_norm
                for c in sql_cols
            )
            if not in_sql:
                best = next(
                    (
                        c for c in catalog_cols
                        if concept_norm in c.replace(" ", "").replace("_", "")
                        or c.replace(" ", "").replace("_", "") in concept_norm
                    ),
                    None,
                )
                suggestion = f' (closest column: "{best}")' if best else ""
                gaps.append(
                    f'  ⚠️  Concept "{concept}" was requested but is not in the SQL result.'
                    f'{suggestion} The query was answered on the available dimensions only.'
                )

    return "\n".join(gaps)


# ---------------------------------------------------------------------------
# Helper: build relationships block for sql_writer prompt
# ---------------------------------------------------------------------------

def _build_relationships_block(table_names: list[str]) -> str:
    try:
        rels = list_relationships(tables=table_names)
    except Exception:
        return ""
    if not rels:
        return ""
    lines = ["## Relationships (authoritative join keys — use ONLY these column pairs)"]
    for r in rels:
        cardinality = r.get("cardinality", "")
        desc = r.get("description", "")
        note = f"  # {desc}" if desc else ""
        lines.append(
            f"  {r['left_table']}.{r['left_column']} "
            f"→ {r['right_table']}.{r['right_column']}"
            f" ({cardinality}){note}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node: orchestrator
# ---------------------------------------------------------------------------

def orchestrator(state: AnalyticsState) -> dict:
    current_query = _resolve_user_query(state)
    all_tables = _all_table_names()
    most_recent = _most_recent_table()
    tables_summary = get_table_summaries()

    base_reset: dict = {
        "user_query": current_query,
        "final_answer": "",
        "plan": [],
        "current_step": None,
        "last_sql": "",
        "last_query_result": "",
        "last_query_metadata": {},
        "verification_verdict": "",
        "verification_feedback": "",
        "retry_count": 0,
        "load_file_path": None,
        "load_file_dataset": None,
        "error": "",
    }

    # ── Synthesis: all steps complete ──────────────────────────────────────
    if state.plan and all(s.status in ("done", "failed") for s in state.plan):
        all_failed = all(s.status == "failed" for s in state.plan)
        if all_failed:
            err = state.error or "The query could not be executed."

            binder_msg = _build_binder_error_message(err, state.last_sql or "")
            if binder_msg:
                return {**base_reset, "final_answer": binder_msg, "messages": [AIMessage(content=binder_msg)]}

            table_msg = _build_table_not_found_message(err, state.last_sql or "", all_tables)
            if table_msg:
                return {**base_reset, "final_answer": table_msg, "messages": [AIMessage(content=table_msg)]}

            final = (
                f"I wasn't able to answer that question due to a SQL error:\n\n"
                f"```\n{err.replace('ERROR: ', '').strip()}\n```\n\n"
                "Could you rephrase, or check that the column names are correct?"
            )
        else:
            row_count = state.last_query_metadata.get("row_count", "?")
            table_preview = _rows_to_markdown(state.last_query_result)
            gap_note = (
                f"\n\nVerifier note: {state.verification_feedback}"
                if state.verification_feedback
                and state.verification_verdict == "warning"
                and "⚠️" in state.verification_feedback
                else ""
            )
            synthesis_prompt = (
                f"User question: {state.user_query}\n\n"
                f"SQL executed: {state.last_sql}\n\n"
                f"Row count: {row_count}\n\n"
                f"Query results:\n{table_preview}\n\n"
                + (gap_note.lstrip() + "\n\n" if gap_note else "")
                + "Write a clear, concise answer for the user. "
                "If the result is a table, present it as markdown. "
                "If it is a single value, describe it in plain English. "
                "If there is a verifier note about a missing dimension, "
                "mention it briefly at the end so the user knows what could not be answered.\n"
                'Output ONLY this JSON: {"final_answer": "<your answer>"}'
            )
            response = _llm().invoke([HumanMessage(content=synthesis_prompt)])
            raw = _extract_text(response.content)
            parsed = _try_parse_json(raw) or {}
            final_val = parsed.get("final_answer")
            if final_val is None or not isinstance(final_val, str):
                final_val = raw if isinstance(raw, str) and raw.strip() else table_preview
            final = str(final_val).strip()
        return {**base_reset, "final_answer": final, "messages": [AIMessage(content=final)]}

    # ── Fast-path 1: file load ──────────────────────────────────────────────
    load_match = _detect_load_intent(current_query)
    if load_match:
        return {
            **base_reset,
            "load_file_path": load_match["path"],
            "load_file_dataset": load_match["dataset"],
        }

    # ── Fast-path 2: schema / column listing ───────────────────────────────
    if _detect_schema_intent(current_query):
        query_lower = current_query.lower()
        target_table: str | None = None
        for t in all_tables:
            if t.lower() in query_lower:
                target_table = t
                break
        if target_table is None and most_recent:
            target_table = most_recent

        if not all_tables:
            reply = "No tables are loaded yet. Load a file first with: `load file at <path> as <name>`."
        elif target_table:
            schema_block = get_schema_context([target_table])
            reply = f"Here is the schema for **{target_table}**:\n\n```\n{schema_block}\n```"
        else:
            reply = f"Here are all loaded tables:\n\n```\n{tables_summary}\n```"
        return {**base_reset, "final_answer": reply, "messages": [AIMessage(content=reply)]}

    # ── Fast-path 3: META — agent-state questions ──────────────────────────
    if _detect_meta_intent(current_query):
        reply = _build_meta_reply(current_query, all_tables)
        return {**base_reset, "final_answer": reply, "messages": [AIMessage(content=reply)]}

    # ── LLM planning ───────────────────────────────────────────────────────
    ORCHESTRATOR_SYSTEM = f"""You are the Orchestrator of a DuckDB analytics agent.

Loaded tables:
{tables_summary}

Most recently loaded table: {most_recent or 'none'}

Classify the user message into exactly ONE of these three intents:

1. ANALYTICS — the question asks to compute, aggregate, filter, or retrieve data
   from the loaded tables. This includes any question that would require SQL.
   Decide which tables are needed, then produce an ordered plan.
   Step types: "profile" (explore a column) | "sql" (SELECT query).
   Every step description MUST name the exact table(s) and columns involved.
   For multi-table queries, state the join tables and join column pairs explicitly
   using the authoritative pairs from _relationships.
   {{"intent": "analytics", "plan": [{{"id": 1, "type": "sql", "description": "..."}}]}}

2. META — the user is asking about the agent's state: which tables are loaded,
   whether a specific table or file is available, what the agent can do, or
   how to use it. Do NOT classify these as chitchat.
   {{"intent": "meta", "final_answer": "<answer using the loaded tables list above>"}}

3. CHITCHAT — pure small talk, greetings, or questions completely unrelated to
   data or the agent (e.g. "what is the capital of France?").
   {{"intent": "chitchat", "final_answer": "<helpful reply>"}}

Decision rules (apply in order):
  • Any question referencing a table name OR asking about data/columns → ANALYTICS
  • Any question about what the agent has loaded, can access, or can do → META
  • Everything else → CHITCHAT

Output ONLY the JSON. No markdown, no explanation.
"""

    response = _llm().invoke(
        [SystemMessage(content=ORCHESTRATOR_SYSTEM), HumanMessage(content=current_query)]
    )
    raw_text = _extract_text(response.content)
    parsed = _try_parse_json(raw_text)

    if parsed is None and most_recent:
        parsed = {
            "intent": "analytics",
            "plan": [
                {
                    "id": 1,
                    "type": "sql",
                    "description": f"Answer the user question using the {most_recent} table: {current_query}",
                }
            ],
        }

    if parsed is None:
        reply = raw_text or "Hi! Ask a data question or say: load file at <path> as <name>."
        return {**base_reset, "final_answer": str(reply), "messages": [AIMessage(content=str(reply))]}

    intent = parsed.get("intent", "chitchat")

    if intent == "analytics":
        raw_steps = parsed.get("plan", [])
        if not raw_steps and most_recent:
            raw_steps = [
                {
                    "id": 1,
                    "type": "sql",
                    "description": f"Answer the user question using the {most_recent} table: {current_query}",
                }
            ]
        plan = [PlanStep(**step) for step in raw_steps]
        return {**base_reset, "plan": plan}

    if intent == "meta":
        reply = _build_meta_reply(current_query, all_tables)
        return {**base_reset, "final_answer": reply, "messages": [AIMessage(content=reply)]}

    reply = str(parsed.get("final_answer", "Hi! Ask a data question or load a file."))
    return {**base_reset, "final_answer": reply, "messages": [AIMessage(content=reply)]}


# ---------------------------------------------------------------------------
# Node: load_file_node
# ---------------------------------------------------------------------------

def load_file_node(state: AnalyticsState) -> dict:
    path = state.load_file_path
    dataset = state.load_file_dataset
    if not path:
        reply = "I couldn't find a file path. Please say: load file at <path> as <name>."
        return {
            "final_answer": reply,
            "messages": [AIMessage(content=reply)],
            "load_file_path": None,
            "load_file_dataset": None,
        }
    if not dataset:
        dataset = Path(path).stem.replace(" ", "_").replace("-", "_").lower()

    # Route NIQ panel files through the NIQ-aware loader automatically
    if _is_niq_file(path, dataset):
        result = load_niq_file.invoke({"path": path, "dataset_name": dataset})
        if result.startswith("ERROR"):
            reply = f"❌ Failed to load NIQ file: {result}"
        else:
            # Extract grain and period info from the result string for a richer reply
            grain_match = re.search(r"NIQ grain:\s*([^|]+)", result)
            grain_info = grain_match.group(1).strip() if grain_match else "detected"
            period_match = re.search(r"Period values:\s*([^|]+)", result)
            period_info = (
                f"\n  • Period values: `{period_match.group(1).strip()}`"
                if period_match
                else ""
            )
            reply = (
                f"✅ {result.split('|')[0].strip()}\n\n"
                f"**NIQ panel loaded** with grain `{grain_info}`.{period_info}\n\n"
                f"You can now ask NIQ questions about `{dataset}` — "
                f"e.g. *Käuferreichweite*, *penetration*, *spend per buyer*, *YoY change*."
            )
    else:
        result = load_file.invoke({"path": path, "dataset_name": dataset})
        reply = (
            f"❌ Failed to load file: {result}"
            if result.startswith("ERROR")
            else f"✅ {result}\n\nYou can now ask questions about the `{dataset}` table!"
        )

    return {
        "final_answer": reply,
        "messages": [AIMessage(content=reply)],
        "load_file_path": None,
        "load_file_dataset": None,
    }


# ---------------------------------------------------------------------------
# Node: profiler
# ---------------------------------------------------------------------------

def profiler(state: AnalyticsState) -> dict:
    step = _next_pending_step(state.plan)
    if step is None:
        return {"current_step": None}
    llm = _llm().bind_tools(PROFILER_TOOLS)
    response = llm.invoke(
        [
            SystemMessage(content=PROFILER_SYSTEM),
            HumanMessage(
                content=f"Profiling task: {step.description}\nUser question: {state.user_query}"
            ),
        ]
    )
    result_text = response.content or "No profiling result."
    updated_plan = [
        s.model_copy(update={"status": "done", "result": result_text})
        if s.id == step.id
        else s
        for s in state.plan
    ]
    return {"plan": updated_plan, "current_step": None}


# ---------------------------------------------------------------------------
# Node: sql_writer
# ---------------------------------------------------------------------------

def sql_writer(state: AnalyticsState) -> dict:
    step = _next_pending_step(state.plan)
    if step is None:
        step = next((s for s in state.plan if s.status == "running"), None)
    if step is None:
        return {"last_sql": "", "error": "No pending step found for sql_writer."}

    all_tables = _all_table_names()
    relevant_tables = _select_relevant_tables(
        state.user_query + " " + step.description, all_tables
    )
    if not relevant_tables:
        relevant_tables = [_most_recent_table()] if _most_recent_table() else all_tables

    schema_context = get_schema_context(relevant_tables)
    cast_warnings = _get_date_cast_warnings(relevant_tables)
    varchar_date_cols = _get_varchar_date_columns_multi(relevant_tables)
    relationships_block = _build_relationships_block(relevant_tables)

    keywords = [
        w
        for w in re.findall(r"[a-zA-Z]{4,}", state.user_query.lower())
        if w
        not in {
            "what", "show", "list", "give", "find", "from", "that", "this",
            "with", "have", "does", "each", "many", "much", "more", "most",
            "last", "year", "month", "week", "date", "time", "when", "where", "which",
        }
    ]
    semantic_hits: list[dict] = []
    seen_cols: set[str] = set()
    for kw in keywords[:6]:
        raw = search_semantic_lookup.invoke({"query": kw})
        for entry in json.loads(raw) if raw and not raw.startswith("ERROR") else []:
            key = f"{entry['dataset']}.{entry['column']}"
            if key not in seen_cols and entry["dataset"] in relevant_tables:
                seen_cols.add(key)
                semantic_hits.append(entry)
    semantic_context = ""
    if semantic_hits:
        lines = ["=== Semantic column hints ==="]
        for h in semantic_hits[:8]:
            lines.append(
                f"  {h['dataset']}.{h['column']} ({h['type']}) — {h.get('description', '')}"
            )
        semantic_context = "\n".join(lines)

    if state.retry_count > 0 and state.last_sql and state.last_sql.strip().upper().startswith("SELECT"):
        sql = _sanitize_sql(state.last_sql, varchar_date_cols)
        updated_plan = [
            s.model_copy(update={"status": "running"}) if s.id == step.id else s
            for s in state.plan
        ]
        return {"last_sql": sql, "plan": updated_plan, "verification_feedback": "", "current_step": step}

    prior_step_notes = "\n".join(
        f"- Step {s.id} ({s.description}): {s.result}"
        for s in state.plan
        if s.status == "done" and s.result and s.id != step.id
    )

    prompt = (
        f"Sub-task: {step.description}\n"
        f"User question: {state.user_query}\n\n"
        f"=== SCHEMA (use these table/column names verbatim) ===\n"
        f"{schema_context}\n\n"
        + (f"{relationships_block}\n\n" if relationships_block else "")
        + (f"{semantic_context}\n\n" if semantic_context else "")
        + f"=== MANDATORY DATE COLUMN RULES ===\n"
        f"{cast_warnings if cast_warnings else '  (no date columns require special handling)'}\n\n"
        f"Prior step results:\n{prior_step_notes or 'None'}\n\n"
        f"Verifier feedback (if retry): {state.verification_feedback or 'None'}\n\n"
        'Output JSON: {"sql": "...", "explanation": "..."}'
    )

    response = _llm().invoke(
        [SystemMessage(content=SQL_WRITER_SYSTEM), HumanMessage(content=prompt)]
    )
    parsed = _try_parse_json(_extract_text(response.content)) or {}
    sql = parsed.get("sql", "")

    if not sql and hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            args = tc.get("args") or {}
            sql = args.get("query") or args.get("sql") or ""
            if sql:
                break

    if not sql or not sql.strip():
        updated_plan = [
            s.model_copy(update={"status": "failed", "result": "sql_writer produced empty SQL"})
            if s.id == step.id
            else s
            for s in state.plan
        ]
        return {
            "last_sql": "",
            "plan": updated_plan,
            "error": "sql_writer produced empty SQL",
            "current_step": step,
        }

    sql = _sanitize_sql(sql, varchar_date_cols)

    updated_plan = [
        s.model_copy(update={"status": "running"}) if s.id == step.id else s
        for s in state.plan
    ]
    return {"last_sql": sql, "plan": updated_plan, "verification_feedback": "", "current_step": step}


# ---------------------------------------------------------------------------
# Node: execute_sql
# ---------------------------------------------------------------------------

def execute_sql(state: AnalyticsState) -> dict:
    if not state.last_sql or not state.last_sql.strip():
        def _fail_plan() -> list[PlanStep]:
            return [
                s.model_copy(update={"status": "failed", "result": "empty SQL"})
                if state.current_step and s.id == state.current_step.id
                else s
                for s in state.plan
            ]
        return {
            "last_query_result": "ERROR: empty SQL",
            "last_query_metadata": {},
            "plan": _fail_plan(),
            "error": "empty SQL",
        }

    result = run_sql.invoke({"query": state.last_sql})

    def _updated_plan(status: str, msg: str = "") -> list[PlanStep]:
        return [
            s.model_copy(update={"status": status, "result": msg})
            if state.current_step and s.id == state.current_step.id
            else s
            for s in state.plan
        ]

    if result.startswith("ERROR"):
        error_body = result[len("ERROR:"):].strip() if result.startswith("ERROR:") else result

        if "Binder Error" in error_body:
            return {
                "last_query_result": result,
                "last_query_metadata": {},
                "plan": _updated_plan("failed", error_body),
                "error": error_body,
            }

        return {
            "last_query_result": result,
            "last_query_metadata": {},
            "plan": _updated_plan("failed", result),
            "error": result,
        }

    parsed = json.loads(result)
    return {
        "last_query_result": json.dumps(parsed.get("rows", []), default=str),
        "last_query_metadata": parsed.get("metadata", {}),
        "plan": _updated_plan("done"),
        "error": "",
    }


# ---------------------------------------------------------------------------
# Node: verifier
# ---------------------------------------------------------------------------

def verifier(state: AnalyticsState) -> dict:
    step = state.current_step
    is_error = state.last_query_result.startswith("ERROR") if state.last_query_result else False
    row_count = state.last_query_metadata.get("row_count", "unknown")
    table_preview = _rows_to_markdown(state.last_query_result)

    if is_error and state.error and "Binder Error" in state.error:
        return {
            "verification_verdict": "fail",
            "verification_feedback": state.error,
            "current_step": None,
            "retry_count": state.retry_count,
        }

    # ── Cardinality check ──────────────────────────────────────────────────
    cardinality_warning = ""
    if not is_error and state.last_query_metadata:
        conn = get_connection()
        all_tables = _all_table_names()
        try:
            max_fact_rows = max(
                (
                    conn.execute(
                        "SELECT row_count FROM _data_registry WHERE dataset_name = ?", [t]
                    ).fetchone() or (0,)
                )[0]
                or 0
                for t in all_tables
            ) if all_tables else 0
            result_rows = state.last_query_metadata.get("row_count", 0) or 0
            if max_fact_rows > 0 and result_rows > max_fact_rows * 2:
                cardinality_warning = (
                    f"⚠️  Result has {result_rows} rows but the largest source table has "
                    f"{max_fact_rows} rows — possible JOIN fan-out (Cartesian product). "
                    "Verify GROUP BY and JOIN ON conditions."
                )
        except Exception:
            pass

    # ── JOIN key check ─────────────────────────────────────────────────────
    join_warnings: list[str] = []
    if not is_error and state.last_sql and "JOIN" in state.last_sql.upper():
        try:
            join_warnings = validate_join_in_sql(state.last_sql)
        except Exception:
            pass

    # ── Schema hint (for error correction) ────────────────────────────────
    schema_hint = ""
    most_recent = _most_recent_table()
    if most_recent:
        try:
            schema_hint = get_schema_context([most_recent])
        except Exception:
            pass

    # ── Semantic gap detection ─────────────────────────────────────────────
    semantic_gap_block = ""
    if not is_error and state.last_sql and state.user_query:
        try:
            all_tables = _all_table_names()
            gap_warnings = _detect_semantic_gaps(
                state.user_query,
                state.last_sql,
                all_tables,
            )
            if gap_warnings:
                semantic_gap_block = (
                    "\n⚠️  SEMANTIC GAP DETECTED (dimension requested but missing from result):\n"
                    + gap_warnings
                    + "\n"
                    "  → Set verdict=warning, do NOT set verdict=fail. "
                    "The result is still valid for the dimensions that ARE present. "
                    "Explain what was answered and what dimension is absent from the data.\n"
                )
        except Exception:
            pass

    if is_error:
        prompt = (
            f"Sub-task: {step.description if step else 'unknown'}\n"
            f"SQL that failed:\n{state.last_sql}\n\n"
            f"DuckDB error:\n{state.last_query_result}\n\n"
            f"Schema hint:\n{schema_hint}\n\n"
            "The SQL produced an error. Provide a corrected SQL query.\n"
            'Output ONLY: {"verdict": "fail", "feedback": "<what was wrong>", "corrected_sql": "<fixed SQL>"}'
        )
    else:
        join_warn_block = ""
        if join_warnings:
            join_warn_block = (
                "\n⚠️  UNREGISTERED JOIN KEYS (must fix):\n"
                + "\n".join(f"  - {w}" for w in join_warnings)
                + "\n\n"
            )
        prompt = (
            f"Sub-task: {step.description if step else 'unknown'}\n"
            f"SQL executed:\n{state.last_sql}\n\n"
            f"Row count: {row_count}\n"
            f"First rows:\n{table_preview[:1500]}\n\n"
            + (f"{cardinality_warning}\n\n" if cardinality_warning else "")
            + join_warn_block
            + semantic_gap_block
            + "Verify whether the SQL correctly answers the sub-task.\n"
            "- Correct → verdict=pass\n"
            "- Clear bug (wrong column, bad filter, JOIN fan-out, unregistered join key) → verdict=fail with corrected_sql\n"
            "- Missing dimension (semantic gap warning above) → verdict=warning with feedback explaining what is present vs absent\n"
            "- Plausible but uncertain → verdict=warning (treated as pass)\n"
            "- Do NOT call any tools.\n"
            'Output ONLY: {"verdict": "pass|fail|warning", "feedback": "...", "corrected_sql": "(only if fail)"}'
        )

    response = _llm().invoke(
        [SystemMessage(content=VERIFIER_SYSTEM), HumanMessage(content=prompt)]
    )
    parsed = _try_parse_json(_extract_text(response.content)) or {"verdict": "pass", "feedback": ""}
    verdict = str(parsed.get("verdict", "pass"))
    feedback = str(parsed.get("feedback", ""))
    corrected_sql = parsed.get("corrected_sql", "")

    new_status = "done" if verdict in ("pass", "warning") else "pending"
    updated_plan = [
        s.model_copy(update={"status": new_status, "result": feedback})
        if step and s.id == step.id
        else s
        for s in state.plan
    ]
    updates: dict = {
        "verification_verdict": verdict,
        "verification_feedback": feedback,
        "plan": updated_plan,
        "current_step": None,
        "retry_count": state.retry_count + 1,
    }
    if corrected_sql and verdict == "fail":
        updates["last_sql"] = str(corrected_sql)
    return updates
