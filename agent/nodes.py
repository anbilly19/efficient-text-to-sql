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
    PROFILER_TOOLS,
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

_NIQ_FILE_PATTERNS = re.compile(
    r"niq|nielsen|panel|penetration|buyer|haushalt|käufer",
    re.IGNORECASE,
)

_YOY_COL_RE = re.compile(
    r'"([^"]*vs\.\s*VJ\s*\(%\s*Ver\.\))"'
    r'|([^\s"()]+\s+vs\.\s*VJ\s*\(%\s*Ver\.\))',
    re.IGNORECASE,
)

_NIQ_METRIC_TERMS: dict[str, str] = {
    "penetration":              "Penetration (%)",
    "käuferreichweite":         "Penetration (%)",
    "kaeuferreichweite":        "Penetration (%)",
    "spend":                    "Ausgaben pro Käuferhaushalt",
    "ausgaben":                 "Ausgaben pro Käuferhaushalt",
    "frequency":                "Einkaufsakte pro Käuferhaushalt",
    "einkaufsakte":             "Einkaufsakte pro Käuferhaushalt",
    "käuferhaushalte":          "Käuferhaushalte",
    "buyers":                   "Käuferhaushalte",
}

_DISTINCT_VALUES_PATTERNS = re.compile(
    r"\b(?:what|which|show|list)\b.{0,40}\b(?:periods?|values?|categories|options|available)\b"
    r"|\b(?:periods?|values?|categories)\b.{0,30}\b(?:available|exist|loaded|in)\b",
    re.IGNORECASE,
)

_NIQ_PERIOD_COL = "Periods"

# Tightened: only explicit meta/grain questions, NOT 'annual figures' data questions
_GRAIN_PATTERNS = re.compile(
    r"\b(?:grain|granularity|temporal\s+grain|how\s+often|frequency\s+of\s+data"
    r"|monthly|quarterly|weekly)\b"
    r"|\bwhat\s+(?:is\s+(?:the\s+)?)?(?:time\s+)?grain\b"
    r"|\bhow\s+(?:is\s+(?:the\s+)?data\s+)?(?:broken\s+down|aggregated|collected|reported)\b",
    re.IGNORECASE,
)

_ALL_FIGURES_PATTERNS = re.compile(
    r"\b(?:show|give|list|all)\b.{0,30}\b(?:figures?|values?|data|numbers?)\b"
    r"|\b(?:annual|yearly|all)\b.{0,20}\b(?:figures?|values?|penetration|spend|ausgaben)\b",
    re.IGNORECASE,
)

# Dimension columns to highlight in empty-SQL suggestions (ordered by importance)
_NIQ_DIMENSION_COLS = ["Products", "Retailers", "Demographics", "Geographies", "People", "Periods"]

# ---------------------------------------------------------------------------
# Fast-path 6: semantic map lookup
# ---------------------------------------------------------------------------

_SEMANTIC_LOOKUP_PATTERNS = re.compile(
    r"\bwhat\s+(?:does|do)\s+['\"']?(?P<term>[\w\s\u00c0-\u024f]+?)['\"']?\s+"
    r"(?:map\s+to|correspond\s+to|mean|stand\s+for|refer\s+to|translate\s+to)\b"
    r"|\blook\s+up\s+['\"']?(?P<term2>[\w\s\u00c0-\u024f]+?)['\"']?\s+in\s+(?:the\s+)?semantic\s+map\b"
    r"|\bwhat\s+column\s+(?:corresponds?\s+to|is|matches?)\s+['\"']?(?P<term3>[\w\s\u00c0-\u024f]+?)['\"']?"
    r"(?:\s+in\b|\s*\?|$)"
    r"|\bsemantic\s+map\b.{0,60}\b['\"']?(?P<term4>[\w\s\u00c0-\u024f]+?)['\"']?\s*(?:for|in|of)\b",
    re.IGNORECASE,
)


def _detect_semantic_lookup_intent(text: str, all_tables: list[str]) -> tuple[str, str | None] | None:
    """Return (term, dataset_name | None) if the query is a semantic-map lookup."""
    m = _SEMANTIC_LOOKUP_PATTERNS.search(text)
    if not m:
        return None
    term = next(
        (v.strip() for v in (m.group("term"), m.group("term2"), m.group("term3"), m.group("term4")) if v),
        None,
    )
    if not term or len(term) < 2:
        return None
    # strip trailing noise words
    term = re.sub(r"\s+(?:in|for|of|the|a|an|table|dataset|column|field)\s*$", "", term, flags=re.IGNORECASE).strip()
    if not term:
        return None
    dataset: str | None = None
    text_lower = text.lower()
    for t in all_tables:
        if t.lower() in text_lower:
            dataset = t
            break
    return (term, dataset)


def _build_semantic_map_reply(term: str, dataset: str | None, all_tables: list[str]) -> str:
    """Query _semantic_map for `term` and return a formatted answer."""
    conn = get_connection()
    datasets_to_search = [dataset] if dataset else all_tables
    if not datasets_to_search:
        return f"No tables are loaded — cannot look up `{term}` in the semantic map."

    # Search with LIKE for flexible partial matching (case-insensitive in DuckDB)
    placeholders = ", ".join("?" * len(datasets_to_search))
    try:
        rows = conn.execute(
            f"SELECT dataset_name, term, column_name, description "
            f"FROM _semantic_map "
            f"WHERE dataset_name IN ({placeholders}) "
            f"  AND LOWER(term) LIKE LOWER(?) "
            f"ORDER BY dataset_name, term",
            datasets_to_search + [f"%{term}%"],
        ).fetchall()
    except Exception as exc:
        return f"❌ Could not query `_semantic_map`: {exc}"

    # Fallback: search by column_name too
    if not rows:
        try:
            rows = conn.execute(
                f"SELECT dataset_name, term, column_name, description "
                f"FROM _semantic_map "
                f"WHERE dataset_name IN ({placeholders}) "
                f"  AND LOWER(column_name) LIKE LOWER(?) "
                f"ORDER BY dataset_name, term",
                datasets_to_search + [f"%{term}%"],
            ).fetchall()
        except Exception:
            pass

    if not rows:
        scope = f"`{dataset}`" if dataset else "any loaded table"
        return (
            f"🔍 No semantic map entry found for **`{term}`** in {scope}.\n\n"
            f"This term is not registered as an alias in `_semantic_map`. "
            f"Try a related term, or check the available columns with: *show columns in {dataset or (all_tables[0] if all_tables else 'table')}*."
        )

    lines = [f"🗺️ Semantic map results for **`{term}`**:\n"]
    lines.append("| Dataset | Alias / Term | → Column | Description |")
    lines.append("|---------|-------------|----------|-------------|")
    for ds, t, col, desc in rows:
        lines.append(f"| `{ds}` | `{t}` | `{col}` | {desc or ''} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def _build_empty_sql_message(explanation: str, user_query: str, all_tables: list[str]) -> str:
    """Build a friendly, informative message when sql_writer returns empty SQL."""
    # Extract the concept the user asked for from the explanation
    concept_match = re.search(r"matching ['\"']?([\w\s]+)['\"']?", explanation, re.IGNORECASE)
    concept = concept_match.group(1).strip() if concept_match else ""

    # Collect dimension columns from the most relevant table
    conn = get_connection()
    dim_cols: list[str] = []
    metric_cols: list[str] = []
    table_used: str = ""
    for tbl in all_tables:
        if tbl.lower() in user_query.lower() or not table_used:
            table_used = tbl
            try:
                rows = conn.execute(
                    "SELECT column_name FROM _column_catalog "
                    "WHERE dataset_name = ? ORDER BY column_name",
                    [tbl],
                ).fetchall()
                all_cols = [r[0] for r in rows]
                dim_cols = [c for c in _NIQ_DIMENSION_COLS if c in all_cols]
                metric_cols = [
                    c for c in all_cols
                    if c not in _NIQ_DIMENSION_COLS
                    and "VJ" not in c
                    and "vs." not in c
                ][:4]
            except Exception:
                pass
            break

    lines: list[str] = []
    if concept and table_used:
        lines.append(
            f"❌ **`{table_used}` doesn't contain a column matching '{concept}'.** "
            f"This table is a household panel dataset — it has no transactional records "
            f"(no customer IDs, invoice numbers, order IDs, or store IDs)."
        )
    elif explanation:
        lines.append(f"❌ {explanation}")
    else:
        lines.append("❌ I couldn't generate a query for that question.")

    if dim_cols:
        lines.append(f"\n**Available dimensions:** {', '.join(f'`{c}`' for c in dim_cols)}")
    if metric_cols:
        lines.append(f"**Available metrics:** {', '.join(f'`{c}`' for c in metric_cols)}")

    if table_used and dim_cols and metric_cols:
        lines.append(
            f"\n💡 Try asking:\n"
            f"  • *Top {dim_cols[0].lower()} by {metric_cols[0]}*\n"
            + (f"  • *{metric_cols[0]} by {dim_cols[1].lower()}*\n" if len(dim_cols) > 1 else "")
            + f"  • *Penetration (%) by {dim_cols[0].lower()}*"
        )

    return "\n".join(lines)


def _infer_niq_grain(cy_label: str) -> str:
    """Derive a human-readable grain string from the NIQ CY period label."""
    label = cy_label.lower()
    if re.search(r"12\s*m|52\s*w|jahr", label):
        return "annual (52-week rolling)"
    if re.search(r"13\s*w|quartal", label):
        return "quarterly"
    if re.search(r"4\s*w|monat", label):
        return "monthly (4-week)"
    if re.search(r"26\s*w", label):
        return "semi-annual (26-week)"
    return "detected"


def _is_niq_file(path: str, dataset_name: str) -> bool:
    combined = f"{path} {dataset_name}".lower()
    return bool(_NIQ_FILE_PATTERNS.search(combined))


def _get_niq_cy_label(dataset_name: str) -> str | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT summary FROM _table_context WHERE dataset_name = ?",
            [dataset_name],
        ).fetchone()
        if row and row[0]:
            m = re.search(r"CY='([^']+)'", row[0])
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def _get_niq_grain(dataset_name: str) -> str | None:
    """Read NIQ grain from _table_context summary string."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT summary FROM _table_context WHERE dataset_name = ?",
            [dataset_name],
        ).fetchone()
        if row and row[0]:
            m = re.search(r"grain='([^']+)'", row[0], re.IGNORECASE)
            if m and m.group(1).lower() != "detected":
                return m.group(1).strip()
            cy_m = re.search(r"CY='([^']+)'", row[0])
            if cy_m:
                inferred = _infer_niq_grain(cy_m.group(1))
                if inferred != "detected":
                    return inferred
    except Exception:
        pass
    return None


def _niq_yoy_fast_path(resolved_query: str, dataset_name: str) -> str | None:
    m = _YOY_COL_RE.search(resolved_query)
    if not m:
        return None
    yoy_col = (m.group(1) or m.group(2) or "").strip()
    if not yoy_col:
        return None
    cy_label = _get_niq_cy_label(dataset_name)
    if not cy_label:
        return None

    dimension_cols = []
    q_lower = resolved_query.lower()
    dim_map = {
        "Products":     ["product", "products", "marke", "brand", "artikel", "produkt", "sku"],
        "Retailers":    ["retailer", "retailers", "händler", "store", "shop", "kanal"],
        "People":       ["people", "panel group"],
        "Demographics": ["demographics", "segment"],
        "Geographies":  ["geograph", "region", "market"],
    }
    for col, keywords in dim_map.items():
        if any(kw in q_lower for kw in keywords):
            dimension_cols.append(col)
    if not dimension_cols:
        dimension_cols = ["Products"]

    select_dims = ", ".join(f'"{c}"' for c in dimension_cols)
    group_by    = ", ".join(f'"{c}"' for c in dimension_cols)
    return (
        f'SELECT {select_dims},\n'
        f'       AVG("{yoy_col}") AS yoy_delta\n'
        f'FROM "{dataset_name}"\n'
        f"WHERE \"Periods\" = '{cy_label}'\n"
        f'  AND "{yoy_col}" IS NOT NULL\n'
        f'GROUP BY {group_by}\n'
        f'ORDER BY yoy_delta DESC'
    )


def _niq_metric_fastpath(user_query: str, dataset_name: str) -> str | None:
    if not _ALL_FIGURES_PATTERNS.search(user_query):
        return None
    cy_label = _get_niq_cy_label(dataset_name)
    if not cy_label:
        return None
    q_lower = user_query.lower()
    metric_col_fragment: str | None = None
    for term, col_fragment in _NIQ_METRIC_TERMS.items():
        if term in q_lower:
            metric_col_fragment = col_fragment
            break
    if not metric_col_fragment:
        return None
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT column_name FROM _column_catalog WHERE dataset_name = ?",
            [dataset_name],
        ).fetchall()
    except Exception:
        return None
    exact_col: str | None = None
    for (col,) in rows:
        if metric_col_fragment.lower() in col.lower():
            if "vs." not in col and "VJ" not in col:
                exact_col = col
                break
    if not exact_col:
        return None
    return (
        f'SELECT "Products",\n'
        f'       AVG("{exact_col}") AS metric_value\n'
        f'FROM "{dataset_name}"\n'
        f"WHERE \"Periods\" = '{cy_label}'\n"
        f'  AND "{exact_col}" IS NOT NULL\n'
        f'GROUP BY "Products"\n'
        f'ORDER BY metric_value DESC'
    )


def _detect_distinct_values_intent(text: str, all_tables: list[str]) -> tuple[str, str] | None:
    if not _DISTINCT_VALUES_PATTERNS.search(text):
        return None
    text_lower = text.lower()
    target_table: str | None = None
    for t in all_tables:
        if t.lower() in text_lower:
            target_table = t
            break
    if target_table is None:
        return None
    if "period" in text_lower:
        return (target_table, _NIQ_PERIOD_COL)
    return None


def _detect_grain_intent(text: str) -> bool:
    return bool(_GRAIN_PATTERNS.search(text))


def _build_grain_reply(query: str, all_tables: list[str]) -> str | None:
    text_lower = query.lower()
    target_table: str | None = None
    for t in all_tables:
        if t.lower() in text_lower:
            target_table = t
            break
    if target_table is None and all_tables:
        target_table = all_tables[-1]
    if not target_table:
        return None

    grain = _get_niq_grain(target_table)
    cy_label = _get_niq_cy_label(target_table)

    if grain:
        cy_note = f" Current year period label: `{cy_label}`" if cy_label else ""
        return f"The grain of **{target_table}** is **{grain}** (sourced from `_table_context`).{cy_note}"

    conn = get_connection()
    try:
        rows = conn.execute(
            f'SELECT DISTINCT "Periods" FROM "{target_table}" ORDER BY 1'
        ).fetchall()
        if rows:
            period_vals = ", ".join(f"`{r[0]}`" for r in rows)
            return (
                f"The grain of **{target_table}** could not be determined from metadata. "
                f"Available period values: {period_vals}"
            )
    except Exception:
        pass
    return None


def _select_relevant_tables(user_query: str, all_tables: list[str]) -> list[str]:
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
        if query_is_niq and tags and "niq" in tags.lower():
            score += 2
        scored.append((score, dataset_name))
    selected = [name for score, name in scored if score > 0]
    if not selected:
        return _select_relevant_tables_llm(user_query, all_tables)
    return selected


def _select_relevant_tables_llm(user_query: str, all_tables: list[str]) -> list[str]:
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
    sql = re.sub(r'"(\w+)""', r'"\1"', sql)
    sql = re.sub(r'""\b(\w+)"', r'"\1"', sql)
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

_TABLE_NOT_FOUND_RE = re.compile(
    r'(?:Table with name|Table|Catalog Error.*?table)\s+["\']?([\w]+)["\']?\s+does not exist'
    r'|(?:relation|table)\s+["\']?([\w.]+)["\']?\s+does not exist'
    r'|([\w"]+)\s+not exist',
    re.IGNORECASE,
)

_BINDER_ERROR_RE = re.compile(
    r'Binder Error[:\s]+Referenced column\s+["\']?([\w ]+)["\']?\s+not found'
    r'|Binder Error[:\s]+.*?Column[\s]+["\']?([\w ]+)["\']?\s+not found',
    re.IGNORECASE | re.DOTALL,
)

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
# NIQ alias resolution
# ---------------------------------------------------------------------------

def _resolve_niq_aliases(text: str, dataset_names: list[str]) -> str:
    if not text or not dataset_names:
        return text
    conn = get_connection()
    try:
        placeholders = ", ".join("?" * len(dataset_names))
        rows = conn.execute(
            f"SELECT term, column_name FROM _semantic_map "
            f"WHERE dataset_name IN ({placeholders}) "
            f"ORDER BY LENGTH(term) DESC",
            dataset_names,
        ).fetchall()
    except Exception:
        return text
    resolved = text
    for term, col_name in rows:
        if not term or not col_name:
            continue
        if term.strip().lower() == col_name.strip().lower():
            continue
        boundary = r"(?<![\w\u00c0-\u024f])" + re.escape(term) + r"(?![\w\u00c0-\u024f])"
        try:
            pattern = re.compile(boundary, re.IGNORECASE)
        except re.error:
            continue
        resolved = pattern.sub(col_name, resolved)
    return resolved


# ---------------------------------------------------------------------------
# Error message builders
# ---------------------------------------------------------------------------

def _build_table_not_found_message(error: str, sql: str, all_tables: list[str]) -> str | None:
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


def _detect_semantic_gaps(user_query: str, sql: str, table_names: list[str]) -> str:
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

            # Empty-SQL path: sql_writer declined to generate SQL
            if err and "empty SQL" in err.lower() or (state.last_sql == ""):
                explanation = err if "No column matching" in err else ""
                friendly = _build_empty_sql_message(explanation, state.user_query, all_tables)
                return {**base_reset, "final_answer": friendly, "messages": [AIMessage(content=friendly)]}

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

    # ── Fast-path 3: distinct column values (e.g. 'what periods are available') ──
    distinct_match = _detect_distinct_values_intent(current_query, all_tables)
    if distinct_match:
        tbl, col = distinct_match
        sql = f'SELECT DISTINCT "{col}" FROM "{tbl}" ORDER BY 1'
        result_raw = run_sql.invoke({"query": sql})
        if result_raw.startswith("ERROR"):
            reply = f"❌ Could not retrieve distinct values: {result_raw}"
        else:
            parsed_rows = json.loads(result_raw).get("rows", [])
            vals = [str(r.get(col, "")) for r in parsed_rows]
            reply = (
                f"Available values for **{col}** in `{tbl}`:\n\n"
                + "\n".join(f"  • `{v}`" for v in vals)
            )
        return {**base_reset, "final_answer": reply, "messages": [AIMessage(content=reply)]}

    # ── Fast-path 4: grain / temporal granularity question ─────────────────
    if _detect_grain_intent(current_query) and all_tables:
        grain_reply = _build_grain_reply(current_query, all_tables)
        if grain_reply:
            return {**base_reset, "final_answer": grain_reply, "messages": [AIMessage(content=grain_reply)]}

    # ── Fast-path 5: META — agent-state questions ──────────────────────────
    if _detect_meta_intent(current_query):
        reply = _build_meta_reply(current_query, all_tables)
        return {**base_reset, "final_answer": reply, "messages": [AIMessage(content=reply)]}

    # ── Fast-path 6: semantic map lookup ───────────────────────────────────
    sem_match = _detect_semantic_lookup_intent(current_query, all_tables)
    if sem_match:
        term, dataset = sem_match
        sem_reply = _build_semantic_map_reply(term, dataset, all_tables)
        return {**base_reset, "final_answer": sem_reply, "messages": [AIMessage(content=sem_reply)]}

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

    if _is_niq_file(path, dataset):
        result = load_niq_file.invoke({"path": path, "dataset_name": dataset})
        if result.startswith("ERROR"):
            reply = f"❌ Failed to load NIQ file: {result}"
        else:
            grain_match = re.search(r"NIQ grain:\s*([^|]+)", result)
            raw_grain = grain_match.group(1).strip() if grain_match else "detected"

            if raw_grain.lower() == "detected":
                cy_match = re.search(r"CY='([^']+)'", result)
                if cy_match:
                    raw_grain = _infer_niq_grain(cy_match.group(1))

            period_match = re.search(r"Period values:\s*([^|]+)", result)
            period_info = (
                f"\n  • Period values: `{period_match.group(1).strip()}`"
                if period_match
                else ""
            )
            reply = (
                f"✅ {result.split('|')[0].strip()}\n\n"
                f"**NIQ panel loaded** with grain `{raw_grain}`.{period_info}\n\n"
                f"You can now ask NIQ questions about `{dataset}` — "
                f"e.g. *Käuferreichweite*, *penetration*, *spend per buyer*, *YoY change*."
            )

            try:
                conn = get_connection()
                row = conn.execute(
                    "SELECT summary FROM _table_context WHERE dataset_name = ?",
                    [dataset],
                ).fetchone()
                if row and row[0]:
                    updated_summary = re.sub(
                        r"grain='[^']*'",
                        f"grain='{raw_grain}'",
                        row[0],
                    )
                    if updated_summary == row[0]:
                        updated_summary = row[0] + f" | grain='{raw_grain}'"
                    conn.execute(
                        "UPDATE _table_context SET summary = ? WHERE dataset_name = ?",
                        [updated_summary, dataset],
                    )
            except Exception:
                pass
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

    resolved_query = _resolve_niq_aliases(state.user_query, relevant_tables)
    resolved_step_desc = _resolve_niq_aliases(step.description, relevant_tables)

    if relevant_tables and state.retry_count == 0:
        fast_sql = _niq_yoy_fast_path(resolved_query, relevant_tables[0])
        if fast_sql:
            updated_plan = [
                s.model_copy(update={"status": "running"}) if s.id == step.id else s
                for s in state.plan
            ]
            return {
                "last_sql": fast_sql,
                "plan": updated_plan,
                "verification_feedback": "",
                "current_step": step,
            }
        metric_sql = _niq_metric_fastpath(state.user_query, relevant_tables[0])
        if metric_sql:
            updated_plan = [
                s.model_copy(update={"status": "running"}) if s.id == step.id else s
                for s in state.plan
            ]
            return {
                "last_sql": metric_sql,
                "plan": updated_plan,
                "verification_feedback": "",
                "current_step": step,
            }

    schema_context = get_schema_context(relevant_tables)
    cast_warnings = _get_date_cast_warnings(relevant_tables)
    varchar_date_cols = _get_varchar_date_columns_multi(relevant_tables)
    relationships_block = _build_relationships_block(relevant_tables)

    keywords = [
        w
        for w in re.findall(r"[a-zA-Z]{4,}", resolved_query.lower())
        if w not in {
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
            if key not in seen_cols:
                seen_cols.add(key)
                semantic_hits.append(entry)

    semantic_block = ""
    if semantic_hits:
        lines = ["## Semantic Map Hints (keyword → actual column name)"]
        for h in semantic_hits[:8]:
            lines.append(
                f"  '{h['column']}' in {h['dataset']}"
                + (f"  # {h['description']}" if h.get("description") else "")
            )
        semantic_block = "\n".join(lines)

    sql_prompt = SQL_WRITER_SYSTEM.format(
        schema=schema_context,
        cast_warnings=cast_warnings or "  (none)",
        relationships=relationships_block or "  (none defined)",
        semantic_map=semantic_block or "  (no hints)",
    )

    response = _llm().invoke(
        [
            SystemMessage(content=sql_prompt),
            HumanMessage(
                content=(
                    f"User question: {resolved_query}\n"
                    f"Step description: {resolved_step_desc}\n"
                    + (
                        f"Previous attempt failed with: {state.verification_feedback}\n"
                        if state.retry_count > 0 and state.verification_feedback
                        else ""
                    )
                )
            ),
        ]
    )

    raw = _extract_text(response.content)
    parsed_resp = _try_parse_json(raw) or {}
    sql = parsed_resp.get("sql", "").strip()
    explanation = parsed_resp.get("explanation", "")

    if not sql:
        updated_plan = [
            s.model_copy(update={"status": "failed"}) if s.id == step.id else s
            for s in state.plan
        ]
        return {
            "last_sql": "",
            "plan": updated_plan,
            "error": explanation or "empty SQL",
            "current_step": None,
        }

    sql = _sanitize_sql(sql, varchar_date_cols)

    updated_plan = [
        s.model_copy(update={"status": "running"}) if s.id == step.id else s
        for s in state.plan
    ]
    return {
        "last_sql": sql,
        "plan": updated_plan,
        "verification_feedback": "",
        "current_step": step,
    }


# ---------------------------------------------------------------------------
# Node: executor
# ---------------------------------------------------------------------------

def executor(state: AnalyticsState) -> dict:
    sql = state.last_sql
    if not sql:
        step = state.current_step
        updated_plan = (
            [
                s.model_copy(update={"status": "failed", "result": "No SQL to execute."})
                if s.id == step.id
                else s
                for s in state.plan
            ]
            if step
            else state.plan
        )
        return {
            "plan": updated_plan,
            "last_query_result": "",
            "last_query_metadata": {},
            "error": "No SQL to execute.",
            "current_step": None,
        }

    result_raw = run_sql.invoke({"query": sql})

    step = state.current_step
    if result_raw.startswith("ERROR"):
        updated_plan = (
            [
                s.model_copy(update={"status": "failed", "result": result_raw})
                if s.id == step.id
                else s
                for s in state.plan
            ]
            if step
            else state.plan
        )
        return {
            "plan": updated_plan,
            "last_query_result": result_raw,
            "last_query_metadata": {},
            "error": result_raw,
            "current_step": None,
        }

    try:
        parsed = json.loads(result_raw)
        rows = parsed.get("rows", [])
        metadata = {
            "row_count": parsed.get("row_count", len(rows)),
            "columns": parsed.get("columns", list(rows[0].keys()) if rows else []),
            "truncated": parsed.get("truncated", False),
        }
        result_str = json.dumps(rows)
    except Exception:
        result_str = result_raw
        metadata = {}

    updated_plan = (
        [
            s.model_copy(update={"status": "done", "result": result_str[:500]})
            if s.id == step.id
            else s
            for s in state.plan
        ]
        if step
        else state.plan
    )
    return {
        "plan": updated_plan,
        "last_query_result": result_str,
        "last_query_metadata": metadata,
        "error": "",
        "current_step": None,
    }


# ---------------------------------------------------------------------------
# Node: verifier
# ---------------------------------------------------------------------------

def verifier(state: AnalyticsState) -> dict:
    sql = state.last_sql
    result = state.last_query_result
    user_query = state.user_query

    if not sql or not result or result.startswith("ERROR"):
        return {"verification_verdict": "skip", "verification_feedback": ""}

    all_tables = _all_table_names()
    relevant_tables = _select_relevant_tables(user_query, all_tables)

    row_count = state.last_query_metadata.get("row_count", "?")
    preview = _rows_to_markdown(result)

    gap_warnings = _detect_semantic_gaps(user_query, sql, relevant_tables)
    gap_section = (
    f"Semantic gap warnings detected:\n{gap_warnings}"
    if gap_warnings
    else ""
)

    VERIFIER_PROMPT = VERIFIER_SYSTEM + f"""

User question: {user_query}

SQL executed:
{sql}

Row count: {row_count}
Result preview:
{preview}

{gap_section}

Evaluate the result. Output ONLY JSON:
{{"verdict": "ok" | "retry" | "warning", "feedback": "..."}}
- "ok": result answers the question correctly
- "retry": result is wrong or empty — include corrected SQL hint in feedback
- "warning": result is technically correct but has a semantic gap noted above
"""

    response = _llm().invoke([HumanMessage(content=VERIFIER_PROMPT)])
    raw = _extract_text(response.content)
    parsed = _try_parse_json(raw) or {}
    verdict = parsed.get("verdict", "ok")
    feedback = parsed.get("feedback", "")

    if verdict == "retry" and state.retry_count >= 2:
        verdict = "warning"
        feedback = f"Max retries reached. Last feedback: {feedback}"

    if verdict == "retry":
        step = state.current_step
        updated_plan = (
            [
                s.model_copy(update={"status": "pending"}) if s.id == step.id else s
                for s in state.plan
            ]
            if step
            else state.plan
        )
        return {
            "verification_verdict": verdict,
            "verification_feedback": feedback,
            "plan": updated_plan,
            "retry_count": state.retry_count + 1,
        }

    return {"verification_verdict": verdict, "verification_feedback": feedback}
