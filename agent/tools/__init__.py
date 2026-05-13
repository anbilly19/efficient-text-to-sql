"""tools sub-package — re-exports every public symbol from agent/tools.py
and adds the NIQ ingest tool and schema_tools.

Python resolves `agent.tools` to this package directory, so anything that
does `from agent.tools import ORCHESTRATOR_TOOLS` (or any other symbol from
the original flat tools.py) would break with an empty __init__.py.

Solution: star-import everything from the sibling tools.py module, which
lives one level up at agent/tools.py.  Because the package directory shadows
the flat file we must import it explicitly via importlib.
"""
import importlib as _importlib
import sys as _sys

# Load the sibling agent/tools.py as a distinct module to avoid the name clash
# with this package.  We register it under a private alias so it is only
# imported once even if __init__.py is re-executed.
_ALIAS = "_agent_tools_flat"
if _ALIAS not in _sys.modules:
    import importlib.util as _util
    import pathlib as _pathlib

    _flat = _pathlib.Path(__file__).parent.parent / "tools.py"
    _spec = _util.spec_from_file_location(_ALIAS, _flat)
    _mod = _util.module_from_spec(_spec)
    _sys.modules[_ALIAS] = _mod
    _spec.loader.exec_module(_mod)

_flat_mod = _sys.modules[_ALIAS]

# Re-export every public name so `from agent.tools import X` keeps working.
from _agent_tools_flat import *  # noqa: F401, F403, E402
from _agent_tools_flat import (  # noqa: E402  (explicit for IDE / type checkers)
    ORCHESTRATOR_TOOLS,
    PROFILER_TOOLS,
    SQL_WRITER_TOOLS,
    get_schema,
    load_file,
    run_sql,
    search_semantic_lookup,
    profile_column,
    get_schema_context_tool,
    # Private helpers needed by niq_ingest — re-exported explicitly so
    # `from agent.tools import _get_parquet_store` resolves through the
    # already-loaded _agent_tools_flat module and never triggers a second
    # importlib load (which would create a second agent.database singleton).
    _get_parquet_store,
    _PROJECT_ROOT,
)

# Schema tools added in this package.
from agent.tools.schema_tools import (  # noqa: E402
    select_tables,
    get_relationships,
    validate_joins,
)

# NIQ ingest tool.
from agent.tools.niq_ingest import (  # noqa: E402
    load_niq_file,
    detect_niq_structure,
    classify_niq_columns,
    seed_niq_semantic_map,
    build_niq_table_context,
    NiqMeta,
    NiqColInfo,
)

__all__ = [
    # ── from tools.py ────────────────────────────────────────────────────────
    "ORCHESTRATOR_TOOLS",
    "PROFILER_TOOLS",
    "SQL_WRITER_TOOLS",
    "get_schema",
    "load_file",
    "run_sql",
    "search_semantic_lookup",
    "profile_column",
    "get_schema_context_tool",
    "_get_parquet_store",
    "_PROJECT_ROOT",
    # ── schema_tools ─────────────────────────────────────────────────────────
    "select_tables",
    "get_relationships",
    "validate_joins",
    # ── niq_ingest ────────────────────────────────────────────────────────────
    "load_niq_file",
    "detect_niq_structure",
    "classify_niq_columns",
    "seed_niq_semantic_map",
    "build_niq_table_context",
    "NiqMeta",
    "NiqColInfo",
]
