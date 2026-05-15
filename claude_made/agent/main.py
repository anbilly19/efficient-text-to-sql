#!/usr/bin/env python3
"""
CLI entrypoint.

Usage:
  python main.py load <path> <table_name>
  python main.py query "<natural language question>"
  python main.py shell       # interactive REPL
"""

import sys


def cmd_load(path: str, table_name: str, fast: bool = False) -> None:
    from agent.ingestion.loader import load_file
    result = load_file(path, table_name, run_llm_enrichment=not fast)
    print(f"\n✓ Loaded '{table_name}'")
    print(f"  Rows: {result['rows']:,}  Columns: {result['columns']}")
    print(f"  Grain: {result['grain']}")
    print(f"  CY label: {result['cy_label']}")
    print(f"  PY label: {result['py_label']}")
    print(f"  Summary: {result['summary']}")


def cmd_query(question: str) -> None:
    from agent.graph.nodes import run_query
    answer = run_query(question)
    print(f"\n{answer}")


def cmd_shell() -> None:
    from agent.graph.nodes import run_query
    print("Analytics agent — type your question or 'exit'")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        print(run_query(q))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    if cmd == "load" and len(args) >= 3:
        cmd_load(args[1], args[2], fast="--fast" in args)
    elif cmd == "query" and len(args) >= 2:
        cmd_query(" ".join(args[1:]))
    elif cmd == "shell":
        cmd_shell()
    else:
        print(__doc__)
        sys.exit(1)
