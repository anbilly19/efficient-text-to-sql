#!/usr/bin/env python3
"""CLI wrapper around `langgraph dev`.

Usage
-----
  uv run cli.py                          # OpenAI, gpt-4o-mini (default)
  uv run cli.py --backend ollama         # Ollama, gemma4:e2b (default model)
  uv run cli.py --backend ollama --model gemma4:e4b
  uv run cli.py --backend ollama --model gemma4:e2b --ollama-url http://localhost:11434
  uv run cli.py --backend openai --model gpt-4o

All flags simply set the corresponding environment variable before handing
off to `langgraph dev`, so every other langgraph CLI option still works:
  uv run cli.py --backend ollama -- --port 8080
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Start the efficient-text-to-sql LangGraph server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run cli.py                            # OpenAI default\n"
            "  uv run cli.py --backend ollama           # Ollama gemma4:e2b\n"
            "  uv run cli.py --backend ollama --model gemma4:e4b\n"
            "  uv run cli.py --backend openai --model gpt-4o\n"
            "  uv run cli.py --backend ollama -- --port 8080  # pass extra langgraph args\n"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "ollama"],
        default=None,
        help="LLM backend to use. Overrides LLM_BACKEND env var. (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. For OpenAI: e.g. gpt-4o-mini, gpt-4o. "
            "For Ollama: e.g. gemma4:e2b, gemma4:e4b. "
            "Overrides OPENAI_MODEL / OLLAMA_MODEL env var."
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        metavar="URL",
        help="Ollama base URL. Overrides OLLAMA_BASE_URL env var. (default: http://localhost:11434)",
    )

    # Everything after '--' is forwarded verbatim to `langgraph dev`
    return parser.parse_known_args()


def main() -> None:
    args, extra = _parse_args()
    env = os.environ.copy()

    # ── Resolve backend ───────────────────────────────────────────────────
    backend = args.backend or env.get("LLM_BACKEND", "openai")
    env["LLM_BACKEND"] = backend

    # ── Resolve model ─────────────────────────────────────────────────────
    if args.model:
        if backend == "ollama":
            env["OLLAMA_MODEL"] = args.model
        else:
            env["OPENAI_MODEL"] = args.model
    else:
        # Apply defaults only when the env var is not already set
        if backend == "ollama" and "OLLAMA_MODEL" not in env:
            env["OLLAMA_MODEL"] = "gemma4:e2b"
        elif backend == "openai" and "OPENAI_MODEL" not in env:
            env["OPENAI_MODEL"] = "gpt-4o-mini"

    # ── Resolve Ollama URL ────────────────────────────────────────────────
    if args.ollama_url:
        env["OLLAMA_BASE_URL"] = args.ollama_url
    elif backend == "ollama" and "OLLAMA_BASE_URL" not in env:
        env["OLLAMA_BASE_URL"] = "http://localhost:11434"

    # ── Print active config ───────────────────────────────────────────────
    print(f"[cli] backend  : {env['LLM_BACKEND']}")
    if backend == "ollama":
        print(f"[cli] model    : {env.get('OLLAMA_MODEL')}")
        print(f"[cli] ollama   : {env.get('OLLAMA_BASE_URL')}")
    else:
        print(f"[cli] model    : {env.get('OPENAI_MODEL')}")
    print()

    # ── Hand off to langgraph dev ─────────────────────────────────────────
    cmd = [sys.executable, "-m", "langgraph", "dev"] + extra
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        pass  # clean Ctrl-C exit
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
