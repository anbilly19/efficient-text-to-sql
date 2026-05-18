"""HITL guided CLI for Phase 3: confirm JOINABLE_ON proposals + author BRIDGES edges.

Usage
-----
    python -m hitl.bridge_author [--db kg/kg.duckdb]

Workflow
--------
Step 1 — Review pending JOINABLE_ON proposals
    For each proposed join (from join_detector):
      [a]ccept  → source='declared', confidence=1.0
      [r]eject  → edge deleted
      [s]kip    → no change

Step 2 — Author BRIDGES edges (cross-file concept equivalence)
    The CLI lists all Concept nodes and accepted JOINABLE_ON file pairs.
    For each pair the user can declare:
        Concept X in File A  ≈≈  Concept Y in File B
    This writes a BRIDGES edge:  Concept_X  --BRIDGES-->  Concept_Y
    with source='declared', confidence=1.0, props={file_a, file_b}

Exit criteria for Phase 3
-------------------------
    At least one confirmed JOINABLE_ON edge exists.
    At least one BRIDGES edge exists (MAT Revenue ↔ Panel Ausgaben).
"""
from __future__ import annotations

import argparse
import json
import os

import duckdb

from kg.concepts.validator import concept_ids, concept_by_id
from kg.models import Edge, EdgeType, NodeType
from kg.store import all_nodes, upsert_edge


def _conn(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


# ---------------------------------------------------------------------------
# Step 1: JOINABLE_ON review
# ---------------------------------------------------------------------------

def _fetch_pending_joins(db_path: str) -> list[dict]:
    con = _conn(db_path)
    rows = con.execute("""
        SELECT e.src_id, e.dst_id, e.confidence, e.props,
               fa.label AS file_a_label, fb.label AS file_b_label
        FROM edges e
        JOIN nodes fa ON fa.id = e.src_id
        JOIN nodes fb ON fb.id = e.dst_id
        WHERE e.edge_type = 'JOINABLE_ON'
          AND e.source = 'proposed'
        ORDER BY e.confidence DESC
    """).fetchall()
    con.close()
    return [
        {
            "src_id":       r[0],
            "dst_id":       r[1],
            "confidence":   r[2],
            "props":        json.loads(r[3]) if r[3] else {},
            "file_a_label": r[4],
            "file_b_label": r[5],
        }
        for r in rows
    ]


def _fetch_confirmed_joins(db_path: str) -> list[dict]:
    con = _conn(db_path)
    rows = con.execute("""
        SELECT e.src_id, e.dst_id, e.props,
               fa.label AS file_a_label, fb.label AS file_b_label
        FROM edges e
        JOIN nodes fa ON fa.id = e.src_id
        JOIN nodes fb ON fb.id = e.dst_id
        WHERE e.edge_type = 'JOINABLE_ON'
          AND e.source = 'declared'
    """).fetchall()
    con.close()
    return [
        {
            "src_id":       r[0],
            "dst_id":       r[1],
            "props":        json.loads(r[2]) if r[2] else {},
            "file_a_label": r[3],
            "file_b_label": r[4],
        }
        for r in rows
    ]


def _delete_edge(db_path: str, src_id: str, dst_id: str, edge_type: str) -> None:
    con = _conn(db_path)
    con.execute(
        "DELETE FROM edges WHERE src_id=? AND dst_id=? AND edge_type=?",
        [src_id, dst_id, edge_type],
    )
    con.close()


def review_joins(db_path: str) -> int:
    """Return number of accepted joins."""
    pending = _fetch_pending_joins(db_path)
    if not pending:
        print("[bridge_author] No pending JOINABLE_ON proposals.")
        return len(_fetch_confirmed_joins(db_path))

    print(f"\n[bridge_author] {len(pending)} pending join proposal(s).")
    print("Commands: [a]ccept  [r]eject  [s]kip\n")

    for i, item in enumerate(pending):
        p = item["props"]
        print(f"{'='*60}")
        print(f"  [{i+1}/{len(pending)}]  {item['file_a_label']}  ⟷  {item['file_b_label']}")
        print(f"  Join key : '{p.get('join_col_a')}' ↔ '{p.get('join_col_b')}'")
        print(f"  Match    : {p.get('match_type')}  (conf={item['confidence']:.2f})")
        action = input("  Action [a/r/s]: ").strip().lower()

        if action == "a":
            upsert_edge(Edge(
                src_id=item["src_id"],
                dst_id=item["dst_id"],
                edge_type=EdgeType.JOINABLE_ON,
                source="declared",
                confidence=1.0,
                props=p,
            ), db_path)
            print(f"  ✅ Accepted.")
        elif action == "r":
            _delete_edge(db_path, item["src_id"], item["dst_id"], "JOINABLE_ON")
            print(f"  ❌ Rejected.")
        else:
            print("  ⏭️  Skipped.")

    confirmed = _fetch_confirmed_joins(db_path)
    print(f"\n[bridge_author] {len(confirmed)} confirmed join(s) total.")
    return len(confirmed)


# ---------------------------------------------------------------------------
# Step 2: BRIDGES authoring
# ---------------------------------------------------------------------------

def _fetch_existing_bridges(db_path: str) -> list[dict]:
    con = _conn(db_path)
    rows = con.execute("""
        SELECT e.src_id, e.dst_id, e.props,
               ca.label AS concept_a_label, cb.label AS concept_b_label
        FROM edges e
        JOIN nodes ca ON ca.id = e.src_id
        JOIN nodes cb ON cb.id = e.dst_id
        WHERE e.edge_type = 'BRIDGES'
    """).fetchall()
    con.close()
    return [
        {
            "src_id": r[0], "dst_id": r[1],
            "props":  json.loads(r[2]) if r[2] else {},
            "concept_a": r[3], "concept_b": r[4],
        }
        for r in rows
    ]


def author_bridges(db_path: str) -> int:
    """Interactive BRIDGES authoring. Returns total bridge count."""
    confirmed_joins = _fetch_confirmed_joins(db_path)
    if not confirmed_joins:
        print("[bridge_author] No confirmed joins yet — run join review first.")
        return 0

    cids   = concept_ids()
    files  = all_nodes(NodeType.FILE, db_path)

    print("\n[bridge_author] Author BRIDGES edges (cross-file concept equivalence).")
    print("A BRIDGE declares that Concept X in File A measures the same economic")
    print("quantity as Concept Y in File B (e.g. MAT Revenue ↔ Panel Ausgaben).\n")
    print("Available concepts:")
    for i, cid in enumerate(cids):
        c = concept_by_id(cid)
        print(f"  [{i}] {cid} — {c['label']}")

    print("\nAvailable files:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f['label']}  (id: {f['id']})")

    while True:
        print("\nEnter a BRIDGE (or blank to finish):")
        concept_a_in = input("  Concept A (number or id): ").strip()
        if not concept_a_in:
            break
        concept_b_in = input("  Concept B (number or id): ").strip()
        file_a_in    = input("  File A    (number or id): ").strip()
        file_b_in    = input("  File B    (number or id): ").strip()

        def _resolve_concept(val: str) -> str | None:
            if val.isdigit() and int(val) < len(cids):
                return cids[int(val)]
            return val if val in cids else None

        def _resolve_file(val: str) -> str | None:
            if val.isdigit() and int(val) < len(files):
                return files[int(val)]["id"]
            for f in files:
                if val == f["id"] or val.lower() == f["label"].lower():
                    return f["id"]
            return None

        ca = _resolve_concept(concept_a_in)
        cb = _resolve_concept(concept_b_in)
        fa = _resolve_file(file_a_in)
        fb = _resolve_file(file_b_in)

        if not all([ca, cb, fa, fb]):
            print("  ⚠️  Could not resolve one or more inputs. Try again.")
            continue

        upsert_edge(Edge(
            src_id=f"concept::{ca}",
            dst_id=f"concept::{cb}",
            edge_type=EdgeType.BRIDGES,
            source="declared",
            confidence=1.0,
            props={"file_a": fa, "file_b": fb},
        ), db_path)
        ca_label = concept_by_id(ca)["label"]
        cb_label = concept_by_id(cb)["label"]
        print(f"  ✅  BRIDGE written: {ca_label} ({fa}) ⟷ {cb_label} ({fb})")

    bridges = _fetch_existing_bridges(db_path)
    print(f"\n[bridge_author] {len(bridges)} BRIDGES edge(s) total.")
    return len(bridges)


# ---------------------------------------------------------------------------
# Programmatic API (used by tests)
# ---------------------------------------------------------------------------

def write_bridge(
    db_path: str,
    concept_a_id: str,
    concept_b_id: str,
    file_a_id: str,
    file_b_id: str,
) -> None:
    """Write a single BRIDGES edge non-interactively (for tests and scripts)."""
    upsert_edge(Edge(
        src_id=f"concept::{concept_a_id}",
        dst_id=f"concept::{concept_b_id}",
        edge_type=EdgeType.BRIDGES,
        source="declared",
        confidence=1.0,
        props={"file_a": file_a_id, "file_b": file_b_id},
    ), db_path)


def confirm_join(
    db_path: str,
    src_id: str,
    dst_id: str,
    props: dict,
) -> None:
    """Confirm a JOINABLE_ON edge non-interactively (for tests and scripts)."""
    upsert_edge(Edge(
        src_id=src_id,
        dst_id=dst_id,
        edge_type=EdgeType.JOINABLE_ON,
        source="declared",
        confidence=1.0,
        props=props,
    ), db_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 HITL: join review + bridge authoring.")
    parser.add_argument("--db", default=None, help="DuckDB path (default: KG_DB_PATH env or kg/kg.duckdb)")
    parser.add_argument("--joins-only",   action="store_true", help="Only run the join review step")
    parser.add_argument("--bridges-only", action="store_true", help="Only run the bridge authoring step")
    args = parser.parse_args()
    db_path = args.db or os.environ.get("KG_DB_PATH", "kg/kg.duckdb")

    if not args.bridges_only:
        review_joins(db_path)
    if not args.joins_only:
        author_bridges(db_path)


if __name__ == "__main__":
    main()
