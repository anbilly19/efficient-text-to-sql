"""HITL review CLI for pending MEASURES edge proposals.

Usage:
    python -m hitl.review_cli [--db kg/kg.duckdb]

For each pending proposal (confidence < 0.85, source='proposed') the CLI shows:
    Column name  |  Proposed concept  |  Confidence
and asks: [a]ccept / [r]eject / [m]ap to different concept / [s]kip

Accepted → source updated to 'ingest', confidence set to 1.0
Rejected → edge deleted
Remapped → new concept written, old edge deleted
Skipped  → no change
"""
from __future__ import annotations

import argparse
import json
import os

import duckdb

from kg.concepts.validator import concept_ids, concept_by_id
from kg.models import Edge, EdgeType
from kg.store import upsert_edge


def _conn(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def _fetch_pending(db_path: str) -> list[dict]:
    con = _conn(db_path)
    rows = con.execute("""
        SELECT e.src_id, e.dst_id, e.confidence, n.label AS col_name,
               c.label AS concept_label, c.props
        FROM edges e
        JOIN nodes n ON n.id = e.src_id
        JOIN nodes c ON c.id = e.dst_id
        WHERE e.edge_type = 'MEASURES'
          AND e.source = 'proposed'
        ORDER BY e.confidence DESC
    """).fetchall()
    con.close()
    return [
        {
            "src_id": r[0],
            "dst_id": r[1],
            "confidence": r[2],
            "col_name": r[3],
            "concept_label": r[4],
            "concept_props": json.loads(r[5]) if r[5] else {},
        }
        for r in rows
    ]


def _delete_edge(db_path: str, src_id: str, dst_id: str, edge_type: str) -> None:
    con = _conn(db_path)
    con.execute(
        "DELETE FROM edges WHERE src_id = ? AND dst_id = ? AND edge_type = ?",
        [src_id, dst_id, edge_type],
    )
    con.close()


def _accept(db_path: str, item: dict) -> None:
    upsert_edge(Edge(
        src_id=item["src_id"],
        dst_id=item["dst_id"],
        edge_type=EdgeType.MEASURES,
        source="ingest",
        confidence=1.0,
    ), db_path)
    print(f"  ✅ Accepted: '{item['col_name']}' → {item['concept_label']}")


def _reject(db_path: str, item: dict) -> None:
    _delete_edge(db_path, item["src_id"], item["dst_id"], "MEASURES")
    print(f"  ❌ Rejected: '{item['col_name']}' will not be mapped to any concept.")


def _remap(db_path: str, item: dict) -> None:
    cids = concept_ids()
    print("  Available concepts:")
    for i, cid in enumerate(cids):
        c = concept_by_id(cid)
        print(f"    [{i}] {cid} — {c['label']}: {c['description'][:60]}...")
    choice = input("  Enter number or concept_id: ").strip()
    # Resolve choice
    new_cid: str | None = None
    if choice.isdigit() and int(choice) < len(cids):
        new_cid = cids[int(choice)]
    elif choice in cids:
        new_cid = choice
    if not new_cid:
        print("  ⚠️  Invalid choice. Skipping.")
        return
    _delete_edge(db_path, item["src_id"], item["dst_id"], "MEASURES")
    new_dst = f"concept::{new_cid}"
    upsert_edge(Edge(
        src_id=item["src_id"],
        dst_id=new_dst,
        edge_type=EdgeType.MEASURES,
        source="ingest",
        confidence=1.0,
    ), db_path)
    c = concept_by_id(new_cid)
    print(f"  ➡️  Remapped: '{item['col_name']}' → {c['label']}")


def run_review(db_path: str) -> None:
    pending = _fetch_pending(db_path)
    if not pending:
        print("[hitl] No pending MEASURES proposals to review. 🎉")
        return

    print(f"\n[hitl] {len(pending)} pending proposal(s) to review.")
    print("Commands: [a]ccept  [r]eject  [m]ap to different concept  [s]kip\n")

    for i, item in enumerate(pending):
        desc = item["concept_props"].get("description", "")[:70]
        print(f"{'='*60}")
        print(f"  [{i+1}/{len(pending)}] Column  : {item['col_name']}")
        print(f"          Concept : {item['concept_label']} (conf={item['confidence']:.2f})")
        if desc:
            print(f"          Def     : {desc}...")
        action = input("  Action [a/r/m/s]: ").strip().lower()

        if action == "a":
            _accept(db_path, item)
        elif action == "r":
            _reject(db_path, item)
        elif action == "m":
            _remap(db_path, item)
        elif action == "s":
            print("  ⏭️  Skipped.")
        else:
            print("  ⚠️  Unknown action, skipping.")

    remaining = len(_fetch_pending(db_path))
    print(f"\n[hitl] Review complete. {remaining} proposal(s) still pending.")


def main() -> None:
    parser = argparse.ArgumentParser(description="HITL review of pending MEASURES proposals.")
    parser.add_argument("--db", default=None, help="DuckDB path (default: KG_DB_PATH or kg/kg.duckdb)")
    args = parser.parse_args()
    db_path = args.db or os.environ.get("KG_DB_PATH", "kg/kg.duckdb")
    run_review(db_path)


if __name__ == "__main__":
    main()
