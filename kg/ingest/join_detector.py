"""Phase 3: Auto-detect JOINABLE_ON edges between files.

Algorithm
---------
For every pair of File nodes in the KG:
  1. Collect the Dimension node labels (column names) attached to each file
     via SLICES edges.
  2. Find exact-match and normalised-match overlaps.
  3. For each overlapping dimension, emit a JOINABLE_ON edge proposal:
       File_A  --JOINABLE_ON-->  File_B
     with props: { join_col_a, join_col_b, match_type }
     and confidence:
       1.0  exact label match
       0.90 normalised match (case/whitespace)
       0.75 fuzzy alias match (e.g. 'Marke' ↔ 'Brand')

All proposals are written to the store with source='proposed' so the
bridge_author CLI can confirm or reject them.

Known shared-key aliases (domain knowledge, petfood NIQ)
--------------------------------------------------------
These are checked when no exact/normalised match exists.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations

from kg.models import Edge, EdgeType, Node, NodeType
from kg.store import all_nodes, get_neighbors, init_store, upsert_edge

# Domain-level dimension aliases: (norm_a, norm_b) pairs that are
# semantically equivalent join keys in the NIQ petfood domain.
_ALIAS_PAIRS: list[tuple[str, str]] = [
    ("marke", "brand"),
    ("marke", "produkt"),
    ("hersteller", "manufacturer"),
    ("kanal", "channel"),
    ("kanal", "retailer"),
    ("handlers", "retailer"),
    ("geographie", "geography"),
    ("geographie", "region"),
    ("segment", "demographic"),
    ("periode", "period"),
    ("periode", "periods"),
]
_ALIAS_LOOKUP: dict[str, list[str]] = {}
for _a, _b in _ALIAS_PAIRS:
    _ALIAS_LOOKUP.setdefault(_a, []).append(_b)
    _ALIAS_LOOKUP.setdefault(_b, []).append(_a)


def _norm(s: str) -> str:
    """Lowercase, strip whitespace and punctuation for comparison."""
    return s.lower().strip().replace(" ", "").replace("_", "").replace("-", "")


@dataclass
class JoinProposal:
    file_a: str          # File node id
    file_b: str          # File node id
    col_a: str           # original Dimension label in file_a
    col_b: str           # original Dimension label in file_b
    match_type: str      # "exact" | "normalised" | "alias"
    confidence: float


def _dimensions_for_file(file_node_id: str, db_path: str) -> list[dict]:
    """Return all Dimension nodes that SLICES into this file (reverse SLICES)."""
    return get_neighbors(file_node_id, EdgeType.SLICES, direction="in", db_path=db_path)


def detect_join_candidates(
    db_path: str | None = None,
) -> list[JoinProposal]:
    """
    Scan all File node pairs, detect shared dimension columns, write
    JOINABLE_ON edge proposals, and return the proposal list.
    """
    db_path = db_path or os.environ.get("KG_DB_PATH", "kg/kg.duckdb")
    init_store(db_path)

    files = all_nodes(NodeType.FILE, db_path)
    if len(files) < 2:
        print("[join_detector] Fewer than 2 files in KG — nothing to detect.")
        return []

    proposals: list[JoinProposal] = []

    for fa, fb in combinations(files, 2):
        dims_a = _dimensions_for_file(fa["id"], db_path)
        dims_b = _dimensions_for_file(fb["id"], db_path)

        if not dims_a or not dims_b:
            continue

        labels_a = {d["label"]: d for d in dims_a}
        labels_b = {d["label"]: d for d in dims_b}
        norms_a  = {_norm(lbl): lbl for lbl in labels_a}
        norms_b  = {_norm(lbl): lbl for lbl in labels_b}

        seen: set[tuple[str, str]] = set()  # avoid duplicate proposals

        # --- exact match ---
        for lbl in labels_a:
            if lbl in labels_b:
                key = (lbl, lbl)
                if key not in seen:
                    seen.add(key)
                    proposals.append(JoinProposal(
                        file_a=fa["id"], file_b=fb["id"],
                        col_a=lbl, col_b=lbl,
                        match_type="exact", confidence=1.0,
                    ))

        # --- normalised match (case/whitespace) ---
        for na, orig_a in norms_a.items():
            if na in norms_b:
                orig_b = norms_b[na]
                key = (orig_a, orig_b)
                if key not in seen and orig_a != orig_b:  # skip if already exact
                    seen.add(key)
                    proposals.append(JoinProposal(
                        file_a=fa["id"], file_b=fb["id"],
                        col_a=orig_a, col_b=orig_b,
                        match_type="normalised", confidence=0.90,
                    ))

        # --- alias match ---
        for na, orig_a in norms_a.items():
            for alias in _ALIAS_LOOKUP.get(na, []):
                if alias in norms_b:
                    orig_b = norms_b[alias]
                    key = (orig_a, orig_b)
                    if key not in seen:
                        seen.add(key)
                        proposals.append(JoinProposal(
                            file_a=fa["id"], file_b=fb["id"],
                            col_a=orig_a, col_b=orig_b,
                            match_type="alias", confidence=0.75,
                        ))

    # Write proposals to store
    for p in proposals:
        upsert_edge(Edge(
            src_id=p.file_a,
            dst_id=p.file_b,
            edge_type=EdgeType.JOINABLE_ON,
            source="proposed",
            confidence=p.confidence,
            props={
                "join_col_a": p.col_a,
                "join_col_b": p.col_b,
                "match_type": p.match_type,
            },
        ), db_path)

    auto   = sum(1 for p in proposals if p.confidence >= 0.90)
    review = len(proposals) - auto
    print(
        f"[join_detector] {len(proposals)} join candidates across {len(files)} files: "
        f"{auto} high-confidence, {review} need review"
    )
    return proposals
