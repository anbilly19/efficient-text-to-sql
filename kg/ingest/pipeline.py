"""End-to-end Phase 1 ingestion pipeline: Excel file → populated KG store.

Usage (standalone):
    python -m kg.ingest.pipeline --excel path/to/file.xlsx [--sheet Sheet1] [--db kg/kg.duckdb]

Usage (programmatic):
    from kg.ingest.pipeline import ingest_excel
    summary = ingest_excel("data/cat_mat.xlsx")
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from kg.ingest.classifier import classify_column
from kg.ingest.excel_reader import read_excel_schema
from kg.ingest.period_parser import extract_periods_from_columns
from kg.ingest.suffix_detector import detect_suffix_pairs
from kg.models import Edge, EdgeType, Node, NodeType
from kg.store import init_store, upsert_edge, upsert_node


def _file_node_id(file_label: str) -> str:
    return f"file::{file_label}"


def _col_node_id(file_label: str, col_name: str, node_type: NodeType) -> str:
    prefix = "metric" if node_type == NodeType.METRIC else "dim"
    return f"{prefix}::{file_label}::{col_name}"


def _period_node_id(label: str) -> str:
    return f"period::{label}"


def ingest_excel(
    filepath: str,
    sheet_name: str | None = None,
    db_path: str | None = None,
) -> dict:
    """
    Ingest an Excel file into the KG store.

    Returns a summary dict:
        { sheet_name: { "nodes": int, "edges": int } }
    """
    db_path = db_path or os.environ.get("KG_DB_PATH", "kg/kg.duckdb")
    init_store(db_path)

    schema = read_excel_schema(filepath, sheet_name)
    file_label = Path(filepath).stem  # e.g. "cat_mat" from "cat_mat.xlsx"
    summary: dict = {}

    for sheet, sheet_meta in schema.items():
        node_count = 0
        edge_count = 0
        row_count = sheet_meta["row_count"]
        columns_meta = sheet_meta["columns"]
        col_names = list(columns_meta.keys())

        # ── File node ─────────────────────────────────────────────
        file_id = _file_node_id(file_label)
        upsert_node(Node(
            id=file_id,
            node_type=NodeType.FILE,
            label=file_label,
            props={
                "filepath": filepath,
                "sheet": sheet,
                "row_count": row_count,
                "col_count": sheet_meta["col_count"],
            },
        ), db_path)
        node_count += 1

        # ── Metric / Dimension nodes + AVAILABLE_IN / SLICES edges ─
        col_node_ids: dict[str, str] = {}
        for col_name, col_meta in columns_meta.items():
            col_meta["_row_count"] = row_count
            ntype = classify_column(col_name, col_meta)
            node_id = _col_node_id(file_label, col_name, ntype)
            col_node_ids[col_name] = node_id

            upsert_node(Node(
                id=node_id,
                node_type=ntype,
                label=col_name,
                props={**col_meta, "file": file_label, "sheet": sheet},
            ), db_path)
            node_count += 1

            edge_type = EdgeType.AVAILABLE_IN if ntype == NodeType.METRIC else EdgeType.SLICES
            upsert_edge(Edge(
                src_id=node_id,
                dst_id=file_id,
                edge_type=edge_type,
            ), db_path)
            edge_count += 1

        # ── CONTAINS edges (Dimension → Entity for low-cardinality cols) ─
        for col_name, col_meta in columns_meta.items():
            if "categories" in col_meta:
                dim_id = col_node_ids.get(col_name)
                if not dim_id:
                    continue
                for val in col_meta["categories"]:
                    entity_id = f"entity::{val}"
                    upsert_node(Node(
                        id=entity_id,
                        node_type=NodeType.ENTITY,
                        label=val,
                    ), db_path)
                    upsert_edge(Edge(
                        src_id=dim_id,
                        dst_id=entity_id,
                        edge_type=EdgeType.CONTAINS,
                    ), db_path)
                    node_count += 1
                    edge_count += 1

        # ── PRIOR_PERIOD_OF / DELTA_OF edges ──────────────────────
        suffix_pairs = detect_suffix_pairs(col_names)
        for derived_col, (base_col, etype_str) in suffix_pairs.items():
            src_id = col_node_ids.get(derived_col)
            dst_id = col_node_ids.get(base_col)
            if src_id and dst_id:
                upsert_edge(Edge(
                    src_id=src_id,
                    dst_id=dst_id,
                    edge_type=EdgeType[etype_str],
                ), db_path)
                edge_count += 1

        # ── Period nodes + COVERS edges ───────────────────────────
        periods = extract_periods_from_columns(col_names)
        for period in periods:
            period_id = _period_node_id(period.label)
            upsert_node(Node(
                id=period_id,
                node_type=NodeType.PERIOD,
                label=period.label,
                props={
                    "window_weeks": period.window_weeks,
                    "end_date": period.end_date,
                    "period_type": period.period_type,
                    "raw": period.raw,
                },
            ), db_path)
            upsert_edge(Edge(
                src_id=period_id,
                dst_id=file_id,
                edge_type=EdgeType.COVERS,
                props={"role": "CY"},
            ), db_path)
            node_count += 1
            edge_count += 1

        summary[sheet] = {"nodes": node_count, "edges": edge_count}
        print(f"[kg] ingested sheet '{sheet}': {node_count} nodes, {edge_count} edges")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest an Excel file into the KG store.")
    parser.add_argument("--excel", required=True, help="Path to .xlsx file")
    parser.add_argument("--sheet", default=None, help="Sheet name (default: all sheets)")
    parser.add_argument("--db", default=None, help="DuckDB path (default: KG_DB_PATH env or kg/kg.duckdb)")
    args = parser.parse_args()
    ingest_excel(args.excel, args.sheet, args.db)


if __name__ == "__main__":
    main()
