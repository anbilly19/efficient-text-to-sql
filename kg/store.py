"""DuckDB-backed adjacency store for the knowledge graph."""
from __future__ import annotations

import os
from typing import Any

import duckdb

from kg.models import Edge, EdgeType, Node, NodeType

_DEFAULT_DB = os.environ.get("KG_DB_PATH", "kg/kg.duckdb")


def _conn(db_path: str = _DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    return duckdb.connect(db_path)


def init_store(db_path: str = _DEFAULT_DB) -> None:
    """Create tables if they don't exist."""
    con = _conn(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id          VARCHAR PRIMARY KEY,
            node_type   VARCHAR NOT NULL,
            label       VARCHAR NOT NULL,
            props       JSON
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            src_id      VARCHAR NOT NULL,
            dst_id      VARCHAR NOT NULL,
            edge_type   VARCHAR NOT NULL,
            source      VARCHAR DEFAULT 'ingest',
            confidence  DOUBLE DEFAULT 1.0,
            props       JSON,
            PRIMARY KEY (src_id, dst_id, edge_type)
        )
    """)
    con.close()


def upsert_node(node: Node, db_path: str = _DEFAULT_DB) -> None:
    import json
    con = _conn(db_path)
    con.execute("""
        INSERT INTO nodes (id, node_type, label, props)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            node_type = excluded.node_type,
            label     = excluded.label,
            props     = excluded.props
    """, [node.id, node.node_type.value, node.label, json.dumps(node.props)])
    con.close()


def upsert_edge(edge: Edge, db_path: str = _DEFAULT_DB) -> None:
    import json
    con = _conn(db_path)
    con.execute("""
        INSERT INTO edges (src_id, dst_id, edge_type, source, confidence, props)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (src_id, dst_id, edge_type) DO UPDATE SET
            source     = excluded.source,
            confidence = excluded.confidence,
            props      = excluded.props
    """, [edge.src_id, edge.dst_id, edge.edge_type.value,
           edge.source, edge.confidence, json.dumps(edge.props)])
    con.close()


def get_neighbors(
    node_id: str,
    edge_type: EdgeType | None = None,
    direction: str = "out",          # "out" | "in" | "both"
    min_confidence: float = 0.0,
    db_path: str = _DEFAULT_DB,
) -> list[dict[str, Any]]:
    """Return neighbour node dicts reachable via edges of the given type."""
    con = _conn(db_path)
    type_filter = "AND e.edge_type = ?" if edge_type else ""
    params_base = [node_id]
    if edge_type:
        params_base.append(edge_type.value)
    params_base.append(min_confidence)

    if direction == "out":
        where = f"e.src_id = ? {type_filter} AND e.confidence >= ?"
        join_col = "e.dst_id"
    elif direction == "in":
        where = f"e.dst_id = ? {type_filter} AND e.confidence >= ?"
        join_col = "e.src_id"
    else:
        # both — union
        rows_out = get_neighbors(node_id, edge_type, "out", min_confidence, db_path)
        rows_in  = get_neighbors(node_id, edge_type, "in",  min_confidence, db_path)
        return rows_out + rows_in

    rows = con.execute(f"""
        SELECT n.id, n.node_type, n.label, n.props, e.edge_type, e.confidence
        FROM edges e
        JOIN nodes n ON n.id = {join_col}
        WHERE {where}
    """, params_base).fetchall()
    con.close()
    cols = ["id", "node_type", "label", "props", "edge_type", "confidence"]
    return [dict(zip(cols, r)) for r in rows]


def all_nodes(node_type: NodeType | None = None, db_path: str = _DEFAULT_DB) -> list[dict]:
    con = _conn(db_path)
    if node_type:
        rows = con.execute(
            "SELECT id, node_type, label, props FROM nodes WHERE node_type = ?",
            [node_type.value]
        ).fetchall()
    else:
        rows = con.execute("SELECT id, node_type, label, props FROM nodes").fetchall()
    con.close()
    cols = ["id", "node_type", "label", "props"]
    return [dict(zip(cols, r)) for r in rows]
