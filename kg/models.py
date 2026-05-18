"""Node and Edge dataclasses + enums for the KG store."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    CONCEPT = "Concept"
    METRIC = "Metric"
    DIMENSION = "Dimension"
    ENTITY = "Entity"
    ENTITY_GROUP = "EntityGroup"
    FILE = "File"
    PERIOD = "Period"


class EdgeType(str, Enum):
    MEASURES = "MEASURES"
    AVAILABLE_IN = "AVAILABLE_IN"
    SLICES = "SLICES"
    PRIOR_PERIOD_OF = "PRIOR_PERIOD_OF"
    DELTA_OF = "DELTA_OF"
    DERIVED_FROM = "DERIVED_FROM"
    CONTAINS = "CONTAINS"
    MEMBER_OF = "MEMBER_OF"
    SCOPED_TO = "SCOPED_TO"
    CHILD_OF = "CHILD_OF"
    LEVEL_IN = "LEVEL_IN"
    BRIDGES = "BRIDGES"
    COVERS = "COVERS"
    JOINABLE_ON = "JOINABLE_ON"


@dataclass
class Node:
    id: str                          # e.g. "metric::cat_mat::Umsatz"
    node_type: NodeType
    label: str                       # human-readable name
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src_id: str
    dst_id: str
    edge_type: EdgeType
    source: str = "ingest"           # "ingest" | "declared"
    confidence: float = 1.0
    props: dict[str, Any] = field(default_factory=dict)
