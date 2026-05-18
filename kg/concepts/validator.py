"""Validate that a concept id exists in the registry."""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


@functools.lru_cache(maxsize=1)
def load_registry() -> dict:
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def concept_ids() -> list[str]:
    return [c["id"] for c in load_registry().get("concepts", [])]


def concept_by_id(concept_id: str) -> dict | None:
    for c in load_registry().get("concepts", []):
        if c["id"] == concept_id:
            return c
    return None


def validate_concept_id(concept_id: str) -> bool:
    return concept_id in concept_ids()


def aliases_for(concept_id: str) -> list[str]:
    c = concept_by_id(concept_id)
    return c.get("aliases", []) if c else []
