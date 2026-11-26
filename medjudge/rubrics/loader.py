"""Loader and SHA-256 validator for versioned YAML rubrics."""

from __future__ import annotations

from pathlib import Path

import yaml

from medjudge.schemas.models import RubricCriterion


class RubricRegistry:
    """In-memory index of validated RubricCriterion objects."""

    def __init__(self, criteria: list[RubricCriterion] | None = None) -> None:
        self._by_id: dict[str, RubricCriterion] = {}
        for item in criteria or []:
            self.register(item)

    def register(self, criterion: RubricCriterion) -> None:
        self._by_id[criterion.id] = criterion

    def get(self, criterion_id: str) -> RubricCriterion:
        if criterion_id not in self._by_id:
            raise KeyError(f"Unknown rubric criterion: '{criterion_id}'")
        return self._by_id[criterion_id]

    def list_all(self) -> list[RubricCriterion]:
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)


def load_rubrics_from_yaml(path: str | Path) -> RubricRegistry:
    """Load and SHA-256 validate rubric criteria from a YAML file."""
    yaml_path = Path(path)
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "criteria" not in raw:
        raise ValueError(f"Rubric YAML '{yaml_path}' must contain a top-level 'criteria' list")

    criteria: list[RubricCriterion] = []
    for entry in raw["criteria"]:
        criterion = RubricCriterion.model_validate(entry)
        criteria.append(criterion)
    return RubricRegistry(criteria)


def load_default_rubrics() -> RubricRegistry:
    """Load the bundled MedJudge-S1 default rubric catalog."""
    default_path = Path(__file__).resolve().parent / "default_rubrics.yaml"
    return load_rubrics_from_yaml(default_path)
