"""Versioned YAML rubric loader and registry."""

from __future__ import annotations

from medjudge.rubrics.loader import RubricRegistry, load_default_rubrics, load_rubrics_from_yaml

__all__ = ["RubricRegistry", "load_default_rubrics", "load_rubrics_from_yaml"]
