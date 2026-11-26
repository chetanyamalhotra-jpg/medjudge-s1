"""6-pillar MedJudge-S1 evaluation suite and Pareto/Risk-Coverage plotting."""

from __future__ import annotations

from medjudge.eval.benchmarks import (
    BenchmarkSummary,
    compute_aurc,
    compute_brier_score,
    compute_ece,
    evaluate_six_pillars,
)
from medjudge.eval.plots import plot_pareto_frontier, plot_risk_coverage_curve

__all__ = [
    "BenchmarkSummary",
    "compute_aurc",
    "compute_brier_score",
    "compute_ece",
    "evaluate_six_pillars",
    "plot_pareto_frontier",
    "plot_risk_coverage_curve",
]
