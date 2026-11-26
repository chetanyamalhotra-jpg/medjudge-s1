"""Pareto frontier and Risk-Coverage curve plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_risk_coverage_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Plot selective Risk-Coverage curve and save PNG."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    order = np.argsort(-confidences)
    sorted_correct = correct[order].astype(np.float64)
    coverages = np.arange(1, len(confidences) + 1) / len(confidences)
    risks = np.cumsum(1.0 - sorted_correct) / np.arange(1, len(confidences) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coverages, risks, linewidth=2.0, label="MedJudge-S1")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective Risk (Error Rate)")
    ax.set_title("Selective Risk-Coverage Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pareto_frontier(
    latencies_ms: list[float],
    accuracies: list[float],
    labels: list[str],
    output_path: str | Path,
) -> Path:
    """Plot Cost/Latency vs Accuracy Pareto frontier and save PNG."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(latencies_ms, accuracies, s=60)
    for x, y, lbl in zip(latencies_ms, accuracies, labels, strict=True):
        ax.annotate(lbl, (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("Agreement / Accuracy")
    ax.set_title("MedJudge-S1 Latency-Accuracy Pareto Frontier")
    ax.grid(True, alpha=0.3)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
