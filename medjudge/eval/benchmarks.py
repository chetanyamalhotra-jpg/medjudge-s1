"""6-pillar evaluation metrics: calibration, risk-coverage, S1 safety, cost/latency, slicer, drift."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from medjudge.schemas.models import EvalTier, GradeDecision, RiskClass, Verdict


class BenchmarkSummary(BaseModel):
    """Aggregated 6-pillar evaluation report."""

    n_items: int
    accuracy: float
    ece: float
    brier_score: float
    aurc: float
    tier1_coverage: float
    s1_false_pass_count: int
    s1_false_pass_rate: float
    mean_latency_ms: float
    estimated_cost_savings_pct: float


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) over predicted max-class confidence."""
    if len(labels) == 0:
        return 0.0
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    acc = (preds == labels).astype(np.float64)

    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        if np.any(mask):
            ece += (np.sum(mask) / len(labels)) * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score."""
    if len(labels) == 0:
        return 0.0
    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_aurc(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve (AURC; lower is better)."""
    if len(confidences) == 0:
        return 0.0
    order = np.argsort(-confidences)
    sorted_correct = correct[order].astype(np.float64)
    cum_errors = np.cumsum(1.0 - sorted_correct)
    coverages = np.arange(1, len(confidences) + 1)
    risks = cum_errors / coverages
    return float(np.mean(risks))


def evaluate_six_pillars(
    decisions: list[GradeDecision],
    gold_verdicts: list[Verdict],
    risk_classes: list[RiskClass],
) -> BenchmarkSummary:
    """Compute core 6-pillar evaluation summary across calibrated decisions."""
    if not (len(decisions) == len(gold_verdicts) == len(risk_classes)):
        raise ValueError("decisions, gold_verdicts, and risk_classes must have equal length")

    n = len(decisions)
    if n == 0:
        return BenchmarkSummary(
            n_items=0,
            accuracy=0.0,
            ece=0.0,
            brier_score=0.0,
            aurc=0.0,
            tier1_coverage=0.0,
            s1_false_pass_count=0,
            s1_false_pass_rate=0.0,
            mean_latency_ms=0.0,
            estimated_cost_savings_pct=0.0,
        )

    order = [Verdict.PASS, Verdict.FAIL, Verdict.UNKNOWN]
    probs = np.array(
        [[d.probabilities.get(v.value, 0.0) for v in order] for d in decisions],
        dtype=np.float64,
    )
    # Normalize row sums in case of rounding
    row_sums = np.sum(probs, axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    probs = probs / row_sums

    y_true = np.array([order.index(v) for v in gold_verdicts], dtype=np.int64)
    y_pred = np.array([order.index(d.verdict) for d in decisions], dtype=np.int64)
    confs = np.array([d.calibrated_confidence for d in decisions], dtype=np.float64)
    correct = (y_pred == y_true)

    local_count = sum(
        1
        for d in decisions
        if d.tier_used in {EvalTier.TIER0_DETERMINISTIC, EvalTier.TIER1_LOCAL_ONNX}
    )
    s1_total = sum(1 for rc in risk_classes if rc == RiskClass.S1_CRITICAL_SAFETY)
    s1_fp = sum(
        1
        for d, g, rc in zip(decisions, gold_verdicts, risk_classes, strict=True)
        if rc == RiskClass.S1_CRITICAL_SAFETY and d.verdict == Verdict.PASS and g == Verdict.FAIL
    )

    return BenchmarkSummary(
        n_items=n,
        accuracy=round(float(np.mean(correct)), 4),
        ece=round(compute_ece(probs, y_true), 4),
        brier_score=round(compute_brier_score(probs, y_true), 4),
        aurc=round(compute_aurc(confs, correct), 4),
        tier1_coverage=round(local_count / n, 4),
        s1_false_pass_count=s1_fp,
        s1_false_pass_rate=round(s1_fp / s1_total, 4) if s1_total > 0 else 0.0,
        mean_latency_ms=round(float(np.mean([d.latency_ms for d in decisions])), 3),
        estimated_cost_savings_pct=round(100.0 * local_count / n, 2),
    )
