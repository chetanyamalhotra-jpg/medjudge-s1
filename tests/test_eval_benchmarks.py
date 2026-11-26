"""Tests for 6-pillar benchmark evaluation metrics."""

from __future__ import annotations

from medjudge.eval.benchmarks import evaluate_six_pillars
from medjudge.schemas.models import EvalTier, GradeDecision, RiskClass, Verdict


def test_evaluate_six_pillars_computes_ece_aurc_coverage_and_s1_fp() -> None:
    decisions = [
        GradeDecision(
            criterion_id="c1",
            criterion_version="1.0.0",
            criterion_sha256="h1",
            verdict=Verdict.PASS,
            probabilities={"PASS": 0.99, "FAIL": 0.01, "UNKNOWN": 0.0},
            calibrated_confidence=0.99,
            tier_used=EvalTier.TIER1_LOCAL_ONNX,
            latency_ms=8.0,
        ),
        GradeDecision(
            criterion_id="c2",
            criterion_version="1.0.0",
            criterion_sha256="h2",
            verdict=Verdict.PASS,
            probabilities={"PASS": 0.90, "FAIL": 0.10, "UNKNOWN": 0.0},
            calibrated_confidence=0.90,
            tier_used=EvalTier.TIER2_FRONTIER_LLM,
            latency_ms=420.0,
        ),
    ]
    gold = [Verdict.PASS, Verdict.FAIL]
    risks = [RiskClass.S1_CRITICAL_SAFETY, RiskClass.S1_CRITICAL_SAFETY]
    summary = evaluate_six_pillars(decisions, gold, risks)
    assert summary.n_items == 2
    assert summary.accuracy == 0.5
    assert summary.tier1_coverage == 0.5
    assert summary.s1_false_pass_count == 1
    assert summary.s1_false_pass_rate == 0.5
    assert summary.ece >= 0.0
    assert summary.brier_score >= 0.0
    assert summary.aurc >= 0.0
