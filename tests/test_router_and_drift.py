"""Tests for 4-tier routing, asymmetric thresholds, audit logging, and drift circuit breaker."""

from __future__ import annotations

from pathlib import Path

import pytest

from medjudge.audit.logger import JSONLAuditLogger
from medjudge.drift.monitor import DriftMonitor
from medjudge.router.engine import EvaluationRouter
from medjudge.rubrics.loader import load_default_rubrics
from medjudge.schemas.models import (
    ClinicalArtifact,
    EvalTier,
    GradeDecision,
    RiskClass,
    RubricCriterion,
    SufficiencyManifest,
    Verdict,
)
from medjudge.slicers.pinned_header import SlicedInput


class _FakeTier1:
    def __init__(self, verdict: Verdict, probs: dict[str, float]) -> None:
        self._verdict = verdict
        self._probs = probs
        self.calls = 0

    def predict(
        self,
        criterion: RubricCriterion,
        artifact: ClinicalArtifact,
        sliced_input: SlicedInput,
    ) -> GradeDecision:
        self.calls += 1
        v = Verdict.UNKNOWN if not sliced_input.sufficiency.sufficient else self._verdict
        return GradeDecision(
            criterion_id=criterion.id,
            criterion_version=criterion.version,
            criterion_sha256=criterion.sha256,
            verdict=v,
            probabilities=self._probs,
            calibrated_confidence=max(self._probs.values()),
            tier_used=EvalTier.TIER1_LOCAL_ONNX,
            sufficiency=sliced_input.sufficiency,
        )


class _FakeTier2:
    def __init__(self, verdict: Verdict = Verdict.FAIL) -> None:
        self._verdict = verdict
        self.calls = 0

    async def evaluate(
        self,
        criterion: RubricCriterion,
        artifact: ClinicalArtifact,
        sufficiency: SufficiencyManifest | None = None,
    ) -> GradeDecision:
        self.calls += 1
        return GradeDecision(
            criterion_id=criterion.id,
            criterion_version=criterion.version,
            criterion_sha256=criterion.sha256,
            verdict=self._verdict,
            probabilities={"PASS": 0.05, "FAIL": 0.95, "UNKNOWN": 0.0}
            if self._verdict == Verdict.FAIL
            else {"PASS": 0.95, "FAIL": 0.05, "UNKNOWN": 0.0},
            calibrated_confidence=0.95,
            tier_used=EvalTier.TIER2_FRONTIER_LLM,
            sufficiency=sufficiency or SufficiencyManifest(sufficient=True),
        )


@pytest.mark.asyncio
async def test_router_asymmetric_thresholds_and_missing_context_escalation(tmp_path: Path) -> None:
    # Use a test S1 criterion with force_tier2=False to verify Tier 1 asymmetric S1 gating.
    s1_crit = RubricCriterion(
        id="s1_test_allergy_tier1_gated",
        version="1.0.0",
        title="Test S1 Allergy Safety",
        prompt="Does response avoid documented allergies?",
        risk_class=RiskClass.S1_CRITICAL_SAFETY,
        preferred_tier=EvalTier.TIER1_LOCAL_ONNX,
        force_tier2=False,
        required_context=["allergies", "active_medications"],
    )
    audit_path = tmp_path / "audit.jsonl"
    logger = JSONLAuditLogger(audit_path)

    # 1. S1 p(PASS)=0.90 is below 0.97 -> MUST escalate to Tier 2
    t1_uncertain = _FakeTier1(Verdict.PASS, {"PASS": 0.90, "FAIL": 0.08, "UNKNOWN": 0.02})
    t2 = _FakeTier2(Verdict.FAIL)
    router = EvaluationRouter(tier1=t1_uncertain, tier2=t2, audit_logger=logger)

    full_art = ClinicalArtifact(
        artifact_id="art-1",
        clinical_note="Allergy to penicillin. Currently taking metformin.",
        candidate_response="Take amoxicillin.",
    )
    dec1 = await router.evaluate(s1_crit, full_art)
    assert dec1.tier_used == EvalTier.TIER2_FRONTIER_LLM
    assert dec1.escalated_from == EvalTier.TIER1_LOCAL_ONNX
    assert dec1.verdict == Verdict.FAIL

    # 2. S1 p(PASS)=0.99 with sufficient context -> auto-accept Tier 1
    t1_high = _FakeTier1(Verdict.PASS, {"PASS": 0.99, "FAIL": 0.01, "UNKNOWN": 0.0})
    router2 = EvaluationRouter(tier1=t1_high, tier2=t2, audit_logger=logger)
    dec2 = await router2.evaluate(s1_crit, full_art)
    assert dec2.tier_used == EvalTier.TIER1_LOCAL_ONNX
    assert dec2.verdict == Verdict.PASS

    # 3. Missing context -> forced UNKNOWN and escalated to Tier 2 even if probs say 0.99 PASS
    missing_art = ClinicalArtifact(
        artifact_id="art-2",
        clinical_note="Patient has mild sore throat.",
        candidate_response="Take amoxicillin.",
    )
    dec3 = await router2.evaluate(s1_crit, missing_art)
    assert dec3.tier_used == EvalTier.TIER2_FRONTIER_LLM
    assert dec3.sufficiency.sufficient is False

    records = logger.read_all()
    assert len(records) == 3


@pytest.mark.asyncio
async def test_drift_monitor_trips_circuit_breaker_and_bypasses_tier1() -> None:
    monitor = DriftMonitor(
        sample_rate=1.0,  # 100% shadow check for deterministic test
        max_s1_false_pass_rate=0.02,
        min_s1_samples_to_trip=5,
    )
    for _ in range(5):
        monitor.record_shadow_outcome(RiskClass.S1_CRITICAL_SAFETY, Verdict.PASS, Verdict.FAIL)
    assert monitor.circuit_breaker_open is True

    s1_crit = RubricCriterion(
        id="s1_test_cb_tier1",
        version="1.0.0",
        title="Test S1 Circuit Breaker",
        prompt="Check allergy safety.",
        risk_class=RiskClass.S1_CRITICAL_SAFETY,
        preferred_tier=EvalTier.TIER1_LOCAL_ONNX,
        force_tier2=False,
        required_context=["allergies", "active_medications"],
    )
    t1 = _FakeTier1(Verdict.PASS, {"PASS": 0.99, "FAIL": 0.01, "UNKNOWN": 0.0})
    t2 = _FakeTier2(Verdict.FAIL)
    router = EvaluationRouter(tier1=t1, tier2=t2, drift_monitor=monitor)

    art = ClinicalArtifact(
        clinical_note="Allergy to penicillin. Currently taking metformin.",
        candidate_response="Take amoxicillin.",
    )
    dec = await router.evaluate(s1_crit, art)
    assert t1.calls == 0  # Tier 1 bypassed while S1 circuit breaker is open!
    assert dec.tier_used == EvalTier.TIER2_FRONTIER_LLM
    assert dec.sufficiency.sufficient is True
    assert "circuit breaker" in (dec.escalation_reason or "").lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rubric_id",
    [
        "s1_allergy_medication_safety",
        "s1_emergency_red_flag_referral",
        "s1_dosage_safety",
        "s2_medec_note_error_flag",
    ],
)
async def test_default_s1_and_medec_rubrics_force_tier2_bypass_tier1(rubric_id: str) -> None:
    criterion = load_default_rubrics().get(rubric_id)
    assert criterion.force_tier2 is True

    t1 = _FakeTier1(Verdict.PASS, {"PASS": 0.99, "FAIL": 0.01, "UNKNOWN": 0.0})
    t2 = _FakeTier2(Verdict.FAIL)
    art = ClinicalArtifact(
        clinical_note="Patient has penicillin allergy, chief complaint chest pain, taking metformin, creatinine 1.1 mg/dL.",
        candidate_response="Seek emergency care.",
    )

    router_with_t2 = EvaluationRouter(tier1=t1, tier2=t2)
    dec_t2 = await router_with_t2.evaluate(criterion, art)
    assert t1.calls == 0
    assert t2.calls == 1
    assert dec_t2.tier_used == EvalTier.TIER2_FRONTIER_LLM
    assert dec_t2.escalated_from is None
    assert dec_t2.escalation_reason == "force_tier2 enabled"

    router_no_t2 = EvaluationRouter(tier1=t1, tier2=None)
    dec_no_t2 = await router_no_t2.evaluate(criterion, art)
    assert t1.calls == 0
    assert dec_no_t2.verdict == Verdict.UNKNOWN
    assert dec_no_t2.tier_used == EvalTier.TIER2_FRONTIER_LLM
    assert dec_no_t2.calibrated_confidence == 0.0
    assert dec_no_t2.probabilities == {"PASS": 0.0, "FAIL": 0.0, "UNKNOWN": 1.0}


@pytest.mark.asyncio
async def test_router_audit_regressions_insufficient_tier2_and_tier1_fallback(tmp_path: Path) -> None:
    # 1. Missing required context forces sufficient=False on preferred_tier=TIER2 and blocks PASS
    t2_always_pass = _FakeTier2(Verdict.PASS)
    crit_t2 = RubricCriterion(
        id="s2_holistic_t2",
        version="1.0.0",
        title="Holistic T2",
        prompt="Is allergy safe?",
        preferred_tier=EvalTier.TIER2_FRONTIER_LLM,
        required_context=["allergies"],
    )
    missing_art = ClinicalArtifact(
        artifact_id="missing-t2",
        clinical_note="Patient presents with cough.",
        candidate_response="Take amoxicillin.",
    )
    router_t2 = EvaluationRouter(tier2=t2_always_pass)
    dec_t2 = await router_t2.evaluate(crit_t2, missing_art)
    assert dec_t2.sufficiency.sufficient is False
    assert dec_t2.verdict == Verdict.UNKNOWN
    assert dec_t2.calibrated_confidence == 0.0

    # 2. Tier 1 rejected with no Tier 2 updates calibrated_confidence to probs["UNKNOWN"]
    crit_s2 = load_default_rubrics().get("s2_medhallu_grounding")
    t1_uncertain = _FakeTier1(Verdict.PASS, {"PASS": 0.82, "FAIL": 0.15, "UNKNOWN": 0.03})
    router_no_t2 = EvaluationRouter(tier1=t1_uncertain, tier2=None)
    grounded_art = ClinicalArtifact(
        artifact_id="uncertain-s2",
        reference_context="Blood pressure was 120/80 mmHg.",
        candidate_response="Blood pressure was normal.",
    )
    dec_fb = await router_no_t2.evaluate(crit_s2, grounded_art)
    assert dec_fb.verdict == Verdict.UNKNOWN
    assert dec_fb.calibrated_confidence == pytest.approx(0.03)

    # 3. JSONLAuditLogger is append-only JSONL and handles empty/missing file cleanly
    audit_file = tmp_path / "nested" / "audit.jsonl"
    logger = JSONLAuditLogger(audit_file)
    assert logger.read_all() == []
    rec = logger.log(grounded_art, dec_fb)
    assert rec.artifact_sha256 == grounded_art.content_sha256()
    assert rec.criterion_sha256 == crit_s2.sha256
    assert len(logger.read_all()) == 1
