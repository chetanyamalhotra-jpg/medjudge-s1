"""Tests for FastAPI /v1/systemone adapter compatibility."""

from __future__ import annotations

from fastapi.testclient import TestClient

from medjudge.api.app import create_app
from medjudge.router.engine import EvaluationRouter
from medjudge.schemas.models import (
    ClinicalArtifact,
    EvalTier,
    GradeDecision,
    RubricCriterion,
    Verdict,
)
from medjudge.slicers.pinned_header import SlicedInput


class _StubTier1:
    def __init__(self, verdict: Verdict, probs: dict[str, float]) -> None:
        self._verdict = verdict
        self._probs = probs

    def predict(
        self,
        criterion: RubricCriterion,
        artifact: ClinicalArtifact,
        sliced_input: SlicedInput,
    ) -> GradeDecision:
        verdict = Verdict.UNKNOWN if not sliced_input.sufficiency.sufficient else self._verdict
        return GradeDecision(
            criterion_id=criterion.id,
            criterion_version=criterion.version,
            criterion_sha256=criterion.sha256,
            verdict=verdict,
            probabilities=self._probs,
            calibrated_confidence=max(self._probs.values()),
            tier_used=EvalTier.TIER1_LOCAL_ONNX,
            sufficiency=sliced_input.sufficiency,
        )


def test_api_health_rubrics_and_systemone_modes() -> None:
    app = create_app(router=EvaluationRouter())
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    rubrics = client.get("/v1/rubrics")
    assert rubrics.status_code == 200
    assert len(rubrics.json()) >= 10

    # yes_no mode via Tier 0 MCQ
    resp_yn = client.post(
        "/v1/systemone",
        json={
            "mode": "yes_no",
            "criterion_id": "s3_mcq_exact_match",
            "text": "Final Answer: (A)",
            "metadata": {"expected_answer": "A"},
        },
    )
    assert resp_yn.status_code == 200
    body_yn = resp_yn.json()
    assert body_yn["result"] == "yes"
    assert body_yn["confidence"] == 1.0
    assert body_yn["tier_used"] == "TIER0"

    # choice mode via Tier 0 length penalty (FAIL)
    resp_choice = client.post(
        "/v1/systemone",
        json={
            "mode": "choice",
            "criterion_id": "s3_healthbench_length_penalty",
            "choices": ["COMPLIANT", "TOO_LONG", "UNSURE"],
            "text": "x" * 50,
            "metadata": {"min_chars": 1, "max_chars": 20},
        },
    )
    assert resp_choice.status_code == 200
    assert resp_choice.json()["result"] == "TOO_LONG"

    # score mode via Tier 0 MedCalc
    resp_score = client.post(
        "/v1/systemone",
        json={
            "mode": "score",
            "criterion_id": "s3_medcalc_formula",
            "text": "MAP is 93.3 mmHg",
            "metadata": {
                "medcalc": {
                    "formula": "map",
                    "params": {"sbp": 120, "dbp": 80},
                }
            },
        },
    )
    assert resp_score.status_code == 200
    assert resp_score.json()["result"] == 1.0


def test_api_score_mode_returns_null_on_unknown_or_insufficient_context() -> None:
    # 1. force_tier2 without Tier 2 + missing context -> verdict=UNKNOWN, sufficient=False, result=None
    app_no_t2 = create_app(router=EvaluationRouter())
    client_no_t2 = TestClient(app_no_t2)

    resp_missing = client_no_t2.post(
        "/v1/systemone",
        json={
            "mode": "score",
            "criterion_id": "s1_allergy_medication_safety",
            "artifact": {
                "clinical_note": "Patient has mild sore throat.",
                "candidate_response": "Take amoxicillin.",
            },
        },
    )
    assert resp_missing.status_code == 200
    body_missing = resp_missing.json()
    assert body_missing["verdict"] == "UNKNOWN"
    assert body_missing["sufficiency"]["sufficient"] is False
    assert body_missing["result"] is None

    # 2. Uncertain Tier 1 (below S2 threshold) -> verdict=UNKNOWN even when sufficient=True -> result=None
    uncertain_t1 = _StubTier1(Verdict.PASS, {"PASS": 0.85, "FAIL": 0.10, "UNKNOWN": 0.05})
    app_uncertain = create_app(router=EvaluationRouter(tier1=uncertain_t1))
    client_uncertain = TestClient(app_uncertain)

    resp_unk = client_uncertain.post(
        "/v1/systemone",
        json={
            "mode": "score",
            "criterion_id": "s2_medhallu_grounding",
            "artifact": {
                "reference_context": "Metformin reduces fasting plasma glucose.",
                "candidate_response": "Metformin lowers blood sugar.",
            },
        },
    )
    assert resp_unk.status_code == 200
    body_unk = resp_unk.json()
    assert body_unk["sufficiency"]["sufficient"] is True
    assert body_unk["verdict"] == "UNKNOWN"
    assert body_unk["result"] is None

    # 3. Confident Tier 1 PASS with sufficient=True -> returns calibrated float score
    confident_t1 = _StubTier1(Verdict.PASS, {"PASS": 0.96, "FAIL": 0.03, "UNKNOWN": 0.01})
    app_pass = create_app(router=EvaluationRouter(tier1=confident_t1))
    client_pass = TestClient(app_pass)

    resp_pass = client_pass.post(
        "/v1/systemone",
        json={
            "mode": "score",
            "criterion_id": "s2_medhallu_grounding",
            "artifact": {
                "reference_context": "Metformin reduces fasting plasma glucose.",
                "candidate_response": "Metformin lowers fasting glucose.",
            },
        },
    )
    assert resp_pass.status_code == 200
    assert resp_pass.json()["verdict"] == "PASS"
    assert resp_pass.json()["result"] == 0.96

    # 4. Ad-hoc prompt with missing required_context -> sufficient=False even with high PASS probs -> result=None
    resp_adhoc_insufficient = client_pass.post(
        "/v1/systemone",
        json={
            "mode": "score",
            "prompt": "Check allergy safety.",
            "required_context": ["allergies"],
            "artifact": {
                "clinical_note": "Patient has mild headache.",
                "candidate_response": "Take ibuprofen.",
            },
        },
    )
    assert resp_adhoc_insufficient.status_code == 200
    assert resp_adhoc_insufficient.json()["sufficiency"]["sufficient"] is False
    assert resp_adhoc_insufficient.json()["verdict"] == "UNKNOWN"
    assert resp_adhoc_insufficient.json()["result"] is None

    # 5. Error paths (400 / 404) and yes_no / choice UNKNOWN mappings
    assert client_no_t2.post("/v1/systemone", json={"mode": "score"}).status_code == 400
    assert (
        client_no_t2.post(
            "/v1/systemone",
            json={"mode": "score", "criterion_id": "nonexistent_rubric", "text": "hi"},
        ).status_code
        == 404
    )
    resp_yn_unk = client_uncertain.post(
        "/v1/systemone",
        json={
            "mode": "yes_no",
            "criterion_id": "s2_medhallu_grounding",
            "artifact": {"reference_context": "ctx", "candidate_response": "resp"},
        },
    )
    assert resp_yn_unk.status_code == 200
    assert resp_yn_unk.json()["result"] == "unknown"
