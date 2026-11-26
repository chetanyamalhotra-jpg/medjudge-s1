"""Tests for Pydantic v2 contracts and versioned SHA-256 YAML rubrics."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


def test_sufficiency_manifest_missing_context_forces_false() -> None:
    manifest = SufficiencyManifest(
        required_context=["allergies"],
        provided_context=[],
        missing_context=["allergies"],
        sufficient=True,  # must be overridden to False
    )
    assert manifest.sufficient is False


def test_tier1_cannot_pass_when_insufficient() -> None:
    with pytest.raises(ValidationError):
        GradeDecision(
            criterion_id="s1_allergy",
            criterion_version="1.0.0",
            criterion_sha256="abc",
            verdict=Verdict.PASS,
            probabilities={"PASS": 0.99, "FAIL": 0.01, "UNKNOWN": 0.0},
            calibrated_confidence=0.99,
            tier_used=EvalTier.TIER1_LOCAL_ONNX,
            sufficiency=SufficiencyManifest(
                required_context=["allergies"],
                missing_context=["allergies"],
                sufficient=False,
            ),
        )


def test_default_rubrics_load_with_valid_sha256_and_semver() -> None:
    registry = load_default_rubrics()
    assert len(registry) >= 10
    allergy = registry.get("s1_allergy_medication_safety")
    assert allergy.risk_class == RiskClass.S1_CRITICAL_SAFETY
    assert allergy.force_tier2 is True
    assert allergy.task_type == "clinical_encounter"
    assert len(allergy.sha256) == 64
    assert allergy.sha256 == allergy.compute_sha256()

    dosage = registry.get("s1_dosage_safety")
    assert dosage.risk_class == RiskClass.S1_CRITICAL_SAFETY
    assert dosage.force_tier2 is True
    assert dosage.required_context == ["active_medications", "renal_hepatic_labs"]

    medhallu = registry.get("s2_medhallu_grounding")
    assert medhallu.task_type == "paper_qa"
    assert medhallu.force_tier2 is False

    medec = registry.get("s2_medec_note_error_flag")
    assert medec.task_type == "clinical_note"
    assert medec.force_tier2 is True

    for rubric in registry.list_all():
        if rubric.id.startswith("s3_"):
            assert rubric.task_type == "deterministic"

    with pytest.raises(ValidationError):
        RubricCriterion(
            id="bad",
            version="v1",
            title="Bad",
            prompt="Bad?",
        )

    art = ClinicalArtifact(artifact_id="a1", candidate_response="Take ibuprofen.")
    assert len(art.content_sha256()) == 64
