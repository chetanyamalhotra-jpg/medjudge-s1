"""Deterministic closed MCQ answer normalization and exact match grader."""

from __future__ import annotations

import re
import time

from medjudge.schemas.models import (
    ClinicalArtifact,
    EvidenceSpan,
    GradeDecision,
    RubricCriterion,
    Verdict,
)
from medjudge.tier0.base import make_tier0_decision

_OPTION_PATTERNS = [
    re.compile(r"(?:final\s+answer|answer|choice|option)\s*(?:is|:)?\s*\(?([A-E])\)?\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-E])\)?(?:[.):\s]|$)", re.IGNORECASE),
]


def normalize_mcq_choice(text: str) -> str | None:
    """Normalize common MCQ phrasing into a canonical uppercase option letter A-E."""
    stripped = text.strip()
    if not stripped:
        return None
    for pat in _OPTION_PATTERNS:
        m = pat.search(stripped)
        if m:
            return m.group(1).upper()
    return None


def verify_mcq_answer(criterion: RubricCriterion, artifact: ClinicalArtifact) -> GradeDecision:
    """Compare normalized candidate MCQ choice against metadata['expected_answer']."""
    t0 = time.perf_counter()
    expected_raw = artifact.metadata.get("expected_answer")
    if not isinstance(expected_raw, str) or not expected_raw.strip():
        return make_tier0_decision(
            criterion,
            Verdict.UNKNOWN,
            "Missing metadata['expected_answer'] for deterministic MCQ grading.",
            t0,
            missing_context=["expected_answer"],
        )

    expected = normalize_mcq_choice(expected_raw) or expected_raw.strip().upper()
    predicted = normalize_mcq_choice(artifact.candidate_response)
    if predicted is None:
        return make_tier0_decision(
            criterion,
            Verdict.FAIL,
            f"Could not extract closed MCQ option (A-E) from candidate; expected '{expected}'.",
            t0,
        )

    verdict = Verdict.PASS if predicted == expected else Verdict.FAIL
    span = EvidenceSpan(
        text=predicted,
        start_char=0,
        end_char=min(len(artifact.candidate_response), 1),
    )
    return make_tier0_decision(
        criterion,
        verdict,
        f"MCQ candidate='{predicted}' vs expected='{expected}'.",
        t0,
        evidence_spans=[span],
    )
