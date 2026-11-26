"""Deterministic HealthBench Professional character-length penalty verifier."""

from __future__ import annotations

import time

from medjudge.schemas.models import ClinicalArtifact, GradeDecision, RubricCriterion, Verdict
from medjudge.tier0.base import make_tier0_decision


def compute_length_penalty(char_count: int, min_chars: int = 0, max_chars: int = 4000) -> float:
    """Compute normalized [0.0, 1.0] penalty for character-length violations."""
    if char_count < min_chars and min_chars > 0:
        return min(1.0, (min_chars - char_count) / min_chars)
    if char_count > max_chars and max_chars > 0:
        return min(1.0, (char_count - max_chars) / max_chars)
    return 0.0


def verify_length_penalty(criterion: RubricCriterion, artifact: ClinicalArtifact) -> GradeDecision:
    """Check candidate response character count against criterion/artifact length bounds."""
    t0 = time.perf_counter()
    min_chars = int(
        artifact.metadata.get("min_chars", criterion.metadata.get("min_chars", 0))
    )
    max_chars = int(
        artifact.metadata.get("max_chars", criterion.metadata.get("max_chars", 4000))
    )
    length = len(artifact.candidate_response.strip())
    penalty = compute_length_penalty(length, min_chars=min_chars, max_chars=max_chars)
    verdict = Verdict.PASS if penalty == 0.0 else Verdict.FAIL
    rationale = (
        f"Character length={length} (bounds [{min_chars}, {max_chars}], penalty={penalty:.3f})."
    )
    return make_tier0_decision(criterion, verdict, rationale, t0)
