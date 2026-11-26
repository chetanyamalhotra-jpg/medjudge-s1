"""Tier 0 handler registry and GradeDecision builder."""

from __future__ import annotations

import time
from collections.abc import Callable

from medjudge.schemas.models import (
    ClinicalArtifact,
    EvalTier,
    EvidenceSpan,
    GradeDecision,
    RubricCriterion,
    SufficiencyManifest,
    Verdict,
)

Tier0Handler = Callable[[RubricCriterion, ClinicalArtifact], GradeDecision]


def make_tier0_decision(
    criterion: RubricCriterion,
    verdict: Verdict,
    rationale: str,
    start_time: float,
    evidence_spans: list[EvidenceSpan] | None = None,
    missing_context: list[str] | None = None,
) -> GradeDecision:
    """Construct a deterministic GradeDecision with confidence 1.0 (or 0.0 if UNKNOWN)."""
    elapsed_ms = max(0.0, (time.perf_counter() - start_time) * 1000.0)
    probs = {
        "PASS": 1.0 if verdict == Verdict.PASS else 0.0,
        "FAIL": 1.0 if verdict == Verdict.FAIL else 0.0,
        "UNKNOWN": 1.0 if verdict == Verdict.UNKNOWN else 0.0,
    }
    missing = missing_context or []
    return GradeDecision(
        criterion_id=criterion.id,
        criterion_version=criterion.version,
        criterion_sha256=criterion.sha256,
        verdict=verdict,
        probabilities=probs,
        calibrated_confidence=1.0 if verdict != Verdict.UNKNOWN else 0.0,
        tier_used=EvalTier.TIER0_DETERMINISTIC,
        evidence_spans=evidence_spans or [],
        sufficiency=SufficiencyManifest(
            required_context=criterion.required_context,
            provided_context=criterion.required_context if not missing else [],
            missing_context=missing,
            sufficient=(len(missing) == 0),
            notes=rationale,
        ),
        latency_ms=elapsed_ms,
        rationale=rationale,
    )


class Tier0Registry:
    """Registry mapping rubric tier0_handler names to deterministic functions."""

    def __init__(self) -> None:
        self._handlers: dict[str, Tier0Handler] = {}

    def register(self, name: str, handler: Tier0Handler) -> None:
        self._handlers[name] = handler

    def has(self, name: str) -> bool:
        return name in self._handlers

    def run(self, name: str, criterion: RubricCriterion, artifact: ClinicalArtifact) -> GradeDecision:
        if name not in self._handlers:
            raise KeyError(f"Unknown Tier 0 handler '{name}'")
        return self._handlers[name](criterion, artifact)


def build_default_tier0_registry() -> Tier0Registry:
    """Build the standard Tier 0 verifier registry."""
    from medjudge.tier0.citations import verify_citations
    from medjudge.tier0.fhir import verify_fhir_json
    from medjudge.tier0.length_penalty import verify_length_penalty
    from medjudge.tier0.mcq import verify_mcq_answer
    from medjudge.tier0.medcalc import verify_medcalc

    registry = Tier0Registry()
    registry.register("medcalc", verify_medcalc)
    registry.register("fhir_schema", verify_fhir_json)
    registry.register("length_penalty", verify_length_penalty)
    registry.register("citation_check", verify_citations)
    registry.register("mcq_match", verify_mcq_answer)
    return registry
