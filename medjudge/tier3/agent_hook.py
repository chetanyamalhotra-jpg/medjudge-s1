"""Tier 3 Agent-as-a-Judge protocol for external PubMed, guideline, and FHIR verification."""

from __future__ import annotations

from typing import Protocol

from medjudge.schemas.models import ClinicalArtifact, GradeDecision, RubricCriterion


class Tier3AgentJudge(Protocol):
    """Protocol for criteria requiring live external tool verification."""

    async def verify_with_tools(
        self, criterion: RubricCriterion, artifact: ClinicalArtifact
    ) -> GradeDecision:
        ...


class UnconfiguredTier3Stub:
    """Explicit placeholder raising NotImplementedError when Tier 3 tools are requested without wiring."""

    async def verify_with_tools(
        self, criterion: RubricCriterion, artifact: ClinicalArtifact
    ) -> GradeDecision:
        raise NotImplementedError(
            f"Criterion '{criterion.id}' requires external Tier 3 tools "
            "(PubMed retrieval / clinical guidelines / stateful FHIR), "
            "but no Tier3AgentJudge implementation is configured."
        )
