"""Request and response schemas for POST /v1/systemone."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from medjudge.schemas.models import (
    ClinicalArtifact,
    EvalTier,
    EvidenceSpan,
    RiskClass,
    SufficiencyManifest,
    Verdict,
)


class SystemOneMode(StrEnum):
    """Supported System One decision primitives."""

    YES_NO = "yes_no"
    CHOICE = "choice"
    SCORE = "score"


class SystemOneRequest(BaseModel):
    """Pydantic AI / Ollama SystemOneModel compatible request payload."""

    mode: SystemOneMode = SystemOneMode.YES_NO
    criterion_id: str | None = None
    prompt: str | None = None
    risk_class: RiskClass = RiskClass.S2_CLINICAL_FACTUALITY
    required_context: list[str] = Field(default_factory=list)
    choices: list[str] = Field(default_factory=lambda: ["PASS", "FAIL", "UNKNOWN"])
    artifact: ClinicalArtifact = Field(default_factory=ClinicalArtifact)
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemOneResponse(BaseModel):
    """Standardized System One response with verdict, typed result, and calibrated confidence."""

    mode: SystemOneMode
    result: str | float | None
    confidence: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    probabilities: dict[str, float]
    tier_used: EvalTier
    escalated_from: EvalTier | None = None
    escalation_reason: str | None = None
    sufficiency: SufficiencyManifest
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    rationale: str = ""
