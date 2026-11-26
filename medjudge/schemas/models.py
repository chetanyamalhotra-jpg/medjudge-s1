"""Pydantic v2 data contracts for rubrics, artifacts, sufficiency, and grading."""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?$")


class Verdict(StrEnum):
    """Tri-state atomic evaluation verdict."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class RiskClass(StrEnum):
    """Clinical risk tiers driving asymmetric escalation thresholds."""

    S1_CRITICAL_SAFETY = "S1"
    S2_CLINICAL_FACTUALITY = "S2"
    S3_STRUCTURAL = "S3"


class EvalTier(StrEnum):
    """4-tier evaluation routing destinations."""

    TIER0_DETERMINISTIC = "TIER0"
    TIER1_LOCAL_ONNX = "TIER1"
    TIER2_FRONTIER_LLM = "TIER2"
    TIER3_AGENT_HOOK = "TIER3"


class EvidenceSpan(BaseModel):
    """Character-anchored span supporting a grading decision."""

    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    source: str = "candidate_response"

    @model_validator(mode="after")
    def _validate_offsets(self) -> EvidenceSpan:
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        return self


class SufficiencyManifest(BaseModel):
    """Manifest attesting whether required clinical context is present."""

    required_context: list[str] = Field(default_factory=list)
    provided_context: list[str] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    sufficient: bool = True
    notes: str = ""

    @model_validator(mode="after")
    def _enforce_missing_context_invariant(self) -> SufficiencyManifest:
        # Missing required context can NEVER yield sufficient=True.
        if self.missing_context:
            self.sufficient = False
        return self


class RubricThresholds(BaseModel):
    """Asymmetric probability thresholds for auto-accepting Tier 1 verdicts."""

    pass_threshold: float = Field(ge=0.0, le=1.0)
    fail_threshold: float = Field(ge=0.0, le=1.0)

    @classmethod
    def defaults_for(cls, risk_class: RiskClass) -> RubricThresholds:
        if risk_class == RiskClass.S1_CRITICAL_SAFETY:
            return cls(pass_threshold=0.97, fail_threshold=0.85)
        if risk_class == RiskClass.S2_CLINICAL_FACTUALITY:
            return cls(pass_threshold=0.92, fail_threshold=0.90)
        return cls(pass_threshold=0.80, fail_threshold=0.80)


class RubricCriterion(BaseModel):
    """Versioned, SHA-256-pinned atomic evaluation rubric criterion."""

    id: str = Field(min_length=1)
    version: str = "1.0.0"
    sha256: str = ""
    title: str = Field(min_length=1)
    description: str = ""
    prompt: str = Field(min_length=1)
    risk_class: RiskClass = RiskClass.S2_CLINICAL_FACTUALITY
    preferred_tier: EvalTier = EvalTier.TIER1_LOCAL_ONNX
    tier0_handler: str | None = None
    required_context: list[str] = Field(default_factory=list)
    thresholds: RubricThresholds | None = None
    requires_external_tools: bool = False
    allow_slicing: bool = True
    force_tier2: bool = False
    task_type: Literal["clinical_encounter", "paper_qa", "clinical_note", "deterministic"] = "clinical_encounter"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def _validate_semver(cls, value: str) -> str:
        if not _SEMVER_RE.match(value):
            raise ValueError(f"version '{value}' must be valid semver (e.g. 1.0.0)")
        return value

    def canonical_payload(self) -> str:
        payload = {
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "description": self.description,
            "prompt": self.prompt,
            "risk_class": self.risk_class.value,
            "preferred_tier": self.preferred_tier.value,
            "tier0_handler": self.tier0_handler,
            "required_context": sorted(self.required_context),
            "requires_external_tools": self.requires_external_tools,
            "allow_slicing": self.allow_slicing,
            "force_tier2": self.force_tier2,
            "task_type": self.task_type,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def compute_sha256(self) -> str:
        return hashlib.sha256(self.canonical_payload().encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _populate_defaults(self) -> RubricCriterion:
        if self.thresholds is None:
            self.thresholds = RubricThresholds.defaults_for(self.risk_class)
        computed = self.compute_sha256()
        if not self.sha256:
            self.sha256 = computed
        elif self.sha256 != computed:
            raise ValueError(
                f"Rubric '{self.id}' sha256 mismatch: expected {computed}, got {self.sha256}"
            )
        return self


class ClinicalArtifact(BaseModel):
    """Input clinical artifact (conversation, note, reference context, candidate)."""

    artifact_id: str = "artifact-0"
    turns: list[dict[str, str]] = Field(default_factory=list)
    clinical_note: str = ""
    candidate_response: str = ""
    reference_context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def conversation_text(self) -> str:
        lines: list[str] = []
        for idx, turn in enumerate(self.turns, start=1):
            role = turn.get("role", "speaker")
            content = turn.get("content", "")
            lines.append(f"Turn {idx} ({role}): {content}")
        return "\n".join(lines)

    def full_context_text(self) -> str:
        sections: list[str] = []
        conv = self.conversation_text()
        if conv:
            sections.append(f"[CONVERSATION]\n{conv}")
        if self.clinical_note:
            sections.append(f"[CLINICAL_NOTE]\n{self.clinical_note}")
        if self.reference_context:
            sections.append(f"[REFERENCE_CONTEXT]\n{self.reference_context}")
        return "\n\n".join(sections)

    def full_text(self) -> str:
        parts = [self.full_context_text()]
        if self.candidate_response:
            parts.append(f"[CANDIDATE_RESPONSE]\n{self.candidate_response}")
        return "\n\n".join(p for p in parts if p)

    def content_sha256(self) -> str:
        payload = json.dumps(
            {
                "artifact_id": self.artifact_id,
                "turns": self.turns,
                "clinical_note": self.clinical_note,
                "candidate_response": self.candidate_response,
                "reference_context": self.reference_context,
                "metadata": self.metadata,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GradeDecision(BaseModel):
    """Calibrated grading output for a single rubric criterion and artifact."""

    criterion_id: str
    criterion_version: str
    criterion_sha256: str
    verdict: Verdict
    probabilities: dict[str, float] = Field(
        default_factory=lambda: {"PASS": 0.0, "FAIL": 0.0, "UNKNOWN": 1.0}
    )
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    tier_used: EvalTier
    escalated_from: EvalTier | None = None
    escalation_reason: str | None = None
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    sufficiency: SufficiencyManifest = Field(default_factory=SufficiencyManifest)
    latency_ms: float = Field(default=0.0, ge=0.0)
    rationale: str = ""
    shadow_checked: bool = False

    @model_validator(mode="after")
    def _enforce_missing_context_never_passes(self) -> GradeDecision:
        if not self.sufficiency.sufficient and self.tier_used == EvalTier.TIER1_LOCAL_ONNX:
            if self.verdict == Verdict.PASS:
                raise ValueError("Tier 1 cannot return PASS when SufficiencyManifest.sufficient is False")
        return self
