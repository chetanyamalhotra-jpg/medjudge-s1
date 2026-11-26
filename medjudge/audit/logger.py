"""JSONL audit logger recording hashes, versions, spans, probabilities, and routing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from medjudge.schemas.models import (
    ClinicalArtifact,
    EvalTier,
    EvidenceSpan,
    GradeDecision,
    SufficiencyManifest,
    Verdict,
)


class AuditRecord(BaseModel):
    """Immutable audit log line for one criterion evaluation."""

    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    artifact_id: str
    artifact_sha256: str
    criterion_id: str
    criterion_version: str
    criterion_sha256: str
    verdict: Verdict
    probabilities: dict[str, float]
    calibrated_confidence: float
    tier_used: EvalTier
    escalated_from: EvalTier | None = None
    escalation_reason: str | None = None
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    sufficiency: SufficiencyManifest
    latency_ms: float
    shadow_checked: bool = False


class JSONLAuditLogger:
    """Append-only JSONL audit writer."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, artifact: ClinicalArtifact, decision: GradeDecision) -> AuditRecord:
        record = AuditRecord(
            artifact_id=artifact.artifact_id,
            artifact_sha256=artifact.content_sha256(),
            criterion_id=decision.criterion_id,
            criterion_version=decision.criterion_version,
            criterion_sha256=decision.criterion_sha256,
            verdict=decision.verdict,
            probabilities=decision.probabilities,
            calibrated_confidence=decision.calibrated_confidence,
            tier_used=decision.tier_used,
            escalated_from=decision.escalated_from,
            escalation_reason=decision.escalation_reason,
            evidence_spans=decision.evidence_spans,
            sufficiency=decision.sufficiency,
            latency_ms=decision.latency_ms,
            shadow_checked=decision.shadow_checked,
        )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        return record

    def read_all(self) -> list[AuditRecord]:
        if not self.log_path.exists():
            return []
        lines = [ln for ln in self.log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [AuditRecord.model_validate_json(ln) for ln in lines]
