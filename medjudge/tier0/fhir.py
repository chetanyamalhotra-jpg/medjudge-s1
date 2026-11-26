"""Deterministic FHIR R4 JSON structural validator."""

from __future__ import annotations

import json
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

_FHIR_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "Patient": ("resourceType",),
    "Observation": ("resourceType", "status", "code"),
    "MedicationRequest": ("resourceType", "status", "intent", "subject"),
    "Condition": ("resourceType", "subject"),
    "AllergyIntolerance": ("resourceType", "patient"),
}
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_payload(text: str) -> str:
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1)
    return text.strip()


def verify_fhir_json(criterion: RubricCriterion, artifact: ClinicalArtifact) -> GradeDecision:
    """Validate FHIR R4 JSON payload structure and required resource fields."""
    t0 = time.perf_counter()
    raw = _extract_json_payload(artifact.candidate_response)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return make_tier0_decision(
            criterion,
            Verdict.FAIL,
            f"Invalid JSON in candidate response: {exc.msg}",
            t0,
        )

    if not isinstance(obj, dict):
        return make_tier0_decision(criterion, Verdict.FAIL, "FHIR payload must be a JSON object.", t0)

    rtype = obj.get("resourceType")
    if not isinstance(rtype, str) or not rtype:
        return make_tier0_decision(criterion, Verdict.FAIL, "Missing string 'resourceType'.", t0)

    expected_type = artifact.metadata.get("expected_resource_type")
    if expected_type and rtype != expected_type:
        return make_tier0_decision(
            criterion,
            Verdict.FAIL,
            f"Expected resourceType '{expected_type}', got '{rtype}'.",
            t0,
        )

    required = _FHIR_REQUIRED_FIELDS.get(rtype)
    if required is None:
        return make_tier0_decision(
            criterion,
            Verdict.UNKNOWN,
            f"Unsupported FHIR resourceType '{rtype}' for deterministic schema validator.",
            t0,
            missing_context=[f"fhir_schema:{rtype}"],
        )

    missing_fields = [f for f in required if f not in obj or obj[f] in (None, "", [], {})]
    if missing_fields:
        return make_tier0_decision(
            criterion,
            Verdict.FAIL,
            f"FHIR {rtype} missing required field(s): {', '.join(missing_fields)}",
            t0,
        )

    span = EvidenceSpan(
        text=f"resourceType={rtype}",
        start_char=0,
        end_char=min(len(artifact.candidate_response), 120),
    )
    return make_tier0_decision(
        criterion,
        Verdict.PASS,
        f"Valid FHIR R4 {rtype} payload with all required fields ({', '.join(required)}).",
        t0,
        evidence_spans=[span],
    )
