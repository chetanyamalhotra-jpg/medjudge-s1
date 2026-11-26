"""Deterministic MedCalc clinical formulas and unit conversion verifier."""

from __future__ import annotations

import math
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

_NUMBER_UNIT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(mg/kg|mcg/kg|mg|mcg|ug|g|kg|lbs?|cm|m|ml/min|meq/l|mmol/l|mmhg)?",
    re.IGNORECASE,
)


def normalize_unit_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert common clinical units deterministically."""
    src = from_unit.lower().strip()
    dst = to_unit.lower().strip()
    if src == dst or not dst:
        return value
    # Mass
    mass_to_mg = {"mcg": 1e-3, "ug": 1e-3, "mg": 1.0, "g": 1e3, "kg": 1e6}
    if src in mass_to_mg and dst in mass_to_mg:
        return value * mass_to_mg[src] / mass_to_mg[dst]
    # Weight (lb <-> kg)
    if src in {"lb", "lbs"} and dst == "kg":
        return value * 0.45359237
    if src == "kg" and dst in {"lb", "lbs"}:
        return value / 0.45359237
    # Length
    len_to_m = {"cm": 0.01, "m": 1.0}
    if src in len_to_m and dst in len_to_m:
        return value * len_to_m[src] / len_to_m[dst]
    raise ValueError(f"Unsupported unit conversion: {from_unit} -> {to_unit}")


def compute_clinical_formula(formula: str, params: dict[str, float | str]) -> tuple[float, str]:
    """Compute exact clinical formulas used in MedCalc-style deterministic verification."""
    name = formula.lower().strip()
    if name == "bmi":
        weight_kg = float(params["weight_kg"])
        height_m = float(params["height_m"])
        if height_m <= 0:
            raise ValueError("height_m must be positive")
        return weight_kg / (height_m**2), "kg/m^2"

    if name == "map":
        sbp = float(params["sbp"])
        dbp = float(params["dbp"])
        return (sbp + 2.0 * dbp) / 3.0, "mmhg"

    if name == "anion_gap":
        na = float(params["na"])
        cl = float(params["cl"])
        hco3 = float(params["hco3"])
        return na - (cl + hco3), "meq/l"

    if name == "corrected_calcium":
        ca = float(params["calcium"])
        alb = float(params["albumin"])
        return ca + 0.8 * (4.0 - alb), "mg/dl"

    if name == "cockcroft_gault":
        age = float(params["age"])
        weight_kg = float(params["weight_kg"])
        scr = float(params["creatinine"])
        sex = str(params.get("sex", "male")).lower().strip()
        if scr <= 0:
            raise ValueError("creatinine must be positive")
        crcl = ((140.0 - age) * weight_kg) / (72.0 * scr)
        if sex in {"f", "female"}:
            crcl *= 0.85
        return crcl, "ml/min"

    if name == "weight_based_dose":
        weight_kg = float(params["weight_kg"])
        dose_per_kg = float(params["dose_per_kg"])
        return weight_kg * dose_per_kg, str(params.get("unit", "mg"))

    raise ValueError(f"Unknown clinical formula: '{formula}'")


def _extract_candidate_number(text: str, target_unit: str) -> tuple[float, str, EvidenceSpan] | None:
    matches = list(_NUMBER_UNIT_RE.finditer(text))
    if not matches:
        return None
    # Prefer a match whose unit matches or converts to target_unit; otherwise take last numeric value.
    for m in reversed(matches):
        val = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if target_unit and unit:
            try:
                converted = normalize_unit_value(val, unit, target_unit)
                return converted, unit, EvidenceSpan(text=m.group(0), start_char=m.start(), end_char=m.end())
            except ValueError:
                continue
    last = matches[-1]
    return (
        float(last.group(1)),
        (last.group(2) or ""),
        EvidenceSpan(text=last.group(0), start_char=last.start(), end_char=last.end()),
    )


def verify_medcalc(criterion: RubricCriterion, artifact: ClinicalArtifact) -> GradeDecision:
    """Verify candidate calculation against metadata['medcalc'] spec."""
    t0 = time.perf_counter()
    spec = artifact.metadata.get("medcalc")
    if not isinstance(spec, dict) or "formula" not in spec or "params" not in spec:
        return make_tier0_decision(
            criterion,
            Verdict.UNKNOWN,
            "Missing metadata['medcalc'] formula/params specification; escalating.",
            t0,
            missing_context=["medcalc_spec"],
        )

    expected_val, expected_unit = compute_clinical_formula(spec["formula"], spec["params"])
    rel_tol = float(spec.get("rel_tol", 0.05))
    abs_tol = float(spec.get("abs_tol", 0.1))

    extracted = _extract_candidate_number(artifact.candidate_response, expected_unit)
    if extracted is None:
        return make_tier0_decision(
            criterion,
            Verdict.FAIL,
            f"No numeric answer found in candidate response (expected {expected_val:.3f} {expected_unit}).",
            t0,
        )

    cand_val, cand_unit, span = extracted
    ok = math.isclose(cand_val, expected_val, rel_tol=rel_tol, abs_tol=abs_tol)
    verdict = Verdict.PASS if ok else Verdict.FAIL
    rationale = (
        f"Formula '{spec['formula']}' expected {expected_val:.3f} {expected_unit}; "
        f"candidate reported {cand_val:.3f} {cand_unit or expected_unit}."
    )
    return make_tier0_decision(criterion, verdict, rationale, t0, evidence_spans=[span])
