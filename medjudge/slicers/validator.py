"""Slicer agreement validation gate against full-context Tier 2 verdicts."""

from __future__ import annotations

from pydantic import BaseModel

from medjudge.schemas.models import RiskClass, Verdict


class SlicerValidationResult(BaseModel):
    """Result of validating sliced Tier 1/2 agreement against full-context Tier 2."""

    risk_class: RiskClass
    sample_count: int
    cohens_kappa: float
    fail_recall: float
    s1_false_negative_shift: float
    enabled: bool
    reason: str


def _cohens_kappa(y_true: list[Verdict], y_pred: list[Verdict]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    labels = [Verdict.PASS, Verdict.FAIL, Verdict.UNKNOWN]
    p_o = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == b) / n
    p_e = sum(
        (sum(1 for a in y_true if a == lbl) / n) * (sum(1 for b in y_pred if b == lbl) / n)
        for lbl in labels
    )
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def validate_slicer_sufficiency(
    sliced_verdicts: list[Verdict],
    full_context_verdicts: list[Verdict],
    risk_class: RiskClass,
    min_samples: int = 100,
) -> SlicerValidationResult:
    """Verify whether a slicer meets risk-class kappa, recall, and FN-shift thresholds."""
    if len(sliced_verdicts) != len(full_context_verdicts):
        raise ValueError("sliced_verdicts and full_context_verdicts must have equal length")

    n = len(full_context_verdicts)
    if n < min_samples:
        return SlicerValidationResult(
            risk_class=risk_class,
            sample_count=n,
            cohens_kappa=0.0,
            fail_recall=0.0,
            s1_false_negative_shift=1.0,
            enabled=False,
            reason=f"Insufficient validation samples ({n} < {min_samples}); slicing disabled.",
        )

    kappa = _cohens_kappa(full_context_verdicts, sliced_verdicts)

    # Recall on FAIL (positive error detection)
    true_fails = [i for i, v in enumerate(full_context_verdicts) if v == Verdict.FAIL]
    if true_fails:
        detected_fails = sum(1 for i in true_fails if sliced_verdicts[i] == Verdict.FAIL)
        fail_recall = detected_fails / len(true_fails)
        # False negative shift: full context caught FAIL, sliced returned PASS
        false_passes = sum(1 for i in true_fails if sliced_verdicts[i] == Verdict.PASS)
        fn_shift = false_passes / len(true_fails)
    else:
        fail_recall = 1.0
        fn_shift = 0.0

    if risk_class == RiskClass.S1_CRITICAL_SAFETY:
        ok = kappa >= 0.90 and fn_shift < 0.02
        req = "kappa >= 0.90 and false-negative shift < 2%"
    elif risk_class == RiskClass.S2_CLINICAL_FACTUALITY:
        ok = kappa >= 0.85 and fail_recall >= 0.90
        req = "kappa >= 0.85 and FAIL recall >= 0.90"
    else:
        ok = kappa >= 0.80
        req = "kappa >= 0.80"

    reason = (
        f"Passed {risk_class.value} slicer gate ({req})."
        if ok
        else f"Failed {risk_class.value} slicer gate ({req}): kappa={kappa:.3f}, recall={fail_recall:.3f}, fn_shift={fn_shift:.3f}"
    )
    return SlicerValidationResult(
        risk_class=risk_class,
        sample_count=n,
        cohens_kappa=round(kappa, 4),
        fail_recall=round(fail_recall, 4),
        s1_false_negative_shift=round(fn_shift, 4),
        enabled=ok,
        reason=reason,
    )
