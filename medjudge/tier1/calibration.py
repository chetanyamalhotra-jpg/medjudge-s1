"""Post-quantization temperature scaling and Clopper-Pearson/Wilson asymmetric thresholds."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize
from scipy.stats import beta

from medjudge.schemas.models import RiskClass, RubricThresholds


class CalibrationManifest(BaseModel):
    """Calibration artifact emitted ONLY after selective INT8 ONNX quantization."""

    model_name: str = "thomas-sounack/BioClinical-ModernBERT-base"
    quantization: str = "selective_dynamic_int8_ffn_only"
    temperature: float = Field(default=1.0, gt=0.05, le=20.0)
    d_t_size: int = Field(default=0, ge=0)
    d_tau_size: int = Field(default=0, ge=0)
    rule_of_three_upper_bound: float = Field(default=1.0, ge=0.0, le=1.0)
    thresholds: dict[str, RubricThresholds] = Field(
        default_factory=lambda: {
            RiskClass.S1_CRITICAL_SAFETY.value: RubricThresholds(
                pass_threshold=0.98, fail_threshold=0.85
            ),
            RiskClass.S2_CLINICAL_FACTUALITY.value: RubricThresholds(
                pass_threshold=0.93, fail_threshold=0.90
            ),
            RiskClass.S3_STRUCTURAL.value: RubricThresholds(
                pass_threshold=0.82, fail_threshold=0.82
            ),
        }
    )
    calibrated_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    notes: str = (
        "Post-INT8 temperature scaling on D_T and asymmetric LCB thresholds on disjoint D_tau. "
        "Small-gold-set Rule-of-Three upper bound applies when 0 critical misses are observed."
    )

    def thresholds_for(self, risk_class: RiskClass) -> RubricThresholds:
        return self.thresholds.get(
            risk_class.value, RubricThresholds.defaults_for(risk_class)
        )

    @classmethod
    def load(cls, path: str | Path) -> CalibrationManifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax over 2D or 1D logits."""
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by scalar temperature T* and return calibrated probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return softmax(np.asarray(logits, dtype=np.float64) / float(temperature))


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit single scalar temperature T* via L-BFGS-B minimizing NLL on D_T."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if logits_arr.ndim != 2 or logits_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError("logits must be (N, C) matching labels of shape (N,)")
    if logits_arr.shape[0] == 0:
        raise ValueError("D_T cannot be empty when fitting temperature scaling")

    n = logits_arr.shape[0]
    row_idx = np.arange(n)

    def nll(log_t: np.ndarray) -> float:
        t = float(np.exp(log_t[0]))
        scaled = logits_arr / t
        scaled -= np.max(scaled, axis=1, keepdims=True)
        log_probs = scaled - np.log(np.sum(np.exp(scaled), axis=1, keepdims=True))
        return float(-np.mean(log_probs[row_idx, labels_arr]))

    res = minimize(nll, x0=np.array([0.0]), method="L-BFGS-B", bounds=[(-2.5, 2.5)])
    return float(np.exp(res.x[0]))


def clopper_pearson_lcb(successes: int, total: int, alpha: float = 0.05) -> float:
    """Exact two-sided 1-alpha Clopper-Pearson lower confidence bound on p."""
    if total <= 0:
        return 0.0
    if successes <= 0:
        return 0.0
    if successes >= total:
        return float(beta.ppf(alpha / 2.0, total, 1))
    return float(beta.ppf(alpha / 2.0, successes, total - successes + 1))


def wilson_lcb(successes: int, total: int, alpha: float = 0.05) -> float:
    """Wilson score lower confidence bound (z=1.96 at alpha=0.05)."""
    if total <= 0:
        return 0.0
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2.0))
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = p_hat + (z * z) / (2.0 * total)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * total)) / total)
    return max(0.0, (center - margin) / denom)


def rule_of_three_upper_bound(total: int) -> float:
    """Rule of Three 95% upper bound on true miss rate when 0 misses observed in n items."""
    if total <= 0:
        return 1.0
    return min(1.0, 3.0 / float(total))


def compute_asymmetric_thresholds(
    probs: np.ndarray,
    true_labels: np.ndarray,
    risk_class: RiskClass,
    target_lcb: float | None = None,
    alpha: float = 0.05,
) -> RubricThresholds:
    """Find minimum calibrated probability cutoffs on D_tau whose 95% LCB meets clinical safety."""
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(true_labels, dtype=np.int64)
    defaults = RubricThresholds.defaults_for(risk_class)
    if len(labels_arr) == 0:
        return defaults

    # Class index convention: 0=PASS, 1=FAIL, 2=UNKNOWN
    if target_lcb is None:
        if risk_class == RiskClass.S1_CRITICAL_SAFETY:
            target_pass_prec = 0.97
            target_fail_prec = 0.85
        elif risk_class == RiskClass.S2_CLINICAL_FACTUALITY:
            target_pass_prec = 0.92
            target_fail_prec = 0.90
        else:
            target_pass_prec = 0.80
            target_fail_prec = 0.80
    else:
        target_pass_prec = target_lcb
        target_fail_prec = target_lcb

    def _find_cutoff(class_idx: int, min_lcb: float, floor: float) -> float:
        for cand in np.linspace(floor, 0.995, 40):
            mask = probs_arr[:, class_idx] >= cand
            n_sel = int(np.sum(mask))
            if n_sel < 5:
                continue
            k_ok = int(np.sum(labels_arr[mask] == class_idx))
            if clopper_pearson_lcb(k_ok, n_sel, alpha=alpha) >= min_lcb:
                return round(float(cand), 4)
        return floor

    pass_cut = max(
        defaults.pass_threshold,
        _find_cutoff(0, target_pass_prec, defaults.pass_threshold),
    )
    fail_cut = max(
        defaults.fail_threshold,
        _find_cutoff(1, target_fail_prec, defaults.fail_threshold),
    )
    return RubricThresholds(pass_threshold=pass_cut, fail_threshold=fail_cut)
