"""Tier 1 pure ONNX Runtime CPU inference and post-quantization calibration (ZERO PyTorch)."""

from __future__ import annotations

from medjudge.tier1.calibration import (
    CalibrationManifest,
    apply_temperature,
    clopper_pearson_lcb,
    compute_asymmetric_thresholds,
    fit_temperature_scaling,
    rule_of_three_upper_bound,
    softmax,
    wilson_lcb,
)
from medjudge.tier1.runtime import (
    LABEL_ORDER,
    Tier1ClassifierProtocol,
    Tier1ONNXClassifier,
    TokenizerProtocol,
)

__all__ = [
    "CalibrationManifest",
    "LABEL_ORDER",
    "Tier1ClassifierProtocol",
    "Tier1ONNXClassifier",
    "TokenizerProtocol",
    "apply_temperature",
    "clopper_pearson_lcb",
    "compute_asymmetric_thresholds",
    "fit_temperature_scaling",
    "rule_of_three_upper_bound",
    "softmax",
    "wilson_lcb",
]
