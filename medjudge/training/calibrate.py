"""Post-quantization calibration pipeline: fit T* on D_T and LCB thresholds on disjoint D_tau."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from medjudge.schemas.models import RiskClass, Verdict
from medjudge.tier1.calibration import (
    CalibrationManifest,
    apply_temperature,
    compute_asymmetric_thresholds,
    fit_temperature_scaling,
    rule_of_three_upper_bound,
)
from medjudge.tier1.runtime import LABEL_ORDER, Tier1ONNXClassifier, TokenizerProtocol


def _verdict_to_idx(verdict: Verdict) -> int:
    return LABEL_ORDER.index(verdict)


def calibrate_int8_onnx_model(
    int8_onnx_path: str | Path,
    tokenizer: TokenizerProtocol | str | Path,
    d_t_texts: list[str],
    d_t_labels: list[Verdict],
    d_tau_texts: list[str],
    d_tau_labels: list[Verdict],
    d_tau_risk_classes: list[RiskClass],
    output_manifest_path: str | Path,
    model_name: str = "thomas-sounack/BioClinical-ModernBERT-base",
    max_length: int = 1024,
) -> CalibrationManifest:
    """Calibrate the final INT8 ONNX model strictly post-quantization using disjoint D_T and D_tau."""
    clf = Tier1ONNXClassifier(
        onnx_model_path=int8_onnx_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    d_t_logits = np.vstack([clf.predict_logits(txt) for txt in d_t_texts])
    d_t_y = np.array([_verdict_to_idx(lbl) for lbl in d_t_labels], dtype=np.int64)
    temperature = fit_temperature_scaling(d_t_logits, d_t_y)

    d_tau_logits = np.vstack([clf.predict_logits(txt) for txt in d_tau_texts])
    d_tau_probs = apply_temperature(d_tau_logits, temperature)
    d_tau_y = np.array([_verdict_to_idx(lbl) for lbl in d_tau_labels], dtype=np.int64)

    thresholds = {}
    for rc in (
        RiskClass.S1_CRITICAL_SAFETY,
        RiskClass.S2_CLINICAL_FACTUALITY,
        RiskClass.S3_STRUCTURAL,
    ):
        mask = np.array([r == rc for r in d_tau_risk_classes], dtype=bool)
        if np.any(mask):
            thresholds[rc.value] = compute_asymmetric_thresholds(
                d_tau_probs[mask], d_tau_y[mask], risk_class=rc
            )
        else:
            thresholds[rc.value] = compute_asymmetric_thresholds(
                np.empty((0, 3)), np.empty((0,), dtype=np.int64), risk_class=rc
            )

    s1_count = sum(1 for r in d_tau_risk_classes if r == RiskClass.S1_CRITICAL_SAFETY)
    manifest = CalibrationManifest(
        model_name=model_name,
        quantization="selective_dynamic_int8_ffn_only",
        temperature=round(temperature, 6),
        d_t_size=len(d_t_texts),
        d_tau_size=len(d_tau_texts),
        rule_of_three_upper_bound=round(rule_of_three_upper_bound(s1_count or len(d_tau_texts)), 6),
        thresholds=thresholds,
    )
    manifest.save(output_manifest_path)
    return manifest
