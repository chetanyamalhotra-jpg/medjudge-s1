"""Tests for selective FFN-only INT8 quantization, real ONNX CPU inference, & calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from medjudge.schemas.models import ClinicalArtifact, RiskClass, RubricCriterion, Verdict
from medjudge.slicers.pinned_header import PinnedHeaderSlicer
from medjudge.tier1.calibration import (
    CalibrationManifest,
    clopper_pearson_lcb,
    compute_asymmetric_thresholds,
    fit_temperature_scaling,
    rule_of_three_upper_bound,
    wilson_lcb,
)
from medjudge.tier1.runtime import Tier1ONNXClassifier
from medjudge.training.export_onnx import (
    select_ffn_quantization_nodes,
    selective_dynamic_int8_quantize,
)


class _StubTokenizer:
    def encode_to_arrays(self, text: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
        arr = np.array([[1.0, 2.0, 0.5, 1.5]], dtype=np.float32)
        mask = np.ones((1, 4), dtype=np.int64)
        return arr, mask


def _build_mini_modernbert_onnx(path: Path) -> Path:
    rng = np.random.default_rng(7)
    w_qkv = numpy_helper.from_array(rng.normal(0, 0.2, (4, 4)).astype(np.float32), "W_qkv")
    w_wi = numpy_helper.from_array(rng.normal(0, 0.2, (4, 8)).astype(np.float32), "W_wi")
    w_wo = numpy_helper.from_array(rng.normal(0, 0.2, (8, 4)).astype(np.float32), "W_wo")
    w_head = numpy_helper.from_array(
        np.array(
            [
                [2.0, -1.0, -1.0],
                [1.5, -0.5, -1.0],
                [-1.0, 0.5, 0.5],
                [0.5, -0.5, 0.0],
            ],
            dtype=np.float32,
        ),
        "W_head",
    )

    n_qkv = helper.make_node("MatMul", ["input_ids", "W_qkv"], ["attn_out"], name="encoder_attention_qkv")
    n_sm = helper.make_node("Softmax", ["attn_out"], ["attn_prob"], axis=1, name="attention_softmax")
    n_wi = helper.make_node("MatMul", ["attn_prob", "W_wi"], ["ffn_mid"], name="encoder_mlp_wi")
    n_relu = helper.make_node("Relu", ["ffn_mid"], ["ffn_act"], name="mlp_relu")
    n_wo = helper.make_node("MatMul", ["ffn_act", "W_wo"], ["ffn_out"], name="encoder_mlp_wo")
    n_head = helper.make_node("MatMul", ["ffn_out", "W_head"], ["logits"], name="classifier_head")

    graph = helper.make_graph(
        [n_qkv, n_sm, n_wi, n_relu, n_wo, n_head],
        "MiniModernBERT",
        [helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
        initializer=[w_qkv, w_wi, w_wo, w_head],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def test_selective_int8_quantization_excludes_attention_softmax_and_head(tmp_path: Path) -> None:
    fp32_path = _build_mini_modernbert_onnx(tmp_path / "mini_fp32.onnx")
    int8_path = tmp_path / "mini_int8.onnx"

    selected = select_ffn_quantization_nodes(fp32_path)
    assert selected == ["encoder_mlp_wi", "encoder_mlp_wo"]
    assert select_ffn_quantization_nodes(fp32_path, exclude_geglu_wo=True) == ["encoder_mlp_wi"]

    sample = {"input_ids": np.array([[1.0, 2.0, 0.5, 1.5]], dtype=np.float32)}
    out_path, quantized_nodes = selective_dynamic_int8_quantize(
        fp32_path, int8_path, per_channel=True, sample_feeds=sample, parity_tol=0.5
    )
    assert out_path.exists()
    assert quantized_nodes == ["encoder_mlp_wi", "encoder_mlp_wo"]

    q_model = onnx.load(str(int8_path))
    q_ops_by_name = {n.name: n.op_type for n in q_model.graph.node}
    # Attention, Softmax, and classification head must NOT be quantized.
    assert q_ops_by_name.get("encoder_attention_qkv") == "MatMul"
    assert q_ops_by_name.get("attention_softmax") == "Softmax"
    assert q_ops_by_name.get("classifier_head") == "MatMul"
    # FFN nodes were replaced by MatMulInteger quantization nodes
    assert any("encoder_mlp_wi" in name for name in q_ops_by_name)
    assert any(n.op_type == "MatMulInteger" for n in q_model.graph.node)

    # Run Tier1ONNXClassifier on CPUExecutionProvider
    manifest = CalibrationManifest(temperature=1.25)
    clf = Tier1ONNXClassifier(int8_path, _StubTokenizer(), calibration_manifest=manifest)
    crit = RubricCriterion(
        id="s2_grounding",
        version="1.0.0",
        title="Grounding",
        prompt="Is it grounded?",
        required_context=["reference_context"],
    )
    art = ClinicalArtifact(
        reference_context="BP is 120/80 mmHg.",
        candidate_response="BP is 120/80 mmHg.",
    )
    sliced = PinnedHeaderSlicer().slice(crit, art)
    dec = clf.predict(crit, art, sliced)
    assert abs(sum(dec.probabilities.values()) - 1.0) < 1e-6

    # Missing required context must force UNKNOWN and calibrated_confidence = probs["UNKNOWN"]
    crit_missing = RubricCriterion(
        id="s2_missing_ctx",
        version="1.0.0",
        title="Missing context",
        prompt="Check allergy",
        required_context=["allergies"],
    )
    sliced_missing = PinnedHeaderSlicer().slice(crit_missing, art)
    assert sliced_missing.sufficiency.sufficient is False
    dec_missing = clf.predict(crit_missing, art, sliced_missing)
    assert dec_missing.verdict == Verdict.UNKNOWN
    assert abs(dec_missing.calibrated_confidence - dec_missing.probabilities["UNKNOWN"]) < 1e-9


def test_post_quantization_calibration_math() -> None:
    # Overconfident logits with realistic misclassified errors should yield temperature T* > 1.5
    rng = np.random.default_rng(12)
    raw = rng.normal(0, 1.0, (160, 3))
    labels = np.argmax(raw, axis=1)
    flip_mask = rng.random(160) < 0.18
    labels[flip_mask] = (labels[flip_mask] + 1) % 3
    overconfident_logits = raw * 3.5

    t_star = fit_temperature_scaling(overconfident_logits, labels)
    assert t_star > 1.5

    assert abs(rule_of_three_upper_bound(150) - 0.02) < 1e-9
    lcb_cp = clopper_pearson_lcb(148, 150)
    lcb_w = wilson_lcb(148, 150)
    assert 0.90 < lcb_cp < 0.99
    assert 0.90 < lcb_w < 0.99

    probs = np.tile(np.array([0.99, 0.005, 0.005]), (60, 1))
    y = np.zeros(60, dtype=np.int64)
    thresh = compute_asymmetric_thresholds(probs, y, RiskClass.S1_CRITICAL_SAFETY)
    assert thresh.pass_threshold >= 0.97
    assert thresh.fail_threshold >= 0.85
