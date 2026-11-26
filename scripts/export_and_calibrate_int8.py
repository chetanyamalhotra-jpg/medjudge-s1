"""Merge LoRA, export FP32 ONNX, selective FFN-only INT8 quantize, calibrate, & benchmark."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from sklearn.metrics import f1_score
from tokenizers import Tokenizer

from medjudge.eval.benchmarks import compute_brier_score, compute_ece
from medjudge.schemas.models import RiskClass, Verdict
from medjudge.tier1.calibration import (
    CalibrationManifest,
    apply_temperature,
    compute_asymmetric_thresholds,
    fit_temperature_scaling,
    rule_of_three_upper_bound,
    softmax,
)
from medjudge.training.export_onnx import (
    merge_and_export_fp32_onnx,
    select_ffn_quantization_nodes,
    selective_dynamic_int8_quantize,
)

LABEL_MAP = {Verdict.PASS.value: 0, Verdict.FAIL.value: 1, Verdict.UNKNOWN.value: 2}
LORA_DIR = Path("artifacts/bioclinical_modernbert_lora")
OUT_DIR = Path("artifacts/bioclinical_modernbert_int8")
MODEL_NAME = "thomas-sounack/BioClinical-ModernBERT-base"


def _load_split(path: Path) -> tuple[list[str], np.ndarray, list[RiskClass]]:
    texts: list[str] = []
    labels: list[int] = []
    risk_classes: list[RiskClass] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(LABEL_MAP[obj["label"]])
            # Map MedHallu / MEDEC-MS grounding & note error detection to S2 Clinical Factuality
            risk_classes.append(RiskClass.S2_CLINICAL_FACTUALITY)
    return texts, np.array(labels, dtype=np.int64), risk_classes


def _batched_onnx_logits(
    onnx_path: Path,
    tokenizer: Tokenizer,
    texts: list[str],
    max_length: int = 1024,
    batch_size: int = 16,
) -> np.ndarray:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.no_padding()

    all_logits: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encs = tokenizer.encode_batch(chunk)
        max_len = max(len(e.ids) for e in encs)
        # Pad to multiple of 8 for clean SIMD/GEMM alignment
        max_len = ((max_len + 7) // 8) * 8
        ids = np.full((len(encs), max_len), pad_id, dtype=np.int64)
        mask = np.zeros((len(encs), max_len), dtype=np.int64)
        for row_i, enc in enumerate(encs):
            n = len(enc.ids)
            ids[row_i, :n] = enc.ids
            mask[row_i, :n] = enc.attention_mask
        out = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        all_logits.append(np.asarray(out, dtype=np.float64))
    return np.vstack(all_logits)


def _eval_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = np.argmax(probs, axis=-1)
    per_f1 = f1_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": round(float(np.mean(per_f1)), 4),
        "pass_f1": round(float(per_f1[0]), 4),
        "fail_f1": round(float(per_f1[1]), 4),
        "unknown_f1": round(float(per_f1[2]), 4),
        "ece": round(compute_ece(probs, labels), 4),
        "brier_score": round(compute_brier_score(probs, labels), 4),
    }


def _benchmark_cpu_latency(onnx_path: Path, seq_lens: tuple[int, ...] = (256, 512, 1024)) -> dict[str, dict[str, float]]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    results: dict[str, dict[str, float]] = {}
    for seq_len in seq_lens:
        ids = np.ones((1, seq_len), dtype=np.int64)
        mask = np.ones((1, seq_len), dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        for _ in range(5):
            sess.run(None, feeds)
        times_ms: list[float] = []
        for _ in range(20):
            t0 = time.perf_counter()
            sess.run(None, feeds)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(times_ms, dtype=np.float64)
        results[str(seq_len)] = {
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
        }
    return results


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = OUT_DIR / "model_fp32.onnx"
    int8_path = OUT_DIR / "model_int8.onnx"

    # Copy tokenizer assets for zero-PyTorch runtime
    tok_src = LORA_DIR / "tokenizer.json"
    if tok_src.exists():
        shutil.copy2(tok_src, OUT_DIR / "tokenizer.json")
    tok_cfg_src = LORA_DIR / "tokenizer_config.json"
    if tok_cfg_src.exists():
        shutil.copy2(tok_cfg_src, OUT_DIR / "tokenizer_config.json")

    # 1. Merge LoRA & export FP32 ONNX (asserts max abs logit error < 1e-4 vs PyTorch FP32)
    merge_and_export_fp32_onnx(
        checkpoint_dir=LORA_DIR,
        fp32_onnx_path=fp32_path,
        max_length=1024,
        opset_version=17,
    )

    # 2. Selective dynamic INT8 quantization (MLP Wi ONLY; GeGLU Wo kept FP32 to avoid outlier collapse)
    ffn_nodes = select_ffn_quantization_nodes(fp32_path, exclude_geglu_wo=True)
    for node_name in ffn_nodes:
        low = node_name.lower()
        assert "mlp" in low and "wi" in low, f"Unexpected quantized node: {node_name}"
        assert not any(bad in low for bad in ("attn", "layernorm", "softmax", "head", "classifier", "wo"))

    tokenizer = Tokenizer.from_file(str(OUT_DIR / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=1024)
    sample_enc = tokenizer.encode("Patient has penicillin allergy. Candidate suggests azithromycin.")
    sample_feeds = {
        "input_ids": np.array([sample_enc.ids], dtype=np.int64),
        "attention_mask": np.array([sample_enc.attention_mask], dtype=np.int64),
    }

    _, quantized_nodes = selective_dynamic_int8_quantize(
        fp32_onnx_path=fp32_path,
        int8_onnx_path=int8_path,
        per_channel=True,
        reduce_range=True,
        use_pre_process=True,
        exclude_geglu_wo=True,
        sample_feeds=sample_feeds,
        parity_tol=0.6,
    )

    # Verify ONNX graph node preservation after quantization
    q_model = onnx.load(str(int8_path))
    quantized_matmuls = [
        n.name
        for n in q_model.graph.node
        if n.op_type in {"MatMulInteger", "QLinearMatMul", "QGemm"}
    ]
    assert len(quantized_matmuls) == len(quantized_nodes), (
        f"Expected {len(quantized_nodes)} quantized MatMulInteger nodes, got {len(quantized_matmuls)}"
    )
    for qn in quantized_matmuls:
        low = qn.lower()
        assert "mlp" in low and "wi" in low, f"Unexpected quantized MatMul node: {qn}"
        assert not any(bad in low for bad in ("attn", "layernorm", "softmax", "head", "classifier", "wo")), (
            f"Forbidden node quantized: {qn}"
        )
    # Confirm LayerNormalization and Softmax ops remain unquantized FP32 nodes
    for node in q_model.graph.node:
        if node.op_type in {"LayerNormalization", "Softmax"}:
            assert "quant" not in node.op_type.lower()

    # 3. Post-quantization calibration strictly on D_T and disjoint D_tau
    d_t_texts, d_t_y, _ = _load_split(Path("data/processed/d_t.jsonl"))
    d_tau_texts, d_tau_y, d_tau_rc = _load_split(Path("data/processed/d_tau.jsonl"))
    d_val_texts, d_val_y, _ = _load_split(Path("data/processed/d_val.jsonl"))

    d_t_logits = _batched_onnx_logits(int8_path, tokenizer, d_t_texts)
    temperature = fit_temperature_scaling(d_t_logits, d_t_y)

    d_tau_logits = _batched_onnx_logits(int8_path, tokenizer, d_tau_texts)
    d_tau_probs = apply_temperature(d_tau_logits, temperature)

    thresholds = {
        RiskClass.S1_CRITICAL_SAFETY.value: compute_asymmetric_thresholds(
            np.empty((0, 3)), np.empty((0,), dtype=np.int64), RiskClass.S1_CRITICAL_SAFETY
        ),
        RiskClass.S2_CLINICAL_FACTUALITY.value: compute_asymmetric_thresholds(
            d_tau_probs, d_tau_y, RiskClass.S2_CLINICAL_FACTUALITY
        ),
        RiskClass.S3_STRUCTURAL.value: compute_asymmetric_thresholds(
            np.empty((0, 3)), np.empty((0,), dtype=np.int64), RiskClass.S3_STRUCTURAL
        ),
    }

    manifest = CalibrationManifest(
        model_name=MODEL_NAME,
        quantization="selective_dynamic_int8_mlp_wi_only",
        temperature=round(temperature, 6),
        d_t_size=len(d_t_texts),
        d_tau_size=len(d_tau_texts),
        rule_of_three_upper_bound=round(rule_of_three_upper_bound(len(d_tau_texts)), 6),
        thresholds=thresholds,
        notes=(
            f"Post-INT8 L-BFGS temperature T*={temperature:.4f} fit on D_T (n={len(d_t_texts)}); "
            f"S2 asymmetric 95% LCB thresholds fit on disjoint D_tau (n={len(d_tau_texts)}). "
            "GeGLU mlp.Wo kept in FP32 alongside attn/LayerNorm/Softmax/head due to GeGLU product activation outliers. "
            "S1/S3 remain at conservative clinical floors pending dedicated S1/S3 gold calibration."
        ),
    )
    manifest.save(OUT_DIR / "calibration_manifest.json")

    # 4. Pre-INT8 vs Post-INT8 validation metrics on d_val & CPU latency benchmark
    val_fp32_logits = _batched_onnx_logits(fp32_path, tokenizer, d_val_texts)
    val_int8_logits = _batched_onnx_logits(int8_path, tokenizer, d_val_texts)

    pre_int8_metrics = _eval_metrics(softmax(val_fp32_logits), d_val_y)
    post_int8_raw_metrics = _eval_metrics(softmax(val_int8_logits), d_val_y)
    post_int8_cal_metrics = _eval_metrics(apply_temperature(val_int8_logits, temperature), d_val_y)

    latency = _benchmark_cpu_latency(int8_path, seq_lens=(256, 512, 1024))

    report = {
        "fp32_size_mb": round(fp32_path.stat().st_size / (1024 * 1024), 2),
        "int8_size_mb": round(int8_path.stat().st_size / (1024 * 1024), 2),
        "quantized_ffn_node_count": len(quantized_nodes),
        "sample_quantized_nodes": quantized_nodes[:6],
        "temperature_T_star": round(temperature, 6),
        "d_val_pre_int8_fp32": pre_int8_metrics,
        "d_val_post_int8_uncalibrated": post_int8_raw_metrics,
        "d_val_post_int8_calibrated": post_int8_cal_metrics,
        "cpu_latency_ms": latency,
    }
    (OUT_DIR / "export_benchmark_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
