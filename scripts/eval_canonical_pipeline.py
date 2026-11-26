"""Canonical end-to-end Tier 1 evaluation on d_val.jsonl through the deployed pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, recall_score
from tokenizers import Tokenizer

from medjudge.eval.benchmarks import compute_aurc, compute_brier_score, compute_ece
from medjudge.router.engine import EvaluationRouter
from medjudge.rubrics.loader import load_default_rubrics
from medjudge.schemas.models import ClinicalArtifact, RubricCriterion, Verdict
from medjudge.slicers.pinned_header import PinnedHeaderSlicer
from medjudge.tier1.calibration import apply_temperature
from medjudge.tier1.runtime import Tier1ONNXClassifier

ARTIFACT_DIR = Path("artifacts/bioclinical_modernbert_int8")
D_VAL_PATH = Path("data/processed/d_val.jsonl")
EXPORT_BENCHMARK_PATH = ARTIFACT_DIR / "export_benchmark_report.json"
OUT_REPORT_PATH = ARTIFACT_DIR / "canonical_eval_report.json"

ORDER: tuple[Verdict, Verdict, Verdict] = (Verdict.PASS, Verdict.FAIL, Verdict.UNKNOWN)
LABEL_MAP: dict[str, int] = {Verdict.PASS.value: 0, Verdict.FAIL.value: 1, Verdict.UNKNOWN.value: 2}
TAU_GRID: tuple[float, ...] = (0.50, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99)


def _get_cpu_info() -> dict[str, str]:
    """Capture exact CPU model name via Win32_Processor (if Windows) and platform.processor()."""
    win32_name = ""
    if platform.system().lower() == "windows":
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            win32_name = proc.stdout.strip()
        except Exception:
            win32_name = ""
    plat_proc = platform.processor().strip()
    return {
        "cpu_model_win32": win32_name or plat_proc,
        "platform_processor": plat_proc,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def _reconstruct_artifact(row: dict[str, Any]) -> ClinicalArtifact:
    """Reconstruct a ClinicalArtifact from a processed d_val record cleanly."""
    dataset = row["dataset_name"]
    text = str(row["text"])
    meta = dict(row.get("metadata") or {})
    if dataset == "medhallu":
        if "\n\n[REFERENCE_CONTEXT]\n" not in text or "\n\n[CANDIDATE_RESPONSE]\n" not in text:
            raise ValueError(f"Unexpected medhallu formatting for {row.get('record_id')}")
        _, rest = text.split("\n\n[REFERENCE_CONTEXT]\n", 1)
        ref_ctx, cand_resp = rest.split("\n\n[CANDIDATE_RESPONSE]\n", 1)
        return ClinicalArtifact(
            artifact_id=str(row["record_id"]),
            reference_context=ref_ctx,
            candidate_response=cand_resp,
            metadata=meta,
        )
    if dataset == "medec-ms":
        if "\n\n[CLINICAL_NOTE]\n" not in text:
            raise ValueError(f"Unexpected medec-ms formatting for {row.get('record_id')}")
        _, note = text.split("\n\n[CLINICAL_NOTE]\n", 1)
        return ClinicalArtifact(
            artifact_id=str(row["record_id"]),
            clinical_note=note,
            metadata=meta,
        )
    raise ValueError(f"Unsupported dataset_name: {dataset}")


_WORKER_CLF: Tier1ONNXClassifier | None = None


def _worker_init(
    onnx_path: str,
    tok_path: str,
    manifest_path: str,
    max_length: int,
    intra_op_threads: int,
) -> None:
    """Initialize a worker-local Tier1ONNXClassifier with zero PyTorch."""
    global _WORKER_CLF
    _WORKER_CLF = Tier1ONNXClassifier(
        onnx_model_path=onnx_path,
        tokenizer=tok_path,
        calibration_manifest=manifest_path,
        max_length=max_length,
        intra_op_num_threads=intra_op_threads,
    )
    assert "torch" not in sys.modules, "torch imported inside worker!"


def _worker_predict(text: str) -> np.ndarray:
    """Run exact single-item unpadded Tier1ONNXClassifier.predict_logits."""
    assert _WORKER_CLF is not None
    return _WORKER_CLF.predict_logits(text)


def _canonical_onnx_logits(
    tokenizer_path: Path,
    texts: list[str],
    max_length: int = 1024,
    workers: int | None = None,
    intra_op_threads: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Run exact single-item unpadded Tier1ONNXClassifier inference (parallelized across processes).

    Note: Batch-padded ONNX execution on dynamic INT8 FFN weights diverges by up to ~0.058
    in calibrated probabilities vs single-item unpadded production inference in
    Tier1ONNXClassifier.predict_logits; using exact single-item unpadded execution guarantees
    <1e-5 (0.0) parity with EvaluationRouter.evaluate(...).
    """
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.enable_truncation(max_length=max_length)
    tok.no_padding()
    encs = tok.encode_batch(texts)
    token_lens = np.array([len(e.ids) for e in encs], dtype=np.int64)

    n_workers = workers or min(4, max(1, (os.cpu_count() or 4) // 4))
    t0 = time.perf_counter()
    logits_list: list[np.ndarray] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(
            str(ARTIFACT_DIR / "model_int8.onnx"),
            str(tokenizer_path),
            str(ARTIFACT_DIR / "calibration_manifest.json"),
            max_length,
            intra_op_threads,
        ),
    ) as ex:
        for i, logits in enumerate(ex.map(_worker_predict, texts, chunksize=16), start=1):
            logits_list.append(logits)
            if i % 400 == 0 or i == len(texts):
                elapsed = time.perf_counter() - t0
                print(
                    f"[canonical-eval] inferred {i}/{len(texts)} items ({elapsed:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )
    return np.vstack(logits_list), token_lens


def _compute_subset_metrics(probs: np.ndarray, labels: np.ndarray, entity_ids: list[str]) -> dict[str, Any]:
    """Compute PASS/FAIL Macro-F1 separately from UNKNOWN F1, 3-class Macro-F1, accuracy, ECE, Brier."""
    preds = np.argmax(probs, axis=-1)
    per_f1 = f1_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    pass_f1 = float(per_f1[0])
    fail_f1 = float(per_f1[1])
    unknown_f1 = float(per_f1[2])
    pf_macro_f1 = 0.5 * (pass_f1 + fail_f1)
    macro_3class = float(np.mean(per_f1))
    fail_recall = float(recall_score(labels, preds, labels=[1], average=None, zero_division=0)[0])
    acc = float(np.mean(preds == labels))
    confs = np.max(probs, axis=-1)
    correct = (preds == labels).astype(np.float64)

    counts = Counter(int(x) for x in labels.tolist())
    return {
        "n": int(len(labels)),
        "n_entities": int(len(set(entity_ids))),
        "label_counts": {
            "PASS": counts.get(0, 0),
            "FAIL": counts.get(1, 0),
            "UNKNOWN": counts.get(2, 0),
        },
        "accuracy": round(acc, 4),
        "pass_fail_macro_f1": round(pf_macro_f1, 4),
        "unknown_f1": round(unknown_f1, 4),
        "macro_f1_3class": round(macro_3class, 4),
        "pass_f1": round(pass_f1, 4),
        "fail_f1": round(fail_f1, 4),
        "fail_recall": round(fail_recall, 4),
        "ece": round(compute_ece(probs, labels, n_bins=10), 4),
        "brier_score": round(compute_brier_score(probs, labels), 4),
        "aurc": round(compute_aurc(confs, correct), 4),
    }


def _cluster_bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    entity_ids: list[str],
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """10,000-resample paired entity-cluster 95% bootstrap CIs using exact per-cluster sufficient stats."""
    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)
    correct = (preds == labels).astype(np.float64)

    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier_per_item = np.sum((probs - one_hot) ** 2, axis=1)

    # Map each item to one of 10 ECE bins (matching medjudge.eval.benchmarks.compute_ece)
    bins = np.linspace(0.0, 1.0, 11)
    bin_idx = np.zeros(len(labels), dtype=np.int64)
    for b in range(10):
        lo, hi = bins[b], bins[b + 1]
        mask = (confs >= lo) & (confs <= hi) if b == 0 else (confs > lo) & (confs <= hi)
        bin_idx[mask] = b

    cluster_to_idx: dict[str, list[int]] = {}
    for idx, eid in enumerate(entity_ids):
        cluster_to_idx.setdefault(eid, []).append(idx)
    clusters = list(cluster_to_idx.values())
    n_clusters = len(clusters)

    # Precompute additive per-cluster statistics
    c_counts = np.zeros(n_clusters, dtype=np.float64)
    c_confmat = np.zeros((n_clusters, 3, 3), dtype=np.float64)  # [true, pred]
    c_brier = np.zeros(n_clusters, dtype=np.float64)
    c_bin_acc = np.zeros((n_clusters, 10), dtype=np.float64)
    c_bin_conf = np.zeros((n_clusters, 10), dtype=np.float64)

    for ci, idxs in enumerate(clusters):
        c_counts[ci] = len(idxs)
        c_brier[ci] = float(np.sum(brier_per_item[idxs]))
        for i in idxs:
            c_confmat[ci, labels[i], preds[i]] += 1.0
            b = int(bin_idx[i])
            c_bin_acc[ci, b] += correct[i]
            c_bin_conf[ci, b] += confs[i]

    rng = np.random.default_rng(seed)
    # Draw cluster multiplicities (n_resamples, n_clusters) via multinomial(n_clusters, uniform)
    # Note: chunked to keep peak memory small while running in <0.2s
    pf_macro_arr = np.empty(n_resamples, dtype=np.float64)
    macro3_arr = np.empty(n_resamples, dtype=np.float64)
    unk_f1_arr = np.empty(n_resamples, dtype=np.float64)
    acc_arr = np.empty(n_resamples, dtype=np.float64)
    fail_rec_arr = np.empty(n_resamples, dtype=np.float64)
    ece_arr = np.empty(n_resamples, dtype=np.float64)
    brier_arr = np.empty(n_resamples, dtype=np.float64)

    p_uniform = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
    flat_cm = c_confmat.reshape(n_clusters, 9)

    chunk_size = 2000
    for start in range(0, n_resamples, chunk_size):
        bs = min(chunk_size, n_resamples - start)
        weights = rng.multinomial(n_clusters, p_uniform, size=bs).astype(np.float64)

        tot_n = weights @ c_counts
        cm = (weights @ flat_cm).reshape(bs, 3, 3)

        f1s = np.zeros((bs, 3), dtype=np.float64)
        for k in range(3):
            tp = cm[:, k, k]
            fp = np.sum(cm[:, :, k], axis=1) - tp
            fn = np.sum(cm[:, k, :], axis=1) - tp
            denom = 2.0 * tp + fp + fn
            f1s[:, k] = np.where(denom > 0.0, (2.0 * tp) / np.where(denom > 0.0, denom, 1.0), 0.0)

        pf_macro_arr[start : start + bs] = 0.5 * (f1s[:, 0] + f1s[:, 1])
        macro3_arr[start : start + bs] = np.mean(f1s, axis=1)
        unk_f1_arr[start : start + bs] = f1s[:, 2]

        diag_sum = cm[:, 0, 0] + cm[:, 1, 1] + cm[:, 2, 2]
        acc_arr[start : start + bs] = diag_sum / tot_n

        fail_total = np.sum(cm[:, 1, :], axis=1)
        fail_rec_arr[start : start + bs] = np.where(
            fail_total > 0.0, cm[:, 1, 1] / np.where(fail_total > 0.0, fail_total, 1.0), 0.0
        )

        brier_arr[start : start + bs] = (weights @ c_brier) / tot_n

        b_acc = weights @ c_bin_acc
        b_conf = weights @ c_bin_conf
        ece_arr[start : start + bs] = np.sum(np.abs(b_acc - b_conf), axis=1) / tot_n

    def _ci(arr: np.ndarray) -> dict[str, float]:
        return {
            "low": round(float(np.percentile(arr, 2.5)), 4),
            "high": round(float(np.percentile(arr, 97.5)), 4),
        }

    return {
        "n_resamples": n_resamples,
        "n_entity_clusters": n_clusters,
        "pass_fail_macro_f1_95ci": _ci(pf_macro_arr),
        "unknown_f1_95ci": _ci(unk_f1_arr),
        "macro_f1_3class_95ci": _ci(macro3_arr),
        "accuracy_95ci": _ci(acc_arr),
        "fail_recall_95ci": _ci(fail_rec_arr),
        "ece_95ci": _ci(ece_arr),
        "brier_score_95ci": _ci(brier_arr),
    }


def _evaluate_gate_mask(
    probs: np.ndarray,
    labels: np.ndarray,
    accepted: np.ndarray,
    pred_pass: np.ndarray,
) -> dict[str, Any]:
    n = len(labels)
    n_acc = int(np.sum(accepted))
    n_esc = n - n_acc
    accepted_preds = np.where(pred_pass[accepted], 0, 1)
    acc_accuracy = round(float(np.mean(accepted_preds == labels[accepted])), 4) if n_acc > 0 else None

    false_pass_fail = int(np.sum(accepted & pred_pass & (labels == 1)))
    false_pass_non_pass = int(np.sum(accepted & pred_pass & (labels != 0)))
    n_true_fails = int(np.sum(labels == 1))

    return {
        "n_total": n,
        "accepted_count": n_acc,
        "escalated_count": n_esc,
        "accepted_coverage_pct": round(100.0 * n_acc / n, 2),
        "escalation_rate_pct": round(100.0 * n_esc / n, 2),
        "accepted_accuracy": acc_accuracy,
        "accepted_false_pass_count": false_pass_fail,
        "accepted_false_pass_rate_pct": (
            round(100.0 * false_pass_fail / n_acc, 4) if n_acc > 0 else None
        ),
        "accepted_false_pass_pct_of_all": round(100.0 * false_pass_fail / n, 4),
        "accepted_false_pass_pct_of_true_fails": (
            round(100.0 * false_pass_fail / n_true_fails, 4) if n_true_fails > 0 else None
        ),
        "accepted_non_pass_error_count": false_pass_non_pass,
        "accepted_non_pass_error_rate_pct": (
            round(100.0 * false_pass_non_pass / n_acc, 4) if n_acc > 0 else None
        ),
    }


def _risk_coverage_report(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Compute symmetric tau sweep AND deployed asymmetric S2 gate (0.92 / 0.90)."""
    preds = np.argmax(probs, axis=-1)
    p_pass = probs[:, 0]
    p_fail = probs[:, 1]
    pf_max = np.maximum(p_pass, p_fail)

    symmetric_rows: list[dict[str, Any]] = []
    for tau in TAU_GRID:
        accepted = (preds != 2) & (pf_max >= tau)
        pred_pass = p_pass >= p_fail
        row = {"tau": tau, **_evaluate_gate_mask(probs, labels, accepted, pred_pass)}
        symmetric_rows.append(row)

    # Deployed asymmetric S2 gate matching EvaluationRouter._meets_asymmetric_gate
    asym_pass = (preds == 0) & (p_pass >= 0.92)
    asym_fail = (preds == 1) & (p_fail >= 0.90)
    asym_accepted = asym_pass | asym_fail
    asym_row = {
        "pass_threshold": 0.92,
        "fail_threshold": 0.90,
        **_evaluate_gate_mask(probs, labels, asym_accepted, asym_pass),
    }

    return {
        "symmetric_tau_sweep": symmetric_rows,
        "deployed_asymmetric_s2_gate": asym_row,
    }


def _benchmark_cpu_latency(
    classifier: Tier1ONNXClassifier,
    seq_lens: tuple[int, ...] = (256, 512, 1024),
) -> dict[str, dict[str, float]]:
    """Measure fresh single-item CPU ONNX latency at fixed sequence lengths."""
    results: dict[str, dict[str, float]] = {}
    for seq_len in seq_lens:
        ids = np.ones((1, seq_len), dtype=np.int64)
        mask = np.ones((1, seq_len), dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        for _ in range(5):
            classifier.session.run(None, feeds)
        times_ms: list[float] = []
        for _ in range(20):
            t0 = time.perf_counter()
            classifier.session.run(None, feeds)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(times_ms, dtype=np.float64)
        results[str(seq_len)] = {
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
        }
    return results


async def _run_router_parity_check(
    router: EvaluationRouter,
    rows: list[dict[str, Any]],
    criteria: list[RubricCriterion],
    artifacts: list[ClinicalArtifact],
    batched_probs: np.ndarray,
    n_items: int = 20,
) -> dict[str, Any]:
    """Verify exact numerical and routing parity between batched ONNX and EvaluationRouter.evaluate."""
    # Deterministic stratified selection across subsets & labels
    chosen_indices: list[int] = []
    buckets: dict[tuple[str, str | None, str], list[int]] = {}
    for i, r in enumerate(rows):
        key = (str(r["dataset_name"]), r.get("metadata", {}).get("config"), str(r["label"]))
        buckets.setdefault(key, []).append(i)
    while len(chosen_indices) < n_items:
        added = False
        for key in sorted(buckets):
            if buckets[key]:
                chosen_indices.append(buckets[key].pop(0))
                added = True
                if len(chosen_indices) == n_items:
                    break
        if not added:
            break

    max_prob_diff = 0.0
    gate_matches = 0
    details: list[dict[str, Any]] = []
    for idx in chosen_indices:
        dec = await router.evaluate(criteria[idx], artifacts[idx])
        p_router = np.array(
            [
                dec.probabilities["PASS"],
                dec.probabilities["FAIL"],
                dec.probabilities["UNKNOWN"],
            ],
            dtype=np.float64,
        )
        p_batch = batched_probs[idx]
        diff = float(np.max(np.abs(p_router - p_batch)))
        max_prob_diff = max(max_prob_diff, diff)

        b_pred = int(np.argmax(p_batch))
        b_accepted = (b_pred == 0 and p_batch[0] >= 0.92) or (b_pred == 1 and p_batch[1] >= 0.90)
        r_accepted = dec.escalation_reason is None
        expected_verdict = ORDER[b_pred].value if b_accepted else Verdict.UNKNOWN.value
        matched = (b_accepted == r_accepted) and (dec.verdict.value == expected_verdict)
        if matched:
            gate_matches += 1

        details.append(
            {
                "row_index": idx,
                "record_id": rows[idx]["record_id"],
                "dataset_name": rows[idx]["dataset_name"],
                "gold_label": rows[idx]["label"],
                "max_abs_prob_diff": round(diff, 9),
                "router_verdict": dec.verdict.value,
                "router_accepted": r_accepted,
                "gate_parity_ok": matched,
            }
        )

    assert max_prob_diff < 1e-5, f"Router parity failed: max_prob_diff={max_prob_diff}"
    assert gate_matches == len(chosen_indices), (
        f"Gate parity mismatch: {gate_matches}/{len(chosen_indices)}"
    )
    return {
        "n_checked": len(chosen_indices),
        "max_abs_prob_diff": max_prob_diff,
        "passed_1e_5_tol": max_prob_diff < 1e-5,
        "gate_parity_matches": gate_matches,
        "sample_checks": details[:5],
    }


def main() -> int:
    assert "torch" not in sys.modules, "torch must NOT be imported before canonical eval!"

    registry = load_default_rubrics()
    s2_medhallu = registry.get("s2_medhallu_grounding")
    s2_medec_prod = registry.get("s2_medec_note_error_flag")
    # Create an isolated copy with force_tier2=False to measure standalone Tier 1 on MEDEC-MS
    medec_payload = s2_medec_prod.model_dump(exclude={"sha256"})
    medec_payload["force_tier2"] = False
    s2_medec_eval = RubricCriterion.model_validate(medec_payload)

    slicer = PinnedHeaderSlicer(max_tokens=1024)
    tier1 = Tier1ONNXClassifier(
        onnx_model_path=ARTIFACT_DIR / "model_int8.onnx",
        tokenizer=ARTIFACT_DIR / "tokenizer.json",
        calibration_manifest=ARTIFACT_DIR / "calibration_manifest.json",
        max_length=1024,
    )
    router = EvaluationRouter(slicer=slicer, tier1=tier1)

    rows = [json.loads(ln) for ln in D_VAL_PATH.open("r", encoding="utf-8") if ln.strip()]
    criteria: list[RubricCriterion] = []
    artifacts: list[ClinicalArtifact] = []
    sliced_texts: list[str] = []
    labels = np.empty(len(rows), dtype=np.int64)
    entity_ids: list[str] = []

    exact_text_matches = 0
    capsule_omitted_count = 0
    sufficient_count = 0
    sliced_overflow_count = 0

    for i, row in enumerate(rows):
        criterion = s2_medhallu if row["dataset_name"] == "medhallu" else s2_medec_eval
        artifact = _reconstruct_artifact(row)
        sliced = slicer.slice(criterion, artifact)

        criteria.append(criterion)
        artifacts.append(artifact)
        sliced_texts.append(sliced.formatted_text)
        labels[i] = LABEL_MAP[row["label"]]
        entity_ids.append(str(row["entity_id"]))

        if sliced.formatted_text == row["text"]:
            exact_text_matches += 1
        if "[GLOBAL_CONTEXT_CAPSULE]" not in sliced.formatted_text:
            capsule_omitted_count += 1
        if sliced.sufficiency.sufficient:
            sufficient_count += 1
        if sliced.sliced:
            sliced_overflow_count += 1

    logits, token_lens = _canonical_onnx_logits(
        ARTIFACT_DIR / "tokenizer.json",
        sliced_texts,
        max_length=1024,
        workers=4,
        intra_op_threads=4,
    )
    probs = apply_temperature(logits, tier1.manifest.temperature)

    parity = asyncio.run(
        _run_router_parity_check(router, rows, criteria, artifacts, probs, n_items=20)
    )

    def _idx(pred_fn: Any) -> np.ndarray:
        return np.array([i for i, r in enumerate(rows) if pred_fn(r)], dtype=np.int64)

    subsets = {
        "d_val_overall": _idx(lambda _: True),
        "medhallu_overall": _idx(lambda r: r["dataset_name"] == "medhallu"),
        "medhallu_pqa_labeled": _idx(
            lambda r: r["dataset_name"] == "medhallu"
            and r.get("metadata", {}).get("config") == "pqa_labeled"
        ),
        "medhallu_pqa_artificial": _idx(
            lambda r: r["dataset_name"] == "medhallu"
            and r.get("metadata", {}).get("config") == "pqa_artificial"
        ),
        "medec_ms": _idx(lambda r: r["dataset_name"] == "medec-ms"),
    }

    metrics_by_split: dict[str, Any] = {}
    for name, idx in subsets.items():
        sub_probs = probs[idx]
        sub_labels = labels[idx]
        sub_eids = [entity_ids[int(j)] for j in idx]
        metrics_by_split[name] = {
            **_compute_subset_metrics(sub_probs, sub_labels, sub_eids),
            "bootstrap_95ci": _cluster_bootstrap_ci(
                sub_probs, sub_labels, sub_eids, n_resamples=10000, seed=42
            ),
        }

    risk_coverage = {
        "d_val_overall": _risk_coverage_report(probs, labels),
        "medhallu_only": _risk_coverage_report(
            probs[subsets["medhallu_overall"]], labels[subsets["medhallu_overall"]]
        ),
    }

    fresh_latency = _benchmark_cpu_latency(tier1, seq_lens=(256, 512, 1024))
    export_latency = (
        json.loads(EXPORT_BENCHMARK_PATH.read_text(encoding="utf-8")).get("cpu_latency_ms", {})
        if EXPORT_BENCHMARK_PATH.exists()
        else {}
    )
    cpu_info = _get_cpu_info()

    assert "torch" not in sys.modules, "torch was imported during canonical evaluation!"

    report = {
        "pipeline": {
            "slicer": "PinnedHeaderSlicer(max_tokens=1024)",
            "classifier": "Tier1ONNXClassifier(artifacts/bioclinical_modernbert_int8/model_int8.onnx)",
            "temperature_T_star": tier1.manifest.temperature,
            "s2_medhallu_rubric_id": s2_medhallu.id,
            "s2_medhallu_task_type": getattr(s2_medhallu, "task_type", "unknown"),
            "s2_medec_rubric_id": s2_medec_prod.id,
            "s2_medec_task_type": getattr(s2_medec_prod, "task_type", "unknown"),
            "s2_medec_production_force_tier2": bool(getattr(s2_medec_prod, "force_tier2", False)),
            "s2_medec_eval_override_note": (
                "Evaluated MEDEC-MS d_val rows using a copy of s2_medec_note_error_flag with "
                "force_tier2=False to measure standalone Tier 1 performance; in production, "
                "s2_medec_note_error_flag has force_tier2=True and routes directly to Tier 2."
            ),
            "torch_in_sys_modules": "torch" in sys.modules,
        },
        "slicer_diagnostics": {
            "n_rows": len(rows),
            "exact_text_matches": exact_text_matches,
            "capsule_omitted_count": capsule_omitted_count,
            "sufficient_count": sufficient_count,
            "sliced_overflow_count": sliced_overflow_count,
            "max_token_len": int(np.max(token_lens)),
            "mean_token_len": round(float(np.mean(token_lens)), 2),
        },
        "router_parity_check": parity,
        "metrics": metrics_by_split,
        "risk_coverage": risk_coverage,
        "hardware_and_latency": {
            **cpu_info,
            "fresh_cpu_latency_ms": fresh_latency,
            "export_benchmark_cpu_latency_ms": export_latency,
        },
    }

    OUT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
