"""Per-dataset d_val breakdown and sealed MEDEC-MS Test (597) holdout evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score, recall_score
from tokenizers import Tokenizer

from medjudge.eval.benchmarks import (
    compute_aurc,
    compute_brier_score,
    compute_ece,
    evaluate_six_pillars,
)
from medjudge.eval.plots import plot_pareto_frontier, plot_risk_coverage_curve
from medjudge.schemas.models import EvalTier, GradeDecision, RiskClass, Verdict
from medjudge.tier1.calibration import CalibrationManifest, apply_temperature

ARTIFACT_DIR = Path("artifacts/bioclinical_modernbert_int8")
HOLDOUT_PATH = Path("data/holdout/medec_ms_test.jsonl")
D_VAL_PATH = Path("data/processed/d_val.jsonl")
PLOTS_DIR = ARTIFACT_DIR / "eval_plots"

LABEL_MAP = {Verdict.PASS.value: 0, Verdict.FAIL.value: 1, Verdict.UNKNOWN.value: 2}
ORDER = (Verdict.PASS, Verdict.FAIL, Verdict.UNKNOWN)


def _format_medec_note(note_text: str) -> str:
    return (
        "[RUBRIC_CRITERION]\n"
        "ID: s2_medec_note_error_flag\n"
        "Prompt: Is the clinical note free of localized factual, diagnostic, medication, or management errors?\n\n"
        f"[CLINICAL_NOTE]\n{note_text.strip()}"
    )


def _length_sorted_onnx_infer(
    sess: ort.InferenceSession,
    tokenizer: Tokenizer,
    texts: list[str],
    max_length: int = 1024,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run CPU ONNX inference using length-sorted batching and return (logits, token_lens, per_item_ms)."""
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.no_padding()

    encs = tokenizer.encode_batch(texts)
    token_lens = np.array([len(e.ids) for e in encs], dtype=np.int64)
    order = np.argsort(token_lens)

    logits_out = np.zeros((len(texts), 3), dtype=np.float64)
    per_item_ms = [0.0] * len(texts)

    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        max_len = max(int(token_lens[i]) for i in batch_idx)
        max_len = ((max_len + 7) // 8) * 8
        ids = np.full((len(batch_idx), max_len), pad_id, dtype=np.int64)
        mask = np.zeros((len(batch_idx), max_len), dtype=np.int64)
        for row_i, orig_i in enumerate(batch_idx):
            n = int(token_lens[orig_i])
            ids[row_i, :n] = encs[orig_i].ids
            mask[row_i, :n] = encs[orig_i].attention_mask
        t0 = time.perf_counter()
        out = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        batch_ms = ((time.perf_counter() - t0) * 1000.0) / len(batch_idx)
        out_arr = np.asarray(out, dtype=np.float64)
        for row_i, orig_i in enumerate(batch_idx):
            logits_out[orig_i] = out_arr[row_i]
            per_item_ms[orig_i] = batch_ms

    return logits_out, token_lens, per_item_ms


def _compute_subset_metrics(probs: np.ndarray, labels: np.ndarray, include_unknown: bool = True) -> dict[str, Any]:
    preds = np.argmax(probs, axis=-1)
    active_labels = [0, 1, 2] if include_unknown and (2 in set(labels.tolist())) else [0, 1]
    per_f1 = f1_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    macro_f1 = float(np.mean([per_f1[i] for i in active_labels]))
    fail_recall = float(recall_score(labels, preds, labels=[1], average=None, zero_division=0)[0])
    acc = float(np.mean(preds == labels))
    return {
        "n": int(len(labels)),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "pass_f1": round(float(per_f1[0]), 4),
        "fail_f1": round(float(per_f1[1]), 4),
        "unknown_f1": round(float(per_f1[2]), 4) if 2 in active_labels else None,
        "fail_recall": round(fail_recall, 4),
        "ece": round(compute_ece(probs, labels), 4),
        "brier_score": round(compute_brier_score(probs, labels), 4),
    }


def _cluster_bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    cluster_ids: list[str],
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """10,000-resample paired source-cluster 95% bootstrap CIs."""
    preds = np.argmax(probs, axis=-1)
    cluster_to_indices: dict[str, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        cluster_to_indices.setdefault(cid, []).append(idx)
    clusters = list(cluster_to_indices.values())
    n_clusters = len(clusters)

    rng = np.random.default_rng(seed)
    macro_f1s = np.empty(n_resamples, dtype=np.float64)
    fail_recalls = np.empty(n_resamples, dtype=np.float64)
    accuracies = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.array([i for c in sampled for i in clusters[c]], dtype=np.int64)
        y_b = labels[idx]
        p_b = preds[idx]
        tp_p = float(np.sum((p_b == 0) & (y_b == 0)))
        fp_p = float(np.sum((p_b == 0) & (y_b != 0)))
        fn_p = float(np.sum((p_b != 0) & (y_b == 0)))
        f1_p = (2 * tp_p / (2 * tp_p + fp_p + fn_p)) if (2 * tp_p + fp_p + fn_p) > 0 else 0.0

        tp_f = float(np.sum((p_b == 1) & (y_b == 1)))
        fp_f = float(np.sum((p_b == 1) & (y_b != 1)))
        fn_f = float(np.sum((p_b != 1) & (y_b == 1)))
        f1_f = (2 * tp_f / (2 * tp_f + fp_f + fn_f)) if (2 * tp_f + fp_f + fn_f) > 0 else 0.0

        macro_f1s[b] = 0.5 * (f1_p + f1_f)
        fail_recalls[b] = (tp_f / (tp_f + fn_f)) if (tp_f + fn_f) > 0 else 0.0
        accuracies[b] = float(np.mean(p_b == y_b))

    return {
        "macro_f1_95ci": {
            "low": round(float(np.percentile(macro_f1s, 2.5)), 4),
            "high": round(float(np.percentile(macro_f1s, 97.5)), 4),
        },
        "fail_recall_95ci": {
            "low": round(float(np.percentile(fail_recalls, 2.5)), 4),
            "high": round(float(np.percentile(fail_recalls, 97.5)), 4),
        },
        "accuracy_95ci": {
            "low": round(float(np.percentile(accuracies, 2.5)), 4),
            "high": round(float(np.percentile(accuracies, 97.5)), 4),
        },
    }


def _cascade_gate_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    pass_thresh: float,
    fail_thresh: float,
) -> dict[str, Any]:
    n = len(labels)
    auto_pass = probs[:, 0] >= pass_thresh
    auto_fail = (~auto_pass) & (probs[:, 1] >= fail_thresh)
    accepted = auto_pass | auto_fail
    escalated = ~accepted

    n_acc = int(np.sum(accepted))
    n_esc = int(np.sum(escalated))
    true_fails = labels == 1
    catastrophic_fp = int(np.sum(auto_pass & true_fails))

    # Cascade outcome assuming Tier 2 resolves escalated items accurately
    cascade_preds = np.where(auto_pass, 0, np.where(auto_fail, 1, labels))
    cascade_f1 = f1_score(labels, cascade_preds, labels=[0, 1], average="macro", zero_division=0)
    cascade_fail_recall = recall_score(labels, cascade_preds, labels=[1], average=None, zero_division=0)[0]

    acc_subset_accuracy = (
        float(np.mean(np.where(auto_pass[accepted], 0, 1) == labels[accepted]))
        if n_acc > 0
        else 0.0
    )

    return {
        "pass_threshold": pass_thresh,
        "fail_threshold": fail_thresh,
        "tier1_coverage_pct": round(100.0 * n_acc / n, 2),
        "tier2_escalation_pct": round(100.0 * n_esc / n, 2),
        "accepted_count": n_acc,
        "escalated_count": n_esc,
        "accepted_subset_accuracy": round(acc_subset_accuracy, 4),
        "catastrophic_false_pass_count": catastrophic_fp,
        "catastrophic_false_pass_pct_of_all": round(100.0 * catastrophic_fp / n, 2),
        "catastrophic_false_pass_pct_of_fails": round(
            100.0 * catastrophic_fp / max(1, int(np.sum(true_fails))), 2
        ),
        "cascade_macro_f1_with_tier2": round(float(cascade_f1), 4),
        "cascade_fail_recall_with_tier2": round(float(cascade_fail_recall), 4),
    }


def main() -> int:
    manifest = CalibrationManifest.load(ARTIFACT_DIR / "calibration_manifest.json")
    tokenizer = Tokenizer.from_file(str(ARTIFACT_DIR / "tokenizer.json"))
    sess = ort.InferenceSession(
        str(ARTIFACT_DIR / "model_int8.onnx"),
        providers=["CPUExecutionProvider"],
    )

    # 1. Per-dataset d_val breakdown
    val_rows = [json.loads(ln) for ln in D_VAL_PATH.open("r", encoding="utf-8") if ln.strip()]
    val_texts = [r["text"] for r in val_rows]
    val_labels = np.array([LABEL_MAP[r["label"]] for r in val_rows], dtype=np.int64)
    val_logits, val_lens, _ = _length_sorted_onnx_infer(sess, tokenizer, val_texts)
    val_probs = apply_temperature(val_logits, manifest.temperature)

    np.savez(
        ARTIFACT_DIR / "d_val_int8_cache.npz",
        logits=val_logits,
        probs=val_probs,
        labels=val_labels,
        token_lens=val_lens,
    )

    def _subset_idx(pred_fn: Any) -> np.ndarray:
        return np.array([i for i, r in enumerate(val_rows) if pred_fn(r)], dtype=np.int64)

    idx_medec = _subset_idx(lambda r: r["dataset_name"] == "medec-ms")
    idx_mh_lab = _subset_idx(
        lambda r: r["dataset_name"] == "medhallu" and r.get("metadata", {}).get("config") == "pqa_labeled"
    )
    idx_mh_art = _subset_idx(
        lambda r: r["dataset_name"] == "medhallu" and r.get("metadata", {}).get("config") == "pqa_artificial"
    )

    d_val_breakdown = {
        "overall": _compute_subset_metrics(val_probs, val_labels, include_unknown=True),
        "medec_ms": _compute_subset_metrics(val_probs[idx_medec], val_labels[idx_medec], include_unknown=True),
        "medhallu_pqa_labeled": _compute_subset_metrics(
            val_probs[idx_mh_lab], val_labels[idx_mh_lab], include_unknown=True
        ),
        "medhallu_pqa_artificial": _compute_subset_metrics(
            val_probs[idx_mh_art], val_labels[idx_mh_art], include_unknown=True
        ),
    }

    # 2. Sealed MEDEC-MS Test (597) Evaluation (read-only)
    holdout_rows = [json.loads(ln) for ln in HOLDOUT_PATH.open("r", encoding="utf-8") if ln.strip()]
    holdout_texts = [_format_medec_note(str(r["text"])) for r in holdout_rows]
    holdout_labels = np.array(
        [1 if str(r["error_flag"]).strip().lower() == "true" else 0 for r in holdout_rows],
        dtype=np.int64,
    )
    cluster_ids = [str(r["text_id"]) for r in holdout_rows]

    h_logits, h_lens, h_latencies = _length_sorted_onnx_infer(sess, tokenizer, holdout_texts)
    h_probs = apply_temperature(h_logits, manifest.temperature)
    h_preds = np.argmax(h_probs, axis=-1)

    standalone_metrics = _compute_subset_metrics(h_probs, holdout_labels, include_unknown=False)
    bootstrap_cis = _cluster_bootstrap_ci(h_probs, holdout_labels, cluster_ids, n_resamples=10000)

    decisions: list[GradeDecision] = []
    gold_verdicts: list[Verdict] = []
    risk_classes: list[RiskClass] = []
    for i in range(len(holdout_rows)):
        p_row = h_probs[i]
        pred_v = ORDER[int(h_preds[i])]
        decisions.append(
            GradeDecision(
                criterion_id="s2_medec_note_error_flag",
                criterion_version="1.0.0",
                criterion_sha256="holdout",
                verdict=pred_v,
                probabilities={"PASS": float(p_row[0]), "FAIL": float(p_row[1]), "UNKNOWN": float(p_row[2])},
                calibrated_confidence=float(np.max(p_row)),
                tier_used=EvalTier.TIER1_LOCAL_ONNX,
                latency_ms=h_latencies[i],
            )
        )
        gold_verdicts.append(ORDER[int(holdout_labels[i])])
        risk_classes.append(RiskClass.S2_CLINICAL_FACTUALITY)

    six_pillars = evaluate_six_pillars(decisions, gold_verdicts, risk_classes)

    s1_thresh = manifest.thresholds_for(RiskClass.S1_CRITICAL_SAFETY)
    s2_thresh = manifest.thresholds_for(RiskClass.S2_CLINICAL_FACTUALITY)
    cascade_s1 = _cascade_gate_metrics(
        h_probs, holdout_labels, s1_thresh.pass_threshold, s1_thresh.fail_threshold
    )
    cascade_s2 = _cascade_gate_metrics(
        h_probs, holdout_labels, s2_thresh.pass_threshold, s2_thresh.fail_threshold
    )

    # Plots: Risk-Coverage and Cost-vs-Accuracy Pareto
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    confs = np.max(h_probs, axis=-1)
    correct = (h_preds == holdout_labels).astype(np.float64)
    aurc = compute_aurc(confs, correct)
    rc_plot = plot_risk_coverage_curve(
        confs, correct, PLOTS_DIR / "risk_coverage_medec_ms_test.png"
    )
    pareto_plot = plot_pareto_frontier(
        latencies_ms=[
            float(np.mean(h_latencies)),
            float(np.mean(h_latencies)) + 1200.0 * (cascade_s2["tier2_escalation_pct"] / 100.0),
            float(np.mean(h_latencies)) + 1200.0 * (cascade_s1["tier2_escalation_pct"] / 100.0),
            1200.0,
        ],
        accuracies=[
            standalone_metrics["accuracy"],
            round(1.0 - (cascade_s2["catastrophic_false_pass_count"] / len(holdout_rows)), 4),
            round(1.0 - (cascade_s1["catastrophic_false_pass_count"] / len(holdout_rows)), 4),
            1.0,
        ],
        labels=["Tier 1 Standalone", "S2 Cascade", "S1 Cascade", "Full Tier 2"],
        output_path=PLOTS_DIR / "pareto_medec_ms_test.png",
    )

    report = {
        "d_val_breakdown": d_val_breakdown,
        "medec_ms_test": {
            "n_notes": len(holdout_rows),
            "true_pass": int(np.sum(holdout_labels == 0)),
            "true_fail": int(np.sum(holdout_labels == 1)),
            "standalone": standalone_metrics,
            "bootstrap_95ci": bootstrap_cis,
            "aurc": round(aurc, 4),
            "six_pillars_summary": six_pillars.model_dump(),
            "cascade_s1_gate": cascade_s1,
            "cascade_s2_gate": cascade_s2,
            "plots": {
                "risk_coverage": str(rc_plot),
                "pareto": str(pareto_plot),
            },
        },
    }
    (ARTIFACT_DIR / "holdout_eval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
