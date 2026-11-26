"""Multi-teacher distillation, minimal-pair hard negatives, MNLI filter, and Style-Only probe."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize

from medjudge.schemas.models import Verdict

DISQUALIFIED_TEACHER_PATTERNS = ("openai/", "anthropic/", "google/gemini", "gpt-4", "claude")


class DistilledExample(BaseModel):
    """Structured multi-teacher distilled training item."""

    example_id: str
    criterion_id: str
    text: str
    verdict: Verdict
    evidence_span: str = ""
    error_type: str = "none"
    sufficient: bool = True
    teacher_model: str


class EntailmentCheckerProtocol(Protocol):
    """Protocol for bidirectional MNLI entailment scoring (e.g., deberta-large-mnli)."""

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        ...


def validate_teacher_models(teacher_models: list[str]) -> None:
    """Require >= 2 permissive open-weight teachers (e.g., Qwen + DeepSeek)."""
    unique = [m.strip() for m in dict.fromkeys(teacher_models) if m.strip()]
    if len(unique) < 2:
        raise ValueError(
            "Multi-teacher distillation requires at least 2 distinct open-weight teacher models."
        )
    for model in unique:
        lower = model.lower()
        if any(pat in lower for pat in DISQUALIFIED_TEACHER_PATTERNS):
            raise ValueError(
                f"Teacher '{model}' violates permissive open-weight distillation policy."
            )


def generate_minimal_pair_hard_negative(
    positive_text: str,
    mutation_type: str = "dosage_slip",
) -> tuple[str, str] | None:
    """Create a +-10% length realistic clinical counterfactual hard negative."""
    orig_len = max(1, len(positive_text))
    mutated = positive_text
    err_label = mutation_type

    if mutation_type == "dosage_slip":
        # Realistic 10x or unit slip: e.g. 0.5 mg -> 5.0 mg or mg -> mcg
        if re.search(r"\b0\.5\s*mg\b", mutated, flags=re.IGNORECASE):
            mutated = re.sub(r"\b0\.5\s*mg\b", "5.0 mg", mutated, count=1, flags=re.IGNORECASE)
        elif re.search(r"\bmg\b", mutated):
            mutated = re.sub(r"\bmg\b", "mcg", mutated, count=1)
        elif re.search(r"\b(\d+(?:\.\d+)?)\s*(mg|ml|units?)\b", mutated, flags=re.IGNORECASE):
            mutated = re.sub(
                r"\b(\d+(?:\.\d+)?)\s*(mg|ml|units?)\b",
                lambda m: f"{float(m.group(1)) * 10:g} {m.group(2)}",
                mutated,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            return None
    elif mutation_type == "pvalue_swap":
        if "p < 0.05" in mutated:
            mutated = mutated.replace("p < 0.05", "p = 0.42", 1)
        elif "95% CI" in mutated:
            mutated = mutated.replace("95% CI", "95% CI (crossing 1.0)", 1)
        else:
            return None
    elif mutation_type == "omit_red_flag":
        new_txt = re.sub(
            r"\b(?:call\s+911|go\s+to\s+the\s+emergency\s+room|seek\s+immediate\s+emergency\s+care)\b",
            "monitor symptoms at home for now",
            mutated,
            count=1,
            flags=re.IGNORECASE,
        )
        if new_txt == mutated:
            return None
        mutated = new_txt
    elif mutation_type == "unsupported_finding":
        mutated = mutated.rstrip() + " Patient also has confirmed pulmonary embolism on CT."
    elif mutation_type == "overconfident_unanswerable":
        mutated = mutated.rstrip() + " Diagnosis is 100% certain without further testing."
    else:
        raise ValueError(f"Unknown mutation_type '{mutation_type}'")

    if mutated == positive_text:
        return None
    length_ratio = abs(len(mutated) - orig_len) / orig_len
    if length_ratio > 0.15 and mutation_type in {"dosage_slip", "pvalue_swap"}:
        return None
    return mutated, err_label


def filter_bidirectional_entailment(
    positive_text: str,
    candidate_negative_text: str,
    checker: EntailmentCheckerProtocol,
    tau: float = 0.75,
) -> bool:
    """Discard harmless paraphrases where bidirectional MNLI entailment >= tau."""
    fwd = checker.entailment_score(positive_text, candidate_negative_text)
    bwd = checker.entailment_score(candidate_negative_text, positive_text)
    return min(fwd, bwd) < tau


def strip_context_for_unknown(
    examples: list[DistilledExample],
    unknown_fraction: float = 0.10,
    seed: int = 42,
) -> list[DistilledExample]:
    """Strip decisive context from ~10% of positives and relabel them UNKNOWN."""
    rng = random.Random(seed)
    positives = [e for e in examples if e.verdict == Verdict.PASS]
    target_n = max(1 if positives and unknown_fraction > 0 else 0, int(round(len(positives) * unknown_fraction)))
    chosen_ids = {e.example_id for e in rng.sample(positives, min(target_n, len(positives)))}

    out: list[DistilledExample] = []
    for ex in examples:
        if ex.example_id in chosen_ids:
            stripped = re.sub(
                r"\[GLOBAL_CONTEXT_CAPSULE\].*?(?=\[CANDIDATE_RESPONSE\]|\Z)",
                "[GLOBAL_CONTEXT_CAPSULE]\nAllergies: UNKNOWN\nActive Medications: UNKNOWN\n",
                ex.text,
                flags=re.DOTALL,
            )
            if stripped == ex.text:
                stripped = "[CONTEXT_OMITTED] " + ex.text
            out.append(
                ex.model_copy(
                    update={
                        "example_id": f"{ex.example_id}_unk",
                        "text": stripped,
                        "verdict": Verdict.UNKNOWN,
                        "sufficient": False,
                        "error_type": "missing_decisive_context",
                    }
                )
            )
        else:
            out.append(ex)
    return out


class StyleProbeResult(BaseModel):
    """Result from bag-of-n-grams + length style-only shortcut probe."""

    cv_accuracy: float
    cv_roc_auc: float
    passed: bool
    reason: str
    top_positive_features: list[tuple[str, float]] = Field(default_factory=list)
    top_negative_features: list[tuple[str, float]] = Field(default_factory=list)


def _compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / (len(pos) * len(neg))


def run_style_only_probe(
    texts: list[str],
    labels: list[Verdict],
    max_auc: float = 0.62,
    n_splits: int = 5,
    seed: int = 42,
    entity_ids: list[str] | None = None,
) -> StyleProbeResult:
    """Bag-of-n-grams + length logistic regression probe; fails if shortcuts separate PASS vs FAIL."""
    if entity_ids is not None:
        pairs = [
            (t, lbl, eid)
            for t, lbl, eid in zip(texts, labels, entity_ids, strict=True)
            if lbl in (Verdict.PASS, Verdict.FAIL)
        ]
        filtered = [(t, lbl) for t, lbl, _ in pairs]
        filtered_entities: list[str] | None = [eid for _, _, eid in pairs]
    else:
        filtered = [
            (t, lbl)
            for t, lbl in zip(texts, labels, strict=True)
            if lbl in (Verdict.PASS, Verdict.FAIL)
        ]
        filtered_entities = None

    if len(filtered) < 10:
        return StyleProbeResult(
            cv_accuracy=0.5,
            cv_roc_auc=0.5,
            passed=True,
            reason="Too few PASS/FAIL samples for style probe.",
        )

    # Polarity: y=1 for PASS, y=0 for FAIL; sigmoid(x @ w) predicts P(PASS).
    y = np.array([1 if lbl == Verdict.PASS else 0 for _, lbl in filtered], dtype=np.int64)
    vocab_counter: Counter[str] = Counter()
    tokenized: list[list[str]] = []
    for txt, _ in filtered:
        toks = re.findall(r"[a-z]+", txt.lower())
        bigrams = [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)]
        feats = toks + bigrams
        tokenized.append(feats)
        vocab_counter.update(set(feats))

    vocab = [tok for tok, _ in vocab_counter.most_common(256)]
    tok_to_idx = {tok: i for i, tok in enumerate(vocab)}

    x_raw = np.zeros((len(filtered), len(vocab) + 2), dtype=np.float64)
    for row, (txt, _) in enumerate(filtered):
        for tok in tokenized[row]:
            if tok in tok_to_idx:
                x_raw[row, tok_to_idx[tok]] += 1.0
        x_raw[row, -2] = math.log1p(len(txt))
        x_raw[row, -1] = 1.0  # bias

    indices = np.arange(len(filtered))
    rng = np.random.default_rng(seed)
    if filtered_entities is not None:
        entity_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, eid in enumerate(filtered_entities):
            entity_to_indices[eid].append(idx)
        unique_entities = list(entity_to_indices.keys())
        n_actual_splits = min(n_splits, len(unique_entities))
        if n_actual_splits < 2:
            return StyleProbeResult(
                cv_accuracy=0.5,
                cv_roc_auc=0.5,
                passed=True,
                reason="Too few unique entities for entity-grouped style probe.",
            )
        ent_arr = np.array(unique_entities, dtype=object)
        rng.shuffle(ent_arr)
        ent_folds = np.array_split(ent_arr, n_actual_splits)
        folds = [
            np.array(
                [idx for eid in fold_ents for idx in entity_to_indices[str(eid)]],
                dtype=np.int64,
            )
            for fold_ents in ent_folds
        ]
    else:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        folds = np.array_split(shuffled, min(n_splits, len(filtered)))

    oof_probs = np.zeros(len(filtered), dtype=np.float64)
    fold_weights: list[np.ndarray] = []
    for val_idx in folds:
        train_idx = np.setdiff1d(indices, val_idx)
        x_tr = x_raw[train_idx].copy()
        x_va = x_raw[val_idx].copy()
        y_tr = y[train_idx]

        # Standardize continuous features strictly inside each fold (no train/val scaling leakage).
        if x_raw.shape[1] > 1:
            mean_tr = np.mean(x_tr[:, :-1], axis=0)
            std_tr = np.std(x_tr[:, :-1], axis=0)
            std_tr[std_tr < 1e-8] = 1.0
            x_tr[:, :-1] = (x_tr[:, :-1] - mean_tr) / std_tr
            x_va[:, :-1] = (x_va[:, :-1] - mean_tr) / std_tr

        def loss_fn(
            w: np.ndarray,
            x_b: np.ndarray = x_tr,
            y_b: np.ndarray = y_tr,
        ) -> float:
            z = np.clip(x_b @ w, -20.0, 20.0)
            p = 1.0 / (1.0 + np.exp(-z))
            bce = -np.mean(y_b * np.log(p + 1e-12) + (1 - y_b) * np.log(1 - p + 1e-12))
            reg = 0.5 * float(np.sum(w[:-1] ** 2))
            return float(bce + reg)

        res = minimize(loss_fn, np.zeros(x_raw.shape[1]), method="L-BFGS-B")
        z_va = np.clip(x_va @ res.x, -20.0, 20.0)
        oof_probs[val_idx] = 1.0 / (1.0 + np.exp(-z_va))
        fold_weights.append(res.x[:-1])

    feature_names = vocab + ["log1p_char_length"]
    mean_weights = np.mean(np.vstack(fold_weights), axis=0)
    feat_weights = list(zip(feature_names, mean_weights.tolist(), strict=True))
    top_positive_features = sorted(
        ((name, round(w, 4)) for name, w in feat_weights if w > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    top_negative_features = sorted(
        ((name, round(w, 4)) for name, w in feat_weights if w < 0.0),
        key=lambda item: item[1],
    )[:10]

    auc = _compute_roc_auc(y, oof_probs)
    acc = float(np.mean((oof_probs >= 0.5).astype(np.int64) == y))
    passed = auc <= max_auc
    reason = (
        f"Style-only probe near chance (AUC={auc:.3f} <= {max_auc:.2f})."
        if passed
        else f"Shortcut detected: style-only probe achieved AUC={auc:.3f} (> {max_auc:.2f}). Fix generator templates."
    )
    return StyleProbeResult(
        cv_accuracy=round(acc, 4),
        cv_roc_auc=round(auc, 4),
        passed=passed,
        reason=reason,
        top_positive_features=top_positive_features,
        top_negative_features=top_negative_features,
    )
