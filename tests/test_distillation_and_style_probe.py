"""Tests for multi-teacher distillation, minimal-pair hard negatives, MNLI filter, & style probe."""

from __future__ import annotations

import pytest

from medjudge.schemas.models import Verdict
from medjudge.training.distillation import (
    DistilledExample,
    filter_bidirectional_entailment,
    generate_minimal_pair_hard_negative,
    run_style_only_probe,
    strip_context_for_unknown,
    validate_teacher_models,
)
from medjudge.training.env_guard import validate_training_guardrails
from medjudge.training.train_lora import build_training_progression, validate_base_model


class _StubMNLI:
    def entailment_score(self, premise: str, hypothesis: str) -> float:
        return 0.90 if premise.lower() == hypothesis.lower() else 0.15


def test_multi_teacher_validation_and_hard_negative_gates() -> None:
    with pytest.raises(ValueError):
        validate_teacher_models(["Qwen/Qwen2.5-72B-Instruct"])
    with pytest.raises(ValueError):
        validate_teacher_models(["Qwen/Qwen2.5-72B-Instruct", "openai/gpt-4o"])
    validate_teacher_models(["Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3"])

    pos = "Administer epinephrine 0.5 mg IM immediately and call 911 for anaphylaxis."
    neg_res = generate_minimal_pair_hard_negative(pos, mutation_type="dosage_slip")
    assert neg_res is not None
    neg_txt, err_type = neg_res
    assert "5.0 mg" in neg_txt
    assert err_type == "dosage_slip"
    assert abs(len(neg_txt) - len(pos)) / len(pos) <= 0.10

    mnli = _StubMNLI()
    assert filter_bidirectional_entailment(pos, neg_txt, mnli, tau=0.75) is True
    assert filter_bidirectional_entailment(pos, pos, mnli, tau=0.75) is False

    # Context stripping ~10%
    exs = [
        DistilledExample(
            example_id=f"ex-{i}",
            criterion_id="s1_allergy",
            text=f"[GLOBAL_CONTEXT_CAPSULE]\nAllergies: penicillin\n[CANDIDATE_RESPONSE]\nUse azithromycin {i}.",
            verdict=Verdict.PASS,
            teacher_model="Qwen/Qwen2.5-72B-Instruct",
        )
        for i in range(10)
    ]
    with_unk = strip_context_for_unknown(exs, unknown_fraction=0.10, seed=1)
    unks = [e for e in with_unk if e.verdict == Verdict.UNKNOWN]
    assert len(unks) == 1
    assert unks[0].sufficient is False


def test_style_only_probe_catches_shortcuts_and_passes_minimal_pairs() -> None:
    # Shortcut-laden dataset: all PASS start with "Certainly!", all FAIL are terse
    shortcut_texts = [f"Certainly! Here is a helpful detailed answer {i}." for i in range(20)] + [
        f"Wrong {i}." for i in range(20)
    ]
    shortcut_labels = [Verdict.PASS] * 20 + [Verdict.FAIL] * 20
    bad_probe = run_style_only_probe(shortcut_texts, shortcut_labels, max_auc=0.62)
    assert bad_probe.passed is False
    assert bad_probe.cv_roc_auc > 0.80

    # Minimal-pair dataset differing only by clinical value -> style probe near chance
    balanced_texts: list[str] = []
    balanced_labels: list[Verdict] = []
    for i in range(20):
        balanced_texts.append(f"Give patient {i} medication dose 0.5 mg once daily.")
        balanced_labels.append(Verdict.PASS)
        balanced_texts.append(f"Give patient {i} medication dose 5.0 mg once daily.")
        balanced_labels.append(Verdict.FAIL)
    good_probe = run_style_only_probe(balanced_texts, balanced_labels, max_auc=0.65)
    assert good_probe.passed is True

    # Shortcut polarity check on feature weights
    assert any(w > 0 for _, w in bad_probe.top_positive_features)
    assert any(w < 0 for _, w in bad_probe.top_negative_features)


def test_style_only_probe_entity_ids_keeps_paired_entities_in_same_fold() -> None:
    # Each entity has a PASS and FAIL item sharing an entity-specific unique vocabulary token
    # and identical [a-z]+ lexical tokens (differing only by numeric dose).
    # Random splitting breaks pairs across folds and causes anti-correlation (AUC < 0.50);
    # entity-grouped splitting keeps paired entities in the same fold and avoids anti-correlation.
    texts: list[str] = []
    labels: list[Verdict] = []
    entity_ids: list[str] = []
    for i in range(30):
        eid = f"ent-{i}"
        tag = "".join(chr(ord("a") + ((i >> (4 * k)) & 0xF)) for k in range(3))
        texts.append(f"Caseword{tag} clinical summary shows stable dose 0.5 mg daily.")
        labels.append(Verdict.PASS)
        entity_ids.append(eid)
        texts.append(f"Caseword{tag} clinical summary shows stable dose 5.0 mg daily.")
        labels.append(Verdict.FAIL)
        entity_ids.append(eid)

    random_probe = run_style_only_probe(
        texts,
        labels,
        max_auc=0.62,
        n_splits=5,
        seed=42,
        entity_ids=None,
    )
    grouped_probe = run_style_only_probe(
        texts,
        labels,
        max_auc=0.62,
        n_splits=5,
        seed=42,
        entity_ids=entity_ids,
    )
    assert random_probe.cv_roc_auc < 0.40
    assert 0.45 <= grouped_probe.cv_roc_auc <= 0.62
    assert grouped_probe.passed is True


def test_windows_training_guardrails_and_model_allow_list() -> None:
    cfg = validate_training_guardrails()
    assert cfg.attn_implementation == "sdpa"
    assert cfg.torch_compile is False
    assert cfg.dataloader_num_workers == 0

    with pytest.raises(RuntimeError):
        validate_training_guardrails(use_flash_attn=True)
    with pytest.raises(RuntimeError):
        validate_training_guardrails(torch_compile=True)
    with pytest.raises(ValueError):
        validate_base_model("google/medgemma-4b-it")
    with pytest.raises(ValueError):
        validate_base_model("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    stages = build_training_progression("answerdotai/ModernBERT-base")
    assert [s.max_length for s in stages] == [512, 512, 1024, 2048]
    assert stages[0].freeze_backbone is True
