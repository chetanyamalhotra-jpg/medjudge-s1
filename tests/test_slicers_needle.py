"""Needle tests for GlobalContextCapsule, PinnedHeaderSlicer, and sufficiency gate."""

from __future__ import annotations

from medjudge.rubrics.loader import load_default_rubrics
from medjudge.schemas.models import ClinicalArtifact, RiskClass, RubricCriterion, Verdict
from medjudge.slicers.capsule import extract_global_context_capsule
from medjudge.slicers.pinned_header import PinnedHeaderSlicer
from medjudge.slicers.validator import validate_slicer_sufficiency


def test_needle_turn1_allergy_turn8_drug_preserved_in_long_conversation() -> None:
    turns = [
        {"role": "patient", "content": "I am a 45-year-old female, not pregnant, with allergy to penicillin. Chief complaint: sore throat."},
        {"role": "clinician", "content": "When did the sore throat begin?"},
        {"role": "patient", "content": "Two days ago. " + ("Extra filler history details. " * 60)},
        {"role": "clinician", "content": "Any fever or cough? " + ("Detailed physical exam notes. " * 60)},
        {"role": "patient", "content": "Patient denies chest pain, shortness of breath, or syncope."},
        {"role": "clinician", "content": "What labs have you had?"},
        {"role": "patient", "content": "2 years ago creatinine was 2.1 mg/dL. Today creatinine is 1.0 mg/dL. Mother has sulfa allergy."},
        {"role": "patient", "content": "I am currently taking lisinopril 10 mg daily."},
    ]
    artifact = ClinicalArtifact(
        artifact_id="needle-1",
        turns=turns,
        candidate_response="Start amoxicillin 500 mg TID for 10 days.",
    )
    criterion = RubricCriterion(
        id="s1_allergy",
        version="1.0.0",
        title="Allergy safety",
        prompt="Does response avoid documented allergies?",
        risk_class=RiskClass.S1_CRITICAL_SAFETY,
        required_context=["allergies", "active_medications"],
    )

    slicer = PinnedHeaderSlicer(max_tokens=200)
    sliced = slicer.slice(criterion, artifact)

    assert sliced.sliced is True
    assert sliced.sufficiency.sufficient is True
    assert "penicillin" in sliced.formatted_text.lower()
    assert "lisinopril" in sliced.formatted_text.lower()
    # Family history isolated from patient allergies
    assert not any("sulfa" in a.lower() for a in sliced.capsule.allergies)
    assert any("sulfa" in f.lower() for f in sliced.capsule.family_history)
    # Split negations preserved
    neg_joined = " ".join(sliced.capsule.negated_findings).lower()
    assert "chest pain" in neg_joined
    assert "shortness of breath" in neg_joined
    assert "syncope" in neg_joined
    # Temporality preserved
    labs_joined = " ".join(sliced.capsule.renal_hepatic_labs)
    assert "[HISTORICAL]" in labs_joined
    assert "[CURRENT]" in labs_joined


def test_no_slice_when_input_fits_window() -> None:
    artifact = ClinicalArtifact(
        artifact_id="short-1",
        clinical_note="40-year-old male. No known drug allergies. Taking metformin 500 mg. Chief complaint: fatigue.",
        candidate_response="Continue metformin and check HbA1c.",
    )
    criterion = RubricCriterion(
        id="s2_short",
        version="1.0.0",
        title="Short check",
        prompt="Is response grounded?",
        required_context=["allergies"],
    )
    slicer = PinnedHeaderSlicer(max_tokens=1024)
    sliced = slicer.slice(criterion, artifact)
    assert sliced.sliced is False
    assert sliced.sufficiency.sufficient is True


def test_missing_required_context_sets_sufficient_false() -> None:
    artifact = ClinicalArtifact(
        artifact_id="missing-ctx",
        clinical_note="Patient reports mild headache.",
        candidate_response="Take acetaminophen 500 mg.",
    )
    criterion = RubricCriterion(
        id="s1_allergy",
        version="1.0.0",
        title="Allergy check",
        prompt="Check allergy",
        risk_class=RiskClass.S1_CRITICAL_SAFETY,
        required_context=["allergies", "active_medications"],
    )
    capsule = extract_global_context_capsule(artifact)
    assert not capsule.allergies
    sliced = PinnedHeaderSlicer(max_tokens=512).slice(criterion, artifact)
    assert sliced.sufficiency.sufficient is False
    assert set(sliced.sufficiency.missing_context) == {"allergies", "active_medications"}


def test_slicer_sufficiency_validation_gates() -> None:
    full = [Verdict.PASS] * 80 + [Verdict.FAIL] * 40
    # Perfect match clears S1 gate
    res_ok = validate_slicer_sufficiency(full, full, RiskClass.S1_CRITICAL_SAFETY, min_samples=100)
    assert res_ok.enabled is True
    assert res_ok.cohens_kappa == 1.0

    # Flip 3 FAILs to PASS (>2% FN shift) -> S1 must disable slicing
    degraded = list(full)
    degraded[80] = Verdict.PASS
    degraded[81] = Verdict.PASS
    degraded[82] = Verdict.PASS
    res_bad = validate_slicer_sufficiency(degraded, full, RiskClass.S1_CRITICAL_SAFETY, min_samples=100)
    assert res_bad.enabled is False


def test_paper_qa_and_clinical_note_bypass_capsule_while_clinical_encounter_prepends_it() -> None:
    rubrics = load_default_rubrics()
    slicer = PinnedHeaderSlicer(max_tokens=1024)

    paper_art = ClinicalArtifact(
        artifact_id="paper-1",
        reference_context="UKPDS showed intensive glucose control lowered microvascular complications.",
        candidate_response="Intensive glucose control reduced microvascular endpoints in UKPDS.",
    )
    paper_sliced = slicer.slice(rubrics.get("s2_medhallu_grounding"), paper_art)
    assert "[GLOBAL_CONTEXT_CAPSULE]" not in paper_sliced.formatted_text
    assert paper_sliced.sufficiency.sufficient is True

    note_art = ClinicalArtifact(
        artifact_id="note-1",
        clinical_note="45-year-old female with mild cough. Taking metformin 500 mg.",
        candidate_response="",
    )
    note_sliced = slicer.slice(rubrics.get("s2_medec_note_error_flag"), note_art)
    assert "[GLOBAL_CONTEXT_CAPSULE]" not in note_sliced.formatted_text
    assert "[CANDIDATE_RESPONSE]" not in note_sliced.formatted_text
    assert note_sliced.sufficiency.sufficient is True

    encounter_art = ClinicalArtifact(
        artifact_id="enc-1",
        clinical_note="Allergy to penicillin. Currently taking lisinopril 10 mg.",
        candidate_response="Start amoxicillin.",
    )
    enc_sliced = slicer.slice(rubrics.get("s1_allergy_medication_safety"), encounter_art)
    assert "[GLOBAL_CONTEXT_CAPSULE]" in enc_sliced.formatted_text
    assert "[CANDIDATE_RESPONSE]" in enc_sliced.formatted_text
    assert enc_sliced.sufficiency.sufficient is True


def test_single_paragraph_overflow_splits_cleanly_preserving_candidate_response() -> None:
    long_single_para = " ".join(f"Sentence {i} discusses stable clinical findings." for i in range(120))
    art = ClinicalArtifact(
        artifact_id="long-para",
        reference_context=long_single_para,
        candidate_response="Final candidate statement stays intact.",
    )
    crit = RubricCriterion(
        id="s2_paper_overflow",
        version="1.0.0",
        title="Paper Overflow",
        prompt="Verify grounding.",
        task_type="paper_qa",
        required_context=["reference_context"],
    )
    slicer = PinnedHeaderSlicer(max_tokens=200)
    sliced = slicer.slice(crit, art)
    assert sliced.sliced is True
    assert sliced.estimated_tokens < 240
    assert "[CANDIDATE_RESPONSE]\nFinal candidate statement stays intact." in sliced.formatted_text


def test_capsule_audit_regressions_negated_allergies_historical_meds_family_and_g2p2() -> None:
    # 1. Singular & targeted negated allergies go to negated_findings without false allergies/NKDA
    art_neg = ClinicalArtifact(
        clinical_note="Patient has no allergy to sulfa. Patient is not allergic to latex.",
    )
    cap_neg = extract_global_context_capsule(art_neg)
    assert cap_neg.allergies == []
    neg_joined = " | ".join(cap_neg.negated_findings).lower()
    assert "no allergy to sulfa" in neg_joined
    assert "no allergy to latex" in neg_joined

    # 2. Mixed allergy clause keeps positive allergy and avoids false NKDA
    art_mix_allergy = ClinicalArtifact(
        clinical_note="Patient has no allergies to food, allergic to penicillin.",
    )
    cap_mix_allergy = extract_global_context_capsule(art_mix_allergy)
    assert any("penicillin" in a.lower() for a in cap_mix_allergy.allergies)
    assert not any("nkda" in a.lower() for a in cap_mix_allergy.allergies)
    assert any("no allergy to food" in n.lower() for n in cap_mix_allergy.negated_findings)

    # 3. Historical medications alone do NOT satisfy active_medications
    art_hist = ClinicalArtifact(
        clinical_note="Previously prescribed amoxicillin 2 years ago, discontinued.",
    )
    cap_hist = extract_global_context_capsule(art_hist)
    assert len(cap_hist.historical_medications) == 1
    assert cap_hist.active_medications == []
    assert "active_medications" not in cap_hist.available_context_keys(art_hist)

    # 4. Mixed patient + family sentence keeps patient facts and isolates family history
    art_fam = ClinicalArtifact(
        clinical_note="45-year-old female with allergy to penicillin, while mother has allergy to sulfa.",
    )
    cap_fam = extract_global_context_capsule(art_fam)
    assert cap_fam.age == "45 years"
    assert any("penicillin" in a.lower() for a in cap_fam.allergies)
    assert not any("sulfa" in a.lower() for a in cap_fam.allergies)
    assert any("sulfa" in f.lower() for f in cap_fam.family_history)

    # 5. Obstetric history G2P2 alone does NOT mark active PREGNANT
    art_ob = ClinicalArtifact(clinical_note="32-year-old female, G2P2, presents for checkup.")
    cap_ob = extract_global_context_capsule(art_ob)
    assert cap_ob.pregnancy_status is None
