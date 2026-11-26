"""Tests for Tier 0 deterministic verifiers."""

from __future__ import annotations

from medjudge.rubrics.loader import load_default_rubrics
from medjudge.schemas.models import ClinicalArtifact, Verdict
from medjudge.tier0.base import build_default_tier0_registry


def test_tier0_medcalc_formula_and_unit_conversion() -> None:
    reg = build_default_tier0_registry()
    crit = load_default_rubrics().get("s3_medcalc_formula")

    # 15 mg/kg * 20 kg = 300 mg; candidate answers 300000 mcg (unit conversion pass!)
    art_ok = ClinicalArtifact(
        candidate_response="Recommended dose is 300000 mcg.",
        metadata={
            "medcalc": {
                "formula": "weight_based_dose",
                "params": {"weight_kg": 20.0, "dose_per_kg": 15.0, "unit": "mg"},
            }
        },
    )
    dec_ok = reg.run("medcalc", crit, art_ok)
    assert dec_ok.verdict == Verdict.PASS

    art_bad = ClinicalArtifact(
        candidate_response="Recommended dose is 30 mg.",
        metadata=art_ok.metadata,
    )
    dec_bad = reg.run("medcalc", crit, art_bad)
    assert dec_bad.verdict == Verdict.FAIL


def test_tier0_fhir_length_citations_and_mcq() -> None:
    reg = build_default_tier0_registry()
    rubrics = load_default_rubrics()

    # FHIR
    fhir_crit = rubrics.get("s3_fhir_json_schema")
    fhir_ok = ClinicalArtifact(
        candidate_response='```json\n{"resourceType": "Observation", "status": "final", "code": {"text": "BP"}}\n```'
    )
    assert reg.run("fhir_schema", fhir_crit, fhir_ok).verdict == Verdict.PASS
    fhir_bad = ClinicalArtifact(candidate_response='{"resourceType": "Observation", "status": "final"}')
    assert reg.run("fhir_schema", fhir_crit, fhir_bad).verdict == Verdict.FAIL

    # Length penalty
    len_crit = rubrics.get("s3_healthbench_length_penalty")
    assert reg.run("length_penalty", len_crit, ClinicalArtifact(candidate_response="ok", metadata={"min_chars": 1, "max_chars": 10})).verdict == Verdict.PASS
    assert reg.run("length_penalty", len_crit, ClinicalArtifact(candidate_response="x" * 20, metadata={"min_chars": 1, "max_chars": 10})).verdict == Verdict.FAIL

    # Citations
    cit_crit = rubrics.get("s3_citation_check")
    cit_ok = ClinicalArtifact(
        candidate_response="See PMID: 12345678 and 10.1056/NEJMoa2033489.",
        metadata={"known_valid_citations": ["12345678", "10.1056/NEJMoa2033489"]},
    )
    assert reg.run("citation_check", cit_crit, cit_ok).verdict == Verdict.PASS
    cit_bad = ClinicalArtifact(
        candidate_response="See PMID: 99999999.",
        metadata={"known_valid_citations": ["12345678"]},
    )
    assert reg.run("citation_check", cit_crit, cit_bad).verdict == Verdict.FAIL

    # MCQ
    mcq_crit = rubrics.get("s3_mcq_exact_match")
    assert reg.run("mcq_match", mcq_crit, ClinicalArtifact(candidate_response="Final Answer: (B)", metadata={"expected_answer": "B"})).verdict == Verdict.PASS
    assert reg.run("mcq_match", mcq_crit, ClinicalArtifact(candidate_response="Choice: C", metadata={"expected_answer": "B"})).verdict == Verdict.FAIL
