"""Tests for license quarantine, PHI scanner, entity-grouped 4-way split, and MinHash LSH dedup."""

from __future__ import annotations

import pytest

from medjudge.schemas.models import Verdict
from medjudge.training.license_quarantine import (
    DatasetQuarantineError,
    DatasetRecord,
    LicenseTier,
    PHIDetectedError,
    assert_dataset_allowed,
    deduplicate_minhash_lsh,
    entity_grouped_four_way_split,
    scan_for_phi,
    validate_and_filter_records,
)


def test_license_quarantine_tiers_and_sealed_holdouts() -> None:
    assert assert_dataset_allowed("medhallu", purpose="train") == LicenseTier.TIER_A_PERMISSIVE
    assert assert_dataset_allowed("medec-ms", purpose="train") == LicenseTier.TIER_A_PERMISSIVE
    assert assert_dataset_allowed("medcalc-bench", purpose="eval") == LicenseTier.TIER_B_SHAREALIKE

    with pytest.raises(DatasetQuarantineError):
        assert_dataset_allowed("medcalc-bench", purpose="train")
    with pytest.raises(DatasetQuarantineError):
        assert_dataset_allowed("medec-uw", purpose="eval")
    with pytest.raises(DatasetQuarantineError):
        assert_dataset_allowed("medexqa", purpose="train")
    with pytest.raises(DatasetQuarantineError):
        assert_dataset_allowed("mimic-iv", purpose="train")
    with pytest.raises(DatasetQuarantineError):
        assert_dataset_allowed("healthbench-professional", purpose="train")
    with pytest.raises(DatasetQuarantineError):
        assert_dataset_allowed("medec-ms-test", purpose="calibrate")


def test_phi_scanner_and_entity_grouped_disjoint_four_way_split() -> None:
    assert "SSN" in scan_for_phi("Patient SSN 123-45-6789")
    assert "EMAIL" in scan_for_phi("Contact jane.doe@hospital.org")

    bad_rec = DatasetRecord(
        record_id="bad-1",
        dataset_name="medhallu",
        entity_id="ent-bad",
        text="Call 555-123-4567 for follow up.",
        label=Verdict.PASS,
    )
    with pytest.raises(PHIDetectedError):
        validate_and_filter_records([bad_rec], purpose="train")

    records: list[DatasetRecord] = []
    for eid in range(20):
        # Correct & error variants share entity_id and MUST stay together
        records.append(
            DatasetRecord(
                record_id=f"r-{eid}-ok",
                dataset_name="medec-ms",
                entity_id=f"note-{eid}",
                text=f"Clean clinical note {eid} without any protected health information.",
                label=Verdict.PASS,
            )
        )
        records.append(
            DatasetRecord(
                record_id=f"r-{eid}-err",
                dataset_name="medec-ms",
                entity_id=f"note-{eid}",
                text=f"Error clinical note {eid} without any protected health information.",
                label=Verdict.FAIL,
            )
        )

    clean = validate_and_filter_records(records, purpose="train")
    splits = entity_grouped_four_way_split(clean)
    e_train = {r.entity_id for r in splits.d_train}
    e_val = {r.entity_id for r in splits.d_val}
    e_t = {r.entity_id for r in splits.d_t}
    e_tau = {r.entity_id for r in splits.d_tau}

    assert e_train and e_val and e_t and e_tau
    assert e_train.isdisjoint(e_val)
    assert e_train.isdisjoint(e_t)
    assert e_train.isdisjoint(e_tau)
    assert e_val.isdisjoint(e_t)
    assert e_val.isdisjoint(e_tau)
    assert e_t.isdisjoint(e_tau)


def test_minhash_lsh_five_gram_deduplication() -> None:
    base_text = "Randomized trial showed systolic blood pressure decreased significantly after treatment with lisinopril daily"
    near_dup = "Randomized trial showed systolic blood pressure decreased significantly after treatment with lisinopril once daily"
    distinct = "Computed tomography angiography of the chest demonstrated no evidence of acute pulmonary embolism"

    recs = [
        DatasetRecord(record_id="1", dataset_name="pubmedqa", entity_id="p1", text=base_text, label=Verdict.PASS),
        DatasetRecord(record_id="2", dataset_name="medhallu", entity_id="m1", text=near_dup, label=Verdict.PASS),
        DatasetRecord(record_id="3", dataset_name="medhallu", entity_id="m2", text=distinct, label=Verdict.FAIL),
    ]
    deduped = deduplicate_minhash_lsh(recs, jaccard_threshold=0.70)
    ids = [r.record_id for r in deduped]
    assert ids == ["1", "3"]
