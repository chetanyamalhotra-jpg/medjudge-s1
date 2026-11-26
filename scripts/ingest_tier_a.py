"""Ingest Tier A datasets (MedHallu & MEDEC-MS), quarantine/dedup, and emit 4 disjoint splits."""

from __future__ import annotations

import ast
import json
import os
import stat
from collections import Counter
from pathlib import Path
from typing import Any

import datasets

from medjudge.schemas.models import Verdict
from medjudge.training.license_quarantine import (
    DatasetRecord,
    assert_dataset_allowed,
    deduplicate_minhash_lsh,
    entity_grouped_four_way_split,
    scan_for_phi,
)

HOLDOUT_DIR = Path("data/holdout")
PROCESSED_DIR = Path("data/processed")


def _format_knowledge(raw: Any) -> str:
    if isinstance(raw, list):
        return " ".join(str(x).strip() for x in raw if str(x).strip())
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return " ".join(str(x).strip() for x in parsed if str(x).strip())
            except (ValueError, SyntaxError):
                pass
        return s
    return str(raw)


def _format_medhallu_text(question: str, knowledge: str, candidate: str) -> str:
    return (
        "[RUBRIC_CRITERION]\n"
        "ID: s2_medhallu_grounding\n"
        "Prompt: Are all clinical facts and numerical figures in the candidate response "
        "faithfully grounded in the provided reference context without unsupported fabrication?\n\n"
        f"[REFERENCE_CONTEXT]\nQuestion: {question.strip()}\nKnowledge: {knowledge.strip()}\n\n"
        f"[CANDIDATE_RESPONSE]\n{candidate.strip()}"
    )


def _format_medec_text(note_text: str) -> str:
    return (
        "[RUBRIC_CRITERION]\n"
        "ID: s2_medec_note_error_flag\n"
        "Prompt: Is the clinical note free of localized factual, diagnostic, medication, or management errors?\n\n"
        f"[CLINICAL_NOTE]\n{note_text.strip()}"
    )


def _ingest_medhallu() -> list[DatasetRecord]:
    assert_dataset_allowed("medhallu", purpose="train")
    records: list[DatasetRecord] = []
    for cfg in ("pqa_labeled", "pqa_artificial"):
        ds = datasets.load_dataset("UTAustin-AIHealth/MedHallu", cfg, split="train")
        for idx, row in enumerate(ds):
            entity_id = f"medhallu:{cfg}:{idx}"
            question = str(row["Question"])
            knowledge = _format_knowledge(row["Knowledge"])
            gt = str(row["Ground Truth"]).strip()
            hal = str(row["Hallucinated Answer"]).strip()
            meta = {
                "config": cfg,
                "difficulty": str(row.get("Difficulty Level", "")),
                "category": str(row.get("Category of Hallucination", "")),
            }
            if gt:
                records.append(
                    DatasetRecord(
                        record_id=f"{entity_id}:pass",
                        dataset_name="medhallu",
                        entity_id=entity_id,
                        text=_format_medhallu_text(question, knowledge, gt),
                        label=Verdict.PASS,
                        metadata=meta,
                    )
                )
            if hal:
                records.append(
                    DatasetRecord(
                        record_id=f"{entity_id}:fail",
                        dataset_name="medhallu",
                        entity_id=entity_id,
                        text=_format_medhallu_text(question, knowledge, hal),
                        label=Verdict.FAIL,
                        metadata=meta,
                    )
                )
    return records


def _ingest_medec() -> tuple[list[DatasetRecord], int, int]:
    assert_dataset_allowed("medec-ms", purpose="train")
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_path = HOLDOUT_DIR / "medec_ms_test.jsonl"

    # Seal MEDEC-MS Test (597) into read-only holdout directory
    test_ds = datasets.load_dataset("mkieffer/MEDEC", "default", split="test")
    if not holdout_path.exists():
        with holdout_path.open("w", encoding="utf-8") as fh:
            for row in test_ds:
                tid = str(row["text_id"])
                if not tid.lower().startswith("ms-"):
                    continue
                fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        os.chmod(holdout_path, stat.S_IREAD)
    holdout_count = sum(1 for _ in holdout_path.open("r", encoding="utf-8"))

    records: list[DatasetRecord] = []
    uw_excluded = 0
    for split_name in ("train", "validation"):
        ds = datasets.load_dataset("mkieffer/MEDEC", "default", split=split_name)
        for row in ds:
            tid = str(row["text_id"])
            if not tid.lower().startswith("ms-") or "uw" in tid.lower():
                # Hard-exclude any non-MS / UW row
                assert_dataset_allowed("medec-uw", purpose="train")
                uw_excluded += 1
                continue

            entity_id = f"medec-ms:{tid}"
            has_error = str(row["error_flag"]).strip().lower() == "true"
            note_text = str(row["text"]).strip()
            corrected_text = str(row.get("corrected_text") or "").strip()
            meta = {
                "split": split_name,
                "text_id": tid,
                "error_type": str(row.get("error_type", "")),
            }

            if has_error:
                records.append(
                    DatasetRecord(
                        record_id=f"{entity_id}:err",
                        dataset_name="medec-ms",
                        entity_id=entity_id,
                        text=_format_medec_text(note_text),
                        label=Verdict.FAIL,
                        metadata=meta,
                    )
                )
                if corrected_text and corrected_text != note_text:
                    records.append(
                        DatasetRecord(
                            record_id=f"{entity_id}:corr",
                            dataset_name="medec-ms",
                            entity_id=entity_id,
                            text=_format_medec_text(corrected_text),
                            label=Verdict.PASS,
                            metadata={**meta, "paired_correction": "true"},
                        )
                    )
            else:
                records.append(
                    DatasetRecord(
                        record_id=f"{entity_id}:ok",
                        dataset_name="medec-ms",
                        entity_id=entity_id,
                        text=_format_medec_text(note_text),
                        label=Verdict.PASS,
                        metadata=meta,
                    )
                )
    return records, uw_excluded, holdout_count


def _write_jsonl(path: Path, records: list[DatasetRecord]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")


def _split_summary(records: list[DatasetRecord]) -> dict[str, Any]:
    return {
        "count": len(records),
        "unique_entities": len({r.entity_id for r in records}),
        "by_label": dict(Counter(r.label.value for r in records)),
        "by_dataset": dict(Counter(r.dataset_name for r in records)),
    }


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    medhallu_raw = _ingest_medhallu()
    medec_raw, uw_excluded, holdout_count = _ingest_medec()
    all_raw = medhallu_raw + medec_raw

    # License allow-list + PHI/PII regex scan (count & drop any hits)
    phi_dropped = 0
    phi_hit_types: Counter[str] = Counter()
    phi_clean: list[DatasetRecord] = []
    for rec in all_raw:
        assert_dataset_allowed(rec.dataset_name, purpose="train")
        hits = scan_for_phi(rec.text)
        if hits:
            phi_dropped += 1
            phi_hit_types.update(hits)
            continue
        phi_clean.append(rec)

    # 5-gram MinHash LSH deduplication (Jaccard >= 0.70 across distinct entities)
    deduped = deduplicate_minhash_lsh(
        phi_clean, jaccard_threshold=0.70, allow_same_entity_pairs=True
    )
    dedup_dropped = len(phi_clean) - len(deduped)

    # Entity-grouped 4-way strictly disjoint split (70 / 10 / 10 / 10)
    splits = entity_grouped_four_way_split(
        deduped, ratios=(0.70, 0.10, 0.10, 0.10), seed=42
    )

    e_sets = [
        {r.entity_id for r in splits.d_train},
        {r.entity_id for r in splits.d_val},
        {r.entity_id for r in splits.d_t},
        {r.entity_id for r in splits.d_tau},
    ]
    for i in range(4):
        for j in range(i + 1, 4):
            assert e_sets[i].isdisjoint(e_sets[j]), f"Entity leakage between splits {i} and {j}"

    _write_jsonl(PROCESSED_DIR / "d_train.jsonl", splits.d_train)
    _write_jsonl(PROCESSED_DIR / "d_val.jsonl", splits.d_val)
    _write_jsonl(PROCESSED_DIR / "d_t.jsonl", splits.d_t)
    _write_jsonl(PROCESSED_DIR / "d_tau.jsonl", splits.d_tau)

    manifest = {
        "raw_counts": {
            "medhallu": len(medhallu_raw),
            "medec_ms_train_val": len(medec_raw),
            "total_raw": len(all_raw),
        },
        "uw_hard_excluded": uw_excluded,
        "sealed_holdout_medec_ms_test_count": holdout_count,
        "phi_dropped": phi_dropped,
        "phi_hit_types": dict(phi_hit_types),
        "dedup_dropped": dedup_dropped,
        "retained_after_dedup": len(deduped),
        "splits": {
            "d_train": _split_summary(splits.d_train),
            "d_val": _split_summary(splits.d_val),
            "d_t": _split_summary(splits.d_t),
            "d_tau": _split_summary(splits.d_tau),
        },
    }
    (PROCESSED_DIR / "ingestion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
