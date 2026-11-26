"""Synthesize ~10% stripped-context UNKNOWN examples per split and run style-only shortcut probe."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from medjudge.schemas.models import Verdict
from medjudge.training.distillation import run_style_only_probe
from medjudge.training.license_quarantine import DatasetRecord

PROCESSED_DIR = Path("data/processed")
SPLIT_FILES = ("d_train.jsonl", "d_val.jsonl", "d_t.jsonl", "d_tau.jsonl")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _load_records(path: Path) -> list[DatasetRecord]:
    rows: list[DatasetRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = DatasetRecord.model_validate_json(line)
                if not rec.record_id.endswith(":unk"):
                    rows.append(rec)
    return rows


def _strip_decisive_context(rec: DatasetRecord) -> str:
    if rec.dataset_name == "medhallu":
        # Strip Knowledge: content while keeping Question: and [CANDIDATE_RESPONSE]
        stripped = re.sub(
            r"Knowledge:\s*.*?(?=\n\n\[CANDIDATE_RESPONSE\])",
            "Knowledge: [OMITTED_REFERENCE_CONTEXT]",
            rec.text,
            flags=re.DOTALL,
        )
        if stripped != rec.text:
            return stripped
    # MEDEC-MS or fallback: drop final ~45% of clinical note sentences (minimum 2 sentences)
    header_marker = "[CLINICAL_NOTE]\n"
    if header_marker in rec.text:
        prefix, note_body = rec.text.split(header_marker, 1)
        sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(note_body) if s.strip()]
        if len(sents) >= 4:
            keep_n = max(2, int(len(sents) * 0.55))
            return f"{prefix}{header_marker}" + " ".join(sents[:keep_n]) + " [DECISIVE_NOTE_CONTEXT_OMITTED]"
    return "[CONTEXT_OMITTED] " + rec.text[: max(40, len(rec.text) // 2)]


def _add_unknown_for_split(records: list[DatasetRecord], fraction: float = 0.10, seed: int = 42) -> list[DatasetRecord]:
    rng = random.Random(seed)
    positives = [r for r in records if r.label == Verdict.PASS]
    target_n = max(1 if positives else 0, int(round(len(positives) * fraction)))
    chosen = rng.sample(positives, min(target_n, len(positives)))
    # Add synthesized UNKNOWN records sharing the same entity_id (zero cross-split leakage)
    synthesized: list[DatasetRecord] = []
    for rec in chosen:
        synthesized.append(
            DatasetRecord(
                record_id=f"{rec.record_id}:unk",
                dataset_name=rec.dataset_name,
                entity_id=rec.entity_id,
                text=_strip_decisive_context(rec),
                label=Verdict.UNKNOWN,
                metadata={**rec.metadata, "synthesized_unknown": "true"},
            )
        )
    combined = list(records) + synthesized
    rng.shuffle(combined)
    return combined


def _write_records(path: Path, records: list[DatasetRecord]) -> None:
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


def _probe_subset(name: str, records: list[DatasetRecord], max_samples: int = 2000) -> dict[str, Any]:
    pf = [r for r in records if r.label in (Verdict.PASS, Verdict.FAIL)]
    if len(pf) > max_samples:
        rng = random.Random(42)
        pf = rng.sample(pf, max_samples)
    res = run_style_only_probe(
        [r.text for r in pf],
        [r.label for r in pf],
        max_auc=0.62,
        n_splits=5,
        seed=42,
        entity_ids=[r.entity_id for r in pf],
    )
    return {
        "subset": name,
        "n_samples": len(pf),
        "unique_entities": len({r.entity_id for r in pf}),
        "cv_accuracy": res.cv_accuracy,
        "cv_roc_auc": res.cv_roc_auc,
        "passed_strict_gate": res.passed,
        "top_positive_features": [[f, w] for f, w in res.top_positive_features],
        "top_negative_features": [[f, w] for f, w in res.top_negative_features],
    }


def main() -> int:
    loaded: dict[str, list[DatasetRecord]] = {}
    for idx, fname in enumerate(SPLIT_FILES):
        p = PROCESSED_DIR / fname
        orig = _load_records(p)
        updated = _add_unknown_for_split(orig, fraction=0.10, seed=100 + idx)
        _write_records(p, updated)
        loaded[fname] = updated

    # Verify zero entity leakage across all 4 splits
    ent_sets = [{r.entity_id for r in loaded[f]} for f in SPLIT_FILES]
    for i in range(4):
        for j in range(i + 1, 4):
            assert ent_sets[i].isdisjoint(ent_sets[j]), f"Entity leakage between {SPLIT_FILES[i]} and {SPLIT_FILES[j]}"

    manifest_path = PROCESSED_DIR / "ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["splits_with_unknown"] = {
        fname.replace(".jsonl", ""): _split_summary(loaded[fname]) for fname in SPLIT_FILES
    }

    # Style probe on d_val (full) and d_train (2k stratified/random sample) separated by dataset & config
    val_recs = loaded["d_val.jsonl"]
    train_recs = loaded["d_train.jsonl"]

    probes = [
        _probe_subset("d_val:medhallu_all", [r for r in val_recs if r.dataset_name == "medhallu"]),
        _probe_subset(
            "d_val:medhallu_pqa_labeled",
            [r for r in val_recs if r.dataset_name == "medhallu" and r.metadata.get("config") == "pqa_labeled"],
        ),
        _probe_subset(
            "d_val:medhallu_pqa_artificial",
            [r for r in val_recs if r.dataset_name == "medhallu" and r.metadata.get("config") == "pqa_artificial"],
        ),
        _probe_subset("d_val:medec_ms", [r for r in val_recs if r.dataset_name == "medec-ms"]),
        _probe_subset("d_train(2k):medhallu_all", [r for r in train_recs if r.dataset_name == "medhallu"]),
        _probe_subset(
            "d_train(2k):medhallu_pqa_artificial",
            [r for r in train_recs if r.dataset_name == "medhallu" and r.metadata.get("config") == "pqa_artificial"],
        ),
        _probe_subset("d_train(2k):medec_ms", [r for r in train_recs if r.dataset_name == "medec-ms"]),
    ]
    manifest["style_probes"] = probes
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Splits with ~10% Synthesized UNKNOWN ===")
    print(json.dumps(manifest["splits_with_unknown"], indent=2))
    print("\n=== Style-Only Probe Results (Bag-of-n-grams + Length LR) ===")
    print(json.dumps(probes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
