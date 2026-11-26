"""Strict dataset license quarantine, PHI/PII regex scan, entity-grouped 4-split, & MinHash LSH dedup."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from enum import StrEnum

from pydantic import BaseModel, Field

from medjudge.schemas.models import Verdict


class LicenseTier(StrEnum):
    TIER_A_PERMISSIVE = "TIER_A"
    TIER_B_SHAREALIKE = "TIER_B"
    TIER_C_HARD_EXCLUDE = "TIER_C"
    SEALED_HOLDOUT = "SEALED_HOLDOUT"


DATASET_CATALOG: dict[str, tuple[LicenseTier, str]] = {
    # Tier A — Permissive Core (Allowed for Training/Distillation, Zero-PHI)
    "healthbench": (LicenseTier.TIER_A_PERMISSIVE, "MIT"),
    "medhallu": (LicenseTier.TIER_A_PERMISSIVE, "MIT"),
    "medec-ms": (LicenseTier.TIER_A_PERMISSIVE, "CC-BY-4.0"),
    "pubmedqa": (LicenseTier.TIER_A_PERMISSIVE, "MIT"),
    "aci-bench": (LicenseTier.TIER_A_PERMISSIVE, "CC-BY-4.0"),
    "medcasereasoning": (LicenseTier.TIER_A_PERMISSIVE, "MIT"),
    "metamedqa": (LicenseTier.TIER_A_PERMISSIVE, "CC-BY-4.0"),
    "longhealth": (LicenseTier.TIER_A_PERMISSIVE, "Apache-2.0"),
    # Tier B — Copyleft ShareAlike (Offline Eval / Tier 0 Logic ONLY)
    "medcalc-bench": (LicenseTier.TIER_B_SHAREALIKE, "CC-BY-SA-4.0"),
    "medr-bench": (LicenseTier.TIER_B_SHAREALIKE, "CC-BY-SA-4.0"),
    # Tier C — Hard Exclude (NEVER ingest)
    "medec-uw": (LicenseTier.TIER_C_HARD_EXCLUDE, "UW-DUA"),
    "meddialog": (LicenseTier.TIER_C_HARD_EXCLUDE, "UNSPECIFIED"),
    "medexqa": (LicenseTier.TIER_C_HARD_EXCLUDE, "CC-BY-NC"),
    "mimic-iv": (LicenseTier.TIER_C_HARD_EXCLUDE, "PHYSIONET-DUA"),
    # Sealed Held-Out Test Sets
    "healthbench-hard": (LicenseTier.SEALED_HOLDOUT, "MIT-SEALED"),
    "healthbench-professional": (LicenseTier.SEALED_HOLDOUT, "MIT-SEALED"),
    "medec-ms-test": (LicenseTier.SEALED_HOLDOUT, "CC-BY-4.0-SEALED"),
}

_PHI_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")),
    ("MRN", re.compile(r"\b(?:MRN|Medical\s+Record\s+Number)\s*[:#]?\s*[A-Z0-9-]{5,}\b", re.IGNORECASE)),
    ("DOB", re.compile(r"\b(?:DOB|Date\s+of\s+Birth)\s*[:#]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE)),
]


class DatasetQuarantineError(PermissionError):
    """Raised when a forbidden, ShareAlike, or sealed dataset is routed to training."""


class PHIDetectedError(ValueError):
    """Raised when PHI/PII regex matches are found in training data."""


class DatasetRecord(BaseModel):
    """Canonical training/calibration record with entity grouping key."""

    record_id: str
    dataset_name: str
    entity_id: str  # patient_id | conversation_id | PMID | scenario_id
    text: str
    label: Verdict
    metadata: dict[str, str] = Field(default_factory=dict)


def assert_dataset_allowed(dataset_name: str, purpose: str = "train") -> LicenseTier:
    """Enforce Tier A/B/C and sealed holdout quarantine rules."""
    key = dataset_name.strip().lower()
    if key not in DATASET_CATALOG:
        raise DatasetQuarantineError(
            f"Dataset '{dataset_name}' is not in the MedJudge-S1 allow-list."
        )
    tier, lic = DATASET_CATALOG[key]
    if tier == LicenseTier.TIER_C_HARD_EXCLUDE:
        raise DatasetQuarantineError(
            f"Dataset '{dataset_name}' ({lic}) is Tier C HARD EXCLUDE and must NEVER be ingested."
        )
    if purpose in {"train", "distill", "calibrate"}:
        if tier == LicenseTier.TIER_B_SHAREALIKE:
            raise DatasetQuarantineError(
                f"Dataset '{dataset_name}' ({lic}) is Tier B ShareAlike and is excluded from training weights."
            )
        if tier == LicenseTier.SEALED_HOLDOUT:
            raise DatasetQuarantineError(
                f"Dataset '{dataset_name}' is a SEALED held-out test set and cannot enter {purpose}."
            )
    return tier


def scan_for_phi(text: str) -> list[str]:
    """Return list of PHI/PII pattern names detected in text."""
    hits: list[str] = []
    for name, pattern in _PHI_PATTERNS:
        if pattern.search(text):
            hits.append(name)
    return hits


def validate_and_filter_records(
    records: list[DatasetRecord], purpose: str = "train", raise_on_phi: bool = True
) -> list[DatasetRecord]:
    """Validate license quarantine and scan every record for PHI/PII."""
    clean: list[DatasetRecord] = []
    for rec in records:
        assert_dataset_allowed(rec.dataset_name, purpose=purpose)
        phi_hits = scan_for_phi(rec.text)
        if phi_hits:
            if raise_on_phi:
                raise PHIDetectedError(
                    f"PHI/PII detected in record '{rec.record_id}': {', '.join(phi_hits)}"
                )
            continue
        clean.append(rec)
    return clean


def _word_5grams(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if len(tokens) < 5:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + 5]) for i in range(len(tokens) - 4)}


def _minhash_signature(shingles: set[str], num_perm: int = 64) -> tuple[int, ...]:
    if not shingles:
        return (0,) * num_perm
    try:
        import xxhash  # noqa: PLC0415

        def _h(salt: bytes, sh_bytes: bytes) -> int:
            return xxhash.xxh64_intdigest(salt + sh_bytes)
    except ImportError:
        def _h(salt: bytes, sh_bytes: bytes) -> int:
            return int.from_bytes(
                hashlib.blake2b(salt + sh_bytes, digest_size=8).digest(),
                "little",
            )

    encoded = [sh.encode("utf-8") for sh in shingles]
    sig: list[int] = []
    for perm_idx in range(num_perm):
        salt = f"{perm_idx}:".encode()
        sig.append(min(_h(salt, sh_b) for sh_b in encoded))
    return tuple(sig)


def deduplicate_minhash_lsh(
    records: list[DatasetRecord],
    jaccard_threshold: float = 0.70,
    num_perm: int = 64,
    bands: int = 16,
    allow_same_entity_pairs: bool = False,
) -> list[DatasetRecord]:
    """5-gram MinHash LSH deduplication removing near-duplicates with Jaccard >= threshold."""
    if not records:
        return []
    rows_per_band = max(1, num_perm // bands)
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    kept: list[DatasetRecord] = []
    kept_shingles: list[set[str]] = []

    for rec in records:
        shingles = _word_5grams(rec.text)
        sig = _minhash_signature(shingles, num_perm=num_perm)

        candidate_indices: set[int] = set()
        for b in range(bands):
            start = b * rows_per_band
            band_key = (b, sig[start : start + rows_per_band])
            candidate_indices.update(buckets[band_key])

        is_dup = False
        for c_idx in candidate_indices:
            if allow_same_entity_pairs and kept[c_idx].entity_id == rec.entity_id:
                continue
            other = kept_shingles[c_idx]
            union = len(shingles | other)
            jaccard = (len(shingles & other) / union) if union > 0 else 1.0
            if jaccard >= jaccard_threshold:
                is_dup = True
                break

        if not is_dup:
            new_idx = len(kept)
            kept.append(rec)
            kept_shingles.append(shingles)
            for b in range(bands):
                start = b * rows_per_band
                band_key = (b, sig[start : start + rows_per_band])
                buckets[band_key].append(new_idx)

    return kept


class FourWaySplits(BaseModel):
    """Four strictly disjoint entity-grouped splits: D_train, D_val, D_T, D_tau."""

    d_train: list[DatasetRecord]
    d_val: list[DatasetRecord]
    d_t: list[DatasetRecord]
    d_tau: list[DatasetRecord]


def entity_grouped_four_way_split(
    records: list[DatasetRecord],
    ratios: tuple[float, float, float, float] = (0.70, 0.10, 0.10, 0.10),
    seed: int = 42,
) -> FourWaySplits:
    """Group by canonical entity_id and split strictly disjointly across 4 splits."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    groups: dict[str, list[DatasetRecord]] = defaultdict(list)
    for rec in records:
        groups[rec.entity_id].append(rec)

    entity_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(entity_ids)

    n = len(entity_ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_t = int(round(n * ratios[2]))
    # Ensure at least 1 entity in d_val, d_t, d_tau when n >= 4
    if n >= 4:
        n_val = max(1, n_val)
        n_t = max(1, n_t)
        n_train = max(1, min(n_train, n - n_val - n_t - 1))

    train_ids = set(entity_ids[:n_train])
    val_ids = set(entity_ids[n_train : n_train + n_val])
    t_ids = set(entity_ids[n_train + n_val : n_train + n_val + n_t])
    tau_ids = set(entity_ids[n_train + n_val + n_t :])

    return FourWaySplits(
        d_train=[r for eid in train_ids for r in groups[eid]],
        d_val=[r for eid in val_ids for r in groups[eid]],
        d_t=[r for eid in t_ids for r in groups[eid]],
        d_tau=[r for eid in tau_ids for r in groups[eid]],
    )
