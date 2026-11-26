# MedJudge-S1 Dataset License Quarantine & Provenance Ledger

MedJudge-S1 enforces a strict automated license allow-list and leakage quarantine in [`medjudge/training/license_quarantine.py`](medjudge/training/license_quarantine.py). Datasets are partitioned into four tiers; only **Tier A** datasets with zero Protected Health Information (PHI) may enter `D_train`, `D_val`, `D_T`, or `D_tau` model training/calibration splits.

---

## v0.1.0 Release Dataset Accounting (`data/processed/ingestion_manifest.json`)

- **Raw Tier A Ingested:** `24,301` records (`20,000` `MedHallu` + `4,301` `MEDEC-MS` train/val; `0` `MEDEC-UW` ingested).
- **PHI/PII & Deduplication Filtering:** `10` records dropped by PHI/PII regex scan (`MRN: 10`), and `1,146` near-duplicates dropped by 5-gram MinHash LSH (`Jaccard >= 0.70`, `allow_same_entity_pairs=True`), leaving **`23,145` retained PASS/FAIL records**.
- **Synthetic `UNKNOWN` Augmentation & Total:** **`23,145 retained PASS/FAIL + 1,161 synthetic UNKNOWN (813 train / 116 val / 116 D_T / 116 D_tau) = 24,306 total`**.

| Split | Total (`n`) | Unique Entities | `PASS` | `FAIL` | Synthetic `UNKNOWN` | `MedHallu` | `MEDEC-MS` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`d_train`** | `17,013` | `8,130` | `8,129` | `8,071` | `813` | `14,668` | `2,345` |
| **`d_val`** | `2,434` | `1,161` | `1,161` | `1,157` | `116` | `2,102` (`212` `pqa_labeled`, `1,890` `pqa_artificial`) | `332` |
| **`D_T` (`d_t`)** | `2,428` | `1,161` | `1,161` | `1,151` | `116` | `2,128` | `300` |
| **`D_tau` (`d_tau`)** | `2,431` | `1,162` | `1,162` | `1,153` | `116` | `2,099` | `332` |
| **Total (`4` Splits)** | **`24,306`** | `11,614` | `11,613` | `11,532` | **`1,161`** | `20,997` | `3,309` |

### Entity-Grouped Style-Only Probe (`run_style_only_probe(..., entity_ids=...)`)
- **`MEDEC-MS`:** Passes strict `<= 0.62` style gate near chance (`AUROC = 0.5088`, `Acc = 0.5093` on `d_val` [`n=322`, `163` entities]; `AUROC = 0.5028`, `Acc = 0.5010` on `d_train(2k)` [`n=2,000`, `1,122` entities]).
- **`MedHallu`:** Exhibits lexical hedging/comparative-phrasing signal (`AUROC = 0.6864`, `Acc = 0.6429` on `d_val:medhallu_pqa_labeled` [`n=196`]; `AUROC = 0.7109`, `Acc = 0.6583` on `d_val:medhallu_pqa_artificial` [`n=1,800`]; `AUROC = 0.7075`, `Acc = 0.6573` on `d_val:medhallu_all` [`n=1,996`]).

---

## Tier A — Permissive Core (Allowed for Training & Distillation, Zero-PHI)

| Dataset | Source / Identifier | Version / Commit Policy | License | Allowed Splits | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HealthBench (non-Hard)** | `openai/healthbench` (standard/consensus subsets excluding Hard & Professional) | Pin commit SHA at ingestion | MIT | `D_train`, `D_val`, `D_T`, `D_tau` | Physician-authored clinical rubrics; must exclude HealthBench Hard & Professional. |
| **MedHallu** | `UTAustin-AIHealth/MedHallu` | Pin commit SHA at ingestion | MIT | `D_train`, `D_val`, `D_T`, `D_tau` | Grounded vs. hallucinated clinical QA (`20,000` raw ingested in v0.1.0); dedup against PubMedQA via 5-gram MinHash LSH (`J >= 0.70`). |
| **MEDEC-MS** | `mkieffer/MEDEC` (MS train/val splits only; **exclude `MEDEC-UW`**) | Pin commit SHA at ingestion | CC-BY-4.0 | `D_train`, `D_val`, `D_T`, `D_tau` | Clinical note error detection (`4,301` raw train/val ingested in v0.1.0); keep error & corrected note pairs in the same entity group. |
| **PubMedQA** | `qiaojin/PubMedQA` | Pin commit SHA at ingestion | MIT | `D_train`, `D_val`, `D_T` | Biomedical literature QA; dedup against MedHallu. |
| **ACI-Bench** | `aci-bench` (Microsoft/UW synthetic clinical encounters) | Pin release tag at ingestion | CC-BY-4.0 | `D_train`, `D_val` | De-identified/synthetic multi-turn ambient clinical encounters; suitable for slicer & history checklists. |
| **MedCaseReasoning** | `MedCaseReasoning` | Pin commit SHA at ingestion | MIT | `D_train`, `D_val` | Structured diagnostic reasoning cases. |
| **MetaMedQA** | `MetaMedQA` | Pin commit SHA at ingestion | CC-BY-4.0 | `D_train`, `D_val` | Closed-ended medical QA; group by canonical question/PMID before splitting. |
| **LongHealth** | `kbressem/LongHealth` | Pin commit SHA at ingestion | Apache-2.0 | `D_train`, `D_val`, `D_tau` | Synthetic multi-document long-context clinical records; primary benchmark for Pinned-Header Slicer validation. |

---

## Tier B — Copyleft ShareAlike (Strictly Excluded from MIT/Apache Training Weights)

| Dataset | Source / Identifier | License | Allowed Usage | Prohibited Usage |
| :--- | :--- | :--- | :--- | :--- |
| **MedCalc-Bench** | `ncbi/MedCalc-Bench` | CC-BY-SA-4.0 | Offline evaluation & validating Tier 0 deterministic math formulas | `D_train`, `D_val`, weight distillation |
| **MedR-Bench** | `MedR-Bench` | CC-BY-SA-4.0 | Offline evaluation only | `D_train`, `D_val`, weight distillation |

---

## Tier C — Hard Exclude (NEVER Ingest)

| Dataset | Reason for Hard Exclusion |
| :--- | :--- |
| **MEDEC-UW** | Governed by University of Washington Data Use Agreement (DUA); incompatible with open permissive redistribution. |
| **MedDialog** | Unspecified / ambiguous upstream license and unverified de-identification provenance. |
| **MedExQA** | Non-commercial license (`CC-BY-NC`), incompatible with commercial/Apache/MIT downstream usage. |
| **MIMIC-IV / Credentialed EHR Notes** | Credentialed access DUA and clinical text restrictions; must NEVER be ingested into MedJudge-S1 datasets. |

---

## Sealed Held-Out Test Sets (Read-Only Holdout Directory)

These sets are locked in a read-only holdout directory and **must never** appear in `D_train`, `D_val`, `D_T`, `D_tau`, or prompt examples:

1. **HealthBench Hard** (`1,000` cases) — Held out for end-to-end Tier 2 + Router evaluation.
2. **HealthBench Professional** (`525` cases) — Held out strictly; **never publicly reproduce examples or text**.
3. **MEDEC-MS Test** (`597` cases: `286` PASS / `311` FAIL) — Held out for final MEDEC error-flagging benchmark reporting (`holdout_eval_report.json`: fail-safe 100% abstention under Tier 1 gates; one-sided 95% upper confidence bound `~0.50%`, `1 - 0.05^(1/597) = 0.005005`).

---

## Anti-Leakage & Privacy Controls

1. **Automated PHI/PII Regex Scan:** All ingested text is scanned for SSNs, phone numbers, email addresses, MRN patterns, dates of birth, and street addresses prior to split assignment (`10` `MRN` hits dropped in v0.1.0).
2. **Entity-Grouped Splitting:** Records are grouped by canonical entity ID (`patient_id`, `conversation_id`, `PMID`, `scenario_id`) before splitting so correct and error versions of the same clinical note or question always remain in the same split.
3. **Cross-Dataset Deduplication:** 5-gram MinHash LSH (`Jaccard >= 0.70`) is enforced across `MedQA <-> MEDEC-MS` and `PubMedQA <-> MedHallu` (`1,146` near-duplicates dropped in v0.1.0).
4. **4 Strictly Disjoint Splits:**
   - `D_train` (`17,013` items): LoRA + classification head weights
   - `D_val` (`2,434` items): Canonical validation & checkpoint selection
   - `D_T` (`2,428` items in v0.1.0; target 150–200 clinician-audited gold items in future releases): Exclusively for post-quantization Temperature Scaling `T*`
   - `D_tau` (`2,431` items) / Sealed Holdouts: Exclusively for asymmetric routing thresholds `tau` and final held-out metrics
