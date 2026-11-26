# Model Card: MedJudge-S1 (`medjudge`)

## Model Overview

**MedJudge-S1 (v0.1.0)** is a **calibrated paper-QA grounding classifier inside a 4-tier medical AI evaluation harness**. Rather than judging an entire clinical consultation with a 150M-parameter encoder, the Tier 1 model cross-encodes atomic propositions:

- **Paper-QA (`task_type: "paper_qa"`, e.g., `s2_medhallu_grounding`):**
  ```text
  [RUBRIC_CRITERION] + [REFERENCE_CONTEXT] + [CANDIDATE_RESPONSE]
  ```
  `[GLOBAL_CONTEXT_CAPSULE]` is omitted for `paper_qa` (and for `clinical_note` rubrics without clinical capsule `required_context` keys) so literature QA inputs match their training format.
- **Clinical Encounters (`task_type: "clinical_encounter"`):**
  ```text
  [RUBRIC_CRITERION] + [GLOBAL_CONTEXT_CAPSULE] + [SLICED_EVIDENCE] + [CANDIDATE_RESPONSE]
  ```

Tier 1 outputs a calibrated **Tri-State verdict** (`PASS` | `FAIL` | `UNKNOWN`) alongside calibrated probabilities, a `SufficiencyManifest`, and evidence spans. Any `UNKNOWN` verdict, `sufficient=False` context manifest, or uncertain probability outside asymmetric risk-class thresholds escalates to a Tier 2 BYOK frontier LLM-as-a-Judge (or returns `UNKNOWN` / `result: null` in `POST /v1/systemone` `mode="score"` when Tier 2 is unconfigured). In production defaults (`medjudge/rubrics/default_rubrics.yaml`), all `S1` critical-safety rubrics (`s1_allergy_medication_safety`, `s1_emergency_red_flag_referral`, `s1_dosage_safety`) and full-note `MEDEC-MS` error detection (`s2_medec_note_error_flag`) set `force_tier2: true` and bypass Tier 1 directly to Tier 2.

---

## Base Model Variants & Pretraining Provenance

1. **Primary Clinical Variant (`thomas-sounack/BioClinical-ModernBERT-base`) — Trained & Released Artifact**
   - Architecture: ModernBERT-base (~150M parameters, 8,192 native context window).
   - Weight License: MIT (on Hugging Face).
   - **Pretraining Provenance Note:** `BioClinical-ModernBERT-base` was pretrained on ~53.5B clinical and biomedical tokens (including MIMIC-IV clinical text and PubMed/PMC literature). Although the published encoder weights carry an MIT license and MedJudge-S1 fine-tuning datasets contain **zero PHI**, downstream enterprise compliance teams requiring zero-EHR pretraining lineage should train the clean-room variant below.
2. **Clean-Room Baseline (`answerdotai/ModernBERT-base`)**
   - Architecture: ModernBERT-base (~149M parameters, 8,192 native context window).
   - Weight License: Apache-2.0.
   - **Zero-EHR Provenance:** Pretrained exclusively on general web/code/text corpora with no EHR or MIMIC pretraining exposure. Verified in our 512-token RTX 3050 smoke test (`0.4068 GB` peak allocated VRAM).

---

## Intended Use & Non-Medical-Device Disclaimer

> **IMPORTANT — NOT A MEDICAL DEVICE:** MedJudge-S1 is an automated **software engineering and offline/shadow AI evaluation tool** for benchmarking, regression testing, and auditing medical AI outputs against explicit rubrics. It is **NOT** a medical device, is **NOT** cleared or approved by the FDA or any regulatory authority, and must **NEVER** be used for clinical diagnosis, treatment decisions, triage, or direct patient care.

---

## Zero-PHI Attestation & Dataset Accounting

- MedJudge-S1 training, distillation, and calibration pipelines ingest **only** Tier A permissive, publicly available synthetic or de-identified research benchmarks (`DATASETS.md`: `MedHallu` MIT, `MEDEC-MS` CC-BY-4.0).
- Credentialed EHR datasets (`MEDEC-UW`, `MIMIC-IV`), ShareAlike datasets (`MedCalc-Bench`, `MedR-Bench`), and non-commercial datasets (`MedExQA`) are hard-excluded in code (`medjudge/training/license_quarantine.py`).
- From `24,301` raw Tier A records (`20,000` `MedHallu` + `4,301` `MEDEC-MS` train/val), automated PHI/PII regex scanning dropped `10` flagged `MRN` patterns and 5-gram MinHash LSH deduplication (`J >= 0.70`, `allow_same_entity_pairs=True`) dropped `1,146` near-duplicates, leaving `23,145` retained `PASS`/`FAIL` records.
- **Explicit Split Accounting:** **`23,145 retained PASS/FAIL + 1,161 synthetic UNKNOWN (813 train / 116 val / 116 D_T / 116 D_tau) = 24,306 total`** across 4 strictly entity-disjoint splits:
  - `d_train`: `17,013` (`8,129` PASS / `8,071` FAIL / `813` UNKNOWN; `8,130` unique entities)
  - `d_val`: `2,434` (`1,161` PASS / `1,157` FAIL / `116` UNKNOWN; `1,161` unique entities)
  - `D_T` (`d_t`): `2,428` (`1,161` PASS / `1,151` FAIL / `116` UNKNOWN; `1,161` unique entities)
  - `D_tau` (`d_tau`): `2,431` (`1,162` PASS / `1,153` FAIL / `116` UNKNOWN; `1,162` unique entities)

---

## Training & Selective GeGLU INT8 Quantization

- **Training Hardware & Recipe:** Trained on a Windows 11 laptop with an **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, Ampere SM 8.6)** using `bf16=True`, `attn_implementation="sdpa"`, `torch_compile=False`, `dataloader_num_workers=0`, `gradient_checkpointing=True` (`use_reentrant=False`), `max_length=1024`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=8`, 3 epochs (`1,596` steps, cosine LR `1e-4`, `warmup_steps=96`), and LoRA (`r=16`, `alpha=32` on `['Wqkv', 'Wo', 'Wi']`) + unfrozen `classifier` head.
  - **Peak VRAM:** `0.8299 GB` allocated / `1.0410 GB` reserved (under the `3.40 GB` ceiling).
  - **Wall-Clock Time:** `6,625.03 s` (`110.4 min`).
- **Selective `mlp.Wi`-Only GeGLU INT8 Finding:**
  - LoRA-merged FP32 ONNX (`model_fp32.onnx`, `571.33 MB`, opset 17) matched PyTorch FP32 within `<= 6.2e-6` max absolute logit error.
  - Dynamic INT8 quantization of **both** `mlp.Wi` and `mlp.Wo` collapsed agreement (`15/30` agree, `max_diff=11.12`) due to heavy-tailed `GELU(gate) * up` GeGLU product activations entering `mlp.Wo`.
  - Restricting dynamic INT8 (`QuantType.QInt8`, `per_channel=True`, `reduce_range=True`, `quant_pre_process`) strictly to the **22 `mlp.Wi` nodes** while keeping `mlp.Wo`, `attn.*`, `LayerNorm`, `Softmax`, and `head.*`/`classifier` in FP32 produced `model_int8.onnx` (`460.22 MB`).
  - **Single-Item Unpadded Inference Requirement:** Canonical router parity checks (`20/20` items) confirmed **`0.0` max probability difference** between standalone `Tier1ONNXClassifier` and `EvaluationRouter.evaluate`, whereas `batch_size=16` padded ONNX diverged up to `0.0583` due to dynamic INT8 activation scaling over padding tokens. Production and canonical evaluation run strictly unpadded single-item (`batch_size=1`) CPU inference.

---

## Calibration & Measured Evaluation Results

All `d_val` numbers below come strictly from [`artifacts/bioclinical_modernbert_int8/canonical_eval_report.json`](artifacts/bioclinical_modernbert_int8/canonical_eval_report.json) (exact deployed pipeline: `PinnedHeaderSlicer(max_tokens=1024)` + single-item unpadded `model_int8.onnx` + `T* = 1.395134`, `torch` absent from `sys.modules`, `0` overflow slices, `2,434/2,434` `[GLOBAL_CONTEXT_CAPSULE]` omitted for `paper_qa` / `clinical_note`).

### Post-INT8 Calibration (`calibration_manifest.json`)
- **Temperature Scaling (`D_T`, `n = 2,428`):** `T* = 1.395134` (L-BFGS minimizing NLL).
- **Disjoint Threshold Split (`D_tau`, `n = 2,431`):** `rule_of_three_upper_bound = 0.001234` (`3 / 2431`).
- **Asymmetric Thresholds:** `S1` (`PASS >= 0.97`, `FAIL >= 0.85`; bypassed via `force_tier2: true` in defaults), `S2` (`PASS >= 0.92`, `FAIL >= 0.90`), `S3` (`PASS >= 0.80`, `FAIL >= 0.80`).

### Canonical `d_val` Evaluation (`n = 2,434`, 10,000-Resample Paired Entity-Cluster 95% CIs)

| Evaluation Slice | `n` (`Entities`) | Accuracy `[95% CI]` | PASS/FAIL Macro-F1 `[95% CI]` | PASS F1 | FAIL F1 | Synthetic UNKNOWN F1 | 3-Class Macro-F1 `[95% CI]` | FAIL Recall `[95% CI]` | ECE `[95% CI]` | Brier `[95% CI]` | AURC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`d_val_overall`** | 2,434 (1,161) | `0.9043` `[0.8928, 0.9157]` | `0.8995` `[0.8874, 0.9115]` | `0.8979` | `0.9011` | `1.0000` | `0.9330` `[0.9249, 0.9410]` | `0.9170` `[0.9012, 0.9326]` | `0.0140` `[0.0088, 0.0212]` | `0.1102` `[0.0988, 0.1220]` | `0.0111` |
| **`medhallu_overall`** | 2,102 (998) | **`0.9657`** `[0.9579, 0.9733]` | **`0.9639`** `[0.9559, 0.9719]` | `0.9641` | `0.9637` | `1.0000` | `0.9760` `[0.9706, 0.9813]` | `0.9579` `[0.9449, 0.9699]` | `0.0102` `[0.0060, 0.0174]` | `0.0504` `[0.0413, 0.0601]` | `0.0026` |
| ↳ `medhallu_pqa_labeled` | 212 (98) | **`0.9670`** `[0.9420, 0.9904]` | **`0.9643`** `[0.9387, 0.9898]` | `0.9648` | `0.9637` | `1.0000` | `0.9762` `[0.9591, 0.9932]` | `0.9490` `[0.8980, 0.9898]` | `0.0151` `[0.0115, 0.0410]` | `0.0470` `[0.0210, 0.0793]` | `0.0045` |
| ↳ `medhallu_pqa_artificial` | 1,890 (900) | **`0.9656`** `[0.9575, 0.9734]` | **`0.9639`** `[0.9556, 0.9722]` | `0.9641` | `0.9637` | `1.0000` | `0.9759` `[0.9704, 0.9815]` | `0.9589` `[0.9456, 0.9711]` | `0.0101` `[0.0059, 0.0178]` | `0.0508` `[0.0414, 0.0608]` | `0.0024` |
| **`medec_ms` (`d_val`, standalone eval*)** | 332 (163) | `0.5151` `[0.4862, 0.5443]` | `0.4881` `[0.4562, 0.5181]` | `0.4103` | `0.5660` | `1.0000` | `0.6588` `[0.6375, 0.6787]` | `0.6604` `[0.5849, 0.7325]` | `0.0408` `[0.0169, 0.0666]` | `0.4884` `[0.4790, 0.4972]` | `0.4416` |

*\*`medec_ms` was evaluated with a `force_tier2=False` override on `s2_medec_note_error_flag` solely to measure standalone Tier 1 performance; in production `s2_medec_note_error_flag` has `force_tier2=True` and routes directly to Tier 2.*

### Risk-Coverage Sweep (`canonical_eval_report.json`)

Evaluated symmetrically (`max(p_PASS, p_FAIL) >= tau`) and under the deployed asymmetric S2 gate (`PASS >= 0.92`, `FAIL >= 0.90`). For all `tau >= 0.70` and the deployed S2 gate, `0` `MEDEC-MS` items are accepted (all `1,908` down to `1,210` and `1,738` accepted items are `MedHallu`):

| Gate / Threshold | `d_val` Accepted Cov (%) | `d_val` Esc Rate (%) | `MedHallu` Accepted Cov (%) | `MedHallu` Esc Rate (%) | Accepted Accuracy | Accepted False-Pass Rate (%) [Count] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tau = 0.50` | `94.41%` | `5.59%` | `94.96%` | `5.04%` | `0.9017` (`d_val`) / `0.9639` (`MedHallu`) | `4.0905%` [94] (`d_val`) / `2.1042%` [42] (`MedHallu`) |
| `tau = 0.70` | `78.39%` | `21.61%` | `90.77%` | `9.23%` | `0.9801` | `1.0482%` [20] |
| `tau = 0.80` | `76.46%` | `23.54%` | `88.53%` | `11.47%` | `0.9860` | `0.8060%` [15] |
| `tau = 0.85` | `74.73%` | `25.27%` | `86.54%` | `13.46%` | `0.9885` | `0.6597%` [12] |
| `tau = 0.90` | `72.23%` | `27.77%` | `83.63%` | `16.37%` | `0.9915` | `0.5119%` [9] |
| `tau = 0.92` | `70.67%` | `29.33%` | `81.83%` | `18.17%` | `0.9948` | `0.4070%` [7] |
| `tau = 0.95` | `66.80%` | `33.20%` | `77.35%` | `22.65%` | `0.9975` | `0.1845%` [3] |
| `tau = 0.97` | `61.96%` | `38.04%` | `71.74%` | `28.26%` | `0.9980` | `0.1989%` [3] |
| `tau = 0.99` | `49.71%` | `50.29%` | `57.56%` | `42.44%` | `0.9992` | `0.0826%` [1] |
| **Deployed S2 Gate (`0.92` / `0.90`)** | **`71.41%`** | **`28.59%`** | **`82.68%`** | **`17.32%`** | **`0.9925`** | **`0.4028%`** [7] |

### Sealed Held-Out `MEDEC-MS Test` (`holdout_eval_report.json`, `n = 597`: `286` PASS / `311` FAIL)
- **Standalone Tier 1 INT8:** `Accuracy = 0.5025` (`95% CI [0.4623, 0.5427]`), `Macro-F1 = 0.4709` (`95% CI [0.4317, 0.5113]`), `FAIL Recall = 0.7170` (`95% CI [0.6677, 0.7669]`), `ECE = 0.0275`, `Brier = 0.5017`, `AURC = 0.4852` (10,000 paired source-cluster bootstrap resamples).
- **Fail-Safe 100% Abstention & Production Routing:**
  - Under standalone confidence gating (`S1` `0.97/0.85` and `S2` `0.92/0.90`), Tier 1 exhibits **fail-safe 100% abstention** (`597 / 597` escalated; `0.0%` auto-accepted; `0 / 597` catastrophic false passes), corresponding to a one-sided 95% upper confidence bound of `~0.50%` (`1 - 0.05^(1/597) = 0.005005`) on the auto-accepted false-pass rate.
  - Because standalone Tier 1 performs near chance on holistic full-note error detection, production `s2_medec_note_error_flag` additionally sets `force_tier2: true` so full-note MEDEC evaluations bypass Tier 1 directly to Tier 2.

### CPU Inference Latency (`CPUExecutionProvider`, `batch_size=1`, `model_int8.onnx`)
- **Measured CPU Hardware:** **`AMD Ryzen 7 170 with Radeon Graphics`** (`AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`, Windows 11, Python 3.11.16)

| Sequence Length | p50 Latency | p95 Latency |
| :---: | :---: | :---: |
| **256 tokens** | `221.09 ms` | `228.92 ms` |
| **512 tokens** | `592.63 ms` | `601.55 ms` |
| **1,024 tokens** | `1,931.84 ms` | `1,975.10 ms` |

### Prompt-Injection Robustness (`robustness_report.json`)
- **Prompt-Injection (`50` `FAIL` items from `d_val` with adversarial override suffixes):**
  - Clean `FAIL` Recall `0.7600` -> Injected `FAIL` Recall `0.4800` (`48.0%` / `24/50` raw argmax flip).
  - **Auto-Accepted Attack Success Rate (ASR):** **`0.0%` (`0 / 50`)** under `S1` (`p >= 0.97`) and **`4.0%` (`2 / 50`)** under `S2` (`p >= 0.92`). (In production defaults, `S1` rubrics also enforce `force_tier2=True`.)

---

## Audit & Logging Guarantees

- **Append-Only JSONL Audit Log (`medjudge/audit/logger.py`):** `JSONLAuditLogger` writes one `AuditRecord` JSON line per decision in append mode (`"a"`), recording `artifact_sha256` and `criterion_sha256`. It is **not hash-chained or tamper-evident** (no `prev_record_sha256`, chain hashing, or HMAC/signatures are computed, so on-disk edits, deletions, or reorderings are not cryptographically detected).
- **Fail-Safe API & Missing-Context Guards:** Whenever `Verdict == UNKNOWN` or `SufficiencyManifest.sufficient == False`, `POST /v1/systemone` returns `result: null` in `mode="score"` (and `"unknown"` in `yes_no` / `"UNKNOWN"` or the 3rd choice in `choice`), and `EvaluationRouter` prevents `PASS` across all tiers when required clinical context is missing.

---

## Known Limitations

1. **Synthetic `UNKNOWN` Tell (`UNKNOWN F1 = 1.0000`):** The `UNKNOWN` class in v0.1.0 was synthesized by stripping decisive context (`[OMITTED_REFERENCE_CONTEXT]`, `[DECISIVE_NOTE_CONTEXT_OMITTED]`), yielding an artificially perfect `1.0000` F1 on synthetic omissions (which inflates 3-class Macro-F1 relative to PASS/FAIL Macro-F1: `0.9330` vs. `0.8995` on `d_val_overall`). Real-world ambiguity relies on `T*` probability calibration and deterministic `SufficiencyManifest` checks.
2. **Holistic Full-Note Error Detection (`MEDEC-MS`):** Trained jointly with ~86% `MedHallu` grounding data, the 150M Tier 1 encoder is a calibrated **paper-QA grounding classifier** (`0.9639` PASS/FAIL Macro-F1 on `MedHallu`) and performs near chance (`0.5025` Acc / `0.4709` Macro-F1 standalone on `MEDEC-MS Test`; `0.5151` Acc / `0.4881` PASS/FAIL Macro-F1 on `MEDEC-MS` `d_val`) on unlocalized full-note `MEDEC-MS` clinical reasoning. In v0.1.0, `s2_medec_note_error_flag` and all `S1` rubrics enforce `force_tier2: true`.
3. **Lexical Style Signal in `MedHallu` (Entity-Grouped Style Probe):** Using entity-grouped cross-validation folds and strictly in-fold feature standardization (`run_style_only_probe(..., entity_ids=...)`), bag-of-n-grams + length logistic regression is at chance on `MEDEC-MS` (`AUROC = 0.5088`, `Acc = 0.5093` on `d_val`; `AUROC = 0.5028`, `Acc = 0.5010` on `d_train(2k)` — passing the `<= 0.62` strict gate), whereas `MedHallu` exhibits lexical hedging/comparative-phrasing signal (`AUROC = 0.6864` on `d_val:medhallu_pqa_labeled`, `0.7109` on `d_val:medhallu_pqa_artificial`, `0.7075` on `d_val:medhallu_all`; top `+` PASS tokens: `may`, `be`, `our`; top `-` FAIL tokens: `significantly`, `through`, `compared_to`). Future iterations (v0.2+) should add paraphrased positives and multi-teacher minimal pairs.
4. **Slicer & Capsule Boundaries:** `PinnedHeaderSlicer` omits `[GLOBAL_CONTEXT_CAPSULE]` for `paper_qa` and non-capsule `clinical_note` rubrics, splits oversized single-paragraph contexts to protect `[CANDIDATE_RESPONSE]` from right-truncation, and forces `sufficient=False` / `UNKNOWN` when required clinical encounter fields are missing.
