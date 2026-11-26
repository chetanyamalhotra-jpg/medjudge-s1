# MedJudge-S1 (`medjudge`)

**MedJudge-S1 (v0.1.0)** is an open-source, locally runnable **calibrated paper-QA grounding classifier inside a 4-tier BYOK (Bring Your Own Key) medical evaluation harness**.

Instead of asking a 150M-parameter model to judge an entire clinical consultation, MedJudge-S1 evaluates **atomic clinical propositions** and emits a **Tri-State verdict (`PASS` | `FAIL` | `UNKNOWN`)**, calibrated probabilities, a `SufficiencyManifest`, and evidence spans—paired with deterministic Tier 0 verifiers, a `GlobalContextCapsule` pinned-header slicer, and asymmetric confidence-gated escalation (plus `force_tier2` safety overrides) to a Tier 2 frontier LLM-as-a-Judge.

## Architecture

1. **Tier 0 (`medjudge/tier0/`):** Zero-ML deterministic verifiers (MedCalc math/units, FHIR R4 schema checks, HealthBench Professional length penalties, DOI/PMID checks, MCQ normalization).
2. **Tier 1 (`medjudge/tier1/` + `medjudge/slicers/`):** Pure CPU **ONNX Runtime** (`CPUExecutionProvider`, **zero PyTorch dependency**) running selective `mlp.Wi`-only INT8 quantized `BioClinical-ModernBERT-base` (`460.22 MB`, `T* = 1.395134`, single-item unpadded `batch_size=1` inference). Default `paper_qa` (`s2_medhallu_grounding`) omits `[GLOBAL_CONTEXT_CAPSULE]` to match training formatting, whereas `clinical_encounter` rubrics prepend it and enforce `SufficiencyManifest`.
3. **Tier 2 (`medjudge/tier2/`):** Async OpenAI-compatible BYOK LLM-as-a-Judge with bounded exponential backoff (raises cleanly on unrecoverable errors; handles `force_tier2` S1 critical-safety and `MEDEC-MS` full-note rubrics, `UNKNOWN`, missing context, uncertain confidence, and holistic criteria).
4. **Tier 3 (`medjudge/tier3/`):** Optional Agent-as-a-Judge hook for live PubMed, clinical guideline, or stateful FHIR verification.
5. **Audit Logger (`medjudge/audit/logger.py`):** `JSONLAuditLogger` writes append-only JSONL `AuditRecord` entries containing `artifact_sha256` and `criterion_sha256` (append-only JSONL, **not** hash-chained or tamper-evident).
6. **FastAPI Adapter (`medjudge/api/`):** Drop-in `POST /v1/systemone` endpoint (`yes_no`, `choice`, `score` + `confidence`) compatible with Pydantic AI `SystemOneModel` and Ollama 0.35+ tooling. In `mode="score"`, `UNKNOWN` verdicts or `sufficient=False` contexts return `result: null`.

## Canonical Benchmark Highlights (`BioClinical-ModernBERT-base` INT8 ONNX)

All `d_val` metrics are from [`artifacts/bioclinical_modernbert_int8/canonical_eval_report.json`](artifacts/bioclinical_modernbert_int8/canonical_eval_report.json) (`2,434` items from `23,145 retained PASS/FAIL + 1,161 synthetic UNKNOWN (813 train / 116 val / 116 D_T / 116 D_tau) = 24,306 total`):

- **Paper-QA Grounding (`MedHallu` `d_val`, `n = 2,102`):** `96.57%` accuracy (`0.9639` PASS/FAIL Macro-F1; `0.9760` 3-class Macro-F1; `ECE = 0.0102`), with `96.70%` accuracy (`0.9643` PASS/FAIL Macro-F1) on `pqa_labeled` (`n = 212`) and `96.56%` accuracy (`0.9639` PASS/FAIL Macro-F1) on `pqa_artificial` (`n = 1,890`). Under the deployed S2 gate (`0.92` PASS / `0.90` FAIL), Tier 1 handles `82.68%` of `MedHallu` locally at `99.25%` accepted accuracy and `0.4028%` (`7/1,738`) accepted false-pass rate.
- **Sealed `MEDEC-MS Test` (`holdout_eval_report.json`, `n = 597`):** Standalone Tier 1 achieves `50.25%` accuracy (`0.4709` Macro-F1) on unlocalized full-note clinical reasoning; confidence gating yields **fail-safe 100% abstention** (`597/597` escalated; `0/597` catastrophic false passes; one-sided 95% upper confidence bound `~0.50%`, `1 - 0.05^(1/597) = 0.005005`), and production `s2_medec_note_error_flag` sets `force_tier2: true`.
- **Prompt-Injection ASR (`robustness_report.json`, `50` `FAIL` items):** `0.0%` (`0/50`) under `S1` (`p >= 0.97`, also `force_tier2: true` in defaults) and `4.0%` (`2/50`) under `S2` (`p >= 0.92`).
- **CPU Latency (`AMD Ryzen 7 170 with Radeon Graphics`, `CPUExecutionProvider`, `batch_size=1`):** `221.09 ms` (`p50`) / `228.92 ms` (`p95`) @ 256 tokens, `592.63 ms` / `601.55 ms` @ 512 tokens, `1,931.84 ms` / `1,975.10 ms` @ 1,024 tokens.

See [MODEL_CARD.md](MODEL_CARD.md) and [DATASETS.md](DATASETS.md) for full canonical metrics, risk-coverage tables, quantization findings, and dataset provenance.

## Installation & Quickstart

> **Model Weights & Datasets Note:** Large ONNX model binaries (`model_int8.onnx`, `model_fp32.onnx`), LoRA checkpoints, and raw/processed/holdout datasets are excluded from Git (only lightweight evaluation reports, calibration manifest, and plots are tracked). Standalone Hugging Face weights hosting is planned separately; you can reproduce the calibrated `model_int8.onnx` artifact locally end-to-end with:
> `scripts/ingest_tier_a.py` -> `scripts/add_unknown_and_probe.py` -> `scripts/train_bioclinical_lora.py` -> `scripts/export_and_calibrate_int8.py`.

### Inference & `/v1/systemone` API (Zero PyTorch)

```powershell
uv venv --python 3.11
.\.venv\Scripts\python.exe -m pip --version
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev]"

# Verify calibrated INT8 ONNX via /v1/systemone (zero PyTorch)
.\.venv\Scripts\python.exe scripts/serve_systemone.py --smoke-test

# Start FastAPI server
.\.venv\Scripts\python.exe scripts/serve_systemone.py --host 127.0.0.1 --port 8000
```

### Optional Training & Export Stack (Windows 11, RTX 3050 4 GB)

Install CUDA PyTorch first (never CPU-only torch), then training/eval dependencies:

```powershell
uv pip install --python .\.venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu126 torch
uv pip install --python .\.venv\Scripts\python.exe "transformers>=4.48.0" "peft>=0.13.0" "datasets>=3.0.0" "accelerate>=1.0.0" "onnxscript>=0.1.0" "scikit-learn>=1.5.0" "datasketch>=1.6.0" "matplotlib>=3.9.0"
```

> **Windows 11 Guardrails:** `medjudge.training.env_guard` enforces `bf16=True` (when supported), `attn_implementation="sdpa"`, `torch_compile=False`, `dataloader_num_workers=0`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`, and a **3.4 GB peak VRAM** ceiling (measured peak during 3-epoch 1024-token LoRA training: `0.83 GB` allocated / `1.04 GB` reserved). Never install `flash-attn` or use OpenVINO for ModernBERT.

## Running Tests & Lint

```powershell
.\.venv\Scripts\pytest.exe -v
.\.venv\Scripts\ruff.exe check medjudge tests scripts
```
