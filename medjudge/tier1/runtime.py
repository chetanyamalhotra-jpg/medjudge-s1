"""Pure CPU ONNX Runtime inference for Tier 1 MedJudge-S1 (ZERO PyTorch imports)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime as ort

from medjudge.schemas.models import (
    ClinicalArtifact,
    EvalTier,
    EvidenceSpan,
    GradeDecision,
    RubricCriterion,
    SufficiencyManifest,
    Verdict,
)
from medjudge.slicers.pinned_header import SlicedInput
from medjudge.tier1.calibration import CalibrationManifest, apply_temperature

LABEL_ORDER: tuple[Verdict, Verdict, Verdict] = (
    Verdict.PASS,
    Verdict.FAIL,
    Verdict.UNKNOWN,
)


class TokenizerProtocol(Protocol):
    """Minimal tokenizer protocol satisfied by tokenizers.Tokenizer or test stubs."""

    def encode_to_arrays(
        self, text: str, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class Tier1ClassifierProtocol(Protocol):
    """Interface expected by EvaluationRouter for Tier 1 inference."""

    def predict(
        self,
        criterion: RubricCriterion,
        artifact: ClinicalArtifact,
        sliced_input: SlicedInput,
    ) -> GradeDecision:
        ...


class HFTokenizerAdapter:
    """Thin wrapper around HuggingFace `tokenizers.Tokenizer` (zero PyTorch)."""

    def __init__(self, tokenizer_json_path: str | Path) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tokenizer_json_path))
        self._tok.no_padding()

    def encode_to_arrays(
        self, text: str, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self._tok.enable_truncation(max_length=max_length)
        enc = self._tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        return ids, mask


class Tier1ONNXClassifier:
    """CPU-only ONNX Runtime classifier with post-quantization temperature scaling."""

    def __init__(
        self,
        onnx_model_path: str | Path,
        tokenizer: TokenizerProtocol | str | Path,
        calibration_manifest: CalibrationManifest | str | Path | None = None,
        max_length: int = 1024,
        intra_op_num_threads: int = 0,
    ) -> None:
        self.onnx_model_path = Path(onnx_model_path)
        self.max_length = max_length

        sess_opts = ort.SessionOptions()
        if intra_op_num_threads > 0:
            sess_opts.intra_op_num_threads = intra_op_num_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Hard requirement: universal CPU-only inference via CPUExecutionProvider.
        self.session = ort.InferenceSession(
            str(self.onnx_model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]

        if isinstance(tokenizer, (str, Path)):
            self.tokenizer: TokenizerProtocol = HFTokenizerAdapter(tokenizer)
        else:
            self.tokenizer = tokenizer

        if calibration_manifest is None:
            default_manifest = Path(__file__).resolve().parent / "calibration_manifest.json"
            self.manifest = (
                CalibrationManifest.load(default_manifest)
                if default_manifest.exists()
                else CalibrationManifest()
            )
        elif isinstance(calibration_manifest, (str, Path)):
            self.manifest = CalibrationManifest.load(calibration_manifest)
        else:
            self.manifest = calibration_manifest

    def predict_logits(self, text: str) -> np.ndarray:
        """Run CPU ONNX Runtime forward pass and return raw (3,) logits."""
        input_ids, attention_mask = self.tokenizer.encode_to_arrays(text, self.max_length)
        feeds: dict[str, Any] = {}
        if "input_ids" in self.input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = attention_mask
        # Support simplified ONNX graphs that expect a single float/int input tensor
        if not feeds and len(self.input_names) == 1:
            feeds[self.input_names[0]] = input_ids

        outputs = self.session.run(None, feeds)
        logits = np.asarray(outputs[0], dtype=np.float64)
        if logits.ndim == 2:
            logits = logits[0]
        return logits

    def predict(
        self,
        criterion: RubricCriterion,
        artifact: ClinicalArtifact,
        sliced_input: SlicedInput,
    ) -> GradeDecision:
        """Classify a sliced input and emit calibrated tri-state probabilities."""
        t0 = time.perf_counter()
        sufficiency: SufficiencyManifest = sliced_input.sufficiency

        logits = self.predict_logits(sliced_input.formatted_text)
        probs_arr = apply_temperature(logits, self.manifest.temperature)
        probs = {
            "PASS": float(probs_arr[0]),
            "FAIL": float(probs_arr[1]),
            "UNKNOWN": float(probs_arr[2]),
        }

        # Missing context is NEVER PASS -> force UNKNOWN.
        if not sufficiency.sufficient:
            verdict = Verdict.UNKNOWN
            confidence = probs["UNKNOWN"]
            rationale = (
                f"Forced UNKNOWN due to insufficient context: {', '.join(sufficiency.missing_context)}"
            )
        else:
            best_idx = int(np.argmax(probs_arr))
            verdict = LABEL_ORDER[best_idx]
            confidence = float(probs_arr[best_idx])
            rationale = (
                f"Tier 1 CPU ONNX predicted {verdict.value} "
                f"(p={confidence:.4f}, T*={self.manifest.temperature:.3f})."
            )

        spans: list[EvidenceSpan] = []
        if artifact.candidate_response:
            snippet = artifact.candidate_response[:160]
            spans.append(
                EvidenceSpan(
                    text=snippet,
                    start_char=0,
                    end_char=len(snippet),
                    source="candidate_response",
                )
            )

        elapsed_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
        return GradeDecision(
            criterion_id=criterion.id,
            criterion_version=criterion.version,
            criterion_sha256=criterion.sha256,
            verdict=verdict,
            probabilities=probs,
            calibrated_confidence=confidence,
            tier_used=EvalTier.TIER1_LOCAL_ONNX,
            evidence_spans=spans,
            sufficiency=sufficiency,
            latency_ms=elapsed_ms,
            rationale=rationale,
        )
