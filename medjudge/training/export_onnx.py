"""LoRA merge, FP32 ONNX export (opset >= 17), and selective FFN-only dynamic INT8 quantization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

# Patterns identifying Attention/QKV, LayerNorm, Softmax, and final head nodes to EXCLUDE from INT8.
EXCLUDED_NODE_SUBSTRINGS = (
    "attn",
    "attention",
    "qkv",
    "wqkv",
    "query",
    "key",
    "value",
    "layernorm",
    "layer_norm",
    "softmax",
    "classifier",
    "head",
    "pooler",
    "out_proj",
)
# Patterns identifying MLP/FFN linear layers eligible for selective dynamic INT8 quantization.
FFN_INCLUDE_SUBSTRINGS = ("mlp", "ffn", "wi", "wo", "intermediate", "fc1", "fc2")


def select_ffn_quantization_nodes(
    onnx_model_path: str | Path,
    exclude_geglu_wo: bool = False,
) -> list[str]:
    """Return ONLY MLP/FFN MatMul/Gemm node names, excluding Attention, LayerNorm, Softmax, and head."""
    model = onnx.load(str(onnx_model_path))
    selected: list[str] = []
    for node in model.graph.node:
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        if not node.name:
            continue
        lower = node.name.lower()
        if any(excl in lower for excl in EXCLUDED_NODE_SUBSTRINGS):
            continue
        if exclude_geglu_wo and "wo" in lower:
            continue
        if any(incl in lower for incl in FFN_INCLUDE_SUBSTRINGS):
            selected.append(node.name)
    return selected


def verify_onnx_logit_parity(
    ref_logits: np.ndarray,
    candidate_onnx_path: str | Path,
    feeds: dict[str, np.ndarray],
    max_abs_tol: float = 1e-4,
) -> float:
    """Assert max absolute logit error between reference logits and CPU ONNX Runtime session."""
    sess = ort.InferenceSession(
        str(candidate_onnx_path), providers=["CPUExecutionProvider"]
    )
    out = np.asarray(sess.run(None, feeds)[0], dtype=np.float64)
    diff = float(np.max(np.abs(np.asarray(ref_logits, dtype=np.float64) - out)))
    if diff > max_abs_tol:
        raise AssertionError(
            f"ONNX logit parity check failed: max_abs_diff={diff:.6e} > tol={max_abs_tol:.6e}"
        )
    return diff


def selective_dynamic_int8_quantize(
    fp32_onnx_path: str | Path,
    int8_onnx_path: str | Path,
    per_channel: bool = True,
    sample_feeds: dict[str, np.ndarray] | None = None,
    parity_tol: float = 0.25,
    exclude_geglu_wo: bool = False,
    use_pre_process: bool = False,
    reduce_range: bool = False,
) -> tuple[Path, list[str]]:
    """Quantize ONLY MLP/FFN MatMul/Gemm nodes to INT8; preserve Attention/Norm/Softmax/Head."""
    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415

    fp32_path = Path(fp32_onnx_path)
    int8_path = Path(int8_onnx_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    pre_path: Path | None = None
    quant_input = fp32_path
    if use_pre_process:
        from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: PLC0415

        pre_path = int8_path.with_name(f"._pre_{int8_path.name}")
        quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=False)
        quant_input = pre_path

    try:
        ffn_nodes = select_ffn_quantization_nodes(
            quant_input, exclude_geglu_wo=exclude_geglu_wo
        )
        if not ffn_nodes:
            raise ValueError(
                f"No eligible MLP/FFN MatMul/Gemm nodes found in '{quant_input}' for selective INT8."
            )

        quantize_dynamic(
            model_input=str(quant_input),
            model_output=str(int8_path),
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=per_channel,
            reduce_range=reduce_range,
            weight_type=QuantType.QInt8,
            nodes_to_quantize=ffn_nodes,
        )
    finally:
        if pre_path is not None and pre_path.exists():
            pre_path.unlink()

    if sample_feeds is not None:
        fp32_sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        int8_sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        fp32_logits = np.asarray(fp32_sess.run(None, sample_feeds)[0], dtype=np.float64)
        int8_logits = np.asarray(int8_sess.run(None, sample_feeds)[0], dtype=np.float64)
        max_diff = float(np.max(np.abs(fp32_logits - int8_logits)))
        if max_diff > parity_tol:
            raise AssertionError(
                f"Post-INT8 logit drift {max_diff:.4f} exceeded parity_tol={parity_tol:.4f}"
            )
        if not np.array_equal(np.argmax(fp32_logits, axis=-1), np.argmax(int8_logits, axis=-1)):
            raise AssertionError("Post-INT8 argmax decision parity failed on sample inputs.")

    return int8_path, ffn_nodes


def merge_and_export_fp32_onnx(
    checkpoint_dir: str | Path,
    fp32_onnx_path: str | Path,
    sample_text: str = "Patient has penicillin allergy. Candidate recommends amoxicillin.",
    max_length: int = 512,
    opset_version: int = 17,
) -> Path:
    """Merge LoRA weights (`merge_and_unload()`), export FP32 ONNX (opset>=17), and verify <1e-4 parity."""
    if opset_version < 17:
        raise ValueError("ONNX opset_version must be >= 17")

    import inspect  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from peft import PeftConfig, PeftModel  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    ckpt = Path(checkpoint_dir)
    out_path = Path(fp32_onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    load_sig = inspect.signature(AutoModelForSequenceClassification.from_pretrained)
    dtype_kw = {"dtype": torch.float32} if "dtype" in load_sig.parameters else {"torch_dtype": torch.float32}

    if (ckpt / "adapter_config.json").exists():
        peft_cfg = PeftConfig.from_pretrained(str(ckpt))
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path,
            num_labels=3,
            attn_implementation="sdpa",
            **dtype_kw,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(ckpt))
        model = peft_model.merge_and_unload()
    else:
        base_or_peft: Any = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt), attn_implementation="sdpa", **dtype_kw
        )
        model = (
            base_or_peft.merge_and_unload()
            if isinstance(base_or_peft, PeftModel)
            else base_or_peft
        )

    model.eval()
    enc = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        ref_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu().numpy()

    export_sig = inspect.signature(torch.onnx.export)
    export_kwargs: dict[str, Any] = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        },
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    if "dynamo" in export_sig.parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(out_path),
        **export_kwargs,
    )

    verify_onnx_logit_parity(
        ref_logits,
        out_path,
        {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        },
        max_abs_tol=1e-4,
    )
    return out_path
