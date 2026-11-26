"""Hardware, Windows guardrail, and ModernBERT backbone verification script."""

from __future__ import annotations

import gc
import sys

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from medjudge.training.env_guard import check_peak_vram_budget, validate_training_guardrails
from medjudge.training.train_lora import inspect_lora_target_modules

BACKBONES = [
    "answerdotai/ModernBERT-base",
    "thomas-sounack/BioClinical-ModernBERT-base",
]


def _inspect_linear_modules(model: nn.Module) -> tuple[set[str], list[str]]:
    leaf_names: set[str] = set()
    sample_paths: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            leaf_names.add(leaf)
            if len(sample_paths) < 20:
                sample_paths.append(f"{name} ({tuple(module.weight.shape)})")
    return leaf_names, sample_paths


def main() -> int:
    env_cfg = validate_training_guardrails(
        use_flash_attn=False,
        torch_compile=False,
        attn_implementation="sdpa",
        dataloader_num_workers=0,
        prefer_bf16=True,
    )
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False!", file=sys.stderr)
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"torch.__version__         : {torch.__version__}")
    print(f"GPU                       : {props.name}")
    print(f"Compute Capability        : {torch.cuda.get_device_capability(0)}")
    print(f"bf16 supported            : {torch.cuda.is_bf16_supported()}")
    print(f"Total VRAM (GB)           : {props.total_memory / (1024**3):.3f}")
    print(f"PYTORCH_CUDA_ALLOC_CONF   : {env_cfg.cuda_alloc_conf}")
    print(f"Peak VRAM ceiling (GB)    : {env_cfg.peak_vram_ceiling_gb:.2f}")

    dtype = torch.bfloat16 if env_cfg.use_bf16 else torch.float16

    for model_id in BACKBONES:
        print(f"\n=== Verifying Backbone: {model_id} ===")
        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"max_position_embeddings   : {config.max_position_embeddings}")
        assert config.max_position_embeddings == 8192, (
            f"Expected max_position_embeddings == 8192, got {config.max_position_embeddings}"
        )
        print(f"vocab_size / tok len      : {config.vocab_size} / {len(tokenizer)}")

        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=3,
            attn_implementation=env_cfg.attn_implementation,
            dtype=dtype,
        ).to("cuda")
        model.eval()

        leaf_names, sample_paths = _inspect_linear_modules(model)
        lora_targets = inspect_lora_target_modules(model)
        print(f"Exact nn.Linear leaf names: {sorted(leaf_names)}")
        print(f"Selected LoRA targets     : {lora_targets}")
        print("Sample qualified Linear layers:")
        for p in sample_paths:
            print(f"  - {p}")

        for seq_len in (512, 1024):
            torch.cuda.reset_peak_memory_stats()
            input_ids = torch.ones((1, seq_len), dtype=torch.long, device="cuda")
            attention_mask = torch.ones((1, seq_len), dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            peak_alloc = check_peak_vram_budget(env_cfg.peak_vram_ceiling_gb)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            print(
                f"Forward seq_len={seq_len:4d} -> logits={tuple(out.logits.shape)}, "
                f"peak_alloc={peak_alloc:.4f} GB, peak_reserved={peak_reserved:.4f} GB"
            )
            del input_ids, attention_mask, out

        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("\nALL BACKBONE VERIFICATION CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
