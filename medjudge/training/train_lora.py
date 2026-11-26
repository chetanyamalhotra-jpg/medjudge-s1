"""Zero-OOM LoRA training recipe for RTX 3050 4 GB Windows 11 (BioClinical-ModernBERT & ModernBERT)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medjudge.training.env_guard import check_peak_vram_budget, validate_training_guardrails

ALLOWED_BASE_MODELS = {
    "thomas-sounack/BioClinical-ModernBERT-base",
    "answerdotai/ModernBERT-base",
}
DISQUALIFIED_BASE_MODELS = {
    "google/medgemma-4b-it",
    "google/medgemma-4b-pt",
    "medgemma-4b",
}


@dataclass(frozen=True)
class TrainingStageConfig:
    """Configuration for a single stage in the 3-stage MedJudge-S1 training progression."""

    stage_name: str
    base_model: str = "thomas-sounack/BioClinical-ModernBERT-base"
    max_length: int = 512
    freeze_backbone: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    group_by_length: bool = True
    pad_to_multiple_of: int = 8
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    output_dir: str = "artifacts/medjudge_s1"
    allow_short_ablation: bool = False
    candidate_linear_names: tuple[str, ...] = field(default=("Wqkv", "Wo", "Wi"))


def validate_base_model(model_name: str, allow_short_ablation: bool = False) -> str:
    """Enforce allowed ModernBERT models and reject disqualified 4B / legacy 512 encoders."""
    lower = model_name.strip().lower()
    if any(bad in lower for bad in DISQUALIFIED_BASE_MODELS) or "medgemma" in lower:
        raise ValueError(
            f"Model '{model_name}' is disqualified (OOMs >4.19 GB on 4-bit QLoRA + restrictive HAI-DEF license)."
        )
    if not allow_short_ablation and any(
        legacy in lower for legacy in ("pubmedbert", "deberta-v3")
    ):
        raise ValueError(
            f"Legacy 512-token model '{model_name}' is only permitted with allow_short_ablation=True."
        )
    if model_name not in ALLOWED_BASE_MODELS and not allow_short_ablation:
        raise ValueError(
            f"Base model '{model_name}' must be one of {sorted(ALLOWED_BASE_MODELS)}."
        )
    return model_name


def inspect_lora_target_modules(
    model: Any,
    preferred_names: tuple[str, ...] = ("Wqkv", "Wo", "Wi"),
) -> list[str]:
    """Inspect actual torch.nn.Linear module names before attaching LoRA adapters."""
    import torch.nn as nn  # noqa: PLC0415

    discovered: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            # Exclude classification head from LoRA; unfreeze head directly.
            if leaf in {"classifier", "score", "head", "out_proj"}:
                continue
            discovered.add(leaf)

    matched = [n for n in preferred_names if n in discovered]
    if matched:
        return matched
    return sorted(discovered)


def build_training_progression(
    base_model: str = "thomas-sounack/BioClinical-ModernBERT-base",
    output_root: str = "artifacts",
) -> list[TrainingStageConfig]:
    """Return the mandatory 512-sanity -> LoRA-512 -> 1024 -> 2048 training progression."""
    validate_base_model(base_model)
    return [
        TrainingStageConfig(
            stage_name="stage1_sanity_frozen_512",
            base_model=base_model,
            max_length=512,
            freeze_backbone=True,
            use_lora=False,
            gradient_accumulation_steps=16,
            learning_rate=1e-3,
            num_train_epochs=1.0,
            output_dir=str(Path(output_root) / "stage1_sanity_512"),
        ),
        TrainingStageConfig(
            stage_name="stage2_lora_512",
            base_model=base_model,
            max_length=512,
            freeze_backbone=False,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            gradient_accumulation_steps=16,
            output_dir=str(Path(output_root) / "stage2_lora_512"),
        ),
        TrainingStageConfig(
            stage_name="stage3_context_ramp_1024",
            base_model=base_model,
            max_length=1024,
            freeze_backbone=False,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            gradient_accumulation_steps=16,
            output_dir=str(Path(output_root) / "stage3_ramp_1024"),
        ),
        TrainingStageConfig(
            stage_name="stage4_context_ramp_2048",
            base_model=base_model,
            max_length=2048,
            freeze_backbone=False,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            gradient_accumulation_steps=32,
            output_dir=str(Path(output_root) / "stage4_ramp_2048"),
        ),
    ]


def run_training_stage(
    cfg: TrainingStageConfig,
    train_dataset: Any,
    eval_dataset: Any,
) -> Path:
    """Execute one zero-OOM training stage on Windows 11 RTX 3050 4 GB."""
    import inspect  # noqa: PLC0415
    import math  # noqa: PLC0415

    validate_base_model(cfg.base_model, allow_short_ablation=cfg.allow_short_ablation)
    env_cfg = validate_training_guardrails(
        use_flash_attn=False,
        torch_compile=False,
        attn_implementation="sdpa",
        dataloader_num_workers=0,
        prefer_bf16=True,
    )

    import torch  # noqa: PLC0415
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    class _NanGuardCallback(TrainerCallback):
        def __init__(self) -> None:
            self.saw_non_finite = False

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if not logs:
                return
            for key in ("loss", "grad_norm", "eval_loss"):
                if key in logs and isinstance(logs[key], (int, float)):
                    if not math.isfinite(float(logs[key])):
                        self.saw_non_finite = True
                        raise FloatingPointError(f"Non-finite training metric {key}={logs[key]}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model_kwargs: dict[str, Any] = {
        "num_labels": 3,
        "attn_implementation": env_cfg.attn_implementation,
    }
    load_sig = inspect.signature(AutoModelForSequenceClassification.from_pretrained)
    dtype_val = torch.bfloat16 if env_cfg.use_bf16 else torch.float32
    if "dtype" in load_sig.parameters:
        model_kwargs["dtype"] = dtype_val
    else:
        model_kwargs["torch_dtype"] = dtype_val

    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, **model_kwargs)

    if cfg.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False
    elif cfg.use_lora:
        target_modules = inspect_lora_target_modules(model, cfg.candidate_linear_names)
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, lora_cfg)

    if cfg.gradient_checkpointing and not cfg.freeze_backbone:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_kwargs: dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "bf16": env_cfg.use_bf16,
        "fp16": False,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "dataloader_num_workers": env_cfg.dataloader_num_workers,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "torch_compile": False,
        "logging_steps": 1,
        "report_to": "none",
    }
    if "group_by_length" in ta_sig.parameters:
        ta_kwargs["group_by_length"] = cfg.group_by_length

    training_args = TrainingArguments(**ta_kwargs)
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, pad_to_multiple_of=cfg.pad_to_multiple_of
    )
    nan_cb = _NanGuardCallback()
    trainer_sig = inspect.signature(Trainer.__init__)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "callbacks": [nan_cb],
    }
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    check_peak_vram_budget(env_cfg.peak_vram_ceiling_gb)

    # Verify all trainable parameters remain finite after training
    for p_name, param in model.named_parameters():
        if param.requires_grad and not torch.isfinite(param).all():
            raise FloatingPointError(f"Non-finite weights detected in '{p_name}'")

    out_path = Path(cfg.output_dir) / "best_checkpoint"
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    return out_path
