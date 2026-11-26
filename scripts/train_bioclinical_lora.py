"""Full 3-epoch LoRA training for BioClinical-ModernBERT-base on RTX 3050 4 GB."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from medjudge.schemas.models import Verdict
from medjudge.training.env_guard import check_peak_vram_budget, validate_training_guardrails
from medjudge.training.train_lora import inspect_lora_target_modules, validate_base_model

LABEL_MAP = {Verdict.PASS.value: 0, Verdict.FAIL.value: 1, Verdict.UNKNOWN.value: 2}
MODEL_ID = "thomas-sounack/BioClinical-ModernBERT-base"
OUTPUT_DIR = Path("artifacts/bioclinical_modernbert_lora")


class _NanGuardCallback(TrainerCallback):
    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        for key in ("loss", "grad_norm", "eval_loss"):
            if key in logs and isinstance(logs[key], (int, float)):
                if not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f"Non-finite training metric {key}={logs[key]}")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                obj = json.loads(line)
                rows.append({"text": obj["text"], "label": LABEL_MAP[obj["label"]]})
    return rows


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    per_class_f1 = f1_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    per_class_recall = recall_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    macro_f1 = float(np.mean(per_class_f1))
    return {
        "macro_f1": macro_f1,
        "pass_f1": float(per_class_f1[0]),
        "fail_f1": float(per_class_f1[1]),
        "unknown_f1": float(per_class_f1[2]),
        "fail_recall": float(per_class_recall[1]),
    }


def main() -> int:
    validate_base_model(MODEL_ID)
    env_cfg = validate_training_guardrails(
        use_flash_attn=False,
        torch_compile=False,
        attn_implementation="sdpa",
        dataloader_num_workers=0,
        prefer_bf16=True,
    )

    probe_path = Path("artifacts/vram_probe_1024.json")
    batch_size = 4
    grad_accum = 8
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        batch_size = int(probe["batch_size"])
        grad_accum = int(probe["grad_accum"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_rows = _load_jsonl(Path("data/processed/d_train.jsonl"))
    val_rows = _load_jsonl(Path("data/processed/d_val.jsonl"))

    train_ds = Dataset.from_list(train_rows).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=1024),
        batched=True,
        remove_columns=["text"],
    )
    val_ds = Dataset.from_list(val_rows).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=1024),
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=3,
        attn_implementation=env_cfg.attn_implementation,
        dtype=torch.bfloat16 if env_cfg.use_bf16 else torch.float32,
    )
    targets = inspect_lora_target_modules(model, ("Wqkv", "Wo", "Wi"))
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=targets,
            modules_to_save=["classifier"],
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    steps_per_epoch = math.ceil(len(train_ds) / (batch_size * grad_accum))
    total_steps = steps_per_epoch * 3
    warmup_steps = max(1, int(round(0.06 * total_steps)))

    work_dir = Path("artifacts/bioclinical_modernbert_checkpoints")
    args = TrainingArguments(
        output_dir=str(work_dir),
        bf16=env_cfg.use_bf16,
        fp16=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=0,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        num_train_epochs=3.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        torch_compile=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
        callbacks=[_NanGuardCallback()],
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    peak_alloc = check_peak_vram_budget(env_cfg.peak_vram_ceiling_gb)
    peak_resvd = torch.cuda.max_memory_reserved() / (1024**3)
    eval_res = trainer.evaluate()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    summary = {
        "model_id": MODEL_ID,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "wall_clock_seconds": round(wall_s, 2),
        "peak_alloc_gb": round(peak_alloc, 4),
        "peak_reserved_gb": round(peak_resvd, 4),
        "val_metrics": {k: float(v) for k, v in eval_res.items()},
    }
    (OUTPUT_DIR / "val_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Full LoRA Training Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
