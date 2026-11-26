"""Micro-probe BioClinical-ModernBERT-base VRAM and speed at max_length=1024."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from medjudge.schemas.models import Verdict
from medjudge.training.env_guard import check_peak_vram_budget, validate_training_guardrails
from medjudge.training.train_lora import inspect_lora_target_modules

LABEL_MAP = {Verdict.PASS.value: 0, Verdict.FAIL.value: 1, Verdict.UNKNOWN.value: 2}
MODEL_ID = "thomas-sounack/BioClinical-ModernBERT-base"


def _load_32(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({"text": obj["text"], "label": LABEL_MAP[obj["label"]]})
            if len(rows) >= 32:
                break
    return rows


def _run_probe(batch_size: int, grad_accum: int, train_ds: Dataset, tokenizer: AutoTokenizer) -> dict[str, object]:
    env_cfg = validate_training_guardrails(
        use_flash_attn=False,
        torch_compile=False,
        attn_implementation="sdpa",
        dataloader_num_workers=0,
        prefer_bf16=True,
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

    args = TrainingArguments(
        output_dir=f"artifacts/probe_bs{batch_size}",
        bf16=env_cfg.use_bf16,
        fp16=False,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=0,
        learning_rate=1e-4,
        num_train_epochs=1.0,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        torch_compile=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_alloc = check_peak_vram_budget(env_cfg.peak_vram_ceiling_gb)
    peak_resvd = torch.cuda.max_memory_reserved() / (1024**3)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "wall_clock_s": round(elapsed, 2),
        "peak_alloc_gb": round(peak_alloc, 4),
        "peak_reserved_gb": round(peak_resvd, 4),
    }


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    rows = _load_32(Path("data/processed/d_train.jsonl"))
    ds = Dataset.from_list(rows).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=1024),
        batched=True,
        remove_columns=["text"],
    )

    res_bs4 = _run_probe(4, 8, ds, tokenizer)
    print("Probe bs=4, grad_accum=8:", json.dumps(res_bs4))
    chosen = res_bs4
    if float(res_bs4["peak_reserved_gb"]) >= 2.8:
        res_bs2 = _run_probe(2, 16, ds, tokenizer)
        print("Fallback probe bs=2, grad_accum=16:", json.dumps(res_bs2))
        chosen = res_bs2

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("artifacts/vram_probe_1024.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(f"Selected batch configuration: bs={chosen['batch_size']}, grad_accum={chosen['grad_accum']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
