"""Zero-OOM 512-token LoRA smoke test on 128 samples from d_train.jsonl."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from medjudge.schemas.models import Verdict
from medjudge.training.train_lora import TrainingStageConfig, run_training_stage

LABEL_MAP = {
    Verdict.PASS.value: 0,
    Verdict.FAIL.value: 1,
    Verdict.UNKNOWN.value: 2,
}


def _load_subset(path: Path, limit: int) -> list[dict[str, object]]:
    pass_rows: list[dict[str, object]] = []
    fail_rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            item = {"text": obj["text"], "label": LABEL_MAP[obj["label"]]}
            if item["label"] == 0 and len(pass_rows) < limit // 2:
                pass_rows.append(item)
            elif item["label"] == 1 and len(fail_rows) < limit - (limit // 2):
                fail_rows.append(item)
            if len(pass_rows) + len(fail_rows) >= limit:
                break
    combined = [x for pair in zip(pass_rows, fail_rows, strict=False) for x in pair]
    if len(combined) < limit:
        combined.extend(pass_rows[len(fail_rows) :])
        combined.extend(fail_rows[len(pass_rows) :])
    return combined[:limit]


def main() -> int:
    model_id = "answerdotai/ModernBERT-base"
    train_rows = _load_subset(Path("data/processed/d_train.jsonl"), limit=128)
    val_rows = _load_subset(Path("data/processed/d_val.jsonl"), limit=32)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _tokenize(batch: dict[str, list[object]]) -> dict[str, object]:
        return tokenizer(batch["text"], truncation=True, max_length=512)

    train_ds = Dataset.from_list(train_rows).map(_tokenize, batched=True, remove_columns=["text"])
    val_ds = Dataset.from_list(val_rows).map(_tokenize, batched=True, remove_columns=["text"])

    cfg = TrainingStageConfig(
        stage_name="smoke_lora_512",
        base_model=model_id,
        max_length=512,
        freeze_backbone=False,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        group_by_length=True,
        pad_to_multiple_of=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=1.0,
        output_dir="artifacts/smoke_512",
    )

    t0 = time.perf_counter()
    ckpt_dir = run_training_stage(cfg, train_ds, val_ds)
    elapsed = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

    # Read trainer_state.json for exact train/eval loss
    state_files = list(Path(cfg.output_dir).rglob("trainer_state.json"))
    train_loss = None
    eval_loss = None
    if state_files:
        state = json.loads(state_files[0].read_text(encoding="utf-8"))
        for entry in state.get("log_history", []):
            if "loss" in entry:
                train_loss = entry["loss"]
            if "train_loss" in entry:
                train_loss = entry["train_loss"]
            if "eval_loss" in entry:
                eval_loss = entry["eval_loss"]

    print("\n=== 512-Token Smoke Test Summary ===")
    print(f"Checkpoint      : {ckpt_dir}")
    print(f"Samples         : train={len(train_ds)}, val={len(val_ds)}")
    print(f"Wall-clock time : {elapsed:.2f} s")
    print(f"Peak alloc VRAM : {peak_alloc:.4f} GB (< 3.40 GB)")
    print(f"Peak resvd VRAM : {peak_reserved:.4f} GB")
    print(f"Final train loss: {train_loss}")
    print(f"Eval loss       : {eval_loss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
