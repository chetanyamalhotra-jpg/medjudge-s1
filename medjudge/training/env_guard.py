"""Windows 11 + NVIDIA RTX 3050 (4 GB VRAM) hard environment guardrails."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

PEAK_VRAM_CEILING_GB = 3.40
REQUIRED_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"


@dataclass(frozen=True)
class WindowsTrainingEnvConfig:
    """Validated runtime settings for 4 GB Ampere training on Windows 11."""

    use_bf16: bool
    attn_implementation: str
    torch_compile: bool
    dataloader_num_workers: int
    cuda_alloc_conf: str
    peak_vram_ceiling_gb: float = PEAK_VRAM_CEILING_GB


def configure_windows_cuda_env() -> str:
    """Apply mandatory PYTORCH_CUDA_ALLOC_CONF setting before CUDA allocation."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = REQUIRED_CUDA_ALLOC_CONF
    return REQUIRED_CUDA_ALLOC_CONF


def validate_training_guardrails(
    *,
    use_flash_attn: bool = False,
    torch_compile: bool = False,
    attn_implementation: str = "sdpa",
    dataloader_num_workers: int = 0,
    prefer_bf16: bool = True,
) -> WindowsTrainingEnvConfig:
    """Enforce Windows 11 4 GB VRAM guardrails and reject forbidden flags."""
    if use_flash_attn:
        raise RuntimeError(
            "flash-attn is strictly forbidden on Windows 11; use attn_implementation='sdpa' ONLY."
        )
    if attn_implementation != "sdpa":
        raise ValueError(
            f"Unsupported attn_implementation='{attn_implementation}'; must be 'sdpa'."
        )
    if torch_compile:
        raise RuntimeError(
            "torch.compile must NEVER be enabled on Windows 11 for MedJudge-S1."
        )
    if dataloader_num_workers != 0 and platform.system() == "Windows":
        raise ValueError(
            "dataloader_num_workers must be 0 on Windows for multiprocessing stability."
        )

    alloc_conf = configure_windows_cuda_env()

    use_bf16 = prefer_bf16
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            bf16_ok = bool(torch.cuda.is_bf16_supported())
            if prefer_bf16 and not bf16_ok:
                use_bf16 = False
    except ImportError:
        pass

    return WindowsTrainingEnvConfig(
        use_bf16=use_bf16,
        attn_implementation="sdpa",
        torch_compile=False,
        dataloader_num_workers=dataloader_num_workers,
        cuda_alloc_conf=alloc_conf,
        peak_vram_ceiling_gb=PEAK_VRAM_CEILING_GB,
    )


def check_peak_vram_budget(ceiling_gb: float = PEAK_VRAM_CEILING_GB) -> float:
    """Verify peak allocated CUDA VRAM stays under the 3.4 GB Windows WDDM ceiling."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return 0.0
    peak_gb = torch.cuda.max_memory_allocated() / (1024.0**3)
    if peak_gb > ceiling_gb:
        raise MemoryError(
            f"Peak CUDA VRAM {peak_gb:.3f} GB exceeded hard ceiling {ceiling_gb:.2f} GB!"
        )
    return peak_gb
