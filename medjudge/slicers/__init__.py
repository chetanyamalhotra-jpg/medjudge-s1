"""Pinned-header clinical slicing, GlobalContextCapsule, and sufficiency validation."""

from __future__ import annotations

from medjudge.slicers.capsule import GlobalContextCapsule, extract_global_context_capsule
from medjudge.slicers.pinned_header import PinnedHeaderSlicer, SlicedInput
from medjudge.slicers.validator import SlicerValidationResult, validate_slicer_sufficiency

__all__ = [
    "GlobalContextCapsule",
    "PinnedHeaderSlicer",
    "SlicedInput",
    "SlicerValidationResult",
    "extract_global_context_capsule",
    "validate_slicer_sufficiency",
]
