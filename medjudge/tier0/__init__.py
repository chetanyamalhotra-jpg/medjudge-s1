"""Tier 0 deterministic zero-ML clinical and structural verifiers."""

from __future__ import annotations

from medjudge.tier0.base import Tier0Registry, build_default_tier0_registry
from medjudge.tier0.citations import verify_citations
from medjudge.tier0.fhir import verify_fhir_json
from medjudge.tier0.length_penalty import verify_length_penalty
from medjudge.tier0.mcq import verify_mcq_answer
from medjudge.tier0.medcalc import verify_medcalc

__all__ = [
    "Tier0Registry",
    "build_default_tier0_registry",
    "verify_citations",
    "verify_fhir_json",
    "verify_length_penalty",
    "verify_mcq_answer",
    "verify_medcalc",
]
