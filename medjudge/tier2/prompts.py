"""Prompt templates and strict JSON parser for Tier 2 frontier LLM-as-a-Judge."""

from __future__ import annotations

import json
import re
from typing import Any

from medjudge.schemas.models import (
    ClinicalArtifact,
    EvidenceSpan,
    RubricCriterion,
    SufficiencyManifest,
    Verdict,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def build_tier2_judge_messages(
    criterion: RubricCriterion,
    artifact: ClinicalArtifact,
    sufficiency: SufficiencyManifest | None = None,
) -> list[dict[str, str]]:
    """Build system + user messages requesting strict JSON evaluation output."""
    system_prompt = (
        "You are a senior clinical AI evaluator (Tier 2 Judge). "
        "Evaluate the candidate clinical response strictly against the provided rubric criterion "
        "and full clinical context. Respond ONLY with a valid JSON object matching:\n"
        '{"verdict": "PASS"|"FAIL"|"UNKNOWN", "confidence": float_0_to_1, '
        '"rationale": string, "evidence_spans": [{"text": string, "source": string}]}.'
    )
    suff_note = (
        f"\n[TIER1_SUFFICIENCY]\nsufficient={sufficiency.sufficient}, missing={sufficiency.missing_context}"
        if sufficiency is not None
        else ""
    )
    user_prompt = (
        f"[RUBRIC_CRITERION]\nID: {criterion.id} (v{criterion.version}, risk={criterion.risk_class.value})\n"
        f"Title: {criterion.title}\n"
        f"Prompt: {criterion.prompt}\n\n"
        f"{artifact.full_text()}"
        f"{suff_note}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_tier2_json_response(raw_content: str) -> dict[str, Any]:
    """Extract and validate structured Tier 2 verdict JSON; raise ValueError on invalid output."""
    text = raw_content.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Tier 2 JSON response must be an object")

    verdict_raw = str(parsed.get("verdict", "")).upper().strip()
    if verdict_raw not in {v.value for v in Verdict}:
        raise ValueError(f"Invalid Tier 2 verdict '{verdict_raw}'")

    conf_raw = parsed.get("confidence", 0.9)
    conf = float(conf_raw)
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"Confidence out of [0,1] bounds: {conf}")

    spans: list[EvidenceSpan] = []
    for item in parsed.get("evidence_spans", []) or []:
        if isinstance(item, dict) and item.get("text"):
            txt = str(item["text"])
            spans.append(
                EvidenceSpan(
                    text=txt,
                    start_char=0,
                    end_char=len(txt),
                    source=str(item.get("source", "tier2_llm")),
                )
            )

    return {
        "verdict": Verdict(verdict_raw),
        "confidence": conf,
        "rationale": str(parsed.get("rationale", "")),
        "evidence_spans": spans,
    }
