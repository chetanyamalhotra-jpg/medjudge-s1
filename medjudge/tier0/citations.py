"""Deterministic DOI and PMID extraction, syntactic checks, and lookup hook."""

from __future__ import annotations

import re
import time

from medjudge.schemas.models import (
    ClinicalArtifact,
    EvidenceSpan,
    GradeDecision,
    RubricCriterion,
    Verdict,
)
from medjudge.tier0.base import make_tier0_decision

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
_PMID_RE = re.compile(r"\bPMID\s*:?\s*(\d{5,9})\b", re.IGNORECASE)


def extract_citations(text: str) -> tuple[list[str], list[str]]:
    """Extract DOI and PMID identifiers from text."""
    dois = [m.group(1).rstrip(".,;)") for m in _DOI_RE.finditer(text)]
    pmids = [m.group(1) for m in _PMID_RE.finditer(text)]
    return dois, pmids


def verify_citations(criterion: RubricCriterion, artifact: ClinicalArtifact) -> GradeDecision:
    """Verify DOI/PMID syntactic validity and optional known-registry membership."""
    t0 = time.perf_counter()
    text = artifact.candidate_response
    dois, pmids = extract_citations(text)
    if not dois and not pmids:
        require_citation = bool(artifact.metadata.get("require_citation", False))
        if require_citation:
            return make_tier0_decision(
                criterion,
                Verdict.FAIL,
                "Citation required but no DOI or PMID was provided.",
                t0,
            )
        return make_tier0_decision(
            criterion,
            Verdict.UNKNOWN,
            "No DOI/PMID identifiers found to verify deterministically.",
            t0,
            missing_context=["citation_identifiers"],
        )

    known_valid = artifact.metadata.get("known_valid_citations")
    if isinstance(known_valid, (list, set)):
        valid_set = {str(x).lower() for x in known_valid}
        invalid = [c for c in [*dois, *pmids] if c.lower() not in valid_set]
        if invalid:
            return make_tier0_decision(
                criterion,
                Verdict.FAIL,
                f"Unverified/fabricated citation ID(s): {', '.join(invalid)}",
                t0,
            )

    spans = [
        EvidenceSpan(text=c, start_char=0, end_char=min(len(text), max(1, len(c))))
        for c in [*dois, *pmids]
    ]
    return make_tier0_decision(
        criterion,
        Verdict.PASS,
        f"Validated {len(dois)} DOI(s) and {len(pmids)} PMID(s).",
        t0,
        evidence_spans=spans,
    )
