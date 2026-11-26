"""Pinned-header clinical slicer with No-Slice-When-Fits and SufficiencyManifest."""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from medjudge.schemas.models import ClinicalArtifact, RubricCriterion, SufficiencyManifest
from medjudge.slicers.capsule import GlobalContextCapsule, extract_global_context_capsule

_CAPSULE_CONTEXT_KEYS = (
    "age",
    "pregnancy_status",
    "allergies",
    "active_medications",
    "renal_hepatic_labs",
    "vitals",
    "chief_complaint",
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def default_token_estimator(text: str) -> int:
    """Fast conservative whitespace/subword token estimator (~1.3 tokens per word)."""
    if not text:
        return 0
    words = len(text.split())
    return max(words, int(words * 1.3))


class SlicedInput(BaseModel):
    """Cross-encoded model input along with pinned capsule and sufficiency manifest."""

    formatted_text: str
    capsule: GlobalContextCapsule
    sufficiency: SufficiencyManifest
    sliced: bool = False
    estimated_tokens: int = Field(ge=0)


class PinnedHeaderSlicer:
    """Clinical slicer that pins GlobalContextCapsule and never drops required context."""

    def __init__(
        self,
        max_tokens: int = 1024,
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        if max_tokens < 128:
            raise ValueError("max_tokens must be >= 128")
        self.max_tokens = max_tokens
        self.estimate_tokens = token_estimator or default_token_estimator

    def build_sufficiency_manifest(
        self,
        criterion: RubricCriterion,
        artifact: ClinicalArtifact,
        capsule: GlobalContextCapsule,
    ) -> SufficiencyManifest:
        available = capsule.available_context_keys(artifact)
        required = list(criterion.required_context)
        provided = [k for k in required if k in available]
        missing = [k for k in required if k not in available]
        notes = (
            "All required context keys satisfied."
            if not missing
            else f"Missing required context: {', '.join(missing)}"
        )
        return SufficiencyManifest(
            required_context=required,
            provided_context=provided,
            missing_context=missing,
            sufficient=(len(missing) == 0),
            notes=notes,
        )

    def slice(self, criterion: RubricCriterion, artifact: ClinicalArtifact) -> SlicedInput:
        """Build cross-encoder input, pinning the capsule header and only slicing on overflow."""
        capsule = extract_global_context_capsule(artifact)
        sufficiency = self.build_sufficiency_manifest(criterion, artifact, capsule)
        omit_capsule = criterion.task_type == "paper_qa" or (
            criterion.task_type == "clinical_note"
            and not any(k in criterion.required_context for k in _CAPSULE_CONTEXT_KEYS)
        )
        header = "" if omit_capsule else capsule.to_header_text()
        rubric_block = f"[RUBRIC_CRITERION]\nID: {criterion.id}\nPrompt: {criterion.prompt}"
        candidate_block = (
            f"[CANDIDATE_RESPONSE]\n{artifact.candidate_response}".strip()
            if artifact.candidate_response.strip()
            else ""
        )
        full_ctx = artifact.full_context_text()

        full_combined = "\n\n".join(
            part for part in [rubric_block, header, full_ctx, candidate_block] if part
        )
        full_tokens = self.estimate_tokens(full_combined)

        # Rule: Never slice inputs that already fit inside the trained token window.
        if full_tokens <= self.max_tokens or not criterion.allow_slicing:
            return SlicedInput(
                formatted_text=full_combined,
                capsule=capsule,
                sufficiency=sufficiency,
                sliced=False,
                estimated_tokens=full_tokens,
            )

        # Overflow path: pin rubric + capsule + candidate response, budget remaining tokens for relevant spans.
        pinned = "\n\n".join(part for part in [rubric_block, header, candidate_block] if part)
        pinned_tokens = self.estimate_tokens(pinned)
        budget = max(64, self.max_tokens - pinned_tokens)

        selected_context = self._select_salient_chunks(
            full_ctx, criterion.prompt + " " + artifact.candidate_response, budget
        )
        sliced_combined = "\n\n".join(
            part
            for part in [
                rubric_block,
                header,
                f"[SLICED_EVIDENCE]\n{selected_context}" if selected_context else "",
                candidate_block,
            ]
            if part
        )
        return SlicedInput(
            formatted_text=sliced_combined,
            capsule=capsule,
            sufficiency=sufficiency,
            sliced=True,
            estimated_tokens=self.estimate_tokens(sliced_combined),
        )

    def _split_oversized_chunk(self, text: str, token_budget: int) -> list[str]:
        if self.estimate_tokens(text) <= token_budget:
            return [text]
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if len(sentences) <= 1:
            sentences = [text]

        chunks: list[str] = []
        max_chunk_words = max(16, int(token_budget / 2.6))
        for sent in sentences:
            if self.estimate_tokens(sent) <= token_budget:
                chunks.append(sent)
                continue
            words = sent.split()
            for start in range(0, len(words), max_chunk_words):
                part = " ".join(words[start : start + max_chunk_words]).strip()
                if part:
                    chunks.append(part)
        return chunks

    def _select_salient_chunks(self, context_text: str, query_text: str, token_budget: int) -> str:
        if not context_text.strip():
            return ""
        query_terms = {
            tok
            for tok in re.findall(r"[a-z0-9]+", query_text.lower())
            if len(tok) > 2
        }
        paragraphs = [c.strip() for c in re.split(r"\n+", context_text) if c.strip()]
        chunks: list[str] = []
        for para in paragraphs:
            chunks.extend(self._split_oversized_chunk(para, token_budget))

        scored: list[tuple[int, int, str]] = []
        for idx, chunk in enumerate(chunks):
            chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.lower()))
            overlap = len(query_terms & chunk_terms)
            scored.append((overlap, -idx, chunk))
        # Sort by overlap descending, keeping original relative order among ties.
        scored.sort(key=lambda x: (-x[0], -x[1]))

        picked: list[tuple[int, str]] = []
        used = 0
        for _, neg_idx, chunk in scored:
            cost = self.estimate_tokens(chunk)
            if used + cost <= token_budget:
                picked.append((-neg_idx, chunk))
                used += cost
            elif not picked:
                max_words = max(1, int(token_budget / 1.3))
                truncated = " ".join(chunk.split()[:max_words])
                picked.append((-neg_idx, truncated))
                break
            if used >= token_budget:
                break
        picked.sort(key=lambda x: x[0])
        return "\n".join(chunk for _, chunk in picked)
