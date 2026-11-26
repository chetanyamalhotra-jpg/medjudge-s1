"""Deterministic GlobalContextCapsule extraction preserving negation, temporality, and family context."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from medjudge.schemas.models import ClinicalArtifact

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")
_AGE_RE = re.compile(
    r"\b(\d{1,3})\s*(?:-\s*year\s*-\s*old|yo|y/o|years?\s+old)\b", re.IGNORECASE
)
_FAMILY_SUBJECT_RE = (
    r"(?:family\s+history|mother|father|sister|brother|sibling|parent|maternal|paternal|aunt|uncle|grandmother|grandfather)"
)
_FAMILY_RE = re.compile(rf"\b{_FAMILY_SUBJECT_RE}\b", re.IGNORECASE)
_MIXED_CLAUSE_SPLIT_RE = re.compile(
    rf";\s*|,\s*(?=(?:patient|pt|he|she|i\s+|{_FAMILY_SUBJECT_RE})\b)|"
    rf"(?:,\s*|\s+)(?=\b(?:and|but|while)\s+(?:patient|pt|he|she|i\s+|{_FAMILY_SUBJECT_RE})\b)",
    re.IGNORECASE,
)
_HISTORICAL_RE = re.compile(
    r"\b(?:history\s+of|prior|previous(?:ly)?|past|former(?:ly)?|years?\s+ago|months?\s+ago|discontinued|stopped|resolved|in\s+20\d{2}|in\s+19\d{2})\b",
    re.IGNORECASE,
)
_CURRENT_RE = re.compile(
    r"\b(?:today|current(?:ly)?|now|active|on\s+admission|present(?:ly)?|this\s+morning|repeat|latest)\b",
    re.IGNORECASE,
)
_NEGATED_ALLERGY_RE = re.compile(
    r"\b(?:no\s+known\s+(?:drug\s+)?allerg(?:y|ies)|nkda|"
    r"(?:denies|no|without|negative\s+for)\s+(?:any\s+|known\s+)?(?:drug\s+)?allerg(?:y|ies)(?!\s+to\b))\b",
    re.IGNORECASE,
)
_NEGATED_ALLERGY_CLAUSE_RE = re.compile(
    r"\b(?:(?:no|denies|without|negative\s+for)\s+(?:known\s+)?(?:drug\s+)?allerg(?:y|ies)\s+to\s+([^,.;\n]+)|"
    r"not\s+allergic\s+to\s+([^,.;\n]+))\b",
    re.IGNORECASE,
)
_ALLERGY_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?:,\s*|\s+(?:and|but)\s+)(?=(?:no|not|denies|without|negative\s+for|nkda|allerg)\b)",
    re.IGNORECASE,
)
_ALLERGY_RE = re.compile(
    r"\b(?:allerg(?:y|ic)\s*(?:to|:)?\s*([^.;\n]+)|allergic\s+reaction\s+to\s+([^.;\n]+))\b",
    re.IGNORECASE,
)
_PREGNANCY_POS_RE = re.compile(
    r"\b(?:currently\s+pregnant|is\s+pregnant|pregnant\s*\(\d+\s*weeks?\)|gestation|trimester|positive\s+pregnancy\s+test)\b",
    re.IGNORECASE,
)
_PREGNANCY_NEG_RE = re.compile(
    r"\b(?:not\s+pregnant|denies\s+pregnancy|negative\s+pregnancy\s+test|non-pregnant)\b",
    re.IGNORECASE,
)
_MED_RE = re.compile(
    r"\b(?:taking|prescribed|started\s+on|active\s+meds?|medications?|current\s+medications?|rx)\s*(?::|include|is|are)?\s*([^.;\n]+)",
    re.IGNORECASE,
)
_LAB_RE = re.compile(
    r"\b(?:creatinine|cr|egfr|bun|alt|ast|bilirubin|inr|alkaline\s+phosphatase|alp)\s*(?:is|was|of|:|=)?\s*\d+(?:\.\d+)?(?:\s*(?:mg/dl|u/l|iu/l|ml/min))?",
    re.IGNORECASE,
)
_VITAL_RE = re.compile(
    r"\b(?:bp|blood\s+pressure|hr|heart\s+rate|spo2|o2\s+sat|temp|temperature|rr|respiratory\s+rate)\s*(?:is|was|:|=)?\s*\d+(?:/\d+)?(?:\.\d+)?(?:\s*(?:mmhg|bpm|%|°?[cf]))?",
    re.IGNORECASE,
)
_CC_RE = re.compile(
    r"\b(?:chief\s+complaint|cc|presents?\s+with|complaining\s+of|reason\s+for\s+visit)\s*(?::|is)?\s*([^.;\n]+)",
    re.IGNORECASE,
)
_SPLIT_NEGATION_RE = re.compile(
    r"\b(denies|no|without|negative\s+for)\s+([^.;\n]+)",
    re.IGNORECASE,
)


class GlobalContextCapsule(BaseModel):
    """Pinned invariant clinical facts extracted deterministically from an artifact."""

    age: str | None = None
    pregnancy_status: str | None = None
    allergies: list[str] = Field(default_factory=list)
    active_medications: list[str] = Field(default_factory=list)
    historical_medications: list[str] = Field(default_factory=list)
    renal_hepatic_labs: list[str] = Field(default_factory=list)
    vitals: list[str] = Field(default_factory=list)
    chief_complaint: str | None = None
    negated_findings: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)

    def available_context_keys(self, artifact: ClinicalArtifact | None = None) -> set[str]:
        """Return context keys populated in this capsule (plus artifact-level sections)."""
        keys: set[str] = set()
        if self.age:
            keys.add("age")
        if self.pregnancy_status:
            keys.add("pregnancy_status")
        if self.allergies:
            keys.add("allergies")
        if self.active_medications:
            keys.add("active_medications")
        if self.renal_hepatic_labs:
            keys.add("renal_hepatic_labs")
        if self.vitals:
            keys.add("vitals")
        if self.chief_complaint:
            keys.add("chief_complaint")
        if artifact is not None:
            if artifact.clinical_note.strip():
                keys.add("clinical_note")
            if artifact.reference_context.strip():
                keys.add("reference_context")
            if artifact.turns:
                keys.add("conversation")
        return keys

    def to_header_text(self) -> str:
        """Render a compact, structured safety header pinned above any slice."""
        lines = ["[GLOBAL_CONTEXT_CAPSULE]"]
        lines.append(f"Age: {self.age or 'UNKNOWN'}")
        lines.append(f"Pregnancy: {self.pregnancy_status or 'UNKNOWN'}")
        lines.append(f"Allergies: {', '.join(self.allergies) if self.allergies else 'UNKNOWN'}")
        lines.append(
            f"Active Medications: {', '.join(self.active_medications) if self.active_medications else 'UNKNOWN'}"
        )
        if self.historical_medications:
            lines.append(f"Historical Medications: {', '.join(self.historical_medications)}")
        lines.append(
            f"Renal/Hepatic Labs: {', '.join(self.renal_hepatic_labs) if self.renal_hepatic_labs else 'NONE_REPORTED'}"
        )
        lines.append(f"Vitals: {', '.join(self.vitals) if self.vitals else 'NONE_REPORTED'}")
        lines.append(f"Chief Complaint: {self.chief_complaint or 'UNKNOWN'}")
        if self.negated_findings:
            lines.append(f"Negated Findings: {', '.join(self.negated_findings)}")
        if self.family_history:
            lines.append(f"Family History: {', '.join(self.family_history)}")
        return "\n".join(lines)


def _tag_temporality(sentence: str) -> str:
    has_current = bool(_CURRENT_RE.search(sentence))
    has_hist = bool(_HISTORICAL_RE.search(sentence))
    if has_hist and not has_current:
        return "[HISTORICAL]"
    if has_current and not has_hist:
        return "[CURRENT]"
    if has_hist and has_current:
        return "[MIXED_TIME]"
    return "[CURRENT]"


def _dedup_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        norm = item.strip()
        if norm and norm.lower() not in seen:
            seen.add(norm.lower())
            out.append(norm)
    return out


def extract_global_context_capsule(artifact: ClinicalArtifact) -> GlobalContextCapsule:
    """Deterministically extract invariant clinical facts from all artifact fields."""
    text = artifact.full_context_text()
    if not text.strip():
        return GlobalContextCapsule()

    capsule = GlobalContextCapsule()
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    for raw_sent in sentences:
        clauses = (
            [c.strip() for c in _MIXED_CLAUSE_SPLIT_RE.split(raw_sent) if c.strip()]
            if _FAMILY_RE.search(raw_sent)
            else [raw_sent]
        )
        for sent in clauses:
            is_family = bool(_FAMILY_RE.search(sent))
            temp_tag = _tag_temporality(sent)

            # Family history clauses must NEVER pollute patient allergies/medications/labs.
            if is_family:
                capsule.family_history.append(f"[FAMILY] {temp_tag} {sent}")
                continue

            if capsule.age is None:
                age_m = _AGE_RE.search(sent)
                if age_m:
                    capsule.age = f"{age_m.group(1)} years"

            if _PREGNANCY_NEG_RE.search(sent):
                capsule.pregnancy_status = "NOT_PREGNANT"
            elif _PREGNANCY_POS_RE.search(sent):
                capsule.pregnancy_status = "PREGNANT"

            neg_allergy_spans: list[tuple[int, int]] = [
                (m.start(), m.end()) for m in _NEGATED_ALLERGY_RE.finditer(sent)
            ]
            for neg_cm in _NEGATED_ALLERGY_CLAUSE_RE.finditer(sent):
                neg_allergy_spans.append((neg_cm.start(), neg_cm.end()))
                target = (neg_cm.group(1) or neg_cm.group(2) or "").strip(" .:;,-")
                if target:
                    capsule.negated_findings.append(f"NO allergy to {target}")

            found_positive_allergy = False
            for m in _ALLERGY_RE.finditer(sent):
                if any(start <= m.start() < end for start, end in neg_allergy_spans):
                    continue
                raw_allergy = (m.group(1) or m.group(2) or "").strip()
                raw_allergy = _ALLERGY_CLAUSE_BOUNDARY_RE.split(raw_allergy, maxsplit=1)[0].strip(" ,.;:-")
                if (
                    raw_allergy
                    and not _NEGATED_ALLERGY_RE.search(raw_allergy)
                    and not _NEGATED_ALLERGY_CLAUSE_RE.search(raw_allergy)
                ):
                    capsule.allergies.append(f"{temp_tag} {raw_allergy}")
                    found_positive_allergy = True

            if _NEGATED_ALLERGY_RE.search(sent) and not found_positive_allergy:
                capsule.allergies.append("NKDA (No Known Drug Allergies)")

            for m in _MED_RE.finditer(sent):
                med_phrase = m.group(1).strip()
                if med_phrase:
                    if temp_tag == "[HISTORICAL]":
                        capsule.historical_medications.append(f"[HISTORICAL] {med_phrase}")
                    else:
                        capsule.active_medications.append(f"{temp_tag} {med_phrase}")

            for m in _LAB_RE.finditer(sent):
                capsule.renal_hepatic_labs.append(f"{temp_tag} {m.group(0).strip()}")

            for m in _VITAL_RE.finditer(sent):
                capsule.vitals.append(f"{temp_tag} {m.group(0).strip()}")

            if capsule.chief_complaint is None:
                cc_m = _CC_RE.search(sent)
                if cc_m:
                    capsule.chief_complaint = cc_m.group(1).strip()

            for neg_m in _SPLIT_NEGATION_RE.finditer(sent):
                neg_phrase = neg_m.group(2).strip()
                neg_phrase = re.split(
                    r",\s*(?=(?:allerg(?:y|ic)|taking|prescribed|started|positive|currently|however|but)\b)",
                    neg_phrase,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]
                # Split compound lists: "chest pain, shortness of breath, or syncope"
                parts = re.split(r",|\bor\b|\band\b", neg_phrase, flags=re.IGNORECASE)
                for part in parts:
                    cleaned = part.strip(" .:;,-")
                    if cleaned and len(cleaned) <= 60:
                        capsule.negated_findings.append(f"NO {cleaned}")

    capsule.allergies = _dedup_preserve(capsule.allergies)
    capsule.active_medications = _dedup_preserve(capsule.active_medications)
    capsule.historical_medications = _dedup_preserve(capsule.historical_medications)
    capsule.renal_hepatic_labs = _dedup_preserve(capsule.renal_hepatic_labs)
    capsule.vitals = _dedup_preserve(capsule.vitals)
    capsule.negated_findings = _dedup_preserve(capsule.negated_findings)
    capsule.family_history = _dedup_preserve(capsule.family_history)
    return capsule
