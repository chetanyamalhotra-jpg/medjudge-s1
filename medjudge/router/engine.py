"""4-tier evaluation router with sufficiency gating, asymmetric thresholds, and drift backchecking."""

from __future__ import annotations

from medjudge.audit.logger import JSONLAuditLogger
from medjudge.drift.monitor import DriftMonitor
from medjudge.schemas.models import (
    ClinicalArtifact,
    EvalTier,
    GradeDecision,
    RiskClass,
    RubricCriterion,
    RubricThresholds,
    SufficiencyManifest,
    Verdict,
)
from medjudge.slicers.capsule import extract_global_context_capsule
from medjudge.slicers.pinned_header import PinnedHeaderSlicer
from medjudge.tier0.base import Tier0Registry, build_default_tier0_registry
from medjudge.tier1.runtime import Tier1ClassifierProtocol
from medjudge.tier2.client import Tier2Protocol
from medjudge.tier3.agent_hook import Tier3AgentJudge, UnconfiguredTier3Stub


class EvaluationRouter:
    """Routes each atomic criterion through the smallest safe evaluation tier."""

    def __init__(
        self,
        tier0: Tier0Registry | None = None,
        slicer: PinnedHeaderSlicer | None = None,
        tier1: Tier1ClassifierProtocol | None = None,
        tier2: Tier2Protocol | None = None,
        tier3: Tier3AgentJudge | None = None,
        audit_logger: JSONLAuditLogger | None = None,
        drift_monitor: DriftMonitor | None = None,
    ) -> None:
        self.tier0 = tier0 or build_default_tier0_registry()
        self.slicer = slicer or PinnedHeaderSlicer()
        self.tier1 = tier1
        self.tier2 = tier2
        self.tier3 = tier3 or UnconfiguredTier3Stub()
        self.audit_logger = audit_logger
        self.drift_monitor = drift_monitor

    def _compute_sufficiency(
        self, criterion: RubricCriterion, artifact: ClinicalArtifact
    ) -> SufficiencyManifest:
        capsule = extract_global_context_capsule(artifact)
        return self.slicer.build_sufficiency_manifest(criterion, artifact, capsule)

    def _guard_insufficient_pass(
        self,
        decision: GradeDecision,
        sufficiency: SufficiencyManifest | None = None,
    ) -> GradeDecision:
        if sufficiency is not None and not sufficiency.sufficient:
            decision.sufficiency = sufficiency
        if not decision.sufficiency.sufficient and decision.verdict == Verdict.PASS:
            decision.verdict = Verdict.UNKNOWN
            decision.calibrated_confidence = float(decision.probabilities.get("UNKNOWN", 0.0))
            reason_prefix = (
                f"Insufficient context ({', '.join(decision.sufficiency.missing_context)}) prevents PASS"
            )
            decision.escalation_reason = (
                f"{decision.escalation_reason}; {reason_prefix}"
                if decision.escalation_reason
                else reason_prefix
            )
        return decision

    def _finish(self, artifact: ClinicalArtifact, decision: GradeDecision) -> GradeDecision:
        self._guard_insufficient_pass(decision)
        if self.audit_logger is not None:
            self.audit_logger.log(artifact, decision)
        return decision

    def _meets_asymmetric_gate(
        self, criterion: RubricCriterion, decision: GradeDecision
    ) -> tuple[bool, str | None]:
        if not decision.sufficiency.sufficient:
            return False, f"Insufficient context: {', '.join(decision.sufficiency.missing_context)}"
        if decision.verdict == Verdict.UNKNOWN:
            return False, "Tier 1 returned UNKNOWN"

        thresholds = criterion.thresholds or RubricThresholds.defaults_for(criterion.risk_class)
        if decision.verdict == Verdict.PASS:
            p_pass = decision.probabilities.get("PASS", 0.0)
            if p_pass < thresholds.pass_threshold:
                return (
                    False,
                    f"p(PASS)={p_pass:.4f} < {criterion.risk_class.value} threshold {thresholds.pass_threshold:.4f}",
                )
            return True, None

        if decision.verdict == Verdict.FAIL:
            p_fail = decision.probabilities.get("FAIL", 0.0)
            if p_fail < thresholds.fail_threshold:
                return (
                    False,
                    f"p(FAIL)={p_fail:.4f} < {criterion.risk_class.value} threshold {thresholds.fail_threshold:.4f}",
                )
            return True, None

        return False, "Unrecognized Tier 1 verdict"

    async def _eval_tier2(
        self,
        criterion: RubricCriterion,
        artifact: ClinicalArtifact,
        sufficiency: SufficiencyManifest,
        *,
        escalated_from: EvalTier | None = None,
        escalation_reason: str | None = None,
    ) -> GradeDecision:
        assert self.tier2 is not None
        decision = await self.tier2.evaluate(criterion, artifact, sufficiency)
        decision.escalated_from = escalated_from
        if escalation_reason is not None:
            decision.escalation_reason = escalation_reason
        return self._guard_insufficient_pass(decision, sufficiency)

    async def evaluate(
        self, criterion: RubricCriterion, artifact: ClinicalArtifact
    ) -> GradeDecision:
        """Evaluate a single criterion and artifact through Tiers 0 -> 1 -> 2 -> 3."""
        # Tier 3: Explicit external-tool verification
        if criterion.requires_external_tools or criterion.preferred_tier == EvalTier.TIER3_AGENT_HOOK:
            decision = await self.tier3.verify_with_tools(criterion, artifact)
            return self._finish(artifact, decision)

        # Tier 0: Deterministic verification
        if criterion.preferred_tier == EvalTier.TIER0_DETERMINISTIC or criterion.tier0_handler:
            handler = criterion.tier0_handler
            if handler and self.tier0.has(handler):
                t0_decision = self.tier0.run(handler, criterion, artifact)
                if t0_decision.verdict != Verdict.UNKNOWN:
                    return self._finish(artifact, t0_decision)
                # Tier 0 could not determine -> escalate to Tier 2 if available
                if self.tier2 is not None:
                    t2_decision = await self._eval_tier2(
                        criterion,
                        artifact,
                        t0_decision.sufficiency,
                        escalated_from=EvalTier.TIER0_DETERMINISTIC,
                        escalation_reason=t0_decision.rationale,
                    )
                    return self._finish(artifact, t2_decision)
                return self._finish(artifact, t0_decision)

        # Mandatory Tier 2 routing (e.g. S1 critical safety and MEDEC clinical notes)
        if criterion.force_tier2:
            sufficiency = self._compute_sufficiency(criterion, artifact)
            if self.tier2 is not None:
                t2_decision = await self._eval_tier2(
                    criterion,
                    artifact,
                    sufficiency,
                    escalated_from=None,
                    escalation_reason="force_tier2 enabled",
                )
                return self._finish(artifact, t2_decision)
            return self._finish(
                artifact,
                GradeDecision(
                    criterion_id=criterion.id,
                    criterion_version=criterion.version,
                    criterion_sha256=criterion.sha256,
                    verdict=Verdict.UNKNOWN,
                    probabilities={"PASS": 0.0, "FAIL": 0.0, "UNKNOWN": 1.0},
                    calibrated_confidence=0.0,
                    tier_used=EvalTier.TIER2_FRONTIER_LLM,
                    escalation_reason="force_tier2 enabled; Tier 2 judge not configured",
                    sufficiency=sufficiency,
                    rationale="force_tier2 enabled; Tier 2 judge not configured",
                ),
            )

        # Tier 2 direct preference (holistic reasoning, red-team traps)
        if criterion.preferred_tier == EvalTier.TIER2_FRONTIER_LLM:
            if self.tier2 is None:
                raise RuntimeError(
                    f"Criterion '{criterion.id}' targets Tier 2, but no Tier 2 judge is configured."
                )
            sufficiency = self._compute_sufficiency(criterion, artifact)
            decision = await self._eval_tier2(criterion, artifact, sufficiency)
            return self._finish(artifact, decision)

        # Circuit breaker check for S1 safety
        if (
            self.drift_monitor is not None
            and self.drift_monitor.circuit_breaker_open
            and criterion.risk_class == RiskClass.S1_CRITICAL_SAFETY
        ):
            if self.tier2 is None:
                raise RuntimeError(
                    "S1 circuit breaker is open due to Tier 1 drift, and no Tier 2 judge is configured."
                )
            sufficiency = self._compute_sufficiency(criterion, artifact)
            t2_decision = await self._eval_tier2(
                criterion,
                artifact,
                sufficiency,
                escalated_from=EvalTier.TIER1_LOCAL_ONNX,
                escalation_reason="S1 drift circuit breaker open (>2% false-pass rate)",
            )
            return self._finish(artifact, t2_decision)

        # Tier 1: Pinned-Header Slicer + Local CPU ONNX
        sliced_input = self.slicer.slice(criterion, artifact)
        if self.tier1 is None:
            if self.tier2 is not None:
                t2_decision = await self._eval_tier2(
                    criterion,
                    artifact,
                    sliced_input.sufficiency,
                    escalated_from=EvalTier.TIER1_LOCAL_ONNX,
                    escalation_reason="Tier 1 ONNX model not loaded",
                )
                return self._finish(artifact, t2_decision)
            raise RuntimeError(
                f"Neither Tier 1 ONNX model nor Tier 2 judge is configured for '{criterion.id}'."
            )

        t1_decision = self.tier1.predict(criterion, artifact, sliced_input)
        accepted, reason = self._meets_asymmetric_gate(criterion, t1_decision)

        if accepted:
            # Optional 2-5% stratified shadow backcheck against Tier 2
            if (
                self.drift_monitor is not None
                and self.tier2 is not None
                and self.drift_monitor.should_shadow_check(criterion.risk_class)
            ):
                shadow_t2 = await self.tier2.evaluate(
                    criterion, artifact, sliced_input.sufficiency
                )
                self.drift_monitor.record_shadow_outcome(
                    criterion.risk_class, t1_decision.verdict, shadow_t2.verdict
                )
                t1_decision.shadow_checked = True
            return self._finish(artifact, t1_decision)

        # Asymmetric confidence or sufficiency gate triggered -> escalate to Tier 2
        if self.tier2 is not None:
            t2_decision = await self._eval_tier2(
                criterion,
                artifact,
                sliced_input.sufficiency,
                escalated_from=EvalTier.TIER1_LOCAL_ONNX,
                escalation_reason=reason,
            )
            return self._finish(artifact, t2_decision)

        # No Tier 2 configured -> never return PASS on insufficient/uncertain Tier 1
        t1_decision.verdict = Verdict.UNKNOWN
        t1_decision.calibrated_confidence = float(t1_decision.probabilities.get("UNKNOWN", 0.0))
        t1_decision.escalation_reason = reason
        return self._finish(artifact, t1_decision)
