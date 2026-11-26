"""Stratified reservoir shadow sampling and S1 >2% false-pass circuit breaker."""

from __future__ import annotations

import random
from collections import defaultdict

from pydantic import BaseModel

from medjudge.schemas.models import RiskClass, Verdict


class DriftStats(BaseModel):
    """Live telemetry snapshot of Tier 1 vs Tier 2 shadow backchecks."""

    total_tier1_accepted: int
    total_shadow_checked: int
    s1_shadow_checked: int
    s1_false_passes: int
    s1_false_pass_rate: float
    circuit_breaker_open: bool


class DriftMonitor:
    """Monitors Tier 1 drift via 2-5% stratified reservoir shadow backchecks against Tier 2."""

    def __init__(
        self,
        sample_rate: float = 0.03,
        max_s1_false_pass_rate: float = 0.02,
        min_s1_samples_to_trip: int = 10,
        seed: int = 42,
    ) -> None:
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        self.sample_rate = sample_rate
        self.max_s1_false_pass_rate = max_s1_false_pass_rate
        self.min_s1_samples_to_trip = min_s1_samples_to_trip
        self._rng = random.Random(seed)

        self._accepted_by_risk: dict[str, int] = defaultdict(int)
        self._shadow_checked_by_risk: dict[str, int] = defaultdict(int)
        self._s1_false_passes: int = 0
        self._circuit_open: bool = False

    @property
    def circuit_breaker_open(self) -> bool:
        return self._circuit_open

    def reset_circuit_breaker(self) -> None:
        self._circuit_open = False

    def should_shadow_check(self, risk_class: RiskClass) -> bool:
        """Stratified probabilistic trigger for Tier 2 shadow backchecking."""
        self._accepted_by_risk[risk_class.value] += 1
        return self._rng.random() < self.sample_rate

    def record_shadow_outcome(
        self,
        risk_class: RiskClass,
        tier1_verdict: Verdict,
        tier2_verdict: Verdict,
    ) -> bool:
        """Record Tier 1 vs Tier 2 shadow comparison and trip breaker if S1 false-pass > 2%."""
        self._shadow_checked_by_risk[risk_class.value] += 1
        if risk_class == RiskClass.S1_CRITICAL_SAFETY:
            if tier1_verdict == Verdict.PASS and tier2_verdict == Verdict.FAIL:
                self._s1_false_passes += 1

        s1_total = self._shadow_checked_by_risk[RiskClass.S1_CRITICAL_SAFETY.value]
        if s1_total >= self.min_s1_samples_to_trip:
            rate = self._s1_false_passes / s1_total
            if rate > self.max_s1_false_pass_rate:
                self._circuit_open = True
        return self._circuit_open

    def stats(self) -> DriftStats:
        s1_total = self._shadow_checked_by_risk[RiskClass.S1_CRITICAL_SAFETY.value]
        rate = (self._s1_false_passes / s1_total) if s1_total > 0 else 0.0
        return DriftStats(
            total_tier1_accepted=sum(self._accepted_by_risk.values()),
            total_shadow_checked=sum(self._shadow_checked_by_risk.values()),
            s1_shadow_checked=s1_total,
            s1_false_passes=self._s1_false_passes,
            s1_false_pass_rate=round(rate, 6),
            circuit_breaker_open=self._circuit_open,
        )
