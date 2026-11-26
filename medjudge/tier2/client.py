"""Async OpenAI-compatible BYOK Tier 2 LLM judge with bounded exponential backoff."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from medjudge.schemas.models import (
    ClinicalArtifact,
    EvalTier,
    GradeDecision,
    RubricCriterion,
    SufficiencyManifest,
    Verdict,
)
from medjudge.tier2.prompts import build_tier2_judge_messages, parse_tier2_json_response

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class Tier2JudgeError(RuntimeError):
    """Base exception for Tier 2 LLM-as-a-Judge failures (never return silent zeros)."""


class Tier2AuthError(Tier2JudgeError):
    """Unrecoverable 401/403 authentication or authorization error."""


class Tier2ResponseError(Tier2JudgeError):
    """Unrecoverable client error (4xx) or malformed/non-JSON judge output."""


class Tier2RetryExhaustedError(Tier2JudgeError):
    """Bounded retry budget exhausted after transient 429/5xx/network errors."""


class Tier2Config(BaseModel):
    """BYOK configuration for an OpenAI-compatible chat completions endpoint."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    base_backoff_seconds: float = Field(default=0.5, ge=0.0)
    max_backoff_seconds: float = Field(default=8.0, gt=0.0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class Tier2Protocol(Protocol):
    """Async interface for Tier 2 escalation."""

    async def evaluate(
        self,
        criterion: RubricCriterion,
        artifact: ClinicalArtifact,
        sufficiency: SufficiencyManifest | None = None,
    ) -> GradeDecision:
        ...


class OpenAICompatibleJudgeClient:
    """Async OpenAI-compatible BYOK client with bounded exponential backoff."""

    def __init__(
        self,
        config: Tier2Config,
        http_client: httpx.AsyncClient | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self._external_client = http_client
        self._sleep = sleep_fn

    def _compute_backoff(self, attempt: int) -> float:
        raw = min(
            self.config.max_backoff_seconds,
            self.config.base_backoff_seconds * (2**attempt),
        )
        jitter = random.uniform(0.0, 0.25 * raw)
        return min(self.config.max_backoff_seconds, raw + jitter)

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        own_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(
            timeout=self.config.timeout_seconds
        )
        try:
            last_error: Exception | None = None
            for attempt in range(self.config.max_retries + 1):
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                    if attempt < self.config.max_retries:
                        await self._sleep(self._compute_backoff(attempt))
                        continue
                    raise Tier2RetryExhaustedError(
                        f"Tier 2 transport error after {attempt + 1} attempts: {exc}"
                    ) from exc

                if resp.status_code in (401, 403):
                    raise Tier2AuthError(
                        f"Tier 2 authentication failed with HTTP {resp.status_code}"
                    )

                if resp.status_code in RETRYABLE_STATUS_CODES:
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    if attempt < self.config.max_retries:
                        await self._sleep(self._compute_backoff(attempt))
                        continue
                    raise Tier2RetryExhaustedError(
                        f"Tier 2 retry budget exhausted (last status={resp.status_code})"
                    )

                if resp.status_code >= 400:
                    raise Tier2ResponseError(
                        f"Unrecoverable Tier 2 HTTP {resp.status_code}: {resp.text[:200]}"
                    )

                try:
                    return resp.json()
                except json.JSONDecodeError as exc:
                    raise Tier2ResponseError(f"Tier 2 returned non-JSON body: {exc}") from exc

            raise Tier2RetryExhaustedError(f"Tier 2 failed: {last_error}")
        finally:
            if own_client:
                await client.aclose()

    async def evaluate(
        self,
        criterion: RubricCriterion,
        artifact: ClinicalArtifact,
        sufficiency: SufficiencyManifest | None = None,
    ) -> GradeDecision:
        """Evaluate criterion + full-context artifact via BYOK Tier 2 judge."""
        t0 = time.perf_counter()
        messages = build_tier2_judge_messages(criterion, artifact, sufficiency)
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        data = await self._post_with_retry(payload)
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = parse_tier2_json_response(content)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise Tier2ResponseError(f"Malformed Tier 2 judge payload: {exc}") from exc

        verdict: Verdict = parsed["verdict"]
        conf: float = parsed["confidence"]
        remainder = max(0.0, 1.0 - conf)
        probs = {
            "PASS": conf if verdict == Verdict.PASS else remainder * 0.5,
            "FAIL": conf if verdict == Verdict.FAIL else remainder * 0.5,
            "UNKNOWN": conf if verdict == Verdict.UNKNOWN else 0.0,
        }
        elapsed_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
        return GradeDecision(
            criterion_id=criterion.id,
            criterion_version=criterion.version,
            criterion_sha256=criterion.sha256,
            verdict=verdict,
            probabilities=probs,
            calibrated_confidence=conf,
            tier_used=EvalTier.TIER2_FRONTIER_LLM,
            evidence_spans=parsed["evidence_spans"],
            sufficiency=sufficiency or SufficiencyManifest(sufficient=True),
            latency_ms=elapsed_ms,
            rationale=parsed["rationale"],
        )
