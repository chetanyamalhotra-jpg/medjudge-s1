"""Tests for Tier 2 BYOK async client bounded backoff and clean error raising."""

from __future__ import annotations

import json

import httpx
import pytest

from medjudge.schemas.models import ClinicalArtifact, RubricCriterion, Verdict
from medjudge.tier2.client import (
    OpenAICompatibleJudgeClient,
    Tier2AuthError,
    Tier2Config,
    Tier2ResponseError,
    Tier2RetryExhaustedError,
)


async def _noop_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_tier2_retries_429_and_503_then_succeeds() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="rate limited")
        if calls == 2:
            return httpx.Response(503, text="upstream unavailable")
        body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "verdict": "PASS",
                                "confidence": 0.94,
                                "rationale": "All criteria satisfied.",
                                "evidence_spans": [{"text": "Avoids penicillin", "source": "candidate"}],
                            }
                        )
                    }
                }
            ]
        }
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenAICompatibleJudgeClient(
            Tier2Config(api_key="test-key", max_retries=3, base_backoff_seconds=0.01),
            http_client=http_client,
            sleep_fn=_noop_sleep,
        )
        crit = RubricCriterion(id="r1", version="1.0.0", title="Safety", prompt="Safe?")
        art = ClinicalArtifact(candidate_response="Use azithromycin.")
        dec = await client.evaluate(crit, art)
        assert calls == 3
        assert dec.verdict == Verdict.PASS
        assert dec.calibrated_confidence == 0.94


@pytest.mark.asyncio
async def test_tier2_raises_cleanly_on_auth_malformed_and_exhausted_retries() -> None:
    crit = RubricCriterion(id="r1", version="1.0.0", title="Safety", prompt="Safe?")
    art = ClinicalArtifact(candidate_response="Use amoxicillin.")

    # 401 -> Tier2AuthError immediately (no retry)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(401, text="bad key"))
    ) as c401:
        client = OpenAICompatibleJudgeClient(
            Tier2Config(api_key="bad", max_retries=3), http_client=c401, sleep_fn=_noop_sleep
        )
        with pytest.raises(Tier2AuthError):
            await client.evaluate(crit, art)

    # Exhausted 503 -> Tier2RetryExhaustedError
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(503, text="down"))
    ) as c503:
        client = OpenAICompatibleJudgeClient(
            Tier2Config(api_key="k", max_retries=2, base_backoff_seconds=0.01),
            http_client=c503,
            sleep_fn=_noop_sleep,
        )
        with pytest.raises(Tier2RetryExhaustedError):
            await client.evaluate(crit, art)

    # Malformed non-JSON content -> Tier2ResponseError
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})
        )
    ) as cbad:
        client = OpenAICompatibleJudgeClient(
            Tier2Config(api_key="k"), http_client=cbad, sleep_fn=_noop_sleep
        )
        with pytest.raises(Tier2ResponseError):
            await client.evaluate(crit, art)
