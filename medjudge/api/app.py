"""FastAPI application exposing POST /v1/systemone, GET /v1/rubrics, and GET /health."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from medjudge.api.schemas import SystemOneMode, SystemOneRequest, SystemOneResponse
from medjudge.router.engine import EvaluationRouter
from medjudge.rubrics.loader import RubricRegistry, load_default_rubrics
from medjudge.schemas.models import ClinicalArtifact, RubricCriterion, Verdict
from medjudge.tier1.runtime import Tier1ONNXClassifier

DEFAULT_INT8_ARTIFACT_DIR = Path("artifacts/bioclinical_modernbert_int8")


def load_default_tier1_classifier(
    artifact_dir: str | Path = DEFAULT_INT8_ARTIFACT_DIR,
) -> Tier1ONNXClassifier | None:
    """Load the calibrated CPU INT8 ONNX classifier if exported artifacts exist (zero PyTorch)."""
    base = Path(artifact_dir)
    onnx_path = base / "model_int8.onnx"
    tok_path = base / "tokenizer.json"
    manifest_path = base / "calibration_manifest.json"
    if not (onnx_path.exists() and tok_path.exists()):
        return None
    return Tier1ONNXClassifier(
        onnx_model_path=onnx_path,
        tokenizer=tok_path,
        calibration_manifest=manifest_path if manifest_path.exists() else None,
        max_length=1024,
    )


def _map_decision_to_result(
    mode: SystemOneMode,
    verdict: Verdict,
    probs: dict[str, float],
    choices: list[str],
    sufficiency_sufficient: bool = True,
) -> str | float | None:
    if mode == SystemOneMode.YES_NO:
        if verdict == Verdict.PASS:
            return "yes"
        if verdict == Verdict.FAIL:
            return "no"
        return "unknown"

    if mode == SystemOneMode.SCORE:
        if verdict == Verdict.UNKNOWN or not sufficiency_sufficient:
            return None
        # Calibrated PASS probability as continuous [0, 1] System One score
        return round(float(probs.get("PASS", 0.0)), 6)

    # SystemOneMode.CHOICE
    mapping = {
        Verdict.PASS: choices[0] if len(choices) >= 1 else "PASS",
        Verdict.FAIL: choices[1] if len(choices) >= 2 else "FAIL",
        Verdict.UNKNOWN: choices[2] if len(choices) >= 3 else "UNKNOWN",
    }
    return mapping[verdict]


def create_app(
    router: EvaluationRouter | None = None,
    rubrics: RubricRegistry | None = None,
    artifact_dir: str | Path = DEFAULT_INT8_ARTIFACT_DIR,
) -> FastAPI:
    """Construct the FastAPI System One adapter app."""
    app = FastAPI(title="MedJudge-S1 SystemOne API", version="0.1.0")
    if router is not None:
        active_router = router
    else:
        tier1 = load_default_tier1_classifier(artifact_dir)
        active_router = EvaluationRouter(tier1=tier1)
    active_rubrics = rubrics or load_default_rubrics()

    @app.get("/health")
    async def health() -> dict[str, object]:
        drift = active_router.drift_monitor.stats().model_dump() if active_router.drift_monitor else None
        return {
            "status": "ok",
            "rubric_count": len(active_rubrics),
            "tier1_loaded": active_router.tier1 is not None,
            "tier2_configured": active_router.tier2 is not None,
            "drift": drift,
        }

    @app.get("/v1/rubrics")
    async def list_rubrics() -> list[dict[str, object]]:
        return [c.model_dump() for c in active_rubrics.list_all()]

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def systemone(req: SystemOneRequest) -> SystemOneResponse:
        if req.criterion_id:
            try:
                criterion = active_rubrics.get(req.criterion_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        elif req.prompt:
            criterion = RubricCriterion(
                id="ad_hoc_systemone",
                version="1.0.0",
                title="Ad-hoc SystemOne Criterion",
                prompt=req.prompt,
                risk_class=req.risk_class,
                required_context=req.required_context,
            )
        else:
            raise HTTPException(
                status_code=400, detail="Provide either 'criterion_id' or 'prompt'."
            )

        artifact = req.artifact
        if req.text and not artifact.candidate_response:
            artifact = ClinicalArtifact(
                artifact_id=artifact.artifact_id,
                turns=artifact.turns,
                clinical_note=artifact.clinical_note,
                candidate_response=req.text,
                reference_context=artifact.reference_context,
                metadata={**artifact.metadata, **req.metadata},
            )
        elif req.metadata:
            artifact = artifact.model_copy(
                update={"metadata": {**artifact.metadata, **req.metadata}}
            )

        try:
            decision = await active_router.evaluate(criterion, artifact)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        result = _map_decision_to_result(
            req.mode,
            decision.verdict,
            decision.probabilities,
            req.choices,
            sufficiency_sufficient=decision.sufficiency.sufficient,
        )
        return SystemOneResponse(
            mode=req.mode,
            result=result,
            confidence=decision.calibrated_confidence,
            verdict=decision.verdict,
            probabilities=decision.probabilities,
            tier_used=decision.tier_used,
            escalated_from=decision.escalated_from,
            escalation_reason=decision.escalation_reason,
            sufficiency=decision.sufficiency,
            evidence_spans=decision.evidence_spans,
            rationale=decision.rationale,
        )

    return app
