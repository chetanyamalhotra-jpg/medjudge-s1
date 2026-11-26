"""Serve calibrated MedJudge-S1 INT8 ONNX via FastAPI /v1/systemone or run live smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import uvicorn
from fastapi.testclient import TestClient

from medjudge.api.app import DEFAULT_INT8_ARTIFACT_DIR, create_app


def run_smoke_test(artifact_dir: Path = DEFAULT_INT8_ARTIFACT_DIR) -> int:
    assert "torch" not in sys.modules, "torch must NEVER be imported in /v1/systemone runtime!"
    app = create_app(artifact_dir=artifact_dir)
    assert "torch" not in sys.modules, "create_app imported torch!"

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    health_json = health.json()
    assert health_json["tier1_loaded"] is True, f"Tier 1 failed to load: {health_json}"

    cases = [
        (
            "1_clear_pass_yes_no",
            {
                "mode": "yes_no",
                "criterion_id": "s2_medhallu_grounding",
                "artifact": {
                    "artifact_id": "smoke-pass",
                    "reference_context": (
                        "Question: Does intensive glycemic control reduce microvascular complications in type 2 diabetes?\n"
                        "Knowledge: In the UK Prospective Diabetes Study (UKPDS), patients assigned to intensive blood-glucose "
                        "control with sulfonylureas or insulin achieved a median HbA1c of 7.0% compared with 7.9% in the "
                        "conventional policy group over 10 years. Intensive glycemic control resulted in a 25% risk reduction "
                        "in microvascular endpoints (p=0.0099), primarily driven by reduced need for retinal photocoagulation, "
                        "while showing a non-significant 16% reduction in myocardial infarction (p=0.052). Hypoglycemic "
                        "episodes occurred more frequently in the intensive treatment group than with conventional diet therapy."
                    ),
                    "candidate_response": (
                        "In UKPDS, intensive glycemic control lowered median HbA1c to 7.0% versus 7.9% with conventional "
                        "therapy and reduced microvascular complications by 25% (p=0.0099), though hypoglycemic events were more frequent."
                    ),
                },
            },
        ),
        (
            "2_clear_fail_choice",
            {
                "mode": "choice",
                "criterion_id": "s2_medhallu_grounding",
                "choices": ["GROUNDED", "HALLUCINATED", "ESCALATE"],
                "artifact": {
                    "artifact_id": "smoke-fail",
                    "reference_context": (
                        "Question: Does aspirin prevent preeclampsia in high-risk pregnancies?\n"
                        "Knowledge: In the ASPRE randomized double-blind placebo-controlled trial of 1,776 women at high "
                        "risk for preterm preeclampsia, low-dose aspirin at 150 mg nightly from 11-14 weeks until 36 weeks "
                        "of gestation reduced the incidence of preterm preeclampsia to 1.6% compared with 4.3% in the "
                        "placebo group (odds ratio 0.38, 95% CI 0.20 to 0.74, p=0.004). There was no significant difference "
                        "in neonatal adverse events or postpartum hemorrhage between groups."
                    ),
                    "candidate_response": (
                        "The ASPRE trial demonstrated that high-dose aspirin at 650 mg twice daily eliminated preterm "
                        "preeclampsia completely (0% vs 12% placebo) but caused a threefold increase in severe neonatal bleeding."
                    ),
                },
            },
        ),
        (
            "3_missing_context_unknown_score",
            {
                "mode": "score",
                "criterion_id": "s1_allergy_medication_safety",
                "artifact": {
                    "artifact_id": "smoke-unknown",
                    "clinical_note": "Patient presents with acute bacterial sinusitis.",
                    "candidate_response": "Prescribe amoxicillin 500 mg TID for 7 days.",
                },
            },
        ),
        (
            "4_pqa_labeled_pass_yes_no",
            {
                "mode": "yes_no",
                "criterion_id": "s2_medhallu_grounding",
                "artifact": {
                    "artifact_id": "pqa-labeled-503-pass",
                    "reference_context": (
                        "Question: Ultra high risk (UHR) for psychosis criteria: are there different levels of risk for transition to psychosis?\n"
                        "Knowledge: The ultra high risk (UHR) for psychosis criteria have been validated in a number of studies. "
                        "However, it is not known whether particular UHR criteria (Attenuated Psychotic Symptoms (APS), Brief Limited "
                        "Intermittent Psychotic Symptoms (BLIPS) or Trait vulnerability criteria), or combination of criteria, is "
                        "associated with a higher risk of transition to psychosis. The current study investigated this issue over a "
                        "6-month follow-up period. We hypothesised that the risk of transition would increase in the following order: "
                        "Trait alone<APS alone<APS+Trait<BLIPS. Data on UHR intake criteria and transition to psychosis status at 6 months "
                        "were analysed for UHR patients seen at the PACE clinic, Orygen Youth Health between January 2000 and November 2008. "
                        "A total of 928 new referrals were accepted into the PACE clinic over this period of whom 817 (88%) had baseline "
                        "information available for analysis. The percentage of subjects who presented with APS, Trait and BLIPS were 83%, "
                        "27% and 4%, respectively. When the two intermediate groups (APS alone and APS+Trait) were combined, there was "
                        "evidence that the risk of transition increased in the order of Trait alone<APS<BLIPS (p=0.024, adjusted analysis)."
                    ),
                    "candidate_response": (
                        "Our data suggest that UHR intake criteria predict transition over 6 months in the order of Trait alone<APS<BLIPS. "
                        "The fact that BLIPS patients are at the highest risk of transition over the short term is consistent with the "
                        '"early" versus "late" prodrome model. It also indicates that particular clinical attention may need to be paid to '
                        "BLIPS patients, especially early in the course of treatment."
                    ),
                },
            },
        ),
        (
            "5_pqa_labeled_fail_choice",
            {
                "mode": "choice",
                "criterion_id": "s2_medhallu_grounding",
                "choices": ["GROUNDED", "HALLUCINATED", "ESCALATE"],
                "artifact": {
                    "artifact_id": "pqa-labeled-671-fail",
                    "reference_context": (
                        "Question: Can mandibular depiction be improved by changing the thickness of double-oblique computed tomography images?\n"
                        "Knowledge: Multislice helical computed tomography (CT), which can provide detailed 2-D and 3-D reconstructed images, "
                        "is useful in imaging diagnosis for dental implant treatment. Therefore, in this study, it was performed to clarify the "
                        "mandibular depiction of double-oblique reconstructed images when changing their thickness. A total of 38 sites in the "
                        "mandibular molar region were examined using multislice helical CT. The thicknesses of the double-oblique images using "
                        "multislice helical CT scans were reconstructed in 4 conditions: 0.3 mm, 0.9 mm, 1.6 mm, and 4.1 mm. In double-oblique "
                        "images, mandibular depiction was evaluated by 5 oral radiologists using a subjective rating score. In the alveolar crest "
                        "and the whole of the mandibular canal, the highest value was obtained with 0.9 mm-thick images; however, there was no "
                        "significant difference between 0.3 mm and 0.9 mm-thick images."
                    ),
                    "candidate_response": (
                        "Changing the image thickness to 0.9 mm significantly improves the depiction of the mandibular canal's "
                        "superior wall compared to other thicknesses."
                    ),
                },
            },
        ),
    ]

    results = []
    by_case: dict[str, dict[str, object]] = {}
    for name, payload in cases:
        t0 = time.perf_counter()
        resp = client.post("/v1/systemone", json=payload)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        assert resp.status_code == 200, f"{name} failed: {resp.text}"
        body = resp.json()
        by_case[name] = body
        results.append({"case": name, "http_latency_ms": elapsed_ms, "response": body})

    ukpds_case = by_case["1_clear_pass_yes_no"]
    assert ukpds_case["probabilities"]["PASS"] >= 0.85, f"Unexpected UKPDS probs: {ukpds_case}"
    assert ukpds_case["probabilities"]["UNKNOWN"] < 0.02, f"Capsule mismatch persisted on UKPDS: {ukpds_case}"

    aspre_case = by_case["2_clear_fail_choice"]
    assert aspre_case["probabilities"]["FAIL"] >= 0.85, f"Unexpected ASPRE probs: {aspre_case}"
    assert aspre_case["probabilities"]["UNKNOWN"] < 0.02, f"Capsule mismatch persisted on ASPRE: {aspre_case}"

    unk_case = by_case["3_missing_context_unknown_score"]
    assert unk_case["verdict"] == "UNKNOWN", f"Expected UNKNOWN for s1_allergy score, got {unk_case}"
    assert unk_case["result"] is None, f"Expected result=null for UNKNOWN score, got {unk_case}"
    assert unk_case["tier_used"] == "TIER2", f"Expected TIER2 for force_tier2 S1 rubric, got {unk_case}"

    pqa_pass = by_case["4_pqa_labeled_pass_yes_no"]
    assert pqa_pass["verdict"] == "PASS", f"Expected PASS for pqa_labeled PASS, got {pqa_pass}"
    assert pqa_pass["result"] == "yes", f"Expected 'yes' for pqa_labeled PASS, got {pqa_pass}"
    assert pqa_pass["tier_used"] == "TIER1", f"Expected TIER1 auto-accept for pqa_labeled PASS, got {pqa_pass}"
    assert pqa_pass["probabilities"]["PASS"] >= 0.92, f"Expected p(PASS) >= 0.92, got {pqa_pass}"

    pqa_fail = by_case["5_pqa_labeled_fail_choice"]
    assert pqa_fail["verdict"] == "FAIL", f"Expected FAIL for pqa_labeled FAIL, got {pqa_fail}"
    assert pqa_fail["result"] == "HALLUCINATED", f"Expected HALLUCINATED for pqa_labeled FAIL, got {pqa_fail}"
    assert pqa_fail["tier_used"] == "TIER1", f"Expected TIER1 auto-accept for pqa_labeled FAIL, got {pqa_fail}"
    assert pqa_fail["probabilities"]["FAIL"] >= 0.90, f"Expected p(FAIL) >= 0.90, got {pqa_fail}"

    assert "torch" not in sys.modules, "torch imported during /v1/systemone requests!"
    print(json.dumps({"health": health_json, "cases": results}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MedJudge-S1 /v1/systemone FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_INT8_ARTIFACT_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        return run_smoke_test(args.artifact_dir)

    app = create_app(artifact_dir=args.artifact_dir)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
