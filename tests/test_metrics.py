from evidence_aware_edu.evaluation.metrics import (
    citation_metrics,
    narration_quality_percent,
    query_coverage_percent,
    reference_evidence_support_percent,
    support_counts,
)
from evidence_aware_edu.schemas import (
    AspectCoverage,
    Claim,
    ClaimVerification,
    QualityEvaluation,
)


def _quality(**overrides):
    data = dict(
        aspect_scores=[],
        coverage_score=0.5,
        relevance_score=0.9,
        opening_orientation_score=4,
        introduction_roadmap_score=4,
        definitions_foundations_score=4,
        explanatory_depth_score=4,
        examples_applications_score=4,
        organization_transitions_score=4,
        clarity_accessibility_score=4,
        spoken_naturalness_score=4,
        conclusion_takeaway_score=4,
        reflective_ending_score=4,
    )
    data.update(overrides)
    return QualityEvaluation(**data)


def test_support_counts():
    verifications = [
        ClaimVerification(claim_id="c1", status="supported", confidence=0.9, evidence_ids=["e1"]),
        ClaimVerification(claim_id="c2", status="unsupported", confidence=0.8),
    ]
    out = support_counts(verifications)
    assert out["total"] == 2
    assert out["supported_claim_rate"] == 0.5


def test_citation_metrics():
    claims = [
        Claim(claim_id="c1", text="A", citation_ids=["e1"]),
        Claim(claim_id="c2", text="B", citation_ids=["wrong"]),
    ]
    verifications = [
        ClaimVerification(claim_id="c1", status="supported", confidence=0.9, evidence_ids=["e1"]),
        ClaimVerification(claim_id="c2", status="supported", confidence=0.9, evidence_ids=["e2"]),
    ]
    precision, recall, f1 = citation_metrics(claims, verifications)
    assert precision == 0.5
    assert recall == 0.5
    assert round(f1, 6) == 0.5


def test_reference_evidence_support_percent():
    verifications = [
        ClaimVerification(claim_id="c1", status="supported", confidence=0.9),
        ClaimVerification(claim_id="c2", status="supported", confidence=0.9),
        ClaimVerification(claim_id="c3", status="partial", confidence=0.8),
        ClaimVerification(claim_id="c4", status="unsupported", confidence=0.8),
    ]
    assert reference_evidence_support_percent(verifications) == 62.5


def test_query_coverage_percent_uses_aspect_scores():
    evaluation = _quality(
        aspect_scores=[
            AspectCoverage(aspect="A", score=1.0),
            AspectCoverage(aspect="B", score=1.0),
            AspectCoverage(aspect="C", score=0.5),
            AspectCoverage(aspect="D", score=0.0),
        ],
        coverage_score=0.1,
    )
    assert query_coverage_percent(evaluation) == 62.5


def test_narration_quality_percent():
    evaluation = _quality(
        opening_orientation_score=4,
        introduction_roadmap_score=4,
        definitions_foundations_score=4,
        explanatory_depth_score=3,
        examples_applications_score=3,
        organization_transitions_score=4,
        clarity_accessibility_score=4,
        spoken_naturalness_score=4,
        conclusion_takeaway_score=3,
        reflective_ending_score=4,
    )
    # 37 out of 40.
    assert round(narration_quality_percent(evaluation), 1) == 92.5


def test_telemetry_cost_and_latency_percentiles():
    from evidence_aware_edu.schemas import LLMCallLog, RunTelemetry

    telemetry = RunTelemetry(
        llm_calls=[
            LLMCallLog(task="a", model="m", latency_seconds=1.0, total_cost_usd=0.01),
            LLMCallLog(task="b", model="m", latency_seconds=2.0, total_cost_usd=0.02),
            LLMCallLog(task="c", model="m", latency_seconds=10.0, total_cost_usd=0.03),
        ]
    )
    assert round(telemetry.estimated_llm_cost_usd, 6) == 0.06
    assert telemetry.llm_call_latency_p50_seconds == 2.0
    assert round(telemetry.llm_call_latency_p90_seconds, 1) == 8.4


def test_gemini_cost_estimate():
    from evidence_aware_edu.costs import estimate_gemini_cost

    out = estimate_gemini_cost(1_000_000, 1_000_000, 0.30, 2.50)
    assert out.input_cost_usd == 0.30
    assert out.output_cost_usd == 2.50
    assert out.total_cost_usd == 2.80
