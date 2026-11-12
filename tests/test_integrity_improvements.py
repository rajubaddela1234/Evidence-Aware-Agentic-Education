from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")
pytest.importorskip("langgraph")

from evidence_aware_edu.agents.narration_requirement_auditor import NarrationRequirementAuditorAgent
from evidence_aware_edu.agents.requirement_completeness import RequirementCompletenessAgent
from evidence_aware_edu.agents.synthesizer import SynthesizerAgent
from evidence_aware_edu.agents.verifier import VerifierAgent
from evidence_aware_edu.evaluation.evaluator import ResearchEvaluator
from evidence_aware_edu.evaluation.metrics import citation_metrics
from evidence_aware_edu.experiments.analysis import analyze
from evidence_aware_edu.experiments.identity import safe_generation_config, sha256_text
from evidence_aware_edu.graph.evidence_aware_graph import AdaptiveFlags, EvidenceAwareGraph
from evidence_aware_edu.schemas import (
    AspectCoverage,
    Claim,
    ClaimVerification,
    Contradiction,
    Evidence,
    LLMCallLog,
    NarrationRequirementAssessment,
    NarrationRequirementAuditBundle,
    PedagogyPlan,
    QualityEvaluation,
    RequirementAssessment,
    RequirementCompletenessResult,
)
from evidence_aware_edu.telemetry import TelemetryCollector


def _log(task: str = "fake") -> LLMCallLog:
    return LLMCallLog(task=task, model="fake", latency_seconds=0.01)


def test_verifier_enforces_one_decision_and_valid_evidence_ids():
    claims = [Claim(claim_id="C1", text="A"), Claim(claim_id="C2", text="B")]
    evidence = [
        Evidence(
            evidence_id="E1",
            source_id="S1",
            source_type="web",
            title="T",
            url="https://example.com",
            passage="A",
            retrieval_query="q",
        )
    ]
    returned = [
        ClaimVerification(
            claim_id="C1", status="supported", confidence=0.9, evidence_ids=["INVALID"]
        ),
        ClaimVerification(
            claim_id="C1", status="contradicted", confidence=0.8, evidence_ids=["E1"]
        ),
    ]
    normalized = VerifierAgent._normalize(claims, evidence, returned)
    assert len(normalized) == 2
    assert normalized[0].claim_id == "C1"
    assert normalized[0].status == "contradicted"
    assert normalized[0].evidence_ids == ["E1"]
    assert normalized[1].claim_id == "C2"
    assert normalized[1].status == "unsupported"


def test_verifier_downgrades_supported_decision_with_invalid_evidence():
    claim = Claim(claim_id="C1", text="A")
    returned = [
        ClaimVerification(
            claim_id="C1", status="supported", confidence=1.0, evidence_ids=["FAKE"]
        )
    ]
    out = VerifierAgent._normalize([claim], [], returned)
    assert out[0].status == "unsupported"
    assert out[0].evidence_ids == []


def test_citation_metric_requires_supported_entailment_and_all_claim_denominator():
    claims = [
        Claim(claim_id="C1", text="A", needs_citation=True, citation_ids=["E1"]),
        Claim(claim_id="C2", text="B", needs_citation=True, citation_ids=["E2"]),
    ]
    verifications = [
        ClaimVerification(
            claim_id="C1", status="partial", confidence=0.9, evidence_ids=["E1"]
        ),
        ClaimVerification(
            claim_id="C2", status="supported", confidence=0.9, evidence_ids=["E2"]
        ),
    ]
    precision, recall, f1 = citation_metrics(claims, verifications)
    assert precision == 0.5
    assert recall == 0.5
    assert round(f1, 6) == 0.5


def _quality(aspects):
    return QualityEvaluation(
        aspect_scores=aspects,
        coverage_score=1.0,
        relevance_score=1.0,
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


def test_coverage_denominator_is_fixed_to_expected_aspects():
    quality = _quality(
        [
            AspectCoverage(aspect="A", score=1.0),
            AspectCoverage(aspect="A", score=0.5),
            AspectCoverage(aspect="EXTRA", score=1.0),
        ]
    )
    out = ResearchEvaluator._normalize_quality_aspects(["A", "B"], quality)
    assert [x.aspect for x in out.aspect_scores] == ["A", "B"]
    # Duplicate A keeps the lower score; missing B is zero.
    assert [x.score for x in out.aspect_scores] == [0.5, 0.0]
    assert out.coverage_score == 0.25


class _FakeCompletenessLLM:
    settings = SimpleNamespace()

    def generate_structured(self, **kwargs):
        return (
            RequirementCompletenessResult(
                final_requirements=[
                    "anode and cathode roles",
                    "lithium-ion movement",
                    "electron flow through the external circuit",
                ],
                added_requirements=["electron flow through the external circuit"],
                rationale="The question asks how energy is released, requiring external electron flow.",
            ),
            _log("requirement_completeness_audit"),
        )


def test_requirement_completeness_can_add_missing_obligation():
    agent = RequirementCompletenessAgent(_FakeCompletenessLLM(), TelemetryCollector())
    out = agent.run(
        "How does a lithium-ion battery store and release energy?",
        ["anode and cathode roles", "lithium-ion movement", "charging and discharging"],
    )
    assert "electron flow through the external circuit" in out.final_requirements
    assert "electron flow through the external circuit" in out.added_requirements


class _FakeNarrationAuditLLM:
    settings = SimpleNamespace()

    def generate_structured(self, **kwargs):
        return (
            NarrationRequirementAuditBundle(
                assessments=[
                    NarrationRequirementAssessment(
                        requirement_id="R1",
                        requirement="A",
                        status="answered_with_evidence",
                        confidence=0.9,
                    ),
                    NarrationRequirementAssessment(
                        requirement_id="R1",
                        requirement="A",
                        status="partial",
                        confidence=0.8,
                    ),
                ]
            ),
            _log("final_narration_requirement_audit"),
        )


def test_final_narration_audit_is_one_per_requirement_and_conservative():
    agent = NarrationRequirementAuditorAgent(_FakeNarrationAuditLLM(), TelemetryCollector())
    evidence_status = [
        RequirementAssessment(
            requirement_id="R1", requirement="A", status="covered", confidence=0.9, evidence_ids=["E1"]
        ),
        RequirementAssessment(
            requirement_id="R2", requirement="B", status="covered", confidence=0.9, evidence_ids=["E2"]
        ),
    ]
    out = agent.run("Q", ["A", "B"], evidence_status, "script", 0.7)
    assert len(out) == 2
    assert out[0].status == "partial"
    assert out[1].status == "omitted"


def test_final_requirement_omission_routes_to_repair_without_extra_retrieval_logic():
    graph = EvidenceAwareGraph.__new__(EvidenceAwareGraph)
    graph.flags = AdaptiveFlags(final_repair=True)
    graph.settings = SimpleNamespace(claim_support_threshold=0.7)
    state = {
        "final_verifications": [],
        "final_requirement_audit": [
            NarrationRequirementAssessment(
                requirement_id="R1",
                requirement="A",
                status="omitted",
                confidence=0.95,
            )
        ],
    }
    assert graph._route_after_final_checks(state) == "repair"


class _FakeTextLLM:
    settings = SimpleNamespace()

    def __init__(self):
        self.last_prompt = ""

    def generate_text(self, **kwargs):
        self.last_prompt = kwargs["prompt"]
        return "narration", _log("p1_synthesis")


def test_contradictions_are_passed_into_synthesis_prompt():
    llm = _FakeTextLLM()
    agent = SynthesizerAgent(llm, TelemetryCollector())
    claim = Claim(claim_id="C1", text="Claim A")
    ev = Evidence(
        evidence_id="E1",
        source_id="S1",
        source_type="web",
        title="T",
        url="https://example.com",
        passage="A",
        retrieval_query="q",
    )
    agent.from_verified_claims(
        "Q",
        [claim],
        {"C1": [ev]},
        PedagogyPlan(),
        contradictions=[
            Contradiction(
                description="Source A and source B disagree on the numerical value.",
                evidence_ids=["E1", "E2"],
                severity="high",
            )
        ],
    )
    assert "Source A and source B disagree" in llm.last_prompt
    assert "CONTRADICTION RULE" in llm.last_prompt


def test_safe_experiment_config_contains_no_credentials():
    settings = SimpleNamespace(
        generation_provider="gemini",
        generation_access="api",
        generation_model="gen",
        evaluator_provider="anthropic",
        evaluator_access="cli",
        claude_access_method="cli",
        evaluator_model="claude-sonnet-5-5",
        claude_evaluator_max_context_chars=16000,
        claude_evaluator_max_output_tokens=6000,
        claude_evaluator_timeout_seconds=300,
        claude_evaluator_retry_attempts=3,
        thinking_level="minimal",
        max_output_tokens=10000,
        max_sources_per_tool=5,
        max_reference_sources_per_tool=5,
        max_context_chars=50000,
        max_evidence_per_query=18,
        arxiv_max_papers_per_query=2,
        youtube_require_transcript=True,
        max_requirement_retrieval_rounds=1,
        max_claim_retrieval_rounds=1,
        requirement_support_threshold=0.7,
        claim_support_threshold=0.7,
        original_max_iterations=3,
        feedback_threshold=7.0,
        use_frozen_reference_pool=True,
        reference_pool_dir="data/reference_pools_final_10q",
        request_sleep_seconds=1.0,
        api_retry_attempts=6,
        gemini_api_key="SECRET",
        tavily_api_key="SECRET2",
    )
    config = safe_generation_config(settings)
    rendered = json.dumps(config)
    assert "SECRET" not in rendered
    assert "api_key" not in rendered.lower()


def test_analysis_rejects_mismatched_script_hash(tmp_path):
    raw_dir = tmp_path / "raw"
    eval_dir = tmp_path / "evaluations"
    analysis_dir = tmp_path / "analysis"
    raw_path = raw_dir / "B1" / "Q01.json"
    eval_path = eval_dir / "B1" / "Q01.json"
    raw_path.parent.mkdir(parents=True)
    eval_path.parent.mkdir(parents=True)
    raw_path.write_text(
        json.dumps(
            {
                "system_id": "B1",
                "prompt_id": "Q01",
                "script": "new script",
                "script_sha256": sha256_text("new script"),
                "reference_pool_sha256": "pool",
                "experiment_id": "EXP1",
                "telemetry": {"llm_calls": []},
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "record": {
                    "experiment_id": "EXP1",
                    "evaluated_script_sha256": sha256_text("old script"),
                    "reference_pool_sha256": "pool",
                    "system_id": "B1",
                    "prompt_id": "Q01",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        analyze(eval_dir, analysis_dir, raw_dir=raw_dir, strict=False)
