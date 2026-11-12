"""Tests for the evaluation-correctness fixes from the external code review."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evidence_aware_edu.agents.verifier import IncompleteVerification, VerifierAgent
from evidence_aware_edu.config import Settings
from evidence_aware_edu.evaluation.additional_metrics import (
    AspectEvidenceRating,
    EvidenceAssessment,
    PassageRelevanceRating,
    _format_passages,
    judge_evidence,
)
from evidence_aware_edu.evaluation.checklist import (
    ChecklistIntegrityError,
    QuestionChecklist,
    load_or_derive_checklist,
)
from evidence_aware_edu.evaluation.evaluator import ResearchEvaluator
from evidence_aware_edu.evaluation.judge import QualityJudge
from evidence_aware_edu.evaluation.rubric import RubricViolation
from evidence_aware_edu.llm.gemini import GeminiClient, TruncatedGeneration
from evidence_aware_edu.schemas import (
    AspectCoverage,
    BenchmarkPrompt,
    Claim,
    ClaimVerification,
    Evidence,
    QualityEvaluation,
    VerificationBundle,
)

CHECKLIST = [f"Aspect {i}" for i in range(1, 7)]
NARRATION = {
    name: 3
    for name in [
        "opening_orientation_score", "introduction_roadmap_score", "definitions_foundations_score",
        "explanatory_depth_score", "examples_applications_score", "organization_transitions_score",
        "clarity_accessibility_score", "spoken_naturalness_score", "conclusion_takeaway_score",
        "reflective_ending_score",
    ]
}


def ev(eid: str, text: str = "passage") -> Evidence:
    return Evidence(evidence_id=eid, source_id="S", source_type="web", title="T",
                    url="https://example.org", passage=text, retrieval_query="q")


class Scripted:
    """Returns queued answers and records every prompt."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.settings = SimpleNamespace(max_context_chars=50000)

    def generate_structured(self, *, prompt, **_):
        self.prompts.append(prompt)
        return self.answers.pop(0), f"log{len(self.prompts)}"


def telemetry():
    return SimpleNamespace(add_llm=lambda log: None)


# --- 1. citation checking keeps each claim's own citations -------------------------------

def test_verifier_prompt_lists_each_claims_allowed_evidence():
    claims = [Claim(claim_id="c1", text="one"), Claim(claim_id="c2", text="two")]
    a = VerifierAgent._prompt(claims, [ev("E-1"), ev("E-2")], 10000,
                              allowed_by_claim={"c1": {"E-1"}, "c2": {"E-2"}})
    b = VerifierAgent._prompt(claims, [ev("E-1"), ev("E-2")], 10000,
                              allowed_by_claim={"c1": {"E-2"}, "c2": {"E-2"}})
    assert "c1: one\n   ALLOWED EVIDENCE: E-1" in a
    assert a != b  # changing a claim's citations now changes what the verifier is told


def test_normalize_rejects_evidence_cited_by_a_different_claim():
    claims = [Claim(claim_id="c1", text="one"), Claim(claim_id="c2", text="two")]
    returned = [
        ClaimVerification(claim_id="c1", status="supported", confidence=0.9, evidence_ids=["E-2"]),
        ClaimVerification(claim_id="c2", status="supported", confidence=0.9, evidence_ids=["E-2"]),
    ]
    out = VerifierAgent._normalize(claims, [ev("E-1"), ev("E-2")], returned,
                                   allowed_by_claim={"c1": {"E-1"}, "c2": {"E-2"}})
    assert out[0].status == "unsupported" and out[0].evidence_ids == []
    assert out[1].status == "supported" and out[1].evidence_ids == ["E-2"]


def test_citation_entailment_skips_uncited_claims_and_passes_per_claim_allow_lists():
    calls = []

    class FakeVerifier:
        def run(self, claims, evidence, **kwargs):
            calls.append(kwargs)
            return [ClaimVerification(claim_id=c.claim_id, status="supported", confidence=0.9,
                                      evidence_ids=sorted(kwargs["allowed_by_claim"][c.claim_id]))
                    for c in claims]

    evaluator = object.__new__(ResearchEvaluator)
    evaluator.settings = SimpleNamespace(claude_evaluator_max_context_chars=40000)
    claims = [Claim(claim_id="c1", text="a", citation_ids=["E-1"]),
              Claim(claim_id="c2", text="b", citation_ids=["E-missing"]),
              Claim(claim_id="c3", text="c", citation_ids=[])]
    out = evaluator._citation_entailment(claims, [ev("E-1")], FakeVerifier())
    assert [v.claim_id for v in out] == ["c1", "c2", "c3"]
    assert [v.status for v in out] == ["supported", "unsupported", "unsupported"]
    assert len(calls) == 1 and calls[0]["allowed_by_claim"] == {"c1": {"E-1"}}
    assert calls[0]["require_complete"] is True


# --- 2. the evidence judge sees complete passages ----------------------------------------

def test_evidence_judge_receives_text_beyond_450_characters():
    passage = "intro " * 100 + "MECHANISM-EXPLAINED-HERE"
    assert len(passage) > 450
    assert "MECHANISM-EXPLAINED-HERE" in _format_passages([ev("E-1", passage)])


# --- 5. omitted judgements are re-asked, never scored as zero -----------------------------

def full_assessment(ids):
    return EvidenceAssessment(
        grounded_aspects=[AspectEvidenceRating(aspect="A", score=1.0)],
        retrieved_aspects=[AspectEvidenceRating(aspect="A", score=1.0)],
        passage_relevance=[PassageRelevanceRating(evidence_id=i, relevant=True) for i in ids],
    )


def test_evidence_judge_re_asks_when_an_aspect_or_passage_is_omitted():
    llm = Scripted(EvidenceAssessment(), full_assessment(["g1"]))
    result, _ = judge_evidence(llm, "q", ["A"], "script", [ev("r1")], [ev("g1")])
    assert len(llm.prompts) == 2 and "omitted" in llm.prompts[1]
    assert result.grounded_aspects[0].score == 1.0


def test_persistently_incomplete_evidence_judgement_fails_the_case():
    llm = Scripted(*[EvidenceAssessment() for _ in range(5)])
    with pytest.raises(RubricViolation, match="omitted"):
        judge_evidence(llm, "q", ["A"], "script", [ev("r1")], [ev("g1")])


def quality(aspects):
    return QualityEvaluation(
        aspect_scores=[AspectCoverage(aspect=a, score=0.9) for a in aspects],
        coverage_score=0.9, relevance_score=0.9, **NARRATION,
    )


def test_quality_judge_re_asks_for_an_unscored_aspect():
    llm = Scripted(quality(["A0"]), quality(["A0", "A1"]))
    result, _ = QualityJudge(llm).run("q", ["A0", "A1"], "script", record=lambda log: None)
    assert len(llm.prompts) == 2 and "'A1' was not scored" in llm.prompts[1]
    assert [a.aspect for a in result.aspect_scores] == ["A0", "A1"]


def test_verifier_re_asks_only_omitted_claims_when_completeness_is_required():
    claims = [Claim(claim_id="c1", text="one"), Claim(claim_id="c2", text="two")]
    first = VerificationBundle(verifications=[
        ClaimVerification(claim_id="c1", status="supported", confidence=0.9, evidence_ids=["E-1"])])
    second = VerificationBundle(verifications=[
        ClaimVerification(claim_id="c2", status="partial", confidence=0.6, evidence_ids=["E-1"])])
    llm = Scripted(first, second)
    out = VerifierAgent(llm, telemetry()).run(claims, [ev("E-1")], require_complete=True)
    assert [v.status for v in out] == ["supported", "partial"]
    assert "c2: two" in llm.prompts[1] and "c1: one" not in llm.prompts[1]


def test_verifier_raises_instead_of_scoring_persistent_omissions_as_unsupported():
    llm = Scripted(*[VerificationBundle() for _ in range(5)])
    with pytest.raises(IncompleteVerification):
        VerifierAgent(llm, telemetry()).run([Claim(claim_id="c1", text="x")], [ev("E-1")],
                                            require_complete=True)
    # Generation-side callers keep the old conservative behaviour.
    llm = Scripted(VerificationBundle())
    out = VerifierAgent(llm, telemetry()).run([Claim(claim_id="c1", text="x")], [ev("E-1")])
    assert out[0].status == "unsupported" and len(llm.prompts) == 1


def test_verifier_splits_many_claims_into_batches_of_ten():
    claims = [Claim(claim_id=f"c{i}", text=f"claim {i}") for i in range(23)]

    def answer(ids):
        return VerificationBundle(verifications=[
            ClaimVerification(claim_id=i, status="supported", confidence=0.9, evidence_ids=["E-1"])
            for i in ids])

    llm = Scripted(answer([f"c{i}" for i in range(10)]), answer([f"c{i}" for i in range(10, 20)]),
                   answer([f"c{i}" for i in range(20, 23)]))
    out = VerifierAgent(llm, telemetry()).run(claims, [ev("E-1")])
    assert len(llm.prompts) == 3
    assert "c9: claim 9" in llm.prompts[0] and "c10: claim 10" not in llm.prompts[0]
    assert [v.claim_id for v in out] == [f"c{i}" for i in range(23)]
    assert all(v.status == "supported" for v in out)


# --- 4. checklists are frozen per experiment ---------------------------------------------

def test_edited_checklist_is_rejected(tmp_path):
    llm = Scripted(QuestionChecklist(aspects=CHECKLIST))
    load_or_derive_checklist(llm, question="Q?", prompt_id="FQ01", directory=tmp_path)
    path = tmp_path / "FQ01.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["aspects"][0] = "A quietly different obligation"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ChecklistIntegrityError, match="edited"):
        load_or_derive_checklist(llm, question="Q?", prompt_id="FQ01", directory=tmp_path)


def test_frozen_checklist_is_never_silently_re_derived(tmp_path):
    with pytest.raises(ChecklistIntegrityError, match="missing"):
        load_or_derive_checklist(Scripted(), question="Q?", prompt_id="FQ01",
                                 directory=tmp_path, allow_derive=False)


def test_freeze_checklists_pins_hashes_and_detects_later_changes(tmp_path):
    evaluator = object.__new__(ResearchEvaluator)
    evaluator.settings = SimpleNamespace(output_dir=tmp_path, evaluator_model="m")
    prompt = BenchmarkPrompt(prompt_id="FQ01", domain="d", reasoning_type="r", question="Q?")
    directory = tmp_path / "experiments" / "EXP" / "checklists"
    _, info = load_or_derive_checklist(Scripted(QuestionChecklist(aspects=CHECKLIST)),
                                       question="Q?", prompt_id="FQ01", directory=directory)
    frozen = evaluator.freeze_checklists([prompt], "EXP", {})
    assert frozen == {"FQ01": info["aspects_sha256"]}
    assert evaluator.freeze_checklists([prompt], "EXP", frozen) == frozen
    (directory / "FQ01.json").unlink()
    with pytest.raises(ChecklistIntegrityError):
        evaluator.freeze_checklists([prompt], "EXP", frozen)


# --- 6. empty or truncated Gemini output is never accepted as a script --------------------

def gemini_with(*responses):
    client = object.__new__(GeminiClient)
    client.settings = Settings(_env_file=None, API_RETRY_ATTEMPTS=2, LANGSMITH_TRACING=False)
    client.model = "gemini-test"
    queue = list(responses)
    client.calls = 0

    def generate_content(**_):
        client.calls += 1
        return queue.pop(0)

    client.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client


def response(text, reason="STOP"):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=reason)],
                           usage_metadata=None)


def test_empty_gemini_text_is_retried_then_fails(monkeypatch):
    monkeypatch.setattr("evidence_aware_edu.llm.gemini.time.sleep", lambda s: None)
    client = gemini_with(response(""), response("   "))
    with pytest.raises(RuntimeError, match="no text"):
        client.generate_text(task="t", system_instruction="s", prompt="p")
    assert client.calls == 2


def test_truncated_gemini_text_fails_immediately(monkeypatch):
    monkeypatch.setattr("evidence_aware_edu.llm.gemini.time.sleep", lambda s: None)
    client = gemini_with(response("half a scri", "FinishReason.MAX_TOKENS"))
    with pytest.raises(TruncatedGeneration):
        client.generate_text(task="t", system_instruction="s", prompt="p")
    assert client.calls == 1
