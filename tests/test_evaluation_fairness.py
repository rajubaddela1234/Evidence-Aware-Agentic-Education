"""Evaluation fairness: clean narration input, complete scripts, enforced 0/0.5/1 rubric."""
from __future__ import annotations

import pytest

from evidence_aware_edu.evaluation.additional_metrics import (
    AspectEvidenceRating,
    EvidenceAssessment,
    PassageRelevanceRating,
    judge_evidence,
)
from evidence_aware_edu.evaluation.checklist import (
    CHECKLIST_SIZE,
    QuestionChecklist,
    checklist_problems,
    load_or_derive_checklist,
)
from evidence_aware_edu.evaluation.judge import QualityJudge
from evidence_aware_edu.evaluation.rubric import (
    RubricViolation,
    generate_with_rubric,
    invalid_aspect_scores,
    is_rubric_score,
)
from evidence_aware_edu.media import did
from evidence_aware_edu.schemas import AspectCoverage, Evidence, QualityEvaluation
from evidence_aware_edu.utils import strip_evidence_markers

CHECKLIST = [f"Aspect {i}" for i in range(1, 7)]

NARRATION_FIELDS = [
    "opening_orientation_score", "introduction_roadmap_score",
    "definitions_foundations_score", "explanatory_depth_score",
    "examples_applications_score", "organization_transitions_score",
    "clarity_accessibility_score", "spoken_naturalness_score",
    "conclusion_takeaway_score", "reflective_ending_score",
]


def make_evidence(eid: str, text: str = "Grounded example passage") -> Evidence:
    return Evidence(evidence_id=eid, source_id="S", source_type="web", title="Source",
                    url="https://example.org", passage=text, retrieval_query="q")


def make_quality(scores: list[float]) -> QualityEvaluation:
    return QualityEvaluation(
        aspect_scores=[AspectCoverage(aspect=f"A{i}", score=s) for i, s in enumerate(scores)],
        coverage_score=sum(scores) / len(scores), relevance_score=0.8,
        **{name: 3 for name in NARRATION_FIELDS},
    )


class ScriptedLLM:
    """Returns queued results and records every prompt it was given."""

    def __init__(self, *results):
        self.results = list(results)
        self.prompts: list[str] = []

    def generate_structured(self, *, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.results.pop(0), f"log{len(self.prompts)}"


# --- 1. markers are removed from narration text -------------------------------------

def test_strip_evidence_markers_leaves_clean_spoken_text():
    text = "Cells use ATP [E-abc123]. Mitochondria help [E-1, E-2] a lot [E-x][E-y] today."
    assert strip_evidence_markers(text) == "Cells use ATP. Mitochondria help a lot today."


def test_strip_evidence_markers_removes_comma_separated_marker_runs():
    text = (
        "RAG grounds answers in facts [E-f7b0], [E-8005]. It conditions generation "
        "[E-11fe], [E-a7a6] and [E-ab12], then answers ([E-1]; [E-2])."
    )
    assert strip_evidence_markers(text) == (
        "RAG grounds answers in facts. It conditions generation, then answers."
    )


def test_strip_evidence_markers_keeps_ordinary_brackets_and_none():
    assert strip_evidence_markers("Array [0] and [Note] stay.") == "Array [0] and [Note] stay."
    assert strip_evidence_markers("") == ""


def test_video_renderer_receives_text_without_markers(monkeypatch):
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["input"] = json["script"]["input"]
        raise RuntimeError("stop after capturing payload")

    monkeypatch.setattr(did.requests, "post", fake_post)
    with pytest.raises(RuntimeError):
        did.create_did_video(script="Fact one [E-123abc]. Fact two.", api_key="k",
                             presenter_url="https://example.org/p.png")
    assert "[E-" not in sent["input"]
    assert sent["input"] == "Fact one. Fact two."


# --- 2. complete scripts, no 5,500-character truncation ------------------------------

def complete_assessment(aspects, passage_ids, grounded=1.0, retrieved=0.5):
    """A judge answer that covers every aspect and every ranked passage, as required."""
    return EvidenceAssessment(
        grounded_aspects=[AspectEvidenceRating(aspect=a, score=grounded) for a in aspects],
        retrieved_aspects=[AspectEvidenceRating(aspect=a, score=retrieved) for a in aspects],
        passage_relevance=[PassageRelevanceRating(evidence_id=i, relevant=True) for i in passage_ids],
    )


def test_judge_evidence_sends_the_entire_script():
    script = "intro " + "x" * 20000 + " FINAL-SENTENCE-AT-THE-END"
    llm = ScriptedLLM(complete_assessment(["A"], ["g1"]))
    judge_evidence(llm, "q", ["A"], script, [make_evidence("r1")], [make_evidence("g1")])
    assert len(llm.prompts) == 1
    assert "FINAL-SENTENCE-AT-THE-END" in llm.prompts[0]


def test_evidence_is_batched_when_prompt_budget_is_exceeded():
    reference = [make_evidence(f"r{i}", "R" * 400) for i in range(12)]
    retrieved = [make_evidence(f"g{i}", "G" * 400) for i in range(12)]
    script = "Full script. END-MARK"

    scores = iter([(0.0, 0.5), (1.0, 0.0), (0.5, 0.0)] + [(0.0, 0.0)] * 30)

    class BatchLLM(ScriptedLLM):
        """Answers every aspect and every ranked passage present in each batch prompt."""

        def generate_structured(self, *, prompt, **kwargs):
            self.prompts.append(prompt)
            ranked = prompt.split("GENERATOR RETRIEVED PASSAGES (RANKED):", 1)[1]
            ids = [f"g{i}" for i in range(12) if f"[g{i}]" in ranked]
            grounded, retrieved_score = next(scores)
            answer = complete_assessment(["A"], ids, grounded, retrieved_score)
            if not ids:
                answer.retrieved_aspects = []
            return answer, f"log{len(self.prompts)}"

    llm = BatchLLM()
    recorded: list = []
    merged, last_log = judge_evidence(
        llm, "q", ["A"], script, reference, retrieved,
        max_prompt_chars=1, record=recorded.append,  # force the minimum evidence batch size
    )
    n_calls = len(llm.prompts)
    assert n_calls > 1
    assert all("END-MARK" in p for p in llm.prompts)  # the script is complete in every batch
    assert len(recorded) == n_calls - 1  # earlier batches still reach telemetry
    assert last_log == f"log{n_calls}"
    # An aspect keeps its best score across batches; relevance rows from all batches survive.
    assert [r.score for r in merged.grounded_aspects] == [1.0]
    assert [r.score for r in merged.retrieved_aspects] == [0.5]
    # Every ranked passage received exactly one relevance judgement across the batches.
    assert sorted(r.evidence_id for r in merged.passage_relevance) == sorted(f"g{i}" for i in range(12))
    # No passage is dropped: every id appears in exactly one batch prompt.
    for evidence_id in [f"[r{i}]" for i in range(12)] + [f"[g{i}]" for i in range(12)]:
        assert sum(evidence_id in p for p in llm.prompts) == 1


def test_small_input_uses_a_single_call():
    llm = ScriptedLLM(complete_assessment(["A"], ["g1"]))
    judge_evidence(llm, "q", ["A"], "short script",
                   [make_evidence("r1")], [make_evidence("g1")])
    assert len(llm.prompts) == 1


# --- 3. the 0 / 0.5 / 1 rubric is enforced -------------------------------------------

@pytest.mark.parametrize("value, ok", [(0, True), (0.5, True), (1, True), (1.0, True),
                                       (0.7, False), (0.25, False), (None, False), ("x", False)])
def test_is_rubric_score(value, ok):
    assert is_rubric_score(value) is ok


def test_invalid_aspect_scores_names_the_offending_aspects():
    items = [AspectCoverage(aspect="Good", score=0.5), AspectCoverage(aspect="Bad", score=0.7)]
    problems = invalid_aspect_scores(items, "coverage")
    assert len(problems) == 1 and "Bad" in problems[0]


def test_quality_judge_keeps_continuous_coverage_scores_without_retry():
    # Coverage is scored on a continuous 0-1 scale, so 0.7 is valid and must not be re-asked.
    llm = ScriptedLLM(make_quality([0.7, 0.95]))
    recorded: list = []
    quality, log = QualityJudge(llm).run("q", ["A0", "A1"], "script", record=recorded.append)
    assert [a.score for a in quality.aspect_scores] == [0.7, 0.95]
    assert len(llm.prompts) == 1
    assert recorded == [] and log == "log1"


def test_rubric_retry_uses_the_callers_rule_and_records_the_rejected_call():
    llm = ScriptedLLM("bad", "good")
    recorded: list = []
    result, log = generate_with_rubric(
        llm,
        prompt="p",
        validate=lambda r: ["problem"] if r == "bad" else [],
        record=recorded.append,
        rule="scores must be between 0.0 and 1.0",
    )
    assert result == "good" and log == "log2" and recorded == ["log1"]
    assert "scores must be between 0.0 and 1.0" in llm.prompts[1]
    assert "0, 0.5 or 1" not in llm.prompts[1]


def test_quality_judge_accepts_valid_scores_without_retry():
    llm = ScriptedLLM(make_quality([0.0, 0.75, 1.0]))
    QualityJudge(llm).run("q", ["A0", "A1", "A2"], "script")
    assert len(llm.prompts) == 1


def test_quality_judge_re_asks_scores_between_absent_and_present():
    # Coverage is 0.0 (absent) or 0.7-1.0 (present); 0.5 is not on the scale.
    llm = ScriptedLLM(make_quality([0.5, 0.9]), make_quality([0.7, 0.9]))
    quality, _ = QualityJudge(llm).run("q", ["A0", "A1"], "script", record=lambda log: None)
    assert len(llm.prompts) == 2 and "0.5" in llm.prompts[1]
    assert [a.score for a in quality.aspect_scores] == [0.7, 0.9]


def test_judge_evidence_retries_invalid_grounded_scores():
    bad = EvidenceAssessment(grounded_aspects=[AspectEvidenceRating(aspect="A", score=0.8)])
    good = EvidenceAssessment(grounded_aspects=[AspectEvidenceRating(aspect="A", score=1.0)])
    llm = ScriptedLLM(bad, good)
    result, _ = judge_evidence(llm, "q", ["A"], "script", [make_evidence("r")], [])
    assert result.grounded_aspects[0].score == 1.0
    assert len(llm.prompts) == 2


def test_persistent_rubric_violation_fails_loudly_instead_of_scoring():
    llm = ScriptedLLM(*[make_quality([0.7]) for _ in range(5)])
    with pytest.raises(RubricViolation, match="off-rubric"):
        generate_with_rubric(
            llm, prompt="p", retries=2,
            validate=lambda q: invalid_aspect_scores(q.aspect_scores, "coverage"),
            task="t", system_instruction="s", schema=QualityEvaluation,
        )
    assert len(llm.prompts) == 3  # initial attempt plus two retries


# --- 4. the evaluator wires it all together -------------------------------------------

def test_evaluator_judges_clean_script_but_verifies_claims_on_the_marked_script(monkeypatch, tmp_path):
    from evidence_aware_edu.config import Settings
    from evidence_aware_edu.evaluation import evaluator as ev
    from evidence_aware_edu.schemas import BenchmarkPrompt, LLMCallLog, PipelineResult

    seen: dict[str, object] = {"quality_scripts": [], "evidence_scripts": [], "claim_scripts": []}

    def log() -> LLMCallLog:
        return LLMCallLog(task="t", model="m", latency_seconds=0.0)

    class FakeClient:
        resolved_model = "fake-model"
        cli_version = "0"
        checklist_calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def generate_structured(self, **kwargs):
            assert kwargs["task"] == "question_coverage_checklist"
            FakeClient.checklist_calls += 1
            return QuestionChecklist(aspects=CHECKLIST), log()

    class FakeExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def from_script(self, script):
            seen["claim_scripts"].append(script)
            return []

    class FakeVerifier:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return []

    class FakeQualityJudge:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, question, aspects, script, *, record=None):
            seen["quality_scripts"].append(script)
            seen.setdefault("quality_aspects", []).append(aspects)
            assert record is not None  # rejected attempts must reach telemetry
            return make_quality([1.0, 0.5]), log()

    def fake_judge_evidence(llm, question, aspects, script, reference, retrieved, **kwargs):
        seen["evidence_scripts"].append(script)
        seen["evidence_aspects"] = aspects
        seen["judge_kwargs"] = kwargs
        return EvidenceAssessment(), log()

    monkeypatch.setattr(ev, "ClaudeEvaluatorClient", FakeClient)
    monkeypatch.setattr(ev, "ClaimExtractorAgent", FakeExtractor)
    monkeypatch.setattr(ev, "VerifierAgent", FakeVerifier)
    monkeypatch.setattr(ev, "QualityJudge", FakeQualityJudge)
    monkeypatch.setattr(ev, "judge_evidence", fake_judge_evidence)

    settings = Settings(_env_file=None)
    settings.langsmith_tracing = False
    settings.output_dir = tmp_path  # the shared checklist is cached under the experiment folder
    evaluator = object.__new__(ev.ResearchEvaluator)
    evaluator.settings = settings
    evaluator.frozen = type("F", (), {"load": staticmethod(lambda prompt_id: [])})()

    marked = "ATP stores energy [E-aaa111]. It is used by cells [E-bbb222]."
    draft = "Draft ATP text [E-aaa111]."
    result = PipelineResult(
        system_id="P1", system_name="P1", prompt_id="Q1", query="q", script=marked,
        evidence=[make_evidence("E-aaa111")], metadata={"pre_repair_script": draft},
    )
    prompt = BenchmarkPrompt(prompt_id="Q1", domain="d", reasoning_type="r", question="q")
    record, details = evaluator.evaluate(prompt, result, experiment_id="EXP_T")

    clean = "ATP stores energy. It is used by cells."
    assert seen["claim_scripts"] == [marked]  # verification keeps the markers
    assert seen["quality_scripts"] == [clean, "Draft ATP text."]  # final and pre-repair draft
    assert seen["evidence_scripts"] == [clean]
    assert seen["judge_kwargs"]["max_prompt_chars"] == settings.claude_evaluator_max_judge_prompt_chars
    assert callable(seen["judge_kwargs"]["record"])
    assert details["narration_judged_script_sha256"] != record.evaluated_script_sha256
    # The judge, the grounded-requirement judge and the revision comparison all use the same
    # six-aspect checklist derived from the question.
    assert seen["quality_aspects"] == [CHECKLIST, CHECKLIST]
    assert seen["evidence_aspects"] == CHECKLIST
    assert details["question_checklist"]["aspects"] == CHECKLIST
    assert record.query_coverage_percent is not None

    # A second system on the same question re-uses the saved checklist: no new derivation.
    other = result.model_copy(update={"system_id": "B2"})
    evaluator.evaluate(prompt, other, experiment_id="EXP_T")
    assert FakeClient.checklist_calls == 1
    assert seen["evidence_aspects"] == CHECKLIST


# --- 5. question-derived checklist replaces hand-written expected aspects -------------

def _checklist_llm(*lists):
    return ScriptedLLM(*[QuestionChecklist(aspects=a) for a in lists])


def test_checklist_is_derived_once_saved_and_reused(tmp_path):
    llm = _checklist_llm(CHECKLIST)
    first, info1 = load_or_derive_checklist(llm, question="Why is the sky blue?",
                                            prompt_id="FQ01", directory=tmp_path)
    second, info2 = load_or_derive_checklist(llm, question="Why is the sky blue?",
                                             prompt_id="FQ01", directory=tmp_path)
    assert first == second == CHECKLIST
    assert len(llm.prompts) == 1  # the second call came from the saved file
    assert info1["derived_in_this_call"] is True and info2["derived_in_this_call"] is False
    assert info1["aspects_sha256"] == info2["aspects_sha256"]
    assert (tmp_path / "FQ01.json").is_file()


def test_checklist_is_rederived_when_the_question_changes(tmp_path):
    llm = _checklist_llm(CHECKLIST, [f"Other {i}" for i in range(6)])
    load_or_derive_checklist(llm, question="Question one?", prompt_id="FQ01", directory=tmp_path)
    changed, _ = load_or_derive_checklist(llm, question="Question two?", prompt_id="FQ01",
                                          directory=tmp_path)
    assert changed[0] == "Other 0" and len(llm.prompts) == 2


def test_checklist_retries_wrong_size_then_succeeds(tmp_path):
    llm = _checklist_llm(["only", "two"], CHECKLIST)
    aspects, _ = load_or_derive_checklist(llm, question="Q?", prompt_id="FQ02", directory=tmp_path)
    assert aspects == CHECKLIST and len(llm.prompts) == 2
    assert "rejected" in llm.prompts[1]


def test_checklist_fails_loudly_when_it_never_becomes_valid(tmp_path):
    llm = _checklist_llm(*[["x"]] * 3)
    with pytest.raises(ValueError, match="Could not derive"):
        load_or_derive_checklist(llm, question="Q?", prompt_id="FQ03", directory=tmp_path)
    assert not (tmp_path / "FQ03.json").exists()


def test_checklist_validation_rules():
    assert checklist_problems(CHECKLIST) == []
    assert checklist_problems(["a", "A", "b", "c", "d", "e"])  # duplicate (case-insensitive)
    assert checklist_problems(["a", "b", "c", "d", "e", " "])  # empty item
    assert checklist_problems(CHECKLIST[: CHECKLIST_SIZE - 1])  # wrong size
