from pathlib import Path

import pytest

from evidence_aware_edu.evaluation.additional_metrics import (
    AspectEvidenceRating, EvidenceAssessment, PassageRelevanceRating,
    calculate_evidence_metrics, compare_revisions, judge_evidence,
)
from evidence_aware_edu.schemas import AspectCoverage, Evidence, QualityEvaluation


def evidence(eid):
    return Evidence(evidence_id=eid, source_id="S", source_type="web", title="Source",
                    url="https://example.org", passage="Grounded example passage",
                    retrieval_query="test question")


def quality(aspects, dim=3):
    return QualityEvaluation(
        aspect_scores=[AspectCoverage(aspect=a, score=s) for a, s in aspects],
        coverage_score=sum(v for _, v in aspects) / len(aspects), relevance_score=0.8,
        **{name: dim for name in [
            "opening_orientation_score", "introduction_roadmap_score",
            "definitions_foundations_score", "explanatory_depth_score",
            "examples_applications_score", "organization_transitions_score",
            "clarity_accessibility_score", "spoken_naturalness_score",
            "conclusion_takeaway_score", "reflective_ending_score"
        ]},
    )


def test_five_metric_types_and_correct_denominators():
    assessment = EvidenceAssessment(
        grounded_aspects=[AspectEvidenceRating(aspect="A", score=1),
                          AspectEvidenceRating(aspect="B", score=.5)],
        passage_relevance=[PassageRelevanceRating(evidence_id="e1", relevant=True),
                           PassageRelevanceRating(evidence_id="e2", relevant=False)],
        retrieved_aspects=[AspectEvidenceRating(aspect="A", score=1),
                           AspectEvidenceRating(aspect="B", score=.5)],
    )
    result = calculate_evidence_metrics(
        assessment, ["A", "B"],
        [AspectCoverage(aspect="A", score=1), AspectCoverage(aspect="B", score=.5)],
        [evidence("e1"), evidence("e2")], k=5,
    )
    assert result["grounded_requirement_coverage_percent"] == 62.5
    assert result["context_precision_at_k"] == .5
    assert result["context_recall_percent"] == 75


def test_missing_and_no_retrieval_values_are_not_fabricated():
    values = calculate_evidence_metrics(EvidenceAssessment(), ["A"],
                    [AspectCoverage(aspect="A", score=1)], [], have_reference=False)
    assert values["grounded_requirement_coverage_percent"] is None
    assert values["context_precision_at_k"] is None
    assert values["context_recall_percent"] is None


def test_missing_aspect_is_scored_zero_conservatively():
    values = calculate_evidence_metrics(
        EvidenceAssessment(grounded_aspects=[AspectEvidenceRating(aspect="A", score=1)]),
        ["A", "B"], [AspectCoverage(aspect="A", score=1), AspectCoverage(aspect="B", score=1)],
        [evidence("e1")], have_reference=True)
    assert values["grounded_requirement_coverage_percent"] == 50.0


def test_repair_quality_and_preservation_calculated_from_paired_drafts():
    before = quality([("A", 1), ("B", .5), ("C", 0)], 2)
    after = quality([("A", 1), ("B", 0), ("C", 1)], 3)
    result = compare_revisions(before, after)
    assert result["requirements_previously_addressed"] == 2
    assert result["requirements_preserved_without_regression"] == 1
    assert result["requirement_preservation_rate"] == .5
    assert result["repair_effectiveness_points"] > 0


def test_judge_prompt_contains_retrieval_and_frozen_reference_evidence():
    class FakeLLM:
        def generate_structured(self, **kwargs):
            assert "Grounded example passage" in kwargs["prompt"]
            assert "EXPECTED ASPECTS" in kwargs["prompt"]
            return EvidenceAssessment(
                grounded_aspects=[AspectEvidenceRating(aspect="A", score=1.0)],
                retrieved_aspects=[AspectEvidenceRating(aspect="A", score=0.5)],
                passage_relevance=[PassageRelevanceRating(evidence_id="retrieved", relevant=True)],
            ), object()
    judgement, _ = judge_evidence(FakeLLM(), "question", ["A"], "script",
                                  [evidence("reference")], [evidence("retrieved")])
    assert isinstance(judgement, EvidenceAssessment)


def test_ten_final_and_ten_pilot_questions_are_distinct():
    import json
    final = [json.loads(x) for x in Path("data/final_benchmark_10.jsonl").read_text().splitlines()]
    pilot = [json.loads(x) for x in Path("data/pilot_benchmark_10.jsonl").read_text().splitlines()]
    assert len(final) == 10 and len(pilot) == 10
    assert not {x["question"] for x in final} & {x["question"] for x in pilot}
    assert all("expected_aspects" not in x and "cutoff_date" not in x for x in final + pilot)


def test_six_figure_definitions_and_no_figure_02():
    import sys
    sys.path.insert(0, str(Path("scripts").resolve()))
    from architecture_figures import ARCHITECTURES
    assert set(ARCHITECTURES) == {"B1", "B2", "B3", "B4", "B5", "P1"}
    assert not Path("research_assets/figures/Figure_02_proposed_architecture.png").exists()


def test_added_metric_plots_are_png_only_when_values_available(tmp_path):
    import pandas as pd
    import sys
    sys.path.insert(0, str(Path("scripts").resolve()))
    from plot_added_metrics import plot_added_metrics

    frame = pd.DataFrame([
        {"system_id": "B2", "prompt_id": "FQ01", "query_coverage_percent": 72,
         "grounded_requirement_coverage_percent": 62, "context_precision_at_k": .9,
         "context_recall_percent": 80, "repair_effectiveness_points": None,
         "requirement_preservation_rate": None},
        {"system_id": "P1", "prompt_id": "FQ01", "query_coverage_percent": 80,
         "grounded_requirement_coverage_percent": 75, "context_precision_at_k": .8,
         "context_recall_percent": 90, "repair_effectiveness_points": 6,
         "requirement_preservation_rate": .75},
    ])
    files = plot_added_metrics(frame, tmp_path)
    assert len(files) == 4
    assert all(p.suffix == ".png" and p.read_bytes().startswith(b"\x89PNG") for p in files)
    assert not list(tmp_path.glob("*.pdf"))


def test_new_gemini_model_and_nonzero_pricing_defaults():
    from evidence_aware_edu.config import Settings
    settings = Settings(_env_file=None)
    assert settings.gemini_model == "gemini-3.8-flash"
    assert settings.thinking_level == "low"
    assert settings.gemini_pricing() == (.75, 3.75)
