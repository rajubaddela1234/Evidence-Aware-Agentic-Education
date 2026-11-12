from pathlib import Path


JUDGE_TEXT = Path("src/evidence_aware_edu/evaluation/judge.py").read_text(encoding="utf-8")


def test_quality_judge_keeps_question_and_fixed_checklist_separate():
    prompt = JUDGE_TEXT.upper()
    assert "JUDGE THE SCRIPT AGAINST THE USER QUESTION" in prompt
    assert "FIXED BENCHMARK CHECKLIST" in prompt
    assert "DO NOT USE OUTSIDE KNOWLEDGE" in prompt


def test_narration_rubric_is_flexible_but_uses_same_standard():
    prompt = JUDGE_TEXT.upper()
    assert "USE THE SAME STANDARD FOR EVERY SYSTEM" in prompt
    assert "EXPLICIT ROADMAP" in prompt
    assert "MANDATORY WHEN" in prompt
    assert "DO NOT REQUIRE A LITERAL QUESTION" in prompt


def test_quality_judge_keeps_the_same_ten_dimensions():
    for field in (
        "opening_orientation_score",
        "introduction_roadmap_score",
        "definitions_foundations_score",
        "explanatory_depth_score",
        "examples_applications_score",
        "organization_transitions_score",
        "clarity_accessibility_score",
        "spoken_naturalness_score",
        "conclusion_takeaway_score",
        "reflective_ending_score",
    ):
        assert field in JUDGE_TEXT
