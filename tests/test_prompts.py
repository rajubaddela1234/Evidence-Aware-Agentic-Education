from evidence_aware_edu.prompts import DIRECT_EDUCATIONAL_SYSTEM, GROUNDED_EDUCATIONAL_SYSTEM


def test_educational_prompts_require_complete_narration_blueprint():
    required = [
        "WELCOME AND TOPIC ORIENTATION",
        "INTRODUCTION AND ROADMAP",
        "DEFINITIONS AND FOUNDATIONS",
        "DETAILED EXPLANATORY MIDDLE",
        "EXAMPLE, APPLICATION, OR ANALOGY",
        "CONCLUSION AND TAKEAWAY",
        "REFLECTIVE ENDING",
    ]
    for prompt in (DIRECT_EDUCATIONAL_SYSTEM, GROUNDED_EDUCATIONAL_SYSTEM):
        upper = prompt.upper()
        for phrase in required:
            assert phrase in upper
        assert "EXACTLY ONE OR TWO" in upper
        assert "DO NOT OUTPUT SECTION LABELS" in upper


def test_grounded_prompt_forbids_invented_facts():
    upper = GROUNDED_EDUCATIONAL_SYSTEM.upper()
    assert "USE ONLY FACTUAL INFORMATION SUPPLIED IN THE PROMPT" in upper
    assert "NEVER INVENT AN EVIDENCE ID" in upper
