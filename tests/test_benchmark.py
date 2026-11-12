import json
from pathlib import Path

from evidence_aware_edu.experiments.io import load_benchmark


BENCHMARK = Path("data/pilot_benchmark_10.jsonl")


def test_benchmark_has_exactly_ten_prompts():
    prompts = load_benchmark(BENCHMARK)
    assert len(prompts) == 10
    assert [p.prompt_id for p in prompts] == [f"Q{i:02d}" for i in range(1, 11)]


def test_benchmark_retains_first_five_sections_and_adds_five():
    prompts = load_benchmark(BENCHMARK)
    assert [p.domain for p in prompts[:5]] == [
        "Computer Science / AI",
        "Electrical / Energy",
        "Environmental Science",
        "Business / Economics",
        "General Knowledge",
    ]
    assert len({p.domain for p in prompts}) == 10


def test_questions_are_short_and_carry_no_hand_written_rubric():
    prompts = load_benchmark(BENCHMARK)
    assert all(len(p.question.split()) <= 20 for p in prompts)
    for p in prompts:
        assert not hasattr(p, "expected_aspects") and not hasattr(p, "cutoff_date")
    for line in BENCHMARK.read_text(encoding="utf-8").splitlines():
        assert set(json.loads(line)) == {"prompt_id", "domain", "reasoning_type", "question"}


def test_only_two_benchmark_question_files():
    files = sorted(p.name for p in Path("data").glob("*benchmark_*.jsonl"))
    assert files == ["final_benchmark_10.jsonl", "pilot_benchmark_10.jsonl"]
