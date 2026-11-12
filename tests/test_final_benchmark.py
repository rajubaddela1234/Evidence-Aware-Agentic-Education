import json
from pathlib import Path

from evidence_aware_edu.experiments.io import load_benchmark


FINAL_BENCHMARK = Path("data/final_benchmark_10.jsonl")


def test_final_benchmark_has_exactly_ten_held_out_prompts():
    prompts = load_benchmark(FINAL_BENCHMARK)
    assert len(prompts) == 10
    assert [p.prompt_id for p in prompts] == [f"FQ{i:02d}" for i in range(1, 11)]


def test_final_questions_are_distinct_from_the_pilot_ids():
    prompts = load_benchmark(FINAL_BENCHMARK)
    assert all(prompt.prompt_id.startswith("FQ") for prompt in prompts)
    assert len({prompt.domain for prompt in prompts}) == 10
    assert len({prompt.question for prompt in prompts}) == 10
    assert len({prompt.reasoning_type for prompt in prompts}) >= 6


def test_final_benchmark_has_no_expected_aspects_or_cutoff_date():
    for line in FINAL_BENCHMARK.read_text(encoding="utf-8").splitlines():
        assert set(json.loads(line)) == {"prompt_id", "domain", "reasoning_type", "question"}
