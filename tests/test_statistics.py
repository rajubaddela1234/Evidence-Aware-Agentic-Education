import pandas as pd

from evidence_aware_edu.evaluation.statistics import compare_p1_to_baselines


def test_pairwise_wins_respect_metric_direction():
    df = pd.DataFrame(
        [
            {"system_id": "P1", "prompt_id": "Q01", "query_coverage_percent": 90, "latency_seconds": 4},
            {"system_id": "B1", "prompt_id": "Q01", "query_coverage_percent": 80, "latency_seconds": 6},
            {"system_id": "P1", "prompt_id": "Q02", "query_coverage_percent": 70, "latency_seconds": 8},
            {"system_id": "B1", "prompt_id": "Q02", "query_coverage_percent": 75, "latency_seconds": 7},
        ]
    )
    out = compare_p1_to_baselines(
        df,
        metrics=["query_coverage_percent", "latency_seconds"],
    )
    coverage = out[out.metric == "query_coverage_percent"].iloc[0]
    latency = out[out.metric == "latency_seconds"].iloc[0]
    assert coverage.p1_wins == 1 and coverage.p1_losses == 1
    assert latency.p1_wins == 1 and latency.p1_losses == 1
    assert latency.preferred_direction == "lower"


def test_cost_is_lower_is_better():
    df = pd.DataFrame(
        [
            {"system_id": "P1", "prompt_id": "Q01", "estimated_llm_cost_usd": 0.02},
            {"system_id": "B1", "prompt_id": "Q01", "estimated_llm_cost_usd": 0.01},
            {"system_id": "P1", "prompt_id": "Q02", "estimated_llm_cost_usd": 0.01},
            {"system_id": "B1", "prompt_id": "Q02", "estimated_llm_cost_usd": 0.03},
        ]
    )
    out = compare_p1_to_baselines(df, metrics=["estimated_llm_cost_usd"])
    row = out.iloc[0]
    assert row.preferred_direction == "lower"
    assert row.p1_wins == 1
    assert row.p1_losses == 1
