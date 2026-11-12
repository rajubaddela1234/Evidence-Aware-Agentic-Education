from __future__ import annotations

import numpy as np
import pandas as pd


def compare_p1_to_baselines(
    df: pd.DataFrame,
    proposed_id: str = "P1",
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compare P1 with each baseline question by question."""
    metrics = metrics or [
        "query_coverage_percent",
        "reference_evidence_support_percent",
        "query_relevance_percent",
        "narration_quality_percent",
        "llm_calls",
        "retrieval_calls",
        "latency_seconds",
        "estimated_llm_cost_usd",
    ]

    higher_is_better = {
        "query_coverage_percent",
        "reference_evidence_support_percent",
        "query_relevance_percent",
        "narration_quality_percent",
        "supported_claim_rate",
        "citation_precision",
        "citation_recall",
        "citation_f1",
    }

    rows: list[dict] = []
    baseline_ids = [x for x in sorted(df.system_id.unique()) if x != proposed_id]

    for metric in metrics:
        if metric not in df.columns:
            continue
        for baseline in baseline_ids:
            left = df[df.system_id == proposed_id][["prompt_id", metric]].rename(
                columns={metric: "p1"}
            )
            right = df[df.system_id == baseline][["prompt_id", metric]].rename(
                columns={metric: "base"}
            )
            merged = left.merge(right, on="prompt_id").dropna()
            if merged.empty:
                continue

            p1 = merged["p1"].astype(float).to_numpy()
            base = merged["base"].astype(float).to_numpy()
            difference = p1 - base
            quality_metric = metric in higher_is_better

            wins = difference > 0 if quality_metric else difference < 0
            losses = difference < 0 if quality_metric else difference > 0
            rows.append(
                {
                    "metric": metric,
                    "preferred_direction": "higher" if quality_metric else "lower",
                    "baseline": baseline,
                    "n_prompts": int(len(merged)),
                    "p1_mean": float(np.mean(p1)),
                    "baseline_mean": float(np.mean(base)),
                    "mean_difference_p1_minus_baseline": float(np.mean(difference)),
                    "median_difference_p1_minus_baseline": float(np.median(difference)),
                    "p1_wins": int(np.sum(wins)),
                    "ties": int(np.sum(difference == 0)),
                    "p1_losses": int(np.sum(losses)),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inferential statistics for the paper: paired tests across questions and bootstrap CIs.
# ---------------------------------------------------------------------------

HEADLINE_METRICS = [
    "query_coverage_percent",
    "reference_evidence_support_percent",
    "query_relevance_percent",
    "narration_quality_percent",
    "citation_f1",
    "grounded_requirement_coverage_percent",
    "context_recall_percent",
    "estimated_llm_cost_usd",
    "latency_seconds",
]
LOWER_IS_BETTER = {"estimated_llm_cost_usd", "latency_seconds", "llm_calls", "retrieval_calls"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261111


def bootstrap_ci(
    values: np.ndarray, *, confidence: float = 0.95, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean (fixed seed, reproducible)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def system_means_with_ci(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Mean, SD and 95% bootstrap CI of each metric for each system."""
    metrics = [m for m in (metrics or HEADLINE_METRICS) if m in df.columns]
    rows: list[dict] = []
    for system_id in sorted(df.system_id.unique()):
        part = df[df.system_id == system_id]
        for metric in metrics:
            values = part[metric].dropna().astype(float).to_numpy()
            if values.size == 0:
                continue
            low, high = bootstrap_ci(values)
            rows.append(
                {
                    "system_id": system_id,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def _holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _rank_biserial(difference: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: +1 means the first system always wins."""
    nonzero = difference[difference != 0]
    if nonzero.size == 0:
        return 0.0
    from scipy.stats import rankdata

    ranks = rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def pairwise_significance(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Wilcoxon signed-rank tests for every pair of systems, paired by question.

    p-values are Holm-adjusted within each metric across all system pairs. The effect size is
    the matched-pairs rank-biserial correlation, signed so that positive favours system_a in
    the metric's preferred direction.
    """
    from scipy.stats import wilcoxon

    metrics = [m for m in (metrics or HEADLINE_METRICS) if m in df.columns]
    systems = sorted(df.system_id.unique())
    rows: list[dict] = []
    for metric in metrics:
        metric_rows: list[dict] = []
        for i, system_a in enumerate(systems):
            for system_b in systems[i + 1:]:
                a = df[df.system_id == system_a][["prompt_id", metric]].rename(columns={metric: "a"})
                b = df[df.system_id == system_b][["prompt_id", metric]].rename(columns={metric: "b"})
                merged = a.merge(b, on="prompt_id").dropna()
                if len(merged) < 2:
                    continue
                difference = merged["a"].astype(float).to_numpy() - merged["b"].astype(float).to_numpy()
                if metric in LOWER_IS_BETTER:
                    difference = -difference  # positive now always means "system_a is better"
                if np.all(difference == 0):
                    p_value = 1.0
                else:
                    p_value = float(wilcoxon(difference, zero_method="zsplit").pvalue)
                metric_rows.append(
                    {
                        "metric": metric,
                        "preferred_direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                        "system_a": system_a,
                        "system_b": system_b,
                        "n_questions": int(len(merged)),
                        "mean_a": float(merged["a"].mean()),
                        "mean_b": float(merged["b"].mean()),
                        "a_better_count": int(np.sum(difference > 0)),
                        "ties": int(np.sum(difference == 0)),
                        "b_better_count": int(np.sum(difference < 0)),
                        "rank_biserial_a_better": _rank_biserial(difference),
                        "p_value": p_value,
                    }
                )
        for row, adjusted in zip(metric_rows, _holm([r["p_value"] for r in metric_rows])):
            row["p_holm"] = adjusted
            row["significant_0_05"] = bool(adjusted < 0.05)
        rows.extend(metric_rows)
    return pd.DataFrame(rows)
