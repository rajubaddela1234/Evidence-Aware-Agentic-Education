"""Extra figures for reporting the final benchmark.

All values come from evaluation_records.csv or the saved generation-call logs.
No chart uses estimated or invented benchmark outcomes.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SYSTEM_ORDER = ["B1", "B2", "B3", "B4", "B5", "P1"]
QUALITY_METRICS = {
    "query_coverage_percent": "Question coverage",
    "reference_evidence_support_percent": "Reference evidence support",
    "query_relevance_percent": "Question relevance",
    "narration_quality_percent": "Narration quality",
}
CLAIM_LABELS = {
    "supported_claims": "Supported",
    "partial_claims": "Partially supported",
    "unsupported_claims": "Unsupported",
    "contradicted_claims": "Contradicted",
}


def _save(fig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _systems(df: pd.DataFrame) -> list[str]:
    return [system for system in SYSTEM_ORDER if system in set(df["system_id"])]


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    prepared = df.copy()
    for column in columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def quality_heatmap(df: pd.DataFrame, metric: str, destination: Path) -> None:
    if metric not in df:
        return
    matrix = df.pivot(index="prompt_id", columns="system_id", values=metric)
    matrix = matrix.reindex(index=sorted(matrix.index), columns=_systems(df))
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(matrix) * 0.7)))
    image = ax.imshow(values, vmin=0, vmax=100, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", color="white")
    ax.set_xlabel("System")
    ax.set_ylabel("Held-out question")
    ax.set_title(f"Per-question {QUALITY_METRICS[metric].lower()}")
    fig.colorbar(image, ax=ax, label="Evaluator score (%)")
    _save(fig, destination)


def paired_difference_plot(df: pd.DataFrame, metric: str, destination: Path) -> pd.DataFrame:
    # Compare the same question on P1 and each baseline. Missing pairs are excluded.
    if metric not in df or "P1" not in set(df.system_id):
        return pd.DataFrame()
    frame = _numeric(df, [metric])
    proposed = frame[frame.system_id == "P1"][["prompt_id", metric]].rename(
        columns={metric: "p1_score"}
    )
    results = []
    for baseline in [s for s in _systems(df) if s != "P1"]:
        base = frame[frame.system_id == baseline][["prompt_id", metric]].rename(
            columns={metric: "baseline_score"}
        )
        paired = proposed.merge(base, on="prompt_id", validate="one_to_one").dropna()
        for row in paired.itertuples(index=False):
            results.append({
                "baseline": baseline,
                "prompt_id": row.prompt_id,
                "metric": metric,
                "p1_score": row.p1_score,
                "baseline_score": row.baseline_score,
                "p1_minus_baseline": row.p1_score - row.baseline_score,
            })
    differences = pd.DataFrame(results)
    if differences.empty:
        return differences

    fig, ax = plt.subplots(figsize=(10, 5.5))
    baselines = [s for s in _systems(df) if s != "P1"]
    for index, baseline in enumerate(baselines):
        y = differences.loc[differences.baseline == baseline, "p1_minus_baseline"]
        y = pd.to_numeric(y, errors="coerce").dropna().to_numpy(float)
        if len(y) == 0:
            continue
        # Each small point is one question; the larger mark is the paired mean.
        offsets = np.linspace(-0.13, 0.13, len(y))
        ax.scatter(np.full(len(y), index) + offsets, y, alpha=0.7, s=27)
        ax.scatter(index, y.mean(), marker="D", s=90, color="black", zorder=5)
    ax.axhline(0, linewidth=1, linestyle="--", color="gray")
    ax.set_xticks(range(len(baselines)), baselines)
    ax.set_xlabel("Baseline")
    ax.set_ylabel("P1 minus baseline (percentage points)")
    ax.set_title(f"Paired differences: {QUALITY_METRICS[metric].lower()}")
    ax.text(0.99, 0.02, "Small dots = questions; black diamond = mean", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9)
    _save(fig, destination)
    return differences


def claim_label_shares(df: pd.DataFrame, destination: Path) -> pd.DataFrame:
    columns = list(CLAIM_LABELS)
    if not set(columns).issubset(df):
        return pd.DataFrame()
    counts = _numeric(df, columns).groupby("system_id")[columns].sum().reindex(_systems(df))
    totals = counts.sum(axis=1)
    shares = counts.div(totals.replace(0, np.nan), axis=0) * 100
    if shares.dropna(how="all").empty:
        return shares
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(shares))
    bottom = np.zeros(len(shares))
    for column in columns:
        values = shares[column].fillna(0).to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=CLAIM_LABELS[column])
        bottom += values
    ax.set_xticks(x, shares.index)
    ax.set_ylim(0, 105)
    ax.set_xlabel("System")
    ax.set_ylabel("Share of extracted claims (%)")
    ax.set_title("Reference-based claim verification outcomes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.text(0.99, 0.02, "Not an independent factual-accuracy measure", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    _save(fig, destination)
    return shares


def citation_scores(df: pd.DataFrame, destination: Path) -> None:
    fields = ["citation_precision", "citation_recall", "citation_f1"]
    if not set(fields).issubset(df):
        return
    means = (_numeric(df, fields).groupby("system_id")[fields].mean()
             .reindex(_systems(df)))
    # These are per-output macro averages, not pooled citation-level precision.
    ax = (100 * means).plot(kind="bar", figsize=(10, 5.5), width=0.8)
    ax.set_ylim(0, 105)
    ax.set_xlabel("System")
    ax.set_ylabel("Mean per-script score (%)")
    ax.set_title("Citation grounding: precision, recall and F1")
    ax.legend(["Precision", "Recall", "F1"])
    _save(ax.figure, destination)


def tradeoff_scatter(df: pd.DataFrame, x_metric: str, y_metric: str,
                     x_label: str, y_label: str, destination: Path) -> None:
    if not {x_metric, y_metric}.issubset(df):
        return
    means = (_numeric(df, [x_metric, y_metric]).groupby("system_id")[[x_metric, y_metric]]
             .mean().reindex(_systems(df)).dropna())
    if means.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(means[x_metric], means[y_metric], s=105)
    for system, row in means.iterrows():
        ax.annotate(system, (row[x_metric], row[y_metric]), xytext=(6, 6),
                    textcoords="offset points")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Evidence support versus generation resources")
    ax.grid(alpha=0.2)
    _save(fig, destination)


def quality_distributions(df: pd.DataFrame, metric: str, destination: Path) -> None:
    if metric not in df:
        return
    frame = _numeric(df, [metric]); systems = _systems(df)
    groups = [frame.loc[frame.system_id == system, metric].dropna().to_numpy() for system in systems]
    if not all(len(group) > 0 for group in groups):
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(groups, tick_labels=systems, patch_artist=False)
    for index, group in enumerate(groups, start=1):
        ax.scatter(np.linspace(index-0.10, index+0.10, len(group)), group,
                   s=26, alpha=0.7)
    ax.set_ylim(0, 105)
    ax.set_xlabel("System")
    ax.set_ylabel("Evaluator score (%)")
    ax.set_title(f"Question-level variation: {QUALITY_METRICS[metric].lower()}")
    ax.text(0.99, 0.02, "Small samples: report descriptive variation only",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    _save(fig, destination)


def call_time_by_task(calls_path: Path, destination: Path) -> None:
    if not calls_path.exists():
        return
    calls = pd.read_csv(calls_path)
    if not {"task", "latency_seconds"}.issubset(calls) or calls.empty:
        return
    calls = calls.copy()
    calls["latency_seconds"] = pd.to_numeric(calls["latency_seconds"], errors="coerce")
    calls = calls.dropna(subset=["latency_seconds"])
    calls = calls[calls.latency_seconds >= 0]
    if calls.empty:
        return
    # Total over the observed generation calls shows where time is spent.
    totals = calls.groupby("task")["latency_seconds"].sum().nlargest(12).sort_values()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(totals.index, totals.values)
    ax.set_xlabel("Cumulative generation-call duration (seconds)")
    ax.set_ylabel("LLM task")
    ax.set_title("Where generation time is spent (observed calls)")
    _save(fig, destination)


def narration_dimension_heatmap(df: pd.DataFrame, destination: Path) -> None:
    columns = [col for col in df.columns
               if col.startswith("narration_") and col.endswith("_percent")
               and col != "narration_quality_percent"]
    if len(columns) != 10:
        return
    means = (df.groupby("system_id")[columns].mean()
             .reindex(_systems(df)))
    if means.isna().any().any():
        return
    fig, ax = plt.subplots(figsize=(12, 5.5))
    values = means.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(columns)), [
        name.removeprefix("narration_").removesuffix("_percent").replace("_", " ")
        for name in columns
    ], rotation=35, ha="right")
    ax.set_yticks(range(len(means)), means.index)
    for i in range(len(means)):
        for j in range(len(columns)):
            ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Mean Claude rubric score (%)")
    ax.set_title("Narration quality by automatically scored rubric dimension")
    _save(fig, destination)


def aspect_status_share_plot(df: pd.DataFrame, destination: Path) -> None:
    fields = ["coverage_aspects_covered", "coverage_aspects_partial", "coverage_aspects_missing"]
    if not set(fields).issubset(df.columns):
        return
    counts = df.groupby("system_id")[fields].sum().reindex(_systems(df))
    totals = counts.sum(axis=1)
    if (totals == 0).any():
        return
    shares = 100.0 * counts.div(totals, axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(shares))
    for field, label in zip(fields, ["Covered", "Partial", "Missing"]):
        values = shares[field].to_numpy(dtype=float)
        ax.bar(shares.index, values, bottom=bottom, label=label)
        bottom += values
    ax.set_ylim(0, 105)
    ax.set_ylabel("Share of benchmark coverage aspects (%)")
    ax.set_xlabel("System")
    ax.set_title("Coverage of the fixed benchmark aspects (Claude evaluation)")
    ax.legend()
    _save(fig, destination)


# ---------------------------------------------------------------------------
# Additional publication figures (25-33). All values come from evaluation_records.csv.
# ---------------------------------------------------------------------------

def radar_profile(df: pd.DataFrame, destination: Path) -> None:
    cols = [c for c in QUALITY_METRICS if c in df]
    if len(cols) < 3:
        return
    means = _numeric(df, cols).groupby("system_id")[cols].mean().reindex(_systems(df)).dropna()
    if means.empty:
        return
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for system, row in means.iterrows():
        vals = row.tolist()
        lw = 3 if system == "P1" else 1.5
        ax.plot(angles + angles[:1], vals + vals[:1], linewidth=lw, label=system)
        if system == "P1":
            ax.fill(angles + angles[:1], vals + vals[:1], alpha=0.15)
    ax.set_xticks(angles, [QUALITY_METRICS[c] for c in cols])
    ax.tick_params(axis="x", pad=22)
    ax.set_ylim(20, 100)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_title("Multi-metric quality profile (mean over questions)", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    _save(fig, destination)


def mean_ci_plot(df: pd.DataFrame, destination: Path, seed: int = 0) -> pd.DataFrame:
    cols = [c for c in QUALITY_METRICS if c in df]
    if not cols:
        return pd.DataFrame()
    frame = _numeric(df, cols)
    systems = _systems(df)
    rng = np.random.default_rng(seed)
    rows = []
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        for i, system in enumerate(systems):
            v = frame.loc[frame.system_id == system, col].dropna().to_numpy(float)
            if len(v) == 0:
                continue
            boots = rng.choice(v, (5000, len(v))).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ax.errorbar(v.mean(), i, xerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o",
                        capsize=4, markersize=8 if system == "P1" else 6)
            rows.append({"metric": col, "system_id": system, "n": len(v),
                         "mean": v.mean(), "ci95_low": lo, "ci95_high": hi})
        ax.set_yticks(range(len(systems)), systems)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Score (%)")
        ax.set_title(QUALITY_METRICS[col])
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    fig.suptitle("Mean with bootstrap 95% CI over questions (descriptive, small n)", y=1.02)
    _save(fig, destination)
    return pd.DataFrame(rows)


def group_breakdown_heatmap(df: pd.DataFrame, group: str, metric: str, destination: Path) -> None:
    if group not in df or metric not in df:
        return
    m = _numeric(df, [metric]).pivot_table(index=group, columns="system_id",
                                          values=metric, aggfunc="mean")
    m = m.reindex(columns=_systems(df))
    if m.empty:
        return
    v = m.to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.6 * len(m) + 2)))
    im = ax.imshow(v, vmin=0, vmax=100, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(m.columns)), m.columns)
    ax.set_yticks(range(len(m.index)), [str(x).replace("_", " ") for x in m.index])
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, f"{v[i, j]:.0f}", ha="center", va="center", color="white")
    ax.set_title(f"{QUALITY_METRICS[metric]} by {group.replace('_', ' ')}")
    fig.colorbar(im, ax=ax, label="Mean score (%)")
    _save(fig, destination)


def token_usage_plot(df: pd.DataFrame, destination: Path) -> None:
    cols = ["prompt_tokens", "output_tokens", "thinking_tokens"]
    if not set(cols).issubset(df):
        return
    m = _numeric(df, cols).groupby("system_id")[cols].mean().reindex(_systems(df)).dropna(how="all")
    if m.empty:
        return
    ax = m.plot(kind="bar", stacked=True, figsize=(10, 5.5), width=0.7)
    ax.set_xlabel("System")
    ax.set_ylabel("Mean tokens per script")
    ax.set_title("Token consumption by system")
    ax.legend(["Prompt", "Output", "Thinking"])
    ax.tick_params(axis="x", rotation=0)
    _save(ax.figure, destination)


def win_tie_loss_plot(df: pd.DataFrame, destination: Path, tol: float = 1.0) -> pd.DataFrame:
    if "P1" not in set(df.system_id):
        return pd.DataFrame()
    cols = [c for c in QUALITY_METRICS if c in df]
    frame = _numeric(df, cols)
    baselines = [s for s in _systems(df) if s != "P1"]
    rows = []
    for b in baselines:
        for c in cols:
            p = frame[frame.system_id == "P1"][["prompt_id", c]].rename(columns={c: "p"})
            q = frame[frame.system_id == b][["prompt_id", c]].rename(columns={c: "b"})
            d = p.merge(q, on="prompt_id").dropna()
            diff = d.p - d.b
            rows.append({"baseline": b, "metric": c, "n": len(d),
                         "p1_wins": int((diff > tol).sum()),
                         "ties": int((diff.abs() <= tol).sum()),
                         "p1_losses": int((diff < -tol).sum())})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        r = res[res.metric == c].set_index("baseline").reindex(baselines)
        left = np.zeros(len(r))
        for field, label, colr in [("p1_wins", "P1 better", "#2a9d8f"),
                                   ("ties", f"Tie (within {tol:g} pt)", "#bbbbbb"),
                                   ("p1_losses", "P1 worse", "#e76f51")]:
            ax.barh(r.index, r[field], left=left, label=label, color=colr)
            left += r[field].to_numpy(float)
        ax.set_title(QUALITY_METRICS[c])
        ax.set_xlabel("Number of questions")
        ax.invert_yaxis()
    axes[0].set_ylabel("Baseline")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Question-level wins, ties and losses of P1 versus each baseline", y=1.02)
    _save(fig, destination)
    return res


def correlation_heatmap(df: pd.DataFrame, destination: Path) -> None:
    cols = [c for c in list(QUALITY_METRICS) + [
        "citation_f1", "grounded_requirement_coverage_percent", "context_precision_at_k",
        "context_recall_percent", "estimated_llm_cost_usd", "latency_seconds", "word_count",
        "source_count"] if c in df]
    if len(cols) < 3:
        return
    corr = _numeric(df, cols)[cols].corr(method="spearman")
    labels = [QUALITY_METRICS.get(c, c.replace("_percent", "").replace("_", " ")) for c in cols]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman correlation (all cases)")
    ax.set_title("Metric inter-correlation across all generated scripts")
    _save(fig, destination)


def pareto_plot(df: pd.DataFrame, destination: Path) -> None:
    x_col = "generation_plus_evaluation_cost_usd"
    y_col = "reference_evidence_support_percent"
    if not {x_col, y_col, "latency_seconds"}.issubset(df):
        return
    m = _numeric(df, [x_col, y_col, "latency_seconds"]).groupby("system_id")[
        [x_col, y_col, "latency_seconds"]].mean().reindex(_systems(df)).dropna()
    if m.empty:
        return
    # Pareto front: no other system is both cheaper and better supported.
    front = [s for s in m.index if not any(
        (m.loc[o, x_col] <= m.loc[s, x_col] and m.loc[o, y_col] >= m.loc[s, y_col]
         and (m.loc[o, x_col] < m.loc[s, x_col] or m.loc[o, y_col] > m.loc[s, y_col]))
        for o in m.index if o != s)]
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = 150 + 900 * m["latency_seconds"] / max(m["latency_seconds"].max(), 1e-9)
    ax.scatter(m[x_col], m[y_col], s=sizes, alpha=0.6,
               c=["#2a9d8f" if s in front else "#999999" for s in m.index])
    for s, r in m.iterrows():
        ax.annotate(s, (r[x_col], r[y_col]), ha="center", va="center", fontweight="bold")
    fr = m.loc[front].sort_values(x_col)
    ax.plot(fr[x_col], fr[y_col], "--", color="#2a9d8f", label="Pareto front")
    ax.set_xlabel("Mean generation + evaluation cost per script (USD)")
    ax.set_ylabel("Mean reference evidence support (%)")
    ax.set_title("Cost-quality Pareto view (bubble area = mean latency)")
    ax.grid(alpha=0.2)
    ax.legend()
    _save(fig, destination)


def case_level_scatter(df: pd.DataFrame, destination: Path) -> None:
    x_col, y_col = "context_recall_percent", "reference_evidence_support_percent"
    if not {x_col, y_col}.issubset(df):
        return
    f = _numeric(df, [x_col, y_col]).dropna(subset=[x_col, y_col])
    if len(f) < 3:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for system in _systems(df):
        g = f[f.system_id == system]
        ax.scatter(g[x_col], g[y_col], label=system, alpha=0.75,
                   s=70 if system == "P1" else 40)
    if f[x_col].nunique() > 1:
        k, b = np.polyfit(f[x_col], f[y_col], 1)
        xs = np.linspace(f[x_col].min(), f[x_col].max(), 50)
        ax.plot(xs, k * xs + b, "k--", linewidth=1, label="Linear trend (all cases)")
    ax.set_xlabel("Context recall (%)")
    ax.set_ylabel("Reference evidence support (%)")
    ax.set_title("Does better retrieval translate to better-supported scripts?")
    ax.grid(alpha=0.2)
    ax.legend()
    _save(fig, destination)


def create_extra_figures(df: pd.DataFrame, exp_root: Path) -> list[Path]:
    figures = exp_root / "figures"
    analysis = exp_root / "analysis"
    outputs = []

    plan = [
        ("Figure_09_evidence_support_by_question.png", lambda dst: quality_heatmap(
            df, "reference_evidence_support_percent", dst)),
        ("Figure_10_coverage_paired_differences.png", lambda dst: _save_pairs(
            df, "query_coverage_percent", dst, analysis)),
        ("Figure_11_support_paired_differences.png", lambda dst: _save_pairs(
            df, "reference_evidence_support_percent", dst, analysis)),
        ("Figure_12_claim_label_proportions.png", lambda dst: _save_claims(df, dst, analysis)),
        ("Figure_13_evidence_support_vs_cost.png", lambda dst: tradeoff_scatter(
            df, "estimated_llm_cost_usd", "reference_evidence_support_percent",
            "Mean estimated generation cost per script (USD)",
            "Mean reference evidence support (%)", dst)),
        ("Figure_14_citation_grounding.png", lambda dst: citation_scores(df, dst)),
        ("Figure_15_coverage_distributions.png", lambda dst: quality_distributions(
            df, "query_coverage_percent", dst)),
        ("Figure_16_evidence_support_vs_latency.png", lambda dst: tradeoff_scatter(
            df, "latency_seconds", "reference_evidence_support_percent",
            "Mean generation latency per script (seconds)",
            "Mean reference evidence support (%)", dst)),
        ("Figure_17_generation_time_by_task.png", lambda dst: call_time_by_task(
            analysis / "llm_call_records.csv", dst)),
        ("Figure_18_evidence_support_vs_total_model_cost.png", lambda dst: tradeoff_scatter(
            df, "generation_plus_evaluation_cost_usd", "reference_evidence_support_percent",
            "Mean estimated generation + evaluation cost per script (USD)",
            "Mean reference evidence support (%)", dst)),
        ("Figure_19_narration_dimensions.png", lambda dst: narration_dimension_heatmap(df, dst)),
        ("Figure_20_aspect_coverage_shares.png", lambda dst: aspect_status_share_plot(df, dst)),
        ("Figure_25_quality_radar.png", lambda dst: radar_profile(df, dst)),
        ("Figure_26_mean_bootstrap_ci.png", lambda dst: _save_table(
            mean_ci_plot(df, dst), "mean_bootstrap_ci95.csv", dst, analysis)),
        ("Figure_27_support_by_domain.png", lambda dst: group_breakdown_heatmap(
            df, "domain", "reference_evidence_support_percent", dst)),
        ("Figure_28_support_by_reasoning_type.png", lambda dst: group_breakdown_heatmap(
            df, "reasoning_type", "reference_evidence_support_percent", dst)),
        ("Figure_29_token_usage.png", lambda dst: token_usage_plot(df, dst)),
        ("Figure_30_win_tie_loss.png", lambda dst: _save_table(
            win_tie_loss_plot(df, dst), "p1_win_tie_loss.csv", dst, analysis)),
        ("Figure_31_metric_correlations.png", lambda dst: correlation_heatmap(df, dst)),
        ("Figure_32_cost_quality_pareto.png", lambda dst: pareto_plot(df, dst)),
        ("Figure_33_retrieval_vs_support.png", lambda dst: case_level_scatter(df, dst)),
    ]
    for name, action in plan:
        path = figures / name
        action(path)
        if path.exists():
            outputs.append(path)
    return outputs


_LONG_PATH_THRESHOLD = 240  # Windows MAX_PATH is 260 including the drive and terminator.
_EXTENDED_PREFIX = "\\\\?\\"  # the characters  \\?\
_UNC_PREFIX = "\\\\"          # the characters  \\


def _io_path(path: Path) -> Path:
    """Return ``path`` unchanged unless it is long enough to hit the Windows MAX_PATH limit.

    Long absolute paths get the extended-length prefix so deep project folders (e.g. inside
    Downloads) can still be written. Short paths stay ordinary ``Path`` objects.
    """
    if os.name != "nt":
        return path
    absolute = os.path.abspath(str(path))
    if len(absolute) < _LONG_PATH_THRESHOLD or absolute.startswith(_EXTENDED_PREFIX):
        return path
    if absolute.startswith(_UNC_PREFIX):
        return Path(_EXTENDED_PREFIX + "UNC" + absolute[1:])
    return Path(_EXTENDED_PREFIX + absolute)


def _write_analysis_csv(
    frame: pd.DataFrame,
    destination: Path,
    *,
    fallback_directory: Path,
    **kwargs,
) -> None:
    """Write a CSV, retrying once after recreating its folder, then falling back."""
    _io_path(destination.parent).mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(_io_path(destination), **kwargs)
    except FileNotFoundError:
        # The folder can vanish between mkdir and write (e.g. cleanup/sync tools).
        _io_path(destination.parent).mkdir(parents=True, exist_ok=True)
        try:
            frame.to_csv(_io_path(destination), **kwargs)
        except FileNotFoundError:
            fallback = fallback_directory / destination.name
            _io_path(fallback_directory).mkdir(parents=True, exist_ok=True)
            warnings.warn(
                f"Could not write {destination}; saving the analysis CSV to {fallback}.",
                RuntimeWarning,
                stacklevel=2,
            )
            frame.to_csv(_io_path(fallback), **kwargs)


def _save_table(table, name: str, destination: Path, analysis: Path) -> None:
    if table is not None and not table.empty:
        _write_analysis_csv(table, analysis / name,
                            fallback_directory=destination.parent, index=False)


def _save_pairs(df: pd.DataFrame, metric: str, destination: Path, analysis: Path) -> None:
    paired = paired_difference_plot(df, metric, destination)
    if not paired.empty:
        _write_analysis_csv(
            paired,
            analysis / f"{metric}_p1_per_question_differences.csv",
            fallback_directory=destination.parent,
            index=False,
        )


def _save_claims(df: pd.DataFrame, destination: Path, analysis: Path) -> None:
    shares = claim_label_shares(df, destination)
    if not shares.empty:
        _write_analysis_csv(
            shares,
            analysis / "reference_claim_status_shares_percent.csv",
            fallback_directory=destination.parent,
        )
