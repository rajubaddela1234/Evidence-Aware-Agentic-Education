from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from plot_paper_results import create_extra_figures
from architecture_figures import generate_six_architectures
from plot_added_metrics import plot_added_metrics


FIG_DIR = Path("research_assets/figures")


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, width, height, text, fontsize=10):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        fill=False,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, start, end, connectionstyle="arc3,rad=0"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=13,
            linewidth=1.2,
            connectionstyle=connectionstyle,
        )
    )


def _design_counts(df):
    """Questions, domains and systems actually present in the results (None if unknown)."""
    if df is None or df.empty:
        return None
    domains = df["domain"].nunique() if "domain" in df else None
    return df["prompt_id"].nunique(), domains, df["system_id"].nunique()


def figure_01_study_design(counts=None):
    if counts:
        n_q, n_dom, n_sys = counts
        domain_line = f"{n_dom} domains\n" if n_dom else ""
        questions_text = f"{n_q} held-out questions\n{domain_line}{n_sys} systems per question"
        scripts_text = f"{n_q * n_sys} scripts total\n{n_q} questions x {n_sys} systems"
    else:
        questions_text = "Held-out questions\n6 systems per question"
        scripts_text = "Scripts: questions x 6 systems"
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _box(ax, (0.3, 2.0), 2.3, 2.0, questions_text, 11)
    _box(ax, (3.1, 0.6), 2.7, 4.8, "Six systems\n\nB1 Direct\nB2 Web RAG\nB3 Multi-source RAG\nB4 Agentic baseline\nB5 Agentic + verifier\nP1 Evidence-aware agentic", 10)
    _box(ax, (6.4, 2.0), 2.5, 2.0, scripts_text + "\n\nGeneration telemetry\ncost + latency + calls", 10)
    _box(ax, (9.6, 1.0), 3.0, 4.0, "Independent Claude\nevaluator\n\nCoverage %\nEvidence Support %\nRelevance %\nNarration Quality %", 10)
    _arrow(ax, (2.6, 3.0), (3.1, 3.0))
    _arrow(ax, (5.8, 3.0), (6.4, 3.0))
    _arrow(ax, (8.9, 3.0), (9.6, 3.0))
    ax.set_title("Figure 1. Gemini script study and measurement boundary", fontsize=14)
    _save(fig, "Figure_01_study_design")


def figure_03_model_separation(counts=None):
    # Claude scores the saved Gemini scripts with a separate model client.
    questions_label = f"{counts[0]} questions" if counts else "Benchmark questions"
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _box(ax, (0.4, 2.0), 3.0, 2.0, f"Gemini generation\n{questions_label}\nB1-B5 and P1", 10.5)
    _box(ax, (4.6, 2.0), 3.0, 2.0, "Saved scripts\nHashed artifacts\nFrozen source corpus", 10.5)
    _box(ax, (8.7, 1.4), 3.8, 3.2, "Independent Claude evaluation\n\nCoverage and relevance\nReference evidence support\nNarration quality\nCitation grounding", 10.5)
    _arrow(ax, (3.4, 3.0), (4.6, 3.0))
    _arrow(ax, (7.6, 3.0), (8.7, 3.0))
    ax.set_title("Figure 3. Separate generation and automated evaluation", fontsize=14)
    _save(fig, "Figure_03_independent_evaluation")


def figure_04_support_not_accuracy():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _box(ax, (0.5, 2.0), 2.6, 2.0, "Frozen reference corpus\nmay contain strong, weak,\nor inaccurate sources", 11)
    _box(ax, (4.0, 2.0), 2.6, 2.0, "Claim-evidence verifier\nasks: does evidence entail\nthe claim?", 11)
    _box(ax, (7.5, 3.2), 3.8, 1.3, "Reference Evidence Support %\nmeasured in this project", 11)
    _box(ax, (7.5, 1.2), 3.8, 1.3, "Independent factual truth /\nauthoritative-source audit\nNOT measured by this score", 11)
    _arrow(ax, (3.1, 3.0), (4.0, 3.0))
    _arrow(ax, (6.6, 3.0), (7.5, 3.85))
    _arrow(ax, (6.6, 3.0), (7.5, 1.85))
    ax.set_title("Figure 4. Evidence support is not factual accuracy", fontsize=14)
    _save(fig, "Figure_04_support_not_accuracy")


def _load_results(exp_root: Path):
    candidates = [
        exp_root / "analysis" / "evaluation_records.csv",
        exp_root / "evaluation_records.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return None


def _experiment_label(exp_root: Path) -> str:
    manifest_path = exp_root / "manifest.json"
    if not manifest_path.exists():
        return exp_root.name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return exp_root.name
    provider = str(manifest.get("generation_provider", "")).strip()
    model = str(manifest.get("generation_model", "")).strip()
    if provider and model:
        return f"{provider.upper()} · {model}"
    if provider:
        return provider.upper()
    return exp_root.name

def figure_05_quality(df: pd.DataFrame):
    metrics = [
        "query_coverage_percent",
        "reference_evidence_support_percent",
        "query_relevance_percent",
        "narration_quality_percent",
    ]
    means = df.groupby("system_id")[metrics].mean().sort_index()
    x = np.arange(len(means.index))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, means[metric], width, label=metric.replace("_percent", "").replace("_", " ").title())
    ax.set_xticks(x, means.index)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent")
    ax.set_title(f"Figure 5. Mean script-quality measures across {df['prompt_id'].nunique()} questions")
    ax.legend()
    _save(fig, "Figure_05_quality_comparison")


def figure_06_claim_labels(df: pd.DataFrame):
    cols = ["supported_claims", "partial_claims", "unsupported_claims", "contradicted_claims"]
    totals = df.groupby("system_id")[cols].sum().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(totals))
    x = np.arange(len(totals))
    for col in cols:
        vals = totals[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=col.replace("_claims", "").title())
        bottom += vals
    ax.set_xticks(x, totals.index)
    ax.set_ylabel("Extracted claim count")
    ax.set_title("Figure 6. Independent evaluator evidence-support labels")
    ax.legend()
    _save(fig, "Figure_06_claim_support_labels")


def figure_07_coverage_heatmap(df: pd.DataFrame):
    pivot = df.pivot(index="prompt_id", columns="system_id", values="query_coverage_percent")
    pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Coverage %")
    ax.set_title("Figure 7. Per-question query coverage")
    _save(fig, "Figure_07_question_coverage")


def figure_08_cost_latency(df: pd.DataFrame):
    means = df.groupby("system_id")[["estimated_llm_cost_usd", "latency_seconds"]].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(means["estimated_llm_cost_usd"], means["latency_seconds"])
    for system_id, row in means.iterrows():
        ax.annotate(system_id, (row["estimated_llm_cost_usd"], row["latency_seconds"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean estimated generation cost (USD)")
    ax.set_ylabel("Mean generation latency (s)")
    ax.set_title("Figure 8. Generation cost-latency trade-off")
    _save(fig, "Figure_08_cost_latency")


def main():
    parser = argparse.ArgumentParser(
        description="Generate methodology and result figures for one benchmark experiment."
    )
    parser.add_argument(
        "--experiment-id",
        default="",
        help="Experiment ID to plot. If omitted, the latest experiment pointer is used.",
    )
    args = parser.parse_args()

    load_dotenv()
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs_final_10q"))
    experiment_id = args.experiment_id.strip()
    if not experiment_id:
        pointer = output_dir / "latest_experiment.txt"
        if not pointer.exists():
            raise SystemExit("No latest experiment pointer found. Run a benchmark first.")
        experiment_id = pointer.read_text(encoding="utf-8").strip()

    exp_root = output_dir / "experiments" / experiment_id
    if not exp_root.exists():
        raise SystemExit(f"Experiment not found: {exp_root}")

    global FIG_DIR
    FIG_DIR = exp_root / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Clear leftover Figure 02 from older versions, plus any PDF copies.
    (FIG_DIR / "Figure_02_proposed_architecture.png").unlink(missing_ok=True)
    # Older runs may have PDF copies of our figures. Keep only the PNG versions.
    for old_figure in FIG_DIR.glob("Figure_*.pdf"):
        old_figure.unlink()

    # Load the results first so the methodology figures can show the real question counts.
    df = _load_results(exp_root)
    counts = _design_counts(df)
    figure_01_study_design(counts)
    generate_six_architectures(FIG_DIR)
    figure_03_model_separation(counts)
    figure_04_support_not_accuracy()

    if df is None:
        print(f"Generated methodology figures in {FIG_DIR}")
        print(
            "No evaluation_records.csv found for this experiment; "
            "run analyze_results.py first, then rerun this command."
        )
        return

    required = {
        "system_id",
        "prompt_id",
        "query_coverage_percent",
        "reference_evidence_support_percent",
        "query_relevance_percent",
        "narration_quality_percent",
        "supported_claims",
        "partial_claims",
        "unsupported_claims",
        "contradicted_claims",
        "estimated_llm_cost_usd",
        "latency_seconds",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Cannot generate result figures; missing columns: {sorted(missing)}")

    label = _experiment_label(exp_root)
    print(f"Generating figures for {experiment_id} ({label})")
    figure_05_quality(df)
    figure_06_claim_labels(df)
    figure_07_coverage_heatmap(df)
    figure_08_cost_latency(df)
    extra = create_extra_figures(df, exp_root)
    extra.extend(plot_added_metrics(df, FIG_DIR))
    print(f"Saved experiment-specific figures to {FIG_DIR}")
    print(f"Additional measured result figures: {len(extra)}")


if __name__ == "__main__":
    main()
