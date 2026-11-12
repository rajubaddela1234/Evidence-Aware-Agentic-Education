"""Report the five additional automated research metrics, when data exists."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _metric(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in df.columns:
        return pd.DataFrame()
    part = df[["system_id", name]].copy()
    part[name] = pd.to_numeric(part[name], errors="coerce")
    return part.dropna(subset=[name])


def _save(fig, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + ".png")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_added_metrics(df: pd.DataFrame, folder: Path) -> list[Path]:
    saved = []
    grounded = _metric(df, "grounded_requirement_coverage_percent")
    coverage = _metric(df, "query_coverage_percent")
    if not grounded.empty and not coverage.empty:
        table = pd.concat([
            coverage.groupby("system_id")["query_coverage_percent"].mean(),
            grounded.groupby("system_id")["grounded_requirement_coverage_percent"].mean(),
        ], axis=1).dropna()
        if not table.empty:
            ax = table.plot(kind="bar", figsize=(10, 5.5), rot=0)
            ax.set_title("Question coverage vs evidence-grounded coverage")
            ax.set_ylabel("Mean percentage of required aspects")
            ax.set_ylim(0, 100)
            ax.legend(["Explained", "Explained and supported"])
            saved.append(_save(ax.figure, folder, "Figure_21_grounded_requirements"))

    precision = _metric(df, "context_precision_at_k")
    recall = _metric(df, "context_recall_percent")
    if not precision.empty and not recall.empty:
        table = pd.concat([
            precision.groupby("system_id")["context_precision_at_k"].mean() * 100,
            recall.groupby("system_id")["context_recall_percent"].mean(),
        ], axis=1).dropna()
        if not table.empty:
            ax = table.plot(kind="bar", figsize=(10, 5.5), rot=0)
            ax.set_title("Relevance and completeness of retrieved context")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Claude-assessed percent")
            ax.legend(["Context precision at 5", "Aspect-based context recall"])
            saved.append(_save(ax.figure, folder, "Figure_22_context_precision_recall"))

    repair = _metric(df, "repair_effectiveness_points")
    if not repair.empty:
        means = repair.groupby("system_id")["repair_effectiveness_points"].mean()
        fig, ax = plt.subplots(figsize=(9, 5))
        means.plot(kind="bar", ax=ax, rot=0)
        ax.axhline(0, color="black", linewidth=.8, linestyle="--")
        ax.set_title("Mean coverage/narration change after actual revisions")
        ax.set_ylabel("Percentage points (positive = improvement)")
        saved.append(_save(fig, folder, "Figure_23_repair_effectiveness"))

    preservation = _metric(df, "requirement_preservation_rate")
    if not preservation.empty:
        means = preservation.groupby("system_id")["requirement_preservation_rate"].mean() * 100
        fig, ax = plt.subplots(figsize=(9, 5))
        means.plot(kind="bar", ax=ax, rot=0)
        ax.set_title("Previously covered requirements retained without regression")
        ax.set_ylabel("Mean preservation rate (%)")
        ax.set_ylim(0, 100)
        saved.append(_save(fig, folder, "Figure_24_requirement_preservation"))
    return saved
