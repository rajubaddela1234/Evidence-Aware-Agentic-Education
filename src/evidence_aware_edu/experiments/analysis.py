from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evidence_aware_edu.evaluation.statistics import (
    compare_p1_to_baselines,
    pairwise_significance,
    system_means_with_ci,
)
from evidence_aware_edu.experiments.identity import sha256_text


# The evaluator already saves these ten 0-4 scores for each script. Report them
# separately, so the overall narration average does not hide weak dimensions.
NARRATION_DIMENSIONS = (
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
)


def _quality_breakdown(record: dict, details: dict) -> None:
    quality = details.get("quality", {})
    if not isinstance(quality, dict):
        return

    for name in NARRATION_DIMENSIONS:
        score = quality.get(name)
        if score is None:
            continue
        value = float(score)
        if not 0 <= value <= 4:
            raise ValueError(f"Invalid {name} in saved evaluation: {value}")
        record[f"narration_{name[:-6]}_percent"] = value * 25.0

    aspects = quality.get("aspect_scores", [])
    if not aspects:
        return
    scores = [float(item["score"]) for item in aspects]
    if not all(0 <= score <= 1 for score in scores):
        raise ValueError("Invalid aspect coverage score in saved evaluation")
    count = len(scores)
    record["coverage_aspect_count"] = count
    record["coverage_aspects_covered"] = sum(score >= 0.8 for score in scores)
    record["coverage_aspects_partial"] = sum(0 < score < 0.8 for score in scores)
    record["coverage_aspects_missing"] = sum(score <= 0 for score in scores)
    record["coverage_aspects_covered_percent"] = 100.0 * record["coverage_aspects_covered"] / count
    record["coverage_aspects_partial_percent"] = 100.0 * record["coverage_aspects_partial"] / count
    record["coverage_aspects_missing_percent"] = 100.0 * record["coverage_aspects_missing"] / count


def _quality_detail_tables(eval_dir: Path, validated_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Read only cases that passed the experiment and script-hash checks.
    aspects: list[dict] = []
    narration: list[dict] = []
    for record in validated_rows:
        system_id, prompt_id = record["system_id"], record["prompt_id"]
        payload = json.loads((eval_dir / system_id / f"{prompt_id}.json").read_text(encoding="utf-8"))
        quality = payload.get("details", {}).get("quality", {})
        if not isinstance(quality, dict):
            continue
        case = {"system_id": system_id, "prompt_id": prompt_id}
        for index, item in enumerate(quality.get("aspect_scores", []), start=1):
            score = float(item["score"])
            if not 0 <= score <= 1:
                raise ValueError(f"Invalid aspect score for {system_id}/{prompt_id}")
            aspects.append({
                **case,
                "aspect_index": index,
                "expected_aspect": item.get("aspect", ""),
                "coverage_score": score,
                "coverage_percent": score * 100.0,
                "status": "covered" if score >= 0.8 else "missing" if score == 0 else "partial",
                "evaluator_rationale": item.get("rationale", ""),
            })
        for name in NARRATION_DIMENSIONS:
            score = quality.get(name)
            if score is None:
                continue
            value = float(score)
            if not 0 <= value <= 4:
                raise ValueError(f"Invalid narration score for {system_id}/{prompt_id}")
            narration.append({
                **case,
                "dimension": name.removesuffix("_score"),
                "score_0_to_4": value,
                "score_percent": value * 25.0,
            })
    return pd.DataFrame(aspects, columns=[
        "system_id", "prompt_id", "aspect_index", "expected_aspect", "coverage_score",
        "coverage_percent", "status", "evaluator_rationale",
    ]), pd.DataFrame(narration, columns=[
        "system_id", "prompt_id", "dimension", "score_0_to_4", "score_percent",
    ])


def _percentile_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "estimated_llm_cost_usd",
        "evaluation_estimated_llm_cost_usd",
        "generation_plus_evaluation_cost_usd",
        "latency_seconds",
        "evaluation_latency_seconds",
        "generation_plus_evaluation_latency_seconds",
        "prompt_tokens",
        "output_tokens",
        "thinking_tokens",
        "billable_output_tokens",
        "llm_calls",
        "retrieval_calls",
        "llm_call_latency_p50_seconds",
        "llm_call_latency_p90_seconds",
    ]
    rows: list[dict] = []
    for system_id, group in df.groupby("system_id"):
        row: dict = {"system_id": system_id, "n_runs": int(len(group))}
        for metric in [m for m in metrics if m in df.columns]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_p50"] = float(values.quantile(0.50))
            row[f"{metric}_p90"] = float(values.quantile(0.90))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        for cost_column in (
            "estimated_llm_cost_usd", "evaluation_estimated_llm_cost_usd",
            "generation_plus_evaluation_cost_usd",
        ):
            if cost_column in group.columns:
                values = pd.to_numeric(group[cost_column], errors="coerce")
                if values.notna().all():
                    row[f"{cost_column}_total"] = float(values.sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("system_id")


def _llm_call_records(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(raw_dir.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for idx, call in enumerate(payload.get("telemetry", {}).get("llm_calls", []), start=1):
            rows.append(
                {
                    "system_id": payload.get("system_id", path.parent.name),
                    "prompt_id": payload.get("prompt_id", path.stem),
                    "call_index": idx,
                    "task": call.get("task", ""),
                    "model": call.get("model", ""),
                    "latency_seconds": call.get("latency_seconds", 0.0),
                    "prompt_tokens": call.get("prompt_tokens", 0),
                    "output_tokens": call.get("output_tokens", 0),
                    "thinking_tokens": call.get("thinking_tokens", 0),
                    "billable_output_tokens": call.get("output_tokens", 0)
                    + call.get("thinking_tokens", 0),
                    "total_tokens": call.get("total_tokens", 0),
                    "input_cost_usd": call.get("input_cost_usd", 0.0),
                    "output_cost_usd": call.get("output_cost_usd", 0.0),
                    "total_cost_usd": call.get("total_cost_usd", 0.0),
                    "attempts": call.get("attempts", 1),
                }
            )
    return pd.DataFrame(rows)


def _llm_call_percentiles(calls: pd.DataFrame) -> pd.DataFrame:
    if calls.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (system_id, task), group in calls.groupby(["system_id", "task"], dropna=False):
        row = {"system_id": system_id, "task": task, "n_calls": int(len(group))}
        for metric in [
            "latency_seconds",
            "total_cost_usd",
            "prompt_tokens",
            "output_tokens",
            "thinking_tokens",
            "billable_output_tokens",
            "attempts",
        ]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_p50"] = float(values.quantile(0.50))
            row[f"{metric}_p90"] = float(values.quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["system_id", "task"])


def _bar_plot(df: pd.DataFrame, metric: str, output: Path) -> None:
    means = df.groupby("system_id")[metric].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    means.plot(kind="bar", ax=ax)
    ax.set_xlabel("System")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Mean {metric.replace('_', ' ')} across {df['prompt_id'].nunique()} questions")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _validated_records(
    eval_dir: Path,
    raw_dir: Path,
    manifest_path: Path | None,
    strict: bool,
) -> list[dict]:
    manifest = None
    complete_cases: set[str] | None = None
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete_cases = {
            key for key, meta in manifest.get("cases", {}).items() if meta.get("status") == "complete"
        }
        if strict and manifest.get("status") != "complete":
            raise ValueError(
                f"Experiment manifest is {manifest.get('status')!r}; "
                "analysis requires a complete experiment unless strict=False."
            )

    rows: list[dict] = []
    for eval_path in sorted(eval_dir.glob("*/*.json")):
        system_id = eval_path.parent.name
        prompt_id = eval_path.stem
        key = f"{system_id}/{prompt_id}"
        if complete_cases is not None and key not in complete_cases:
            continue
        raw_path = raw_dir / system_id / f"{prompt_id}.json"
        if not raw_path.exists():
            raise ValueError(f"Missing raw output for evaluation {eval_path}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        record = ev["record"]
        actual_hash = sha256_text(raw.get("script", ""))
        if raw.get("script_sha256") and raw.get("script_sha256") != actual_hash:
            raise ValueError(f"Raw script hash mismatch for {key}")
        evaluated_hash = record.get("evaluated_script_sha256", "")
        if evaluated_hash and evaluated_hash != actual_hash:
            raise ValueError(f"Evaluation/script hash mismatch for {key}")
        if raw.get("reference_pool_sha256") != record.get("reference_pool_sha256"):
            raise ValueError(f"Reference-pool hash mismatch for {key}")
        if raw.get("experiment_id") != record.get("experiment_id"):
            raise ValueError(f"Experiment ID mismatch for {key}")

        # Evaluation is another set of model calls. Keep its overhead visible.
        details = ev.get("details", {})
        _quality_breakdown(record, details)
        eval_cost = details.get("evaluation_estimated_llm_cost_usd")
        eval_telemetry = details.get("evaluation_telemetry", {})
        if eval_cost is not None:
            record["evaluation_estimated_llm_cost_usd"] = float(eval_cost)
            generation_cost = record.get("estimated_llm_cost_usd")
            if generation_cost is not None:
                record["generation_plus_evaluation_cost_usd"] = (
                    float(generation_cost) + float(eval_cost)
                )
        if eval_telemetry.get("wall_seconds") is not None:
            record["evaluation_latency_seconds"] = float(eval_telemetry["wall_seconds"])
            generation_time = record.get("latency_seconds")
            if generation_time is not None:
                record["generation_plus_evaluation_latency_seconds"] = (
                    float(generation_time) + float(eval_telemetry["wall_seconds"])
                )
        rows.append(record)

    if strict and manifest is not None:
        expected = int(manifest.get("expected_primary_runs", 0))
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} complete validated cases, found {len(rows)}.")
    return rows


def analyze(
    eval_dir: Path,
    output_dir: Path,
    *,
    raw_dir: Path | None = None,
    manifest_path: Path | None = None,
    strict: bool = True,
) -> dict[str, Path]:
    files = sorted(eval_dir.glob("*/*.json"))
    if not files:
        raise FileNotFoundError(f"No evaluation JSON files found in {eval_dir}")

    raw_dir = raw_dir or (eval_dir.parent / "raw")
    rows = _validated_records(eval_dir, raw_dir, manifest_path, strict)
    if not rows:
        raise ValueError("No validated evaluation records are available for analysis.")
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export the already recorded Claude aspect/narration detail. No new judge calls.
    aspect_scores, narration_scores = _quality_detail_tables(eval_dir, rows)
    aspects_path = output_dir / "aspect_coverage_by_case.csv"
    narration_path = output_dir / "narration_dimensions_by_case.csv"
    aspect_scores.to_csv(aspects_path, index=False)
    narration_scores.to_csv(narration_path, index=False)

    raw_csv = output_dir / "evaluation_records.csv"
    df.to_csv(raw_csv, index=False)

    numeric = [
        "query_coverage_percent",
        "reference_evidence_support_percent",
        "query_relevance_percent",
        "narration_quality_percent",
        "claim_count",
        "supported_claims",
        "partial_claims",
        "unsupported_claims",
        "contradicted_claims",
        "supported_claim_rate",
        "citation_precision",
        "citation_recall",
        "citation_f1",
        "grounded_requirement_coverage_percent",
        "context_precision_at_k",
        "context_recall_percent",
        "repair_effectiveness_points",
        "requirement_preservation_rate",
        "llm_calls",
        "prompt_tokens",
        "output_tokens",
        "thinking_tokens",
        "billable_output_tokens",
        "retrieval_calls",
        "latency_seconds",
        "estimated_llm_cost_usd",
        "evaluation_estimated_llm_cost_usd",
        "generation_plus_evaluation_cost_usd",
        "evaluation_latency_seconds",
        "generation_plus_evaluation_latency_seconds",
        "llm_call_latency_p50_seconds",
        "llm_call_latency_p90_seconds",
        "source_count",
        "source_type_count",
        "coverage_aspect_count",
        "coverage_aspects_covered",
        "coverage_aspects_partial",
        "coverage_aspects_missing",
        "coverage_aspects_covered_percent",
        "coverage_aspects_partial_percent",
        "coverage_aspects_missing_percent",
        *(f"narration_{name[:-6]}_percent" for name in NARRATION_DIMENSIONS),
        "word_count",
    ]
    numeric = [c for c in numeric if c in df.columns]

    summary = df.groupby("system_id")[numeric].agg(["mean", "median", "std"])
    summary_path = output_dir / "system_summary.csv"
    summary.to_csv(summary_path)

    per_question = df[[
        c
        for c in ["prompt_id", "domain", "reasoning_type", "system_id", *numeric]
        if c in df.columns
    ]].sort_values(["prompt_id", "system_id"])
    per_question_path = output_dir / "per_question_results.csv"
    per_question.to_csv(per_question_path, index=False)

    pairwise = compare_p1_to_baselines(df)
    pairwise_path = output_dir / "p1_pairwise_descriptive.csv"
    pairwise.to_csv(pairwise_path, index=False)

    means_ci = system_means_with_ci(df)
    means_ci_path = output_dir / "system_means_ci95.csv"
    means_ci.to_csv(means_ci_path, index=False)

    significance = pairwise_significance(df)
    significance_path = output_dir / "pairwise_wilcoxon_holm.csv"
    significance.to_csv(significance_path, index=False)

    percentiles = _percentile_table(df)
    percentiles_path = output_dir / "system_cost_latency_percentiles.csv"
    percentiles.to_csv(percentiles_path, index=False)

    calls = _llm_call_records(raw_dir)
    calls_path = output_dir / "llm_call_records.csv"
    calls.to_csv(calls_path, index=False)

    call_percentiles = _llm_call_percentiles(calls)
    call_percentiles_path = output_dir / "llm_call_percentiles_by_task.csv"
    call_percentiles.to_csv(call_percentiles_path, index=False)

    for metric in [
        "query_coverage_percent",
        "reference_evidence_support_percent",
        "query_relevance_percent",
        "narration_quality_percent",
        "estimated_llm_cost_usd",
        "latency_seconds",
    ]:
        if metric in df.columns:
            _bar_plot(df, metric, output_dir / f"{metric}.png")

    return {
        "evaluation_records": raw_csv,
        "aspect_coverage_by_case": aspects_path,
        "narration_dimensions_by_case": narration_path,
        "system_summary": summary_path,
        "per_question": per_question_path,
        "pairwise": pairwise_path,
        "system_means_ci95": means_ci_path,
        "pairwise_significance": significance_path,
        "system_percentiles": percentiles_path,
        "llm_calls": calls_path,
        "llm_call_percentiles": call_percentiles_path,
    }

