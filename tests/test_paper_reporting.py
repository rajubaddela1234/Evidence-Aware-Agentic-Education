from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_paper_results import _save_pairs, claim_label_shares, paired_difference_plot

from evidence_aware_edu.experiments.analysis import _validated_records
from evidence_aware_edu.experiments.identity import sha256_text


def test_figures_pair_the_same_questions_and_save_real_plot_files(tmp_path):
    df = pd.DataFrame([
        {"system_id": "B1", "prompt_id": "FQ01", "query_coverage_percent": 40},
        {"system_id": "B1", "prompt_id": "FQ02", "query_coverage_percent": 60},
        {"system_id": "P1", "prompt_id": "FQ01", "query_coverage_percent": 70},
        {"system_id": "P1", "prompt_id": "FQ02", "query_coverage_percent": 65},
    ])
    output = tmp_path / "paired.png"
    paired = paired_difference_plot(df, "query_coverage_percent", output)
    assert paired.sort_values("prompt_id")["p1_minus_baseline"].tolist() == [30, 5]
    assert output.exists()
    assert not output.with_suffix(".pdf").exists()


def test_paired_analysis_csv_recreates_directory_if_it_disappears(
    tmp_path, monkeypatch
):
    df = pd.DataFrame([
        {"system_id": "B1", "prompt_id": "FQ01", "query_coverage_percent": 40},
        {"system_id": "P1", "prompt_id": "FQ01", "query_coverage_percent": 70},
    ])
    analysis = tmp_path / "analysis"
    original_to_csv = pd.DataFrame.to_csv
    attempts = 0

    def remove_directory_once(frame, destination, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            analysis.rmdir()
            raise FileNotFoundError(destination)
        return original_to_csv(frame, destination, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", remove_directory_once)
    _save_pairs(df, "query_coverage_percent", tmp_path / "figures" / "paired.png", analysis)

    assert attempts == 2
    assert (analysis / "query_coverage_percent_p1_per_question_differences.csv").exists()


def test_paired_analysis_csv_falls_back_to_figure_directory(tmp_path, monkeypatch):
    df = pd.DataFrame([
        {"system_id": "B1", "prompt_id": "FQ01", "query_coverage_percent": 40},
        {"system_id": "P1", "prompt_id": "FQ01", "query_coverage_percent": 70},
    ])
    analysis = tmp_path / "analysis"
    figures = tmp_path / "figures"
    original_to_csv = pd.DataFrame.to_csv

    def fail_analysis_path(frame, destination, **kwargs):
        # Compare folder names: on Windows a very long path may arrive with the \\?\ prefix.
        if Path(destination).parent.name == analysis.name:
            raise FileNotFoundError(destination)
        return original_to_csv(frame, destination, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_analysis_path)
    with pytest.warns(RuntimeWarning, match="saving the analysis CSV"):
        _save_pairs(df, "query_coverage_percent", figures / "paired.png", analysis)

    assert (figures / "query_coverage_percent_p1_per_question_differences.csv").exists()


def test_claim_shares_normalize_across_unequal_script_lengths(tmp_path):
    frame = pd.DataFrame([
        {"system_id": "B1", "supported_claims": 2, "partial_claims": 0,
         "unsupported_claims": 2, "contradicted_claims": 0},
        {"system_id": "P1", "supported_claims": 8, "partial_claims": 1,
         "unsupported_claims": 1, "contradicted_claims": 0},
    ])
    result = claim_label_shares(frame, tmp_path / "shares.png")
    assert round(result.loc["B1", "supported_claims"], 1) == 50.0
    assert round(result.loc["P1", "supported_claims"], 1) == 80.0
    assert (result.sum(axis=1).round(6) == 100).all()


def test_analysis_keeps_evaluation_cost_separate_from_generation(tmp_path):
    raw_dir = tmp_path / "raw"
    eval_dir = tmp_path / "evaluations"
    (raw_dir / "B1").mkdir(parents=True)
    (eval_dir / "B1").mkdir(parents=True)
    script = "A short educational script."
    (raw_dir / "B1" / "FQ01.json").write_text(json.dumps({
        "script": script, "script_sha256": sha256_text(script), "experiment_id": "EXP_TEST",
        "reference_pool_sha256": "POOL_TEST",
    }), encoding="utf-8")
    (eval_dir / "B1" / "FQ01.json").write_text(json.dumps({
        "record": {"system_id": "B1", "prompt_id": "FQ01",
                   "experiment_id": "EXP_TEST", "reference_pool_sha256": "POOL_TEST",
                   "evaluated_script_sha256": sha256_text(script),
                   "estimated_llm_cost_usd": 0.10, "latency_seconds": 6},
        "details": {"evaluation_estimated_llm_cost_usd": 0.03,
                    "evaluation_telemetry": {"wall_seconds": 4}},
    }), encoding="utf-8")

    records = _validated_records(eval_dir, raw_dir, None, strict=False)
    assert len(records) == 1
    assert round(records[0]["generation_plus_evaluation_cost_usd"], 3) == 0.13
    assert records[0]["generation_plus_evaluation_latency_seconds"] == 10
