from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from evidence_aware_edu.experiments.analysis import (
    NARRATION_DIMENSIONS,
    _quality_detail_tables,
    _validated_records,
)
from evidence_aware_edu.experiments.identity import sha256_text


def _write_case(root: Path, system: str, prompt: str, quality: dict) -> None:
    raw = root / "raw" / system / f"{prompt}.json"
    evaluated = root / "evaluations" / system / f"{prompt}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    evaluated.parent.mkdir(parents=True, exist_ok=True)
    script = "An educational explanation with a cited example."
    digest = sha256_text(script)
    raw.write_text(json.dumps({
        "script": script,
        "script_sha256": digest,
        "experiment_id": "EXP_TEST",
        "reference_pool_sha256": "POOL",
    }), encoding="utf-8")
    evaluated.write_text(json.dumps({
        "record": {
            "system_id": system, "prompt_id": prompt,
            "experiment_id": "EXP_TEST", "reference_pool_sha256": "POOL",
            "evaluated_script_sha256": digest,
            "query_coverage_percent": 50.0,
            "reference_evidence_support_percent": 75.0,
            "citation_f1": 0.75,
            "narration_quality_percent": 75.0,
        },
        "details": {
            "quality": quality,
            "evaluation_estimated_llm_cost_usd": 0.05,
            "evaluation_telemetry": {"wall_seconds": 3.0},
        },
    }), encoding="utf-8")


def test_automated_granular_reporting_preserves_original_metrics(tmp_path):
    quality = {
        "aspect_scores": [
            {"aspect": "Concept A", "score": 1.0, "rationale": "Explained"},
            {"aspect": "Concept B", "score": 0.5, "rationale": "Needs detail"},
            {"aspect": "Concept C", "score": 0.0, "rationale": "Absent"},
        ],
        **{name: (4 if name != "reflective_ending_score" else 2)
           for name in NARRATION_DIMENSIONS},
    }
    _write_case(tmp_path, "B1", "FQ01", quality)
    records = _validated_records(tmp_path / "evaluations", tmp_path / "raw", None, False)
    assert len(records) == 1
    record = records[0]
    assert record["query_coverage_percent"] == 50.0
    assert record["reference_evidence_support_percent"] == 75.0
    assert record["citation_f1"] == 0.75
    assert record["narration_quality_percent"] == 75.0
    assert record["coverage_aspects_covered"] == 1
    assert record["coverage_aspects_partial"] == 1
    assert record["coverage_aspects_missing"] == 1
    assert record["narration_reflective_ending_percent"] == 50.0
    assert record["narration_opening_orientation_percent"] == 100.0
    assert record["evaluation_estimated_llm_cost_usd"] == 0.05

    aspect_table, narration_table = _quality_detail_tables(tmp_path / "evaluations", records)
    assert len(aspect_table) == 3
    assert aspect_table.status.tolist() == ["covered", "partial", "missing"]
    assert len(narration_table) == 10
    assert set(narration_table.dimension) == {
        name.removesuffix("_score") for name in NARRATION_DIMENSIONS
    }


def test_automated_quality_figures_use_only_existing_scores(tmp_path):
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(script_dir))
    from plot_paper_results import aspect_status_share_plot, narration_dimension_heatmap

    rows = []
    for system in ("B1", "P1"):
        row = {
            "system_id": system,
            "coverage_aspects_covered": 3 if system == "B1" else 5,
            "coverage_aspects_partial": 2 if system == "B1" else 1,
            "coverage_aspects_missing": 1 if system == "B1" else 0,
        }
        row.update({
            f"narration_{name[:-6]}_percent": 75 if system == "B1" else 100
            for name in NARRATION_DIMENSIONS
        })
        rows.append(row)
    frame = pd.DataFrame(rows)
    narration_destination = tmp_path / "narration.png"
    aspect_destination = tmp_path / "aspects.png"
    narration_dimension_heatmap(frame, narration_destination)
    aspect_status_share_plot(frame, aspect_destination)
    for path in (narration_destination, aspect_destination):
        assert path.exists()
        assert not path.with_suffix(".pdf").exists()


def _load_final_study():
    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("final_study_test", project / "scripts" / "run_final_study.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_study_prepares_missing_pools_before_the_60_cases(monkeypatch, tmp_path):
    study = _load_final_study()
    calls: list[str] = []
    checks = iter([["FQ01: missing"], []])
    monkeypatch.setattr(study, "get_settings", lambda: SimpleNamespace(output_dir=tmp_path))
    monkeypatch.setattr(study, "load_benchmark", lambda _: [
        SimpleNamespace(prompt_id=f"FQ{i:02d}") for i in range(1, 11)
    ])
    monkeypatch.setattr(study, "_check_pools", lambda *_: next(checks))
    monkeypatch.setattr(study, "make_arm_experiment_id", lambda _: "EXP_TEST")

    def run_script(script, *args):
        calls.append(script)
        if script == "run_benchmark.py":
            path = tmp_path / "experiments" / "EXP_TEST" / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"status":"complete"}', encoding="utf-8")

    monkeypatch.setattr(study, "_run_script", run_script)
    monkeypatch.setattr(sys, "argv", ["run_final_study.py"])
    study.main()
    assert calls == [
        "build_reference_pool.py", "run_benchmark.py", "analyze_results.py",
        "generate_research_figures.py",
    ]


def test_final_study_does_not_rebuild_pools_during_resume(monkeypatch, tmp_path):
    study = _load_final_study()
    manifest = tmp_path / "experiments" / "EXP_OLD" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"status":"running"}', encoding="utf-8")
    monkeypatch.setattr(study, "get_settings", lambda: SimpleNamespace(output_dir=tmp_path))
    monkeypatch.setattr(study, "load_benchmark", lambda _: [
        SimpleNamespace(prompt_id=f"FQ{i:02d}") for i in range(1, 11)
    ])
    monkeypatch.setattr(study, "_check_pools", lambda *_: ["FQ01: incompatible"])
    monkeypatch.setattr(study, "_run_script", lambda *_: pytest.fail("Must not rebuild a resumed pool"))
    monkeypatch.setattr(sys, "argv", ["run_final_study.py", "--experiment-id", "EXP_OLD"])
    with pytest.raises(SystemExit, match="Do not rebuild"):
        study.main()


def test_final_study_auto_resumes_latest_incomplete_experiment(monkeypatch, tmp_path):
    study = _load_final_study()
    experiment_id = "EXP_GEMINI_INCOMPLETE"
    experiment_root = tmp_path / "experiments" / experiment_id
    experiment_root.mkdir(parents=True)
    (tmp_path / "latest_gemini_experiment.txt").write_text(experiment_id, encoding="utf-8")
    manifest_path = experiment_root / "manifest.json"
    manifest_path.write_text(json.dumps({
        "status": "incomplete",
        "benchmark_sha256": "benchmark-hash",
        "systems": study.SYSTEMS.split(","),
        "prompt_ids": [f"FQ{i:02d}" for i in range(1, 11)],
    }), encoding="utf-8")

    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        study,
        "get_settings",
        lambda: SimpleNamespace(output_dir=tmp_path, generation_provider="gemini"),
    )
    monkeypatch.setattr(study, "sha256_file", lambda _: "benchmark-hash")
    monkeypatch.setattr(study, "load_benchmark", lambda _: [
        SimpleNamespace(prompt_id=f"FQ{i:02d}") for i in range(1, 11)
    ])
    monkeypatch.setattr(study, "_check_pools", lambda *_: [])

    def run_script(script, *args):
        calls.append((script, args))
        if script == "run_benchmark.py":
            manifest_path.write_text('{"status":"complete"}', encoding="utf-8")

    monkeypatch.setattr(study, "_run_script", run_script)
    monkeypatch.setattr(sys, "argv", ["run_final_study.py", "--claude-access", "cli"])
    study.main()

    assert calls[0][0] == "run_benchmark.py"
    assert "--experiment-id" in calls[0][1]
    experiment_arg = calls[0][1].index("--experiment-id")
    assert calls[0][1][experiment_arg + 1] == experiment_id
    assert [script for script, _ in calls] == [
        "run_benchmark.py", "analyze_results.py", "generate_research_figures.py",
    ]


def test_final_study_prefers_completed_experiment_with_unfinished_reporting(
    monkeypatch, tmp_path
):
    study = _load_final_study()
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text("benchmark", encoding="utf-8")
    prompts = [SimpleNamespace(prompt_id=f"FQ{i:02d}") for i in range(1, 11)]
    previous_id = "EXP_GEMINI_REPORTS_PENDING"
    previous_dir = tmp_path / "experiments" / previous_id
    previous_dir.mkdir(parents=True)
    (previous_dir / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "updated_at": "2026-11-10T16:00:00+00:00",
        "benchmark_sha256": "benchmark-hash",
        "systems": study.SYSTEMS.split(","),
        "prompt_ids": [prompt.prompt_id for prompt in prompts],
    }), encoding="utf-8")

    interrupted_id = "EXP_GEMINI_INTERRUPTED"
    interrupted_dir = tmp_path / "experiments" / interrupted_id
    interrupted_dir.mkdir()
    (interrupted_dir / "manifest.json").write_text(json.dumps({
        "status": "running",
        "benchmark_sha256": "benchmark-hash",
        "systems": study.SYSTEMS.split(","),
        "prompt_ids": [prompt.prompt_id for prompt in prompts],
    }), encoding="utf-8")
    (tmp_path / "latest_gemini_experiment.txt").write_text(
        interrupted_id, encoding="utf-8"
    )

    monkeypatch.setattr(
        study,
        "get_settings",
        lambda: SimpleNamespace(output_dir=tmp_path, generation_provider="gemini"),
    )
    monkeypatch.setattr(study, "sha256_file", lambda _: "benchmark-hash")
    monkeypatch.setattr(study, "load_benchmark", lambda _: prompts)
    monkeypatch.setattr(study, "_check_pools", lambda *_: [])
    calls: list[tuple[str, tuple[str, ...]]] = []

    def run_script(script, *args):
        calls.append((script, args))

    monkeypatch.setattr(study, "_run_script", run_script)
    monkeypatch.setattr(sys, "argv", ["run_final_study.py", "--skip-figures"])
    study.main()

    run_args = calls[0][1]
    assert run_args[run_args.index("--experiment-id") + 1] == previous_id
    assert [script for script, _ in calls] == ["run_benchmark.py", "analyze_results.py"]
    updated = json.loads((previous_dir / "manifest.json").read_text(encoding="utf-8"))
    assert updated["reporting_status"] == "complete"
