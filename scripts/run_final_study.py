from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evidence_aware_edu.config import get_settings, use_benchmark_directories
from evidence_aware_edu.experiments.identity import make_arm_experiment_id, sha256_file
from evidence_aware_edu.experiments.io import load_benchmark
from evidence_aware_edu.experiments.reference_pool import reference_pool_compatibility

SYSTEMS = "B1,B2,B3,B4,B5,P1"


def _check_pools(benchmark: Path, settings) -> list[str]:
    # Use the same saved evidence for every system. Do not fetch new sources mid-study.
    missing = []
    for prompt in load_benchmark(benchmark):
        path = settings.reference_pool_dir / f"{prompt.prompt_id}.json"
        valid, reason = reference_pool_compatibility(
            path, prompt, transcript_only=settings.youtube_require_transcript
        )
        if not valid:
            missing.append(f"{prompt.prompt_id}: {reason}")

    return missing


def _compatible_experiment_manifest(settings, experiment_id: str, benchmark: Path, prompts):
    if not experiment_id or Path(experiment_id).name != experiment_id:
        return None
    manifest_path = settings.output_dir / "experiments" / experiment_id / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_prompt_ids = [prompt.prompt_id for prompt in prompts]
    if (
        manifest.get("benchmark_sha256") != sha256_file(benchmark)
        or manifest.get("systems") != SYSTEMS.split(",")
        or manifest.get("prompt_ids") != expected_prompt_ids
    ):
        return None
    return manifest


def _latest_resumable_experiment_id(settings, benchmark: Path, prompts) -> str:
    experiments_dir = settings.output_dir / "experiments"
    if experiments_dir.exists():
        unfinished_reports = []
        for experiment_dir in experiments_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
            manifest = _compatible_experiment_manifest(
                settings, experiment_dir.name, benchmark, prompts
            )
            if (
                manifest
                and manifest.get("status") == "complete"
                and manifest.get("reporting_status") != "complete"
            ):
                unfinished_reports.append(
                    (manifest.get("updated_at", ""), experiment_dir.name)
                )
        if unfinished_reports:
            return max(unfinished_reports)[1]

    provider = getattr(settings, "generation_provider", "gemini")
    pointer = settings.output_dir / f"latest_{provider}_experiment.txt"
    if not pointer.exists():
        return ""

    experiment_id = pointer.read_text(encoding="utf-8").strip()
    if not experiment_id or Path(experiment_id).name != experiment_id:
        raise SystemExit(f"Invalid experiment ID in {pointer}.")
    manifest = _compatible_experiment_manifest(settings, experiment_id, benchmark, prompts)
    if not manifest or manifest.get("status") not in {"incomplete", "running"}:
        return ""
    return experiment_id


def _set_reporting_status(manifest_path: Path, status: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reporting_status"] = status
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _run_script(script: str, *args: str) -> None:
    command = [sys.executable, str(Path(__file__).with_name(script)), *args]
    print(f"\nRunning {script}...", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen evidence, run ten questions, score with Claude and create figures."
    )
    parser.add_argument("--benchmark", default="data/final_benchmark_10.jsonl")
    parser.add_argument("--experiment-id", default="", help="Reuse this ID to resume a run.")
    parser.add_argument("--claude-access", choices=["cli", "api"], default="")
    parser.add_argument(
        "--require-existing-pools", action="store_true",
        help="Do not retrieve or build evidence pools; fail if any are missing.",
    )
    parser.add_argument(
        "--skip-figures", action="store_true", help="Keep the evaluated results without plotting."
    )
    args = parser.parse_args()
    use_benchmark_directories(args.benchmark)

    settings = get_settings()
    benchmark = Path(args.benchmark)
    prompts = load_benchmark(benchmark)
    if len(prompts) != 10:
        raise SystemExit(f"Expected ten final questions; found {len(prompts)}.")
    experiment_id = args.experiment_id.strip() or _latest_resumable_experiment_id(
        settings, benchmark, prompts
    )
    if not experiment_id:
        experiment_id = make_arm_experiment_id("gemini")
    elif not args.experiment_id.strip():
        print(f"Resuming incomplete experiment: {experiment_id}", flush=True)

    missing = _check_pools(benchmark, settings)
    if missing:
        previous = (
            settings.output_dir / "experiments" / experiment_id / "manifest.json"
            if (settings.output_dir / "experiments" / experiment_id / "manifest.json").exists()
            else None
        )
        if previous is not None:
            raise SystemExit(
                "A frozen reference pool changed or is unavailable for an existing experiment. "
                "Do not rebuild it while resuming; restore the original pool or begin a new experiment."
            )
        if args.require_existing_pools:
            raise SystemExit(
                "Missing/incompatible frozen reference pools: " + "; ".join(missing)
            )
        print("Preparing frozen reference pools for all questions before any generation calls.", flush=True)
        _run_script("build_reference_pool.py", "--benchmark", str(benchmark))
        missing = _check_pools(benchmark, settings)
        if missing:
            raise SystemExit("Reference pool preparation incomplete: " + "; ".join(missing))
    # The lower-level runner checks credentials and hashes again before any model call.
    run_args = [
        "--benchmark", str(benchmark),
        "--experiment-id", experiment_id,
        "--systems", SYSTEMS,
        "--require-existing-pools",
    ]
    if args.claude_access:
        run_args.extend(["--claude-access", args.claude_access])

    print(f"Experiment: {experiment_id}", flush=True)
    _run_script("run_benchmark.py", *run_args)

    manifest_path = settings.output_dir / "experiments" / experiment_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit(
            f"Experiment is {manifest.get('status')!r}. Resume with "
            f"--experiment-id {experiment_id} once the failure is resolved."
        )

    _set_reporting_status(manifest_path, "in_progress")
    _run_script("analyze_results.py", "--experiment-id", experiment_id)
    if not args.skip_figures:
        _run_script("generate_research_figures.py", "--experiment-id", experiment_id)
    _set_reporting_status(manifest_path, "complete")

    print(f"\nFinished {len(prompts) * len(SYSTEMS.split(','))} cases. "
          f"Results: {settings.output_dir / 'experiments' / experiment_id}")


if __name__ == "__main__":
    main()
