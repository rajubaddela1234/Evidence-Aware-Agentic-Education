from __future__ import annotations

import argparse
from pathlib import Path

from evidence_aware_edu.config import get_settings
from evidence_aware_edu.experiments.analysis import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one immutable benchmark experiment.")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Analyze complete hash-valid cases from an incomplete experiment.",
    )
    args = parser.parse_args()

    settings = get_settings()
    experiment_id = args.experiment_id.strip()
    if not experiment_id:
        pointer = settings.output_dir / "latest_experiment.txt"
        if not pointer.exists():
            raise SystemExit("No latest experiment pointer found. Run the benchmark first.")
        experiment_id = pointer.read_text(encoding="utf-8").strip()

    exp_root = settings.output_dir / "experiments" / experiment_id
    outputs = analyze(
        exp_root / "evaluations",
        exp_root / "analysis",
        raw_dir=exp_root / "raw",
        manifest_path=exp_root / "manifest.json",
        strict=not args.allow_incomplete,
    )
    print(f"Experiment: {experiment_id}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
