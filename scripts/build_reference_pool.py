from __future__ import annotations

import argparse
from pathlib import Path

from evidence_aware_edu.config import get_settings, use_benchmark_directories
from evidence_aware_edu.experiments.io import load_benchmark
from evidence_aware_edu.experiments.reference_pool import (
    build_reference_pool,
    reference_pool_compatibility,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen reference corpora for the ten-question final benchmark."
    )
    parser.add_argument("--benchmark", default="data/final_benchmark_10.jsonl")
    parser.add_argument("--prompt-ids", default="", help="Comma-separated subset, e.g. FQ01,FQ10")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild an existing pool. Without --force compatible pools are kept to save API quota.",
    )
    args = parser.parse_args()
    use_benchmark_directories(args.benchmark)

    settings = get_settings()
    try:
        settings.validate_reference_pool_builder()
    except ValueError as exc:
        raise SystemExit(f"Reference-pool configuration error: {exc}") from exc

    prompts = load_benchmark(Path(args.benchmark))
    wanted = {value.strip() for value in args.prompt_ids.split(",") if value.strip()}
    if wanted:
        prompts = [prompt for prompt in prompts if prompt.prompt_id in wanted]

    if not prompts:
        raise SystemExit("No benchmark questions matched the supplied --prompt-ids.")

    for index, prompt in enumerate(prompts, start=1):
        destination = settings.reference_pool_dir / f"{prompt.prompt_id}.json"
        compatible, reason = reference_pool_compatibility(
            destination,
            prompt,
            transcript_only=settings.youtube_require_transcript,
        )

        if compatible and not args.force:
            print(f"[{index}/{len(prompts)}] SKIP {prompt.prompt_id}: {destination} already exists")
            continue

        action = "REBUILD" if destination.exists() else "BUILD"
        why = "forced rebuild" if args.force and destination.exists() else reason
        print(f"[{index}/{len(prompts)}] {action} {prompt.prompt_id}: {prompt.question} ({why})")
        path = build_reference_pool(prompt, settings)
        print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
