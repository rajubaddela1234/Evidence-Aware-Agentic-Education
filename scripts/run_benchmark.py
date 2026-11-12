from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from evidence_aware_edu.config import get_settings, use_benchmark_directories
from evidence_aware_edu.evaluation.checklist import ChecklistIntegrityError
from evidence_aware_edu.evaluation.evaluator import ResearchEvaluator
from evidence_aware_edu.experiments.identity import (
    EXPERIMENT_SCHEMA_VERSION,
    PROJECT_VERSION,
    PROMPT_VERSION,
    case_key,
    make_arm_experiment_id,
    safe_generation_config,
    sha256_file,
    sha256_json,
    sha256_text,
    utc_now,
)
from evidence_aware_edu.experiments.io import load_benchmark
from evidence_aware_edu.experiments.reference_pool import (
    build_reference_pool,
    reference_pool_compatibility,
)
from evidence_aware_edu.observability import trace_span
from evidence_aware_edu.pipelines.factory import PIPELINES, make_pipeline
from evidence_aware_edu.schemas import PipelineResult
from evidence_aware_edu.utils import json_dump


def prepare_missing_reference_pools(prompts, settings, require_existing: bool = False) -> None:
    if not settings.use_frozen_reference_pool:
        return

    to_build = []
    reasons: dict[str, str] = {}
    for prompt in prompts:
        pool_path = settings.reference_pool_dir / f"{prompt.prompt_id}.json"
        compatible, reason = reference_pool_compatibility(
            pool_path,
            prompt,
            transcript_only=settings.youtube_require_transcript,
        )
        if not compatible:
            to_build.append(prompt)
            reasons[prompt.prompt_id] = reason

    if not to_build:
        return

    ids = ", ".join(p.prompt_id for p in to_build)
    if require_existing:
        detail = "; ".join(f"{p.prompt_id}: {reasons[p.prompt_id]}" for p in to_build)
        raise SystemExit(
            "Frozen reference pools are missing or incompatible: "
            f"{detail}. Run `uv run python scripts/build_reference_pool.py --force`."
        )

    print(
        f"Preparing {len(to_build)} missing/incompatible frozen reference pool(s) before "
        f"the primary benchmark: {ids}"
    )
    print(
        "Pool preparation uses live retrieval once and is excluded from primary generation "
        "cost/latency. YouTube evidence is transcript-only."
    )
    for i, prompt in enumerate(to_build, start=1):
        print(
            f"[POOL {i}/{len(to_build)}] BUILD {prompt.prompt_id}: {prompt.question} "
            f"({reasons[prompt.prompt_id]})"
        )
        path = build_reference_pool(prompt, settings)
        print(f"  -> {path}")


def _load_or_create_manifest(
    exp_root: Path,
    experiment_id: str,
    benchmark_path: Path,
    prompts,
    systems: list[str],
    settings,
    study_id: str = "",
) -> dict:
    manifest_path = exp_root / "manifest.json"
    config = safe_generation_config(settings)
    config_hash = sha256_json(config)
    pool_hashes = {
        p.prompt_id: sha256_file(settings.reference_pool_dir / f"{p.prompt_id}.json")
        for p in prompts
    }

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generation_config_sha256") != config_hash:
            raise SystemExit(
                "Existing experiment uses a different generation/evaluation configuration. "
                "Create a new experiment instead of mixing configurations."
            )
        if manifest.get("benchmark_sha256") != sha256_file(benchmark_path):
            raise SystemExit(
                "Existing experiment uses a different benchmark file. Create a new experiment."
            )
        if manifest.get("systems") != systems or manifest.get("prompt_ids") != [p.prompt_id for p in prompts]:
            raise SystemExit(
                "Existing experiment was created for a different system/question selection. "
                "Create a new experiment ID instead of mixing cases."
            )
        if manifest.get("reference_pool_sha256") != pool_hashes:
            raise SystemExit(
                "One or more frozen reference pools changed after this experiment was created. "
                "Create a new experiment ID so corpus versions are not mixed."
            )
        if study_id and manifest.get("study_id", "") != study_id:
            raise SystemExit("Existing experiment belongs to a different parent study.")
        return manifest

    expected_cases = [case_key(s, p.prompt_id) for p in prompts for s in systems]
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "project_version": PROJECT_VERSION,
        "study_id": study_id,
        "generation_provider": settings.generation_provider,
        "generation_access": settings.generation_access,
        "generation_model": settings.generation_model,
        "prompt_version": PROMPT_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "generation_config": config,
        "generation_config_sha256": config_hash,
        "reference_pool_sha256": pool_hashes,
        "systems": systems,
        "prompt_ids": [p.prompt_id for p in prompts],
        "expected_primary_runs": len(expected_cases),
        "expected_cases": expected_cases,
        "cases": {
            key: {
                "status": "pending",
                "generation_attempts": 0,
                "evaluation_attempts": 0,
            }
            for key in expected_cases
        },
    }
    json_dump(manifest_path, manifest)
    return manifest


def _save_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = utc_now()
    json_dump(path, manifest)


def _validate_existing_pair(raw_path: Path, eval_path: Path, experiment_id: str) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    ev = json.loads(eval_path.read_text(encoding="utf-8"))
    record = ev["record"]
    actual_hash = sha256_text(raw.get("script", ""))
    if raw.get("experiment_id") != experiment_id or record.get("experiment_id") != experiment_id:
        raise ValueError("Experiment ID mismatch in saved artifacts.")
    if raw.get("script_sha256") != actual_hash:
        raise ValueError("Saved raw script hash does not match the script text.")
    if record.get("evaluated_script_sha256") != actual_hash:
        raise ValueError("Evaluation does not belong to the saved script hash.")
    if raw.get("reference_pool_sha256") != record.get("reference_pool_sha256"):
        raise ValueError("Raw/evaluation reference-pool hash mismatch.")
    return record


def _validate_reuse_source(
    source_root: Path,
    benchmark_path: Path,
    prompts,
    systems: list[str],
    current_manifest: dict,
) -> dict:
    """Check that saved raw generations can be reused in a new evaluator experiment."""
    manifest_path = source_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Reuse source has no manifest: {manifest_path}")
    source = json.loads(manifest_path.read_text(encoding="utf-8"))

    if source.get("benchmark_sha256") != sha256_file(benchmark_path):
        raise SystemExit("Cannot reuse raw generations: benchmark hash differs.")

    source_pools = source.get("reference_pool_sha256", {})
    current_pools = current_manifest.get("reference_pool_sha256", {})
    for prompt in prompts:
        pid = prompt.prompt_id
        if source_pools.get(pid) != current_pools.get(pid):
            raise SystemExit(
                f"Cannot reuse raw generation for {pid}: frozen reference-pool hash differs."
            )

    source_cfg = source.get("generation_config", {})
    current_cfg = current_manifest.get("generation_config", {})
    generation_keys = [
        "generation_provider",
        "generation_access",
        "generation_model",
        "generation_reasoning_setting",
        "max_output_tokens",
        "max_sources_per_tool",
        "max_reference_sources_per_tool",
        "max_context_chars",
        "max_evidence_per_query",
        "arxiv_max_papers_per_query",
        "youtube_require_transcript",
        "max_requirement_retrieval_rounds",
        "max_claim_retrieval_rounds",
        "requirement_support_threshold",
        "claim_support_threshold",
        "original_max_iterations",
        "feedback_threshold",
        "use_frozen_reference_pool",
    ]
    changed = [
        key for key in generation_keys
        if key in source_cfg and source_cfg.get(key) != current_cfg.get(key)
    ]
    if changed:
        raise SystemExit(
            "Cannot reuse raw generations because generation settings changed: "
            + ", ".join(changed)
        )

    available_systems = set(source.get("systems", []))
    missing_systems = [system for system in systems if system not in available_systems]
    if missing_systems:
        print(
            "Reuse source does not contain these systems, so they will be generated normally: "
            + ", ".join(missing_systems)
        )
    return source


def _copy_reused_raw(
    source_path: Path,
    target_path: Path,
    *,
    source_experiment_id: str,
    target_experiment_id: str,
    pool_hash: str,
    config_hash: str,
    study_id: str = "",
    generation_provider: str = "",
    generation_access: str = "",
    generation_model: str = "",
) -> PipelineResult:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    script_hash = sha256_text(payload.get("script", ""))
    if payload.get("script_sha256") != script_hash:
        raise RuntimeError(f"Reuse source has an invalid script hash: {source_path}")
    if payload.get("reference_pool_sha256") != pool_hash:
        raise RuntimeError(f"Reuse source has a different reference pool: {source_path}")

    result = PipelineResult.model_validate(payload)
    new_payload = result.model_dump()
    new_payload.update(
        {
            "experiment_id": target_experiment_id,
            "script_sha256": script_hash,
            "reference_pool_sha256": pool_hash,
            "generation_config_sha256": config_hash,
            "prompt_version": PROMPT_VERSION,
            "study_id": study_id,
            "generation_provider": generation_provider,
            "generation_access": generation_access,
            "generation_model": generation_model,
            "artifact_status": "generation_complete",
            "generation_reused": True,
            "generation_reused_from_experiment": source_experiment_id,
            "generation_reused_from_path": str(source_path),
        }
    )
    json_dump(target_path, new_payload)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the six systems on the chosen benchmark (Gemini generation, Claude evaluation)."
    )
    parser.add_argument("--benchmark", default="data/final_benchmark_10.jsonl")
    parser.add_argument("--systems", default="B1,B2,B3,B4,B5,P1")
    parser.add_argument(
        "--claude-access",
        choices=["cli", "api"],
        default="",
        help="Override CLAUDE_ACCESS_METHOD for the independent evaluator.",
    )
    parser.add_argument(
        "--study-id",
        default="",
        help="Optional study ID for tracking an external research protocol.",
    )
    parser.add_argument("--prompt-ids", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-existing-pools", action="store_true")
    parser.add_argument(
        "--experiment-id",
        default="",
        help="Resume a named experiment. If omitted, a new immutable experiment ID is created.",
    )
    parser.add_argument(
        "--reuse-raw-from",
        default="",
        help=(
            "Create a new experiment but reuse hash-validated raw generations from an older "
            "experiment when generation settings and frozen pools match. Accepts an experiment "
            "ID in this project or a path to an experiment directory."
        ),
    )
    args = parser.parse_args()
    use_benchmark_directories(args.benchmark)

    settings = get_settings()
    overrides = {}
    if args.claude_access:
        overrides["claude_access_method"] = args.claude_access
    if overrides:
        settings = settings.model_copy(update=overrides)
        settings.ensure_dirs()

    try:
        settings.validate_generation()
        settings.validate_independent_evaluator()
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    benchmark_path = Path(args.benchmark)
    prompts = load_benchmark(benchmark_path)
    wanted_prompts = {x.strip() for x in args.prompt_ids.split(",") if x.strip()}
    if wanted_prompts:
        prompts = [p for p in prompts if p.prompt_id in wanted_prompts]
    if not prompts:
        raise SystemExit("No benchmark prompts matched the requested prompt IDs.")

    systems = [x.strip().upper() for x in args.systems.split(",") if x.strip()]
    unknown = [s for s in systems if s not in PIPELINES]
    if unknown:
        raise SystemExit(f"Unknown systems: {unknown}")

    prepare_missing_reference_pools(prompts, settings, args.require_existing_pools)

    experiment_id = args.experiment_id.strip() or make_arm_experiment_id(settings.generation_provider)
    exp_root = settings.output_dir / "experiments" / experiment_id
    exp_root.mkdir(parents=True, exist_ok=True)
    latest_pointer = settings.output_dir / "latest_experiment.txt"
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(experiment_id, encoding="utf-8")
    generator_pointer = settings.output_dir / f"latest_{settings.generation_provider}_experiment.txt"
    generator_pointer.write_text(experiment_id, encoding="utf-8")

    manifest = _load_or_create_manifest(
        exp_root,
        experiment_id,
        benchmark_path,
        prompts,
        systems,
        settings,
        study_id=args.study_id.strip(),
    )
    manifest_path = exp_root / "manifest.json"
    config_hash = manifest["generation_config_sha256"]

    reuse_arg = args.reuse_raw_from.strip()
    reuse_root = None
    reuse_experiment_id = ""
    if reuse_arg:
        supplied_path = Path(reuse_arg)
        reuse_root = (
            supplied_path
            if supplied_path.exists()
            else settings.output_dir / "experiments" / reuse_arg
        )
        reuse_experiment_id = reuse_root.name
        if reuse_experiment_id == experiment_id:
            raise SystemExit("--reuse-raw-from must name a different experiment.")
        _validate_reuse_source(reuse_root, benchmark_path, prompts, systems, manifest)
        manifest["generation_reuse_source"] = str(reuse_root)
        _save_manifest(manifest_path, manifest)
        print(f"Raw-generation reuse source: {reuse_root}")

    raw_root = exp_root / "raw"
    eval_root = exp_root / "evaluations"
    error_path = exp_root / "errors.jsonl"
    evaluator = ResearchEvaluator(settings)
    # Derive all coverage checklists up front and pin their hashes in the manifest, so every
    # system in this experiment is judged against the same obligations even across resumes.
    try:
        manifest["checklist_sha256"] = evaluator.freeze_checklists(
            prompts, experiment_id, manifest.get("checklist_sha256", {})
        )
    except ChecklistIntegrityError as exc:
        raise SystemExit(f"Checklist integrity failure: {exc}") from exc
    _save_manifest(manifest_path, manifest)
    records = []

    print(f"Experiment ID: {experiment_id}")
    if args.study_id.strip():
        print(f"Study ID: {args.study_id.strip()}")
    print(
        f"Generation: {settings.generation_provider} "
        f"({settings.generation_access}) · {settings.generation_model}"
    )
    print(f"Experiment directory: {exp_root}")

    total = len(prompts) * len(systems)
    counter = 0
    for prompt in prompts:
        pool_hash = manifest["reference_pool_sha256"][prompt.prompt_id]
        for system_id in systems:
            counter += 1
            key = case_key(system_id, prompt.prompt_id)
            raw_path = raw_root / system_id / f"{prompt.prompt_id}.json"
            eval_path = eval_root / system_id / f"{prompt.prompt_id}.json"

            if raw_path.exists() and eval_path.exists() and not args.force:
                try:
                    record = _validate_existing_pair(raw_path, eval_path, experiment_id)
                    print(f"[{counter}/{total}] SKIP {key} (hash-validated)")
                    records.append(record)
                    manifest["cases"][key]["status"] = "complete"
                    continue
                except Exception as exc:
                    raise SystemExit(f"Existing artifact integrity failure for {key}: {exc}") from exc

            print(f"[{counter}/{total}] RUN {key}")
            case_meta = manifest["cases"].setdefault(key, {})
            case_meta.update({"status": "running", "started_at": utc_now(), "error": None})
            # Remove the old evaluation before replacing this case.
            if args.force and eval_path.exists():
                eval_path.unlink()
            _save_manifest(manifest_path, manifest)

            try:
                with trace_span(
                    settings,
                    f"benchmark_case:{system_id}:{prompt.prompt_id}",
                    run_type="chain",
                    inputs={
                        "system_id": system_id,
                        "prompt_id": prompt.prompt_id,
                        "question": prompt.question
                        if settings.langsmith_trace_content
                        else "[redacted]",
                    },
                    metadata={
                        "phase": "primary_benchmark_case",
                        "experiment_id": experiment_id,
                    },
                    tags=["evidence-aware-education", "benchmark", system_id, prompt.prompt_id],
                ) as case_span:
                    reuse_raw_generation = raw_path.exists() and not eval_path.exists() and not args.force
                    if reuse_raw_generation:
                        raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
                        script_hash = sha256_text(raw_payload.get("script", ""))
                        if raw_payload.get("experiment_id") != experiment_id:
                            raise RuntimeError("Existing raw artifact belongs to another experiment.")
                        if raw_payload.get("script_sha256") != script_hash:
                            raise RuntimeError("Existing raw artifact script hash is invalid.")
                        if raw_payload.get("reference_pool_sha256") != pool_hash:
                            raise RuntimeError("Existing raw artifact uses a different reference pool.")
                        if raw_payload.get("generation_config_sha256") != config_hash:
                            raise RuntimeError("Existing raw artifact uses a different configuration.")
                        result = PipelineResult.model_validate(raw_payload)
                        print("  Reusing hash-validated generation; running independent evaluation only.")
                    else:
                        reused = False
                        source_raw_path = (
                            reuse_root / "raw" / system_id / f"{prompt.prompt_id}.json"
                            if reuse_root is not None
                            else None
                        )
                        if source_raw_path is not None and source_raw_path.exists():
                            result = _copy_reused_raw(
                                source_raw_path,
                                raw_path,
                                source_experiment_id=reuse_experiment_id,
                                target_experiment_id=experiment_id,
                                pool_hash=pool_hash,
                                config_hash=config_hash,
                                study_id=args.study_id.strip(),
                                generation_provider=settings.generation_provider,
                                generation_access=settings.generation_access,
                                generation_model=settings.generation_model,
                            )
                            reused = True
                            print(
                                f"  Reused hash-validated generation from "
                                f"{reuse_experiment_id}; running independent evaluation."
                            )
                        else:
                            result = make_pipeline(system_id, settings).run(prompt)
                            if not result.script.strip():
                                # Never save or score an empty script: it would be evaluated
                                # as a real answer with zero coverage.
                                raise RuntimeError(f"{key} produced an empty script.")

                        script_hash = sha256_text(result.script)
                        if not reused:
                            raw_payload = result.model_dump()
                            raw_payload.update(
                                {
                                    "experiment_id": experiment_id,
                                    "script_sha256": script_hash,
                                    "reference_pool_sha256": pool_hash,
                                    "generation_config_sha256": config_hash,
                                    "prompt_version": PROMPT_VERSION,
                                    "study_id": args.study_id.strip(),
                                    "generation_provider": settings.generation_provider,
                                    "generation_access": settings.generation_access,
                                    "generation_model": settings.generation_model,
                                    "artifact_status": "generation_complete",
                                }
                            )
                            json_dump(raw_path, raw_payload)

                        case_meta["status"] = "generation_complete"
                        case_meta["generation_reused"] = reused
                        if reused:
                            case_meta["generation_reused_from_experiment"] = reuse_experiment_id
                        case_meta["script_sha256"] = script_hash
                        case_meta["reference_pool_sha256"] = pool_hash
                        case_meta["generation_attempts"] = sum(
                            call.attempts for call in result.telemetry.llm_calls
                        )
                        case_meta["generation_completed_at"] = utc_now()
                        _save_manifest(manifest_path, manifest)

                    record, details = evaluator.evaluate(
                        prompt,
                        result,
                        experiment_id=experiment_id,
                        script_sha256=script_hash,
                        reference_pool_sha256=pool_hash,
                        generation_config_sha256=config_hash,
                        prompt_version=PROMPT_VERSION,
                    )
                    if record.evaluated_script_sha256 != script_hash:
                        raise RuntimeError("Evaluator returned a mismatched script hash.")

                    json_dump(eval_path, {"record": record.model_dump(), "details": details})
                    records.append(record.model_dump())
                    evaluation_calls = details.get("evaluation_telemetry", {}).get("llm_calls", [])
                    case_meta.update(
                        {
                            "status": "complete",
                            "evaluation_attempts": sum(
                                int(call.get("attempts", 1)) for call in evaluation_calls
                            ),
                            "evaluation_completed_at": utc_now(),
                            "evaluated_script_sha256": record.evaluated_script_sha256,
                        }
                    )
                    case_span.set_outputs(
                        {
                            "experiment_id": experiment_id,
                            "script_sha256": script_hash,
                            "coverage_percent": record.query_coverage_percent,
                            "reference_evidence_support_percent": (
                                record.reference_evidence_support_percent
                            ),
                            "relevance_percent": record.query_relevance_percent,
                            "narration_quality_percent": record.narration_quality_percent,
                            "generation_cost_usd": record.estimated_llm_cost_usd,
                            "generation_latency_seconds": record.latency_seconds,
                        }
                    )
                print(
                    f"  Coverage={record.query_coverage_percent:.1f}% "
                    f"EvidenceSupport={record.reference_evidence_support_percent:.1f}% "
                    f"Relevance={record.query_relevance_percent:.1f}% "
                    f"Narration={record.narration_quality_percent:.1f}% "
                    f"Cost=${record.estimated_llm_cost_usd:.5f} "
                    f"Latency={record.latency_seconds:.1f}s"
                )
            except Exception as exc:
                error_path.parent.mkdir(parents=True, exist_ok=True)
                with error_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "experiment_id": experiment_id,
                                "system_id": system_id,
                                "prompt_id": prompt.prompt_id,
                                "time": utc_now(),
                                "error": repr(exc),
                            }
                        )
                        + "\n"
                    )
                case_meta.update({"status": "failed", "error": repr(exc), "failed_at": utc_now()})
                print(f"  ERROR: {exc!r}")
            finally:
                _save_manifest(manifest_path, manifest)
            time.sleep(settings.request_sleep_seconds)

    complete = sum(1 for x in manifest["cases"].values() if x.get("status") == "complete")
    manifest["completed_primary_runs"] = complete
    manifest["failed_primary_runs"] = sum(
        1 for x in manifest["cases"].values() if x.get("status") == "failed"
    )
    manifest["status"] = (
        "complete" if complete == manifest["expected_primary_runs"] else "incomplete"
    )
    manifest["completed_at"] = utc_now()
    _save_manifest(manifest_path, manifest)

    if records:
        df = pd.DataFrame(records)
        out = exp_root / "evaluation_records.csv"
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} evaluation rows to {out}")
    print(
        f"Experiment status: {manifest['status']} "
        f"({complete}/{manifest['expected_primary_runs']} complete cases)"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nBenchmark interrupted by user. Completed artifacts are saved; "
            "resume the same experiment without --force.",
            flush=True,
        )
