from __future__ import annotations

import json
from pathlib import Path

from evidence_aware_edu.experiments.reference_pool import (
    POOL_SCHEMA_VERSION,
    _public_prompt_metadata,
    reference_pool_compatibility,
)
from evidence_aware_edu.schemas import BenchmarkPrompt


def _prompt(question: str = "How does a solar panel work?") -> BenchmarkPrompt:
    return BenchmarkPrompt(
        prompt_id="FQ03",
        domain="Energy / Engineering",
        reasoning_type="mechanism",
        question=question,
    )


def _write_pool(path: Path, prompt: BenchmarkPrompt) -> None:
    path.write_text(
        json.dumps(
            {
                "pool_schema_version": POOL_SCHEMA_VERSION,
                "benchmark_prompt": {
                    "prompt_id": prompt.prompt_id,
                    "domain": prompt.domain,
                    "reasoning_type": prompt.reasoning_type,
                    "question": prompt.question,
                },
                "documents": [],
                "retrieval_summary": {"youtube_transcript_only": True},
            }
        ),
        encoding="utf-8",
    )


def test_reference_pool_matches_the_current_question(tmp_path: Path) -> None:
    prompt = _prompt()
    path = tmp_path / "FQ03.json"
    _write_pool(path, prompt)

    compatible, reason = reference_pool_compatibility(
        path,
        prompt,
        transcript_only=True,
    )

    assert compatible is True
    assert reason == "compatible"


def test_reference_pool_rejects_a_changed_benchmark_question(tmp_path: Path) -> None:
    original = _prompt()
    path = tmp_path / "FQ03.json"
    _write_pool(path, original)

    compatible, reason = reference_pool_compatibility(
        path,
        _prompt("How does a wind turbine work?"),
        transcript_only=True,
    )

    assert compatible is False
    assert "different benchmark" in reason


def test_pool_metadata_contains_only_public_question_fields() -> None:
    metadata = _public_prompt_metadata(_prompt())

    assert set(metadata) == {"prompt_id", "domain", "reasoning_type", "question"}
    assert metadata["question"] == "How does a solar panel work?"


def test_existing_pools_that_still_store_cutoff_date_remain_compatible(tmp_path: Path) -> None:
    # Pools built before cutoff_date was removed carry "cutoff_date": null.
    prompt = _prompt()
    path = tmp_path / "FQ03.json"
    payload = {
        "pool_schema_version": POOL_SCHEMA_VERSION,
        "benchmark_prompt": {**_public_prompt_metadata(prompt), "cutoff_date": None},
        "documents": [],
        "retrieval_summary": {"youtube_transcript_only": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    compatible, reason = reference_pool_compatibility(path, prompt, transcript_only=True)

    assert compatible is True, reason
