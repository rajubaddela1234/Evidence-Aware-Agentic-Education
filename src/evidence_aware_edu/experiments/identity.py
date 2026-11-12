from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from evidence_aware_edu.config import Settings


PROJECT_VERSION = "1.4.0"
PROMPT_VERSION = "evidence-aware-10q-shared-checklist-clean-narration-v1.6.0"
EXPERIMENT_SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_experiment_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"EXP_{stamp}_{uuid4().hex[:6]}"


def make_study_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"STUDY_{stamp}_{uuid4().hex[:6]}"


def make_arm_experiment_id(generator: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = generator.strip().upper()
    return f"EXP_{label}_{stamp}_{uuid4().hex[:6]}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(data) -> str:
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


_PACKAGE_DIR = Path(__file__).resolve().parents[1]

# Source files that contain prompts or change what a script/score is. Hashing the code (not a
# hand-edited version string) ties every result to the exact prompts and rubric that made it,
# including prompts written inline in agent methods.
GENERATION_CODE_PATHS = (
    "prompts.py",
    "agents",
    "pipelines",
    "graph",
    "evidence",
    "retrieval/frozen.py",
)
EVALUATION_CODE_PATHS = (
    "evaluation",
    "agents/verifier.py",
    "agents/claim_extractor.py",
    "retrieval/frozen.py",
    "utils.py",
)


def code_sha256(paths: tuple[str, ...]) -> str:
    """Hash the listed package files/folders (line endings normalised, order fixed)."""
    digest = hashlib.sha256()
    for part in paths:
        target = _PACKAGE_DIR / part
        files = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for file in files:
            if "__pycache__" in file.parts or not file.exists():
                continue
            digest.update(file.relative_to(_PACKAGE_DIR).as_posix().encode("utf-8") + b"\0")
            digest.update(file.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def safe_generation_config(settings: Settings) -> dict:
    """Return the non-secret settings that define one immutable experiment arm."""
    generation = {
        "generation_provider": settings.generation_provider,
        "generation_access": settings.generation_access,
        "generation_model": settings.generation_model,
        "max_output_tokens": settings.max_output_tokens,
    }
    generation["generation_reasoning_setting"] = settings.thinking_level

    return {
        "project_version": PROJECT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generation_code_sha256": code_sha256(GENERATION_CODE_PATHS),
        "evaluation_code_sha256": code_sha256(EVALUATION_CODE_PATHS),
        **generation,
        "evaluator_provider": settings.evaluator_provider,
        "evaluator_access": settings.evaluator_access,
        "claude_access_method": settings.claude_access_method,
        "evaluator_model": settings.evaluator_model,
        "claude_evaluator_max_context_chars": settings.claude_evaluator_max_context_chars,
        "claude_evaluator_max_output_tokens": settings.claude_evaluator_max_output_tokens,
        "claude_evaluator_timeout_seconds": settings.claude_evaluator_timeout_seconds,
        "claude_evaluator_retry_attempts": settings.claude_evaluator_retry_attempts,
        "max_sources_per_tool": settings.max_sources_per_tool,
        "max_reference_sources_per_tool": settings.max_reference_sources_per_tool,
        "max_context_chars": settings.max_context_chars,
        "max_evidence_per_query": settings.max_evidence_per_query,
        "arxiv_max_papers_per_query": settings.arxiv_max_papers_per_query,
        "youtube_require_transcript": settings.youtube_require_transcript,
        "max_requirement_retrieval_rounds": settings.max_requirement_retrieval_rounds,
        "max_claim_retrieval_rounds": settings.max_claim_retrieval_rounds,
        "requirement_support_threshold": settings.requirement_support_threshold,
        "claim_support_threshold": settings.claim_support_threshold,
        "original_max_iterations": settings.original_max_iterations,
        "feedback_threshold": settings.feedback_threshold,
        "use_frozen_reference_pool": settings.use_frozen_reference_pool,
        "reference_pool_dir": str(settings.reference_pool_dir),
        "request_sleep_seconds": settings.request_sleep_seconds,
        "api_retry_attempts": settings.api_retry_attempts,
    }


def case_key(system_id: str, prompt_id: str) -> str:
    return f"{system_id}/{prompt_id}"
