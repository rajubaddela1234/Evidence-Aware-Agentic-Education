from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


GenerationProvider = Literal["gemini"]
ClaudeAccessMethod = Literal["cli", "api"]


class Settings(BaseSettings):
    """Project settings loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini writes the scripts. Claude scores the saved scripts independently.
    generation_provider: GenerationProvider = Field(
        default="gemini",
        alias="GENERATION_PROVIDER",
    )

    # Gemini generation and reference-pool planning.
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.8-flash", alias="GEMINI_MODEL")
    thinking_level: Literal["minimal", "low", "medium", "high"] = Field(
        default="low",
        alias="GEMINI_THINKING_LEVEL",
    )
    max_output_tokens: int = Field(default=10000, alias="MAX_OUTPUT_TOKENS")

    # Claude is the independent evaluator. CLI uses a Claude Code login; API uses
    # ANTHROPIC_API_KEY. Both paths expose the same evaluator interface to the study.
    claude_access_method: ClaudeAccessMethod = Field(default="cli", alias="CLAUDE_ACCESS_METHOD")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    claude_evaluator_model: str = Field(
        default="claude-sonnet-5-5",
        alias="CLAUDE_EVALUATOR_MODEL",
    )
    claude_evaluator_max_context_chars: int = Field(
        default=40000,
        alias="CLAUDE_EVALUATOR_MAX_CONTEXT_CHARS",
    )
    # Prompt budget for the grounded-requirements judge. The script is always sent in full;
    # only the evidence is split into batches when this budget would be exceeded.
    claude_evaluator_max_judge_prompt_chars: int = Field(
        default=60000,
        alias="CLAUDE_EVALUATOR_MAX_JUDGE_PROMPT_CHARS",
    )
    claude_evaluator_max_output_tokens: int = Field(
        default=6000,
        alias="CLAUDE_EVALUATOR_MAX_OUTPUT_TOKENS",
    )
    claude_evaluator_timeout_seconds: int = Field(
        default=300,
        alias="CLAUDE_EVALUATOR_TIMEOUT_SECONDS",
    )
    claude_evaluator_retry_attempts: int = Field(
        default=3,
        alias="CLAUDE_EVALUATOR_RETRY_ATTEMPTS",
    )
    claude_evaluator_show_progress: bool = Field(
        default=False,
        alias="CLAUDE_EVALUATOR_SHOW_PROGRESS",
    )
    # Claude Sonnet 5.5 standard API token rates (USD per million tokens).
    # Update these when the evaluator model or Anthropic pricing changes.
    anthropic_input_usd_per_1m: float = Field(
        default=2.0,
        alias="ANTHROPIC_INPUT_USD_PER_1M",
    )
    anthropic_output_usd_per_1m: float = Field(
        default=10.0,
        alias="ANTHROPIC_OUTPUT_USD_PER_1M",
    )

    # Gemini cost estimates used by the generation telemetry.
    gemini_input_usd_per_1m: float = Field(default=0.75, alias="GEMINI_INPUT_USD_PER_1M")
    gemini_output_usd_per_1m: float = Field(default=3.75, alias="GEMINI_OUTPUT_USD_PER_1M")

    # LangSmith tracing is optional and off by default.
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="evidence-aware-agentic-education-final-study",
        alias="LANGSMITH_PROJECT",
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT",
    )
    langsmith_trace_content: bool = Field(default=True, alias="LANGSMITH_TRACE_CONTENT")

    # Retrieval credentials are needed while preparing frozen pools and for the demo.
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    youtube_api_key: str | None = Field(default=None, alias="YOUTUBE_API_KEY")
    youtube_require_transcript: bool = Field(default=True, alias="YOUTUBE_REQUIRE_TRANSCRIPT")
    youtube_transcript_timeout_seconds: float = Field(
        default=10.0, alias="YOUTUBE_TRANSCRIPT_TIMEOUT_SECONDS", gt=0
    )
    youtube_max_transcript_candidates: int = Field(
        default=8, alias="YOUTUBE_MAX_TRANSCRIPT_CANDIDATES", ge=1, le=50
    )

    # arXiv uses PDF text only. No OCR or figure/image understanding is used.
    arxiv_max_papers_per_query: int = Field(default=2, alias="ARXIV_MAX_PAPERS_PER_QUERY")
    arxiv_full_text_cache_dir: Path = Field(
        default=Path("data/arxiv_text_cache"),
        alias="ARXIV_FULL_TEXT_CACHE_DIR",
    )
    arxiv_pdf_timeout_seconds: float = Field(default=20.0, alias="ARXIV_PDF_TIMEOUT_SECONDS")
    arxiv_fallback_to_abstract: bool = Field(default=True, alias="ARXIV_FALLBACK_TO_ABSTRACT")

    # Keep these limits fixed across the six systems.
    max_sources_per_tool: int = Field(default=4, alias="MAX_SOURCES_PER_TOOL")
    max_reference_sources_per_tool: int = Field(default=2, alias="MAX_REFERENCE_SOURCES_PER_TOOL")
    max_context_chars: int = Field(default=50000, alias="MAX_CONTEXT_CHARS")
    max_evidence_per_query: int = Field(default=16, alias="MAX_EVIDENCE_PER_QUERY")
    max_requirement_retrieval_rounds: int = Field(
        default=1,
        alias="MAX_REQUIREMENT_RETRIEVAL_ROUNDS",
    )
    max_claim_retrieval_rounds: int = Field(default=1, alias="MAX_CLAIM_RETRIEVAL_ROUNDS")
    original_max_iterations: int = Field(default=3, alias="ORIGINAL_MAX_ITERATIONS")
    feedback_threshold: float = Field(default=7.0, alias="FEEDBACK_THRESHOLD")
    claim_support_threshold: float = Field(default=0.70, alias="CLAIM_SUPPORT_THRESHOLD")
    requirement_support_threshold: float = Field(
        default=0.70,
        alias="REQUIREMENT_SUPPORT_THRESHOLD",
    )

    # Final-study defaults.
    use_frozen_reference_pool: bool = Field(default=True, alias="USE_FROZEN_REFERENCE_POOL")
    reference_pool_dir: Path = Field(
        default=Path("data/reference_pools_final_10q"),
        alias="REFERENCE_POOL_DIR",
    )
    output_dir: Path = Field(default=Path("outputs_final_10q"), alias="OUTPUT_DIR")
    request_sleep_seconds: float = Field(default=1.0, alias="REQUEST_SLEEP_SECONDS")
    api_retry_attempts: int = Field(default=6, alias="API_RETRY_ATTEMPTS")

    # Optional Streamlit media layer. It is not part of the benchmark.
    did_api_key: str | None = Field(default=None, alias="DID_API_KEY")
    did_presenter_url: str = Field(
        default="https://create-images-results.d-id.com/DefaultPresenters/Noelle_f/image.jpeg",
        alias="DID_PRESENTER_URL",
    )
    did_voice_id: str = Field(default="en-GB-SoniaNeural", alias="DID_VOICE_ID")

    def validate_generation(self) -> None:
        """Fail early when the selected generation path is not configured."""
        key = (self.gemini_api_key or "").strip()
        if not key or key.lower().startswith("your_"):
            raise ValueError("GEMINI_API_KEY is required for generation.")
        if not self.gemini_model.strip():
            raise ValueError("GEMINI_MODEL must be configured.")

    def validate_reference_pool_builder(self) -> None:
        """The shared frozen pool is planned once with Gemini."""
        key = (self.gemini_api_key or "").strip()
        if not key or key.lower().startswith("your_"):
            raise ValueError(
                "GEMINI_API_KEY is required while building the shared frozen reference pools."
            )

    def validate_independent_evaluator(self) -> None:
        """Check the selected Claude evaluator access path."""
        model = (self.claude_evaluator_model or "").strip()
        if not model or model.lower().startswith("your_"):
            raise ValueError("CLAUDE_EVALUATOR_MODEL must name a Claude model or alias.")
        if self.claude_evaluator_timeout_seconds < 30:
            raise ValueError("CLAUDE_EVALUATOR_TIMEOUT_SECONDS must be at least 30 seconds.")
        if self.claude_evaluator_retry_attempts < 1:
            raise ValueError("CLAUDE_EVALUATOR_RETRY_ATTEMPTS must be at least 1.")
        if self.claude_evaluator_max_output_tokens < 256:
            raise ValueError("CLAUDE_EVALUATOR_MAX_OUTPUT_TOKENS must be at least 256.")

        if self.claude_access_method == "api":
            key = (self.anthropic_api_key or "").strip()
            if not key or key.lower().startswith("your_"):
                raise ValueError("ANTHROPIC_API_KEY is required when CLAUDE_ACCESS_METHOD=api.")

    @property
    def generation_model(self) -> str:
        return self.gemini_model

    @property
    def generation_access(self) -> str:
        return "google_api"

    @property
    def evaluator_provider(self) -> str:
        return "anthropic"

    @property
    def evaluator_access(self) -> str:
        return "claude_code_cli" if self.claude_access_method == "cli" else "anthropic_api"

    @property
    def evaluator_model(self) -> str:
        self.validate_independent_evaluator()
        return self.claude_evaluator_model

    def gemini_pricing(self) -> tuple[float, float]:
        return self.gemini_input_usd_per_1m, self.gemini_output_usd_per_1m

    def anthropic_pricing(self) -> tuple[float, float]:
        return self.anthropic_input_usd_per_1m, self.anthropic_output_usd_per_1m

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_pool_dir.mkdir(parents=True, exist_ok=True)
        self.arxiv_full_text_cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


PILOT_REFERENCE_POOL_DIR = "data/reference_pools_pilot_10q"
PILOT_OUTPUT_DIR = "outputs_pilot_10q"


def use_benchmark_directories(benchmark: str | Path) -> bool:
    """Send the pilot benchmark to its own pool and output folders.

    Pilot runs are for development and must never mix with final-study pools, experiments or
    reports. Environment variables override ``.env`` and are inherited by the sub-scripts that
    ``run_final_study.py`` launches, so the whole pipeline follows the redirect.
    Returns True when the pilot folders were selected.
    """
    if "pilot" not in Path(benchmark).stem.lower():
        return False
    os.environ["REFERENCE_POOL_DIR"] = PILOT_REFERENCE_POOL_DIR
    os.environ["OUTPUT_DIR"] = PILOT_OUTPUT_DIR
    get_settings.cache_clear()
    return True
