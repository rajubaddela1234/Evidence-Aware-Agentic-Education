from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evidence_aware_edu.agents.planner import PlannerAgent
from evidence_aware_edu.config import Settings
from evidence_aware_edu.experiments.pool_filter import (
    MIN_DOCUMENTS,
    MIN_RELEVANCE,
    extra_queries,
    filter_relevant,
)
from evidence_aware_edu.observability import configure_langsmith, trace_span
from evidence_aware_edu.schemas import BenchmarkPrompt
from evidence_aware_edu.telemetry import TelemetryCollector
from evidence_aware_edu.utils import dedupe_by, json_dump


POOL_SCHEMA_VERSION = "2.0"


def _public_prompt_metadata(prompt: BenchmarkPrompt) -> dict:
    """Keep pool provenance without copying the hidden evaluation rubric.

    The pool builder only needs the question.  Saving ``expected_aspects`` beside
    generation evidence would create an unnecessary route for benchmark leakage,
    even though the frozen retriever currently ignores that field.
    """
    return {
        "prompt_id": prompt.prompt_id,
        "domain": prompt.domain,
        "reasoning_type": prompt.reasoning_type,
        "question": prompt.question,
    }


def reference_pool_compatibility(
    path: Path,
    prompt: BenchmarkPrompt,
    *,
    transcript_only: bool,
) -> tuple[bool, str]:
    """Check that an existing frozen pool belongs to the current benchmark item."""
    if not path.exists():
        return False, "missing"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, "pool metadata could not be read"

    if payload.get("pool_schema_version") != POOL_SCHEMA_VERSION:
        return False, "pool was created with an older pool format"

    saved_prompt = payload.get("benchmark_prompt") or {}
    expected_prompt = _public_prompt_metadata(prompt)
    if any(saved_prompt.get(key) != value for key, value in expected_prompt.items()):
        return False, "pool belongs to a different benchmark question/version"

    if transcript_only:
        summary = payload.get("retrieval_summary", {})
        if summary.get("youtube_transcript_only") is not True:
            return False, "pool is not transcript-only"

    if not isinstance(payload.get("documents"), list):
        return False, "pool has no document list"

    return True, "compatible"


def build_reference_pool(prompt: BenchmarkPrompt, settings: Settings) -> Path:
    """Build one live multi-source corpus that can later be searched locally."""
    # Import live-provider clients lazily. This keeps local metadata checks usable
    # even when optional retrieval SDKs have not been installed in the test shell.
    from evidence_aware_edu.llm.gemini import GeminiClient
    from evidence_aware_edu.retrieval.hub import RetrievalHub

    live_settings = settings.model_copy(update={"use_frozen_reference_pool": False})
    configure_langsmith(live_settings)

    with trace_span(
        live_settings,
        f"reference_pool:{prompt.prompt_id}",
        run_type="chain",
        inputs={
            "prompt_id": prompt.prompt_id,
            "question": (
                prompt.question if live_settings.langsmith_trace_content else "[redacted]"
            ),
        },
        metadata={"phase": "reference_pool_build", "pool_schema_version": POOL_SCHEMA_VERSION},
        tags=["evidence-aware-education", "reference-pool", prompt.prompt_id],
    ) as span:
        telemetry = TelemetryCollector()
        llm = GeminiClient(live_settings)
        planner = PlannerAgent(llm, telemetry)
        hub = RetrievalHub(live_settings, telemetry=telemetry, prompt_id=None)

        plan = planner.run(prompt.question)
        queries = dedupe_by([prompt.question] + plan.subquestions, lambda value: value)
        documents = []
        for query in queries:
            documents.extend(
                hub.search(
                    query,
                    ["web", "academic", "video"],
                    max_per_source=live_settings.max_reference_sources_per_tool,
                )
            )

        documents = dedupe_by(documents, lambda document: document.url or document.source_id)

        # Drop documents that only share keywords with the question, then top up with extra
        # web queries if too few relevant documents remain.
        retrieved_count = len(documents)
        documents, relevance_audit = filter_relevant(
            llm, prompt.question, documents, record=telemetry.add_llm
        )
        top_up_queries: list[str] = []
        if len(documents) < MIN_DOCUMENTS:
            top_up_queries = extra_queries(llm, prompt.question, record=telemetry.add_llm)
            extra = []
            for query in top_up_queries:
                extra.extend(
                    hub.search(
                        query,
                        ["web"],
                        max_per_source=live_settings.max_reference_sources_per_tool,
                    )
                )
            seen = {d.url or d.source_id for d in documents}
            seen.update(a["url"] for a in relevance_audit)  # do not re-score rejected pages
            extra = [d for d in dedupe_by(extra, lambda d: d.url or d.source_id)
                     if (d.url or d.source_id) not in seen]
            if extra:
                more, more_audit = filter_relevant(
                    llm, prompt.question, extra, record=telemetry.add_llm
                )
                documents.extend(more)
                relevance_audit.extend(more_audit)
            retrieved_count += len(extra)

        run_telemetry = telemetry.finish()
        source_counts = Counter(document.source_type for document in documents)
        academic_modes = Counter(
            str(document.metadata.get("content_mode", "unknown"))
            for document in documents
            if document.source_type == "academic"
        )
        retrieval_summary = {
            "query_count": len(queries),
            "document_count": len(documents),
            "source_counts": {
                source_type: source_counts.get(source_type, 0)
                for source_type in ("web", "academic", "video")
            },
            # Keep provider diagnostics in the saved JSON rather than the terminal.
            "retrieval_errors": hub.retrieval_errors,
            "relevance_filter": {
                "min_relevance": MIN_RELEVANCE,
                "retrieved_before_filter": retrieved_count,
                "kept": len(documents),
                "top_up_queries": top_up_queries,
                "audit": relevance_audit,
            },
            "youtube_ip_blocked": bool(getattr(hub._live.get("video"), "ip_blocked", False)),
            "academic_content_modes": dict(academic_modes),
            "arxiv_max_papers_per_query": live_settings.arxiv_max_papers_per_query,
            "arxiv_text_only": True,
            "youtube_transcript_only": live_settings.youtube_require_transcript,
            "video_transcript_documents": sum(
                1
                for document in documents
                if document.source_type == "video"
                and document.metadata.get("has_transcript") is True
            ),
        }

        path = settings.reference_pool_dir / f"{prompt.prompt_id}.json"
        json_dump(
            path,
            {
                "pool_schema_version": POOL_SCHEMA_VERSION,
                "benchmark_prompt": _public_prompt_metadata(prompt),
                "planner_provider": "google",
                "planner_model": settings.gemini_model,
                "plan": plan.model_dump(),
                "documents": [document.model_dump() for document in documents],
                "retrieval_summary": retrieval_summary,
                "telemetry": run_telemetry.model_dump(),
            },
        )

        span.set_outputs(
            {
                "queries": len(queries),
                "documents": len(documents),
                "source_counts": dict(source_counts),
                "academic_content_modes": dict(academic_modes),
                "youtube_transcript_only": live_settings.youtube_require_transcript,
                "retrieval_calls": run_telemetry.retrieval_calls,
                "llm_calls": run_telemetry.total_llm_calls,
                "estimated_llm_cost_usd": run_telemetry.estimated_llm_cost_usd,
                "output_path": str(path),
            }
        )
        return path
