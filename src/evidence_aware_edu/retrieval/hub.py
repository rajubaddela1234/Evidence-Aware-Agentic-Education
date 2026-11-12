from __future__ import annotations

import logging
from evidence_aware_edu.config import Settings
from evidence_aware_edu.retrieval.arxiv_retriever import ArxivRetriever
from evidence_aware_edu.retrieval.frozen import FrozenCorpusHub
from evidence_aware_edu.retrieval.tavily import TavilyRetriever
from evidence_aware_edu.observability import trace_span
from evidence_aware_edu.retrieval.youtube import YouTubeRetriever
from evidence_aware_edu.schemas import SourceDocument, SourceType
from evidence_aware_edu.telemetry import TelemetryCollector
from evidence_aware_edu.utils import dedupe_by


logger = logging.getLogger(__name__)


class RetrievalHub:
    def __init__(
        self,
        settings: Settings,
        telemetry: TelemetryCollector | None = None,
        prompt_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.telemetry = telemetry
        self.prompt_id = prompt_id
        self.frozen = FrozenCorpusHub(
            settings.reference_pool_dir,
            require_video_transcript=settings.youtube_require_transcript,
        )
        self._live: dict[SourceType, object] = {}
        self.retrieval_errors: list[dict[str, str]] = []
        if settings.tavily_api_key:
            self._live["web"] = TavilyRetriever(settings.tavily_api_key)
        self._live["academic"] = ArxivRetriever(
            max_papers_per_query=settings.arxiv_max_papers_per_query,
            cache_dir=settings.arxiv_full_text_cache_dir,
            pdf_timeout_seconds=settings.arxiv_pdf_timeout_seconds,
            fallback_to_abstract=settings.arxiv_fallback_to_abstract,
        )
        if settings.youtube_api_key:
            self._live["video"] = YouTubeRetriever(
                settings.youtube_api_key,
                transcript_required=settings.youtube_require_transcript,
                transcript_timeout_seconds=settings.youtube_transcript_timeout_seconds,
                max_transcript_candidates=settings.youtube_max_transcript_candidates,
            )

    def search(
        self,
        query: str,
        source_types: list[SourceType],
        max_per_source: int,
    ) -> list[SourceDocument]:
        if self.settings.use_frozen_reference_pool and self.prompt_id:
            with trace_span(
                self.settings,
                "retrieval:frozen_pool",
                run_type="retriever",
                inputs={
                    "query": query,
                    "source_types": source_types,
                    "max_per_source": max_per_source,
                    "prompt_id": self.prompt_id,
                },
                metadata={"retrieval_mode": "frozen", "prompt_id": self.prompt_id},
                tags=["evidence-aware-education", "retrieval", "frozen"],
            ) as span:
                pool = self.frozen.load(self.prompt_id)
                docs = self.frozen.rank(
                    query,
                    pool,
                    source_types,
                    max_per_source,
                    per_source_limits={
                        "academic": self.settings.arxiv_max_papers_per_query,
                    },
                )
                if self.telemetry:
                    self.telemetry.add_retrieval(len(docs))
                span.set_outputs({
                    "document_count": len(docs),
                    "sources": [
                        {
                            "source_type": d.source_type,
                            "title": d.title,
                            "url": d.url,
                            "content_mode": d.metadata.get("content_mode"),
                            "pages_extracted": d.metadata.get("pages_extracted"),
                        }
                        for d in docs
                    ],
                })
                return docs

        docs: list[SourceDocument] = []
        for source_type in source_types:
            retriever = self._live.get(source_type)
            if retriever is None:
                continue
            with trace_span(
                self.settings,
                f"retrieval:{source_type}",
                run_type="retriever",
                inputs={"query": query, "max_results": max_per_source},
                metadata={"retrieval_mode": "live", "source_type": source_type},
                tags=["evidence-aware-education", "retrieval", source_type],
            ) as span:
                try:
                    found = retriever.search(query, max_results=max_per_source)
                    docs.extend(found)
                    if self.telemetry:
                        self.telemetry.add_retrieval(len(found))
                    span.set_outputs({
                        "document_count": len(found),
                        "sources": [
                            {
                                "title": d.title,
                                "url": d.url,
                                "content_mode": d.metadata.get("content_mode"),
                                "pages_extracted": d.metadata.get("pages_extracted"),
                            }
                            for d in found
                        ],
                    })
                except Exception as exc:
                    # Keep the run going if one retrieval tool fails.
                    span.add_metadata({"retrieval_error": repr(exc)})
                    self.retrieval_errors.append({
                        "source_type": source_type,
                        "error_type": type(exc).__name__,
                    })
                    logger.debug("%s retrieval failed", source_type, exc_info=True)
        return dedupe_by(docs, lambda d: d.url or d.source_id)
