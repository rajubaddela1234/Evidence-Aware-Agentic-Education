from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

from evidence_aware_edu.config import Settings
from evidence_aware_edu.llm.factory import make_generation_client
from evidence_aware_edu.retrieval.hub import RetrievalHub
from evidence_aware_edu.schemas import BenchmarkPrompt, Evidence, PipelineResult, SourceDocument
from evidence_aware_edu.telemetry import TelemetryCollector
from evidence_aware_edu.utils import clip, dedupe_by


class PipelineBase(ABC):
    system_id: str
    system_name: str

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def source_context(sources: list[SourceDocument], max_chars: int = 24000) -> str:
        parts: list[str] = []
        used = 0
        for doc in sources:
            item = (
                f"SOURCE {doc.source_id} | {doc.source_type} | {doc.title}\n"
                f"URL: {doc.url}\n{clip(doc.text, 2200)}\n"
            )
            if used + len(item) > max_chars:
                break
            parts.append(item)
            used += len(item)
        return "\n\n".join(parts)

    @staticmethod
    def evidence_context(evidence: list[Evidence], max_chars: int = 24000) -> str:
        parts: list[str] = []
        used = 0
        for e in evidence:
            item = f"[{e.evidence_id}] {e.title} ({e.source_type})\nURL: {e.url}\n{clip(e.passage, 1800)}\n"
            if used + len(item) > max_chars:
                break
            parts.append(item)
            used += len(item)
        return "\n\n".join(parts)

    @abstractmethod
    def run(self, prompt: BenchmarkPrompt) -> PipelineResult:
        raise NotImplementedError

    def _components(self, prompt: BenchmarkPrompt):
        telemetry = TelemetryCollector()
        llm = make_generation_client(self.settings)
        hub = RetrievalHub(self.settings, telemetry=telemetry, prompt_id=prompt.prompt_id)
        return telemetry, llm, hub

    def _result(
        self,
        prompt: BenchmarkPrompt,
        telemetry: TelemetryCollector,
        script: str,
        sources: list[SourceDocument] | None = None,
        evidence: list[Evidence] | None = None,
        **kwargs,
    ) -> PipelineResult:
        metadata = dict(kwargs.pop("metadata", {}) or {})
        metadata.setdefault("generation_provider", self.settings.generation_provider)
        metadata.setdefault("generation_access", self.settings.generation_access)
        metadata.setdefault("generation_model_requested", self.settings.generation_model)
        return PipelineResult(
            system_id=self.system_id,
            system_name=self.system_name,
            prompt_id=prompt.prompt_id,
            query=prompt.question,
            domain=prompt.domain,
            reasoning_type=prompt.reasoning_type,
            script=script,
            sources=dedupe_by(sources or [], lambda x: x.url or x.source_id),
            evidence=evidence or [],
            metadata=metadata,
            telemetry=telemetry.finish(),
            **kwargs,
        )
