from __future__ import annotations

from typing import Any

from evidence_aware_edu.config import Settings
from evidence_aware_edu.observability import configure_langsmith, trace_span
from evidence_aware_edu.pipelines.b1_direct import DirectLLMPipeline
from evidence_aware_edu.pipelines.b2_rag import StandardRAGPipeline
from evidence_aware_edu.pipelines.b3_multisource import MultiSourceRAGPipeline
from evidence_aware_edu.pipelines.b4_original_agentic import OriginalAgenticPipeline
from evidence_aware_edu.pipelines.b5_verified_agentic import VerifiedAgenticPipeline
from evidence_aware_edu.pipelines.p1_evidence_aware_agentic import EvidenceAwareAgenticPipeline
from evidence_aware_edu.schemas import BenchmarkPrompt, PipelineResult


PIPELINES = {
    "B1": DirectLLMPipeline,
    "B2": StandardRAGPipeline,
    "B3": MultiSourceRAGPipeline,
    "B4": OriginalAgenticPipeline,
    "B5": VerifiedAgenticPipeline,
    "P1": EvidenceAwareAgenticPipeline,
}


class TracedPipeline:
    """Wrap a pipeline in one LangSmith root trace."""

    def __init__(self, pipeline: Any, settings: Settings) -> None:
        self._pipeline = pipeline
        self.settings = settings
        self.system_id = pipeline.system_id
        self.system_name = pipeline.system_name

    def __getattr__(self, name: str):
        return getattr(self._pipeline, name)

    def run(self, prompt: BenchmarkPrompt) -> PipelineResult:
        trace_inputs = {
            "prompt_id": prompt.prompt_id,
            "domain": prompt.domain,
            "reasoning_type": prompt.reasoning_type,
            "question": prompt.question if self.settings.langsmith_trace_content else "[redacted]",
        }
        with trace_span(
            self.settings,
            f"pipeline:{self.system_id}:{prompt.prompt_id}",
            run_type="chain",
            inputs=trace_inputs,
            metadata={
                "system_id": self.system_id,
                "system_name": self.system_name,
                "prompt_id": prompt.prompt_id,
                "frozen_retrieval": self.settings.use_frozen_reference_pool,
                "generation_provider": self.settings.generation_provider,
                "generation_access": self.settings.generation_access,
                "generation_model": self.settings.generation_model,
            },
            tags=["evidence-aware-education", "pipeline", self.system_id, prompt.prompt_id],
        ) as span:
            result = self._pipeline.run(prompt)
            outputs = {
                "source_count": len(result.sources),
                "evidence_count": len(result.evidence),
                "llm_calls": result.telemetry.total_llm_calls,
                "retrieval_calls": result.telemetry.retrieval_calls,
                "latency_seconds": result.telemetry.wall_seconds,
                "estimated_llm_cost_usd": result.telemetry.estimated_llm_cost_usd,
            }
            if self.settings.langsmith_trace_content:
                outputs["script"] = result.script
            else:
                outputs["script_chars"] = len(result.script)
            span.set_outputs(outputs)
            return result


def make_pipeline(system_id: str, settings: Settings):
    key = system_id.upper()
    if key not in PIPELINES:
        raise KeyError(f"Unknown system {system_id}. Choose from {', '.join(PIPELINES)}")
    configure_langsmith(settings)
    return TracedPipeline(PIPELINES[key](settings), settings)
