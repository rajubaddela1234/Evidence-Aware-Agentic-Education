from __future__ import annotations

from evidence_aware_edu.agents.synthesizer import SynthesizerAgent
from evidence_aware_edu.evidence.extractor import extract_ranked_evidence
from evidence_aware_edu.pipelines.base import PipelineBase
from evidence_aware_edu.schemas import BenchmarkPrompt


class StandardRAGPipeline(PipelineBase):
    system_id = "B2"
    system_name = "Standard single-source RAG"

    def run(self, prompt: BenchmarkPrompt):
        telemetry, llm, hub = self._components(prompt)
        sources = hub.search(
            prompt.question,
            ["web"],
            max_per_source=self.settings.max_sources_per_tool,
        )
        evidence = extract_ranked_evidence(
            prompt.question, sources, max_total=min(10, self.settings.max_evidence_per_query)
        )
        synthesizer = SynthesizerAgent(llm, telemetry)
        script = synthesizer.from_context(
            prompt.question,
            self.evidence_context(evidence, self.settings.max_context_chars),
            task="b2_rag_synthesis",
        )
        return self._result(prompt, telemetry, script, sources=sources, evidence=evidence)
