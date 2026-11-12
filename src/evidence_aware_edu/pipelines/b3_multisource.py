from __future__ import annotations

from evidence_aware_edu.agents.synthesizer import SynthesizerAgent
from evidence_aware_edu.evidence.extractor import extract_ranked_evidence
from evidence_aware_edu.pipelines.base import PipelineBase
from evidence_aware_edu.schemas import BenchmarkPrompt


class MultiSourceRAGPipeline(PipelineBase):
    system_id = "B3"
    system_name = "Fixed multi-source RAG"

    def run(self, prompt: BenchmarkPrompt):
        telemetry, llm, hub = self._components(prompt)
        sources = hub.search(
            prompt.question,
            ["web", "academic", "video"],
            max_per_source=self.settings.max_sources_per_tool,
        )
        evidence = extract_ranked_evidence(
            prompt.question, sources, max_total=self.settings.max_evidence_per_query
        )
        synthesizer = SynthesizerAgent(llm, telemetry)
        script = synthesizer.from_context(
            prompt.question,
            self.evidence_context(evidence, self.settings.max_context_chars),
            task="b3_multisource_synthesis",
        )
        return self._result(prompt, telemetry, script, sources=sources, evidence=evidence)
