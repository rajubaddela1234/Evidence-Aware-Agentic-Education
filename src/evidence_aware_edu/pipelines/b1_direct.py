from __future__ import annotations

from evidence_aware_edu.pipelines.base import PipelineBase
from evidence_aware_edu.prompts import DIRECT_EDUCATIONAL_SYSTEM
from evidence_aware_edu.schemas import BenchmarkPrompt


class DirectLLMPipeline(PipelineBase):
    system_id = "B1"
    system_name = "Direct LLM"

    def run(self, prompt: BenchmarkPrompt):
        telemetry, llm, _ = self._components(prompt)
        script, log = llm.generate_text(
            task="b1_direct_generation",
            system_instruction=DIRECT_EDUCATIONAL_SYSTEM,
            prompt=prompt.question,
            max_output_tokens=4500,
        )
        telemetry.add_llm(log)
        return self._result(prompt, telemetry, script)
