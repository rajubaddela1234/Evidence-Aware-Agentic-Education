from __future__ import annotations

from evidence_aware_edu.agents.decomposer import DecomposerAgent
from evidence_aware_edu.agents.feedback import FeedbackAgent
from evidence_aware_edu.agents.synthesizer import SynthesizerAgent
from evidence_aware_edu.evidence.extractor import extract_ranked_evidence
from evidence_aware_edu.pipelines.base import PipelineBase
from evidence_aware_edu.schemas import BenchmarkPrompt
from evidence_aware_edu.utils import dedupe_by


class OriginalAgenticPipeline(PipelineBase):
    """Agentic baseline based on the earlier MSc workflow."""

    system_id = "B4"
    system_name = "Original 8-agent-style Agentic Synthesizer"

    def run(self, prompt: BenchmarkPrompt):
        telemetry, llm, hub = self._components(prompt)
        decomposer = DecomposerAgent(llm, telemetry)
        synthesizer = SynthesizerAgent(llm, telemetry)
        feedback_agent = FeedbackAgent(llm, telemetry)

        current_query = prompt.question
        final_script = ""
        final_sources = []
        final_evidence = []
        feedback_history = []
        first_draft = ""

        for iteration in range(1, self.settings.original_max_iterations + 1):
            subquestions = decomposer.run(current_query)
            all_sources = []
            for subq in subquestions:
                all_sources.extend(
                    hub.search(
                        subq,
                        ["video", "web", "academic"],
                        max_per_source=self.settings.max_sources_per_tool,
                    )
                )
            all_sources = dedupe_by(all_sources, lambda d: d.url or d.source_id)
            evidence = extract_ranked_evidence(
                current_query,
                all_sources,
                max_total=self.settings.max_evidence_per_query,
            )
            context = self.evidence_context(evidence, self.settings.max_context_chars)
            script = synthesizer.from_context(
                prompt.question,
                context,
                task=f"b4_synthesis_iter_{iteration}",
            )
            if not first_draft:
                first_draft = script
            feedback = feedback_agent.run(
                prompt.question,
                script,
                context,
                self.settings.feedback_threshold,
            )
            feedback_history.append(feedback.model_dump())
            final_script, final_sources, final_evidence = script, all_sources, evidence
            if not feedback.revision_needed:
                break
            # Keep the original baseline's query-refinement step.
            rewritten, log = llm.generate_text(
                task=f"b4_query_rewrite_{iteration}",
                system_instruction=(
                    "Rewrite the research query to address the feedback. Return one concise query only."
                ),
                prompt=f"Original query: {prompt.question}\nFeedback: {feedback.feedback}",
                max_output_tokens=300,
            )
            telemetry.add_llm(log)
            current_query = rewritten.strip() or current_query

        return self._result(
            prompt,
            telemetry,
            final_script,
            sources=final_sources,
            evidence=final_evidence,
            metadata={"feedback_history": feedback_history, "pre_repair_script": first_draft if first_draft != final_script else None},
        )
