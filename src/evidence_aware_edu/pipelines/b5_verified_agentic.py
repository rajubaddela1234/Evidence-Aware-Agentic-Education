from __future__ import annotations

from evidence_aware_edu.agents.claim_extractor import ClaimExtractorAgent
from evidence_aware_edu.agents.repair import RepairAgent
from evidence_aware_edu.agents.verifier import VerifierAgent
from evidence_aware_edu.llm.factory import make_generation_client
from evidence_aware_edu.pipelines.b4_original_agentic import OriginalAgenticPipeline
from evidence_aware_edu.pipelines.base import PipelineBase
from evidence_aware_edu.schemas import BenchmarkPrompt
from evidence_aware_edu.telemetry import TelemetryCollector


class VerifiedAgenticPipeline(PipelineBase):
    system_id = "B5"
    system_name = "Agentic RAG + whole-script verifier"

    def run(self, prompt: BenchmarkPrompt):
        base = OriginalAgenticPipeline(self.settings).run(prompt)

        telemetry = TelemetryCollector()
        telemetry.data.llm_calls.extend(base.telemetry.llm_calls)
        telemetry.data.retrieval_calls = base.telemetry.retrieval_calls
        telemetry.data.retrieved_documents = base.telemetry.retrieved_documents

        # B5 uses the same generation-side model as the rest of its experiment arm.
        # Claude remains outside the system and is used only for common evaluation.
        system_llm = make_generation_client(self.settings)
        extractor = ClaimExtractorAgent(system_llm, telemetry)
        verifier = VerifierAgent(system_llm, telemetry)
        repair = RepairAgent(system_llm, telemetry)

        claims = extractor.from_script(base.script)
        verifications = verifier.run(claims, base.evidence, task="b5_verify")
        repaired = repair.whole_script_repair(
            prompt.question,
            base.script,
            claims,
            verifications,
            base.evidence,
        )
        if repaired != base.script:
            claims = extractor.from_script(repaired)
            verifications = verifier.run(claims, base.evidence, task="b5_reverify")

        result = self._result(
            prompt,
            telemetry,
            repaired,
            sources=base.sources,
            evidence=base.evidence,
            claims=claims,
            verifications=verifications,
            metadata={"b4_metadata": base.metadata, "pre_repair_script": base.script if repaired != base.script else None},
        )
        result.telemetry.wall_seconds += base.telemetry.wall_seconds
        return result
