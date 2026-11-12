from __future__ import annotations

from evidence_aware_edu.graph.evidence_aware_graph import EvidenceAwareGraph
from evidence_aware_edu.pipelines.base import PipelineBase
from evidence_aware_edu.schemas import BenchmarkPrompt
from evidence_aware_edu.telemetry import TelemetryCollector


class EvidenceAwareAgenticPipeline(PipelineBase):
    system_id = "P1"
    system_name = "Evidence-Aware Agentic System"

    def run(self, prompt: BenchmarkPrompt):
        telemetry = TelemetryCollector()
        graph = EvidenceAwareGraph(self.settings, prompt, telemetry)
        state = graph.invoke()
        return self._result(
            prompt,
            telemetry,
            state.get("script", ""),
            sources=state.get("sources", []),
            evidence=state.get("evidence", []),
            claims=state.get("final_claims", []),
            verifications=state.get("final_verifications", []),
            contradictions=state.get("contradictions", []),
            metadata={
                "research_plan": state.get("plan").model_dump() if state.get("plan") else None,
                "requirement_completeness": state.get("requirement_completeness", {}),
                "requirement_assessments": [
                    x.model_dump() for x in state.get("requirement_assessments", [])
                ],
                "requirement_audit_history": state.get("requirement_audit_history", []),
                "candidate_claims": [x.model_dump() for x in state.get("candidate_claims", [])],
                "candidate_verifications": [
                    x.model_dump() for x in state.get("candidate_verifications", [])
                ],
                "candidate_history": state.get("candidate_history", []),
                "claim_evidence_graph": state.get("ceg", {}),
                "requirement_retrieval_rounds": state.get("requirement_retrieval_round", 0),
                "claim_retrieval_rounds": state.get("claim_retrieval_round", 0),
                "retrieval_events": state.get("retrieval_events", []),
                "pedagogy_plan": (
                    state.get("pedagogy").model_dump() if state.get("pedagogy") else None
                ),
                "final_requirement_audit": [
                    x.model_dump() for x in state.get("final_requirement_audit", [])
                ],
                "final_requirement_audit_history": state.get(
                    "final_requirement_audit_history", []
                ),
                "repair_performed": bool(state.get("repair_performed", False)),
                "pre_repair_script": state.get("pre_repair_script") or None,
            },
        )
