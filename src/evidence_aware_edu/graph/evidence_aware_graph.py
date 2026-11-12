from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from evidence_aware_edu.agents.claim_extractor import ClaimExtractorAgent
from evidence_aware_edu.agents.contradiction import ContradictionAgent
from evidence_aware_edu.agents.narration_requirement_auditor import NarrationRequirementAuditorAgent
from evidence_aware_edu.agents.pedagogy import PedagogyAgent
from evidence_aware_edu.agents.planner import PlannerAgent
from evidence_aware_edu.agents.repair import RepairAgent
from evidence_aware_edu.agents.requirement_auditor import RequirementAuditorAgent
from evidence_aware_edu.agents.requirement_completeness import RequirementCompletenessAgent
from evidence_aware_edu.agents.router import RouterAgent
from evidence_aware_edu.agents.synthesizer import SynthesizerAgent
from evidence_aware_edu.agents.verifier import VerifierAgent
from evidence_aware_edu.adaptive_logic import claim_needs_more_evidence, requirement_needs_more_evidence
from evidence_aware_edu.config import Settings
from evidence_aware_edu.evidence.extractor import extract_ranked_evidence
from evidence_aware_edu.evidence.graph import ClaimEvidenceGraph
from evidence_aware_edu.llm.factory import make_generation_client
from evidence_aware_edu.retrieval.hub import RetrievalHub
from evidence_aware_edu.schemas import (
    BenchmarkPrompt,
    Claim,
    ClaimVerification,
    Contradiction,
    Evidence,
    NarrationRequirementAssessment,
    PedagogyPlan,
    RequirementAssessment,
    ResearchPlan,
    RouteDecision,
    SourceDocument,
)
from evidence_aware_edu.telemetry import TelemetryCollector
from evidence_aware_edu.utils import dedupe_by


GAP_PASSAGES_PER_QUERY = 3  # frozen-pool passages added per weak requirement/claim query


@dataclass(frozen=True)
class AdaptiveFlags:
    adaptive_routing: bool = True
    requirement_gap_retrieval: bool = True
    claim_gap_retrieval: bool = True
    contradiction_detection: bool = True
    final_repair: bool = True


class EvidenceAwareState(TypedDict, total=False):
    prompt: BenchmarkPrompt
    plan: ResearchPlan
    sources: list[SourceDocument]
    evidence: list[Evidence]
    requirement_completeness: dict
    requirement_assessments: list[RequirementAssessment]
    requirement_audit_history: list[list[dict]]
    candidate_claims: list[Claim]
    candidate_verifications: list[ClaimVerification]
    candidate_history: list[dict]
    contradictions: list[Contradiction]
    pedagogy: PedagogyPlan
    script: str
    final_claims: list[Claim]
    final_verifications: list[ClaimVerification]
    final_requirement_audit: list[NarrationRequirementAssessment]
    final_requirement_audit_history: list[list[dict]]
    repair_performed: bool
    pre_repair_script: str
    requirement_retrieval_round: int
    claim_retrieval_round: int
    retrieval_events: list[dict]
    ceg: dict


class EvidenceAwareGraph:
    """Build and run the P1 evidence-aware workflow."""

    def __init__(
        self,
        settings: Settings,
        prompt: BenchmarkPrompt,
        telemetry: TelemetryCollector,
        flags: AdaptiveFlags | None = None,
    ) -> None:
        self.settings = settings
        self.prompt = prompt
        self.telemetry = telemetry
        self.flags = flags or AdaptiveFlags()

        # P1 uses the generation model selected for this study arm.
        # The Claude benchmark evaluator is kept separate from the system under test.
        self.llm = make_generation_client(settings)
        self.hub = RetrievalHub(settings, telemetry=telemetry, prompt_id=prompt.prompt_id)

        self.planner = PlannerAgent(self.llm, telemetry)
        self.requirement_completeness = RequirementCompletenessAgent(self.llm, telemetry)
        self.router = RouterAgent(self.llm, telemetry)
        self.requirement_auditor = RequirementAuditorAgent(self.llm, telemetry)
        self.narration_requirement_auditor = NarrationRequirementAuditorAgent(self.llm, telemetry)
        self.claim_extractor = ClaimExtractorAgent(self.llm, telemetry)
        self.verifier = VerifierAgent(self.llm, telemetry)
        self.contradiction = ContradictionAgent(self.llm, telemetry)
        self.pedagogy = PedagogyAgent(self.llm, telemetry)
        self.synthesizer = SynthesizerAgent(self.llm, telemetry)
        self.repair = RepairAgent(self.llm, telemetry)

        self.app = self._build()

    def _build(self):
        g = StateGraph(EvidenceAwareState)
        g.add_node("plan", self._plan)
        g.add_node("complete_requirements", self._complete_requirements)
        g.add_node("retrieve", self._retrieve)
        g.add_node("audit_requirements", self._audit_requirements)
        g.add_node("targeted_requirement_retrieve", self._targeted_requirement_retrieve)
        g.add_node("claims", self._claims)
        g.add_node("verify", self._verify)
        g.add_node("targeted_claim_retrieve", self._targeted_claim_retrieve)
        g.add_node("contradictions", self._contradictions)
        g.add_node("pedagogy", self._pedagogy)
        g.add_node("synthesize", self._synthesize)
        g.add_node("final_verify", self._final_verify)
        g.add_node("final_requirement_audit", self._final_requirement_audit)
        g.add_node("repair", self._repair)

        g.add_edge(START, "plan")
        g.add_edge("plan", "complete_requirements")
        g.add_edge("complete_requirements", "retrieve")
        g.add_edge("retrieve", "audit_requirements")
        g.add_conditional_edges(
            "audit_requirements",
            self._route_after_requirement_audit,
            {"retrieve_requirement": "targeted_requirement_retrieve", "continue": "claims"},
        )
        g.add_edge("targeted_requirement_retrieve", "audit_requirements")
        g.add_edge("claims", "verify")
        g.add_conditional_edges(
            "verify",
            self._route_after_candidate_verify,
            {"retrieve_claim": "targeted_claim_retrieve", "continue": "contradictions"},
        )
        # New evidence can change the candidate claim set.
        g.add_edge("targeted_claim_retrieve", "claims")
        g.add_edge("contradictions", "pedagogy")
        g.add_edge("pedagogy", "synthesize")
        g.add_edge("synthesize", "final_verify")
        g.add_edge("final_verify", "final_requirement_audit")
        g.add_conditional_edges(
            "final_requirement_audit",
            self._route_after_final_checks,
            {"repair": "repair", "finish": END},
        )
        g.add_edge("repair", END)
        return g.compile()

    def _plan(self, state: EvidenceAwareState):
        return {
            "plan": self.planner.run(self.prompt.question),
            "requirement_retrieval_round": 0,
            "claim_retrieval_round": 0,
            "retrieval_events": [],
            "requirement_audit_history": [],
            "candidate_history": [],
            "final_requirement_audit_history": [],
            "repair_performed": False,
        }

    def _complete_requirements(self, state: EvidenceAwareState):
        plan = state["plan"]
        audit = self.requirement_completeness.run(
            self.prompt.question,
            list(plan.answer_requirements),
        )
        updated = plan.model_copy(update={"answer_requirements": audit.final_requirements})
        return {"plan": updated, "requirement_completeness": audit.model_dump()}

    def _route(self, query: str) -> RouteDecision:
        if self.flags.adaptive_routing:
            return self.router.run(query)
        return RouteDecision(
            sources=["web", "academic", "video"],
            max_per_source=self.settings.max_sources_per_tool,
            rationale="Fixed-source routing for ablation/control.",
        )

    def _retrieve(self, state: EvidenceAwareState):
        plan = state["plan"]
        sources: list[SourceDocument] = []
        for subq in plan.subquestions:
            route = self._route(subq)
            sources.extend(
                self.hub.search(
                    subq,
                    route.sources,
                    max_per_source=route.max_per_source,
                )
            )
        sources = dedupe_by(sources, lambda d: d.url or d.source_id)
        evidence = extract_ranked_evidence(
            self.prompt.question + " " + " ".join(plan.answer_requirements),
            sources,
            max_total=self.settings.max_evidence_per_query,
        )
        events = list(state.get("retrieval_events", []))
        events.append(
            {
                "stage": "initial",
                "queries": list(plan.subquestions),
                "source_count_after": len(sources),
                "evidence_count_after": len(evidence),
            }
        )
        return {"sources": sources, "evidence": evidence, "retrieval_events": events}

    def _gap_retrieve(
        self,
        state: EvidenceAwareState,
        queries: list[str],
        focus: str,
        *,
        cap: int,
    ) -> tuple[list[SourceDocument], list[Evidence]]:
        """Run one targeted retrieval round for weak requirements or claims.

        Live mode searches the providers again. Frozen mode cannot fetch new documents, and
        re-running the document ranker returns the same top documents, so the round instead
        ranks passages across the whole frozen pool for each gap query. That is the frozen
        equivalent of a new targeted search.
        """
        existing_sources = state.get("sources", [])
        existing_evidence = state.get("evidence", [])

        if not self.settings.use_frozen_reference_pool:
            new_sources: list[SourceDocument] = []
            for query in queries:
                route = self._route(query)
                new_sources.extend(
                    self.hub.search(query, route.sources, max_per_source=route.max_per_source)
                )
            sources = dedupe_by(existing_sources + new_sources, lambda d: d.url or d.source_id)
            evidence = extract_ranked_evidence(
                self.prompt.question + " " + focus, sources, max_total=cap
            )
            return sources, evidence

        pool = self.hub.frozen.load(self.prompt.prompt_id)
        known = {e.evidence_id for e in existing_evidence}
        gap_evidence: list[Evidence] = []
        for query in queries:
            found = extract_ranked_evidence(
                query, pool, max_total=GAP_PASSAGES_PER_QUERY, per_document=1
            )
            if self.telemetry:
                self.telemetry.add_retrieval(len(found))
            for item in found:
                if item.evidence_id not in known:
                    known.add(item.evidence_id)
                    gap_evidence.append(item)

        # Keep every gap passage; trim the lowest-ranked initial passages if over the cap.
        keep = max(0, cap - len(gap_evidence))
        evidence = existing_evidence[:keep] + gap_evidence[:cap]
        used = {e.source_id for e in gap_evidence}
        sources = dedupe_by(
            existing_sources + [d for d in pool if d.source_id in used],
            lambda d: d.url or d.source_id,
        )
        return sources, evidence

    def _audit_requirements(self, state: EvidenceAwareState):
        assessments = self.requirement_auditor.run(
            self.prompt.question,
            state["plan"].answer_requirements,
            state.get("evidence", []),
        )
        history = list(state.get("requirement_audit_history", []))
        history.append([x.model_dump() for x in assessments])
        return {"requirement_assessments": assessments, "requirement_audit_history": history}

    def _requirement_needs_more(self, assessment: RequirementAssessment) -> bool:
        return requirement_needs_more_evidence(
            assessment, self.settings.requirement_support_threshold
        )

    def _route_after_requirement_audit(self, state: EvidenceAwareState) -> str:
        if not self.flags.requirement_gap_retrieval:
            return "continue"
        rounds = int(state.get("requirement_retrieval_round", 0))
        if rounds >= self.settings.max_requirement_retrieval_rounds:
            return "continue"
        weak = [
            a for a in state.get("requirement_assessments", []) if self._requirement_needs_more(a)
        ]
        return "retrieve_requirement" if weak else "continue"

    def _targeted_requirement_retrieve(self, state: EvidenceAwareState):
        weak = [
            a for a in state.get("requirement_assessments", []) if self._requirement_needs_more(a)
        ][:4]
        if not weak:
            return {
                "requirement_retrieval_round": int(
                    state.get("requirement_retrieval_round", 0)
                )
                + 1
            }

        queries = [a.suggested_query.strip() or a.requirement for a in weak]
        focus = " ".join(a.requirement for a in weak)
        sources, evidence = self._gap_retrieve(
            state, queries, focus, cap=self.settings.max_evidence_per_query + 8
        )
        events = list(state.get("retrieval_events", []))
        events.append(
            {
                "stage": "requirement_gap",
                "trigger_requirements": [a.requirement_id for a in weak],
                "queries": queries,
                "source_count_after": len(sources),
                "evidence_count_after": len(evidence),
            }
        )
        return {
            "sources": sources,
            "evidence": evidence,
            "requirement_retrieval_round": int(state.get("requirement_retrieval_round", 0)) + 1,
            "retrieval_events": events,
        }

    def _claims(self, state: EvidenceAwareState):
        claims = self.claim_extractor.from_evidence(
            self.prompt.question,
            state.get("evidence", []),
            answer_requirements=state["plan"].answer_requirements,
            requirement_assessments=state.get("requirement_assessments", []),
        )
        return {"candidate_claims": claims}

    def _verify(self, state: EvidenceAwareState):
        verifications = self.verifier.run(
            state.get("candidate_claims", []),
            state.get("evidence", []),
            task="p1_candidate_verify",
        )
        graph = ClaimEvidenceGraph()
        graph.add_claims(state.get("candidate_claims", []))
        graph.add_evidence(state.get("evidence", []))
        graph.apply_verifications(verifications)

        history = list(state.get("candidate_history", []))
        history.append(
            {
                "claim_retrieval_round": int(state.get("claim_retrieval_round", 0)),
                "claims": [c.model_dump() for c in state.get("candidate_claims", [])],
                "verifications": [v.model_dump() for v in verifications],
            }
        )
        return {
            "candidate_verifications": verifications,
            "ceg": graph.to_dict(),
            "candidate_history": history,
        }

    def _claim_needs_more(self, verification: ClaimVerification) -> bool:
        return claim_needs_more_evidence(
            verification, self.settings.claim_support_threshold
        )

    def _weak_claims(self, state: EvidenceAwareState) -> list[Claim]:
        claim_map = {c.claim_id: c for c in state.get("candidate_claims", [])}
        weak_pairs = [
            (claim_map[v.claim_id], v)
            for v in state.get("candidate_verifications", [])
            if v.claim_id in claim_map and self._claim_needs_more(v)
        ]
        # Retry core claims first, then supporting claims.
        weak_pairs.sort(key=lambda pair: 0 if pair[0].importance == "core" else 1)
        return [pair[0] for pair in weak_pairs[:3]]

    def _route_after_candidate_verify(self, state: EvidenceAwareState) -> str:
        if not self.flags.claim_gap_retrieval:
            return "continue"
        rounds = int(state.get("claim_retrieval_round", 0))
        if rounds >= self.settings.max_claim_retrieval_rounds:
            return "continue"
        return "retrieve_claim" if self._weak_claims(state) else "continue"

    def _targeted_claim_retrieve(self, state: EvidenceAwareState):
        weak = self._weak_claims(state)
        if not weak:
            return {
                "claim_retrieval_round": int(state.get("claim_retrieval_round", 0)) + 1
            }

        queries = [c.text for c in weak]
        sources, evidence = self._gap_retrieve(
            state,
            queries,
            " ".join(c.text for c in weak),
            cap=self.settings.max_evidence_per_query + 10,
        )
        events = list(state.get("retrieval_events", []))
        events.append(
            {
                "stage": "weak_claim",
                "trigger_claims": [c.claim_id for c in weak],
                "queries": queries,
                "source_count_after": len(sources),
                "evidence_count_after": len(evidence),
            }
        )
        return {
            "sources": sources,
            "evidence": evidence,
            "claim_retrieval_round": int(state.get("claim_retrieval_round", 0)) + 1,
            "retrieval_events": events,
        }

    def _contradictions(self, state: EvidenceAwareState):
        contradictions = (
            self.contradiction.run(self.prompt.question, state.get("evidence", []))
            if self.flags.contradiction_detection
            else []
        )
        return {"contradictions": contradictions}

    def _supported_candidate_claims(self, state: EvidenceAwareState) -> list[Claim]:
        vmap = {v.claim_id: v for v in state.get("candidate_verifications", [])}
        return [
            c
            for c in state.get("candidate_claims", [])
            if vmap.get(c.claim_id)
            and vmap[c.claim_id].status == "supported"
            and vmap[c.claim_id].confidence >= self.settings.claim_support_threshold
        ]

    def _pedagogy(self, state: EvidenceAwareState):
        claims = self._supported_candidate_claims(state)
        return {
            "pedagogy": self.pedagogy.run(
                self.prompt.question,
                claims,
                answer_requirements=state["plan"].answer_requirements,
                requirement_assessments=state.get("requirement_assessments", []),
                contradictions=state.get("contradictions", []),
            )
        }

    def _synthesize(self, state: EvidenceAwareState):
        claims = self._supported_candidate_claims(state)
        evidence_map = {e.evidence_id: e for e in state.get("evidence", [])}
        vmap = {v.claim_id: v for v in state.get("candidate_verifications", [])}
        claim_to_evidence: dict[str, list[Evidence]] = {}
        for claim in claims:
            verification = vmap[claim.claim_id]
            claim_to_evidence[claim.claim_id] = [
                evidence_map[eid] for eid in verification.evidence_ids if eid in evidence_map
            ]
        script = self.synthesizer.from_verified_claims(
            self.prompt.question,
            claims,
            claim_to_evidence,
            state["pedagogy"],
            answer_requirements=state["plan"].answer_requirements,
            requirement_assessments=state.get("requirement_assessments", []),
            contradictions=state.get("contradictions", []),
        )
        return {"script": script}

    def _final_verify(self, state: EvidenceAwareState):
        claims = self.claim_extractor.from_script(state["script"])
        verifications = self.verifier.run(
            claims,
            state.get("evidence", []),
            task="p1_final_verify",
        )
        return {"final_claims": claims, "final_verifications": verifications}

    def _final_requirement_audit(self, state: EvidenceAwareState):
        assessments = self.narration_requirement_auditor.run(
            self.prompt.question,
            state["plan"].answer_requirements,
            state.get("requirement_assessments", []),
            state.get("script", ""),
            support_threshold=self.settings.requirement_support_threshold,
        )
        history = list(state.get("final_requirement_audit_history", []))
        history.append([x.model_dump() for x in assessments])
        return {
            "final_requirement_audit": assessments,
            "final_requirement_audit_history": history,
        }

    def _route_after_final_checks(self, state: EvidenceAwareState) -> str:
        if not self.flags.final_repair:
            return "finish"
        factual_issue = any(
            self._claim_needs_more(v) for v in state.get("final_verifications", [])
        )
        narration_issue = any(
            a.status in {"partial", "omitted"}
            for a in state.get("final_requirement_audit", [])
        )
        return "repair" if factual_issue or narration_issue else "finish"

    def _repair(self, state: EvidenceAwareState):
        repaired = self.repair.whole_script_repair(
            self.prompt.question,
            state["script"],
            state.get("final_claims", []),
            state.get("final_verifications", []),
            state.get("evidence", []),
            requirement_audit=state.get("final_requirement_audit", []),
            contradictions=state.get("contradictions", []),
        )
        claims = self.claim_extractor.from_script(repaired)
        verifications = self.verifier.run(
            claims,
            state.get("evidence", []),
            task="p1_post_repair_verify",
        )
        post_audit = self.narration_requirement_auditor.run(
            self.prompt.question,
            state["plan"].answer_requirements,
            state.get("requirement_assessments", []),
            repaired,
            support_threshold=self.settings.requirement_support_threshold,
        )
        history = list(state.get("final_requirement_audit_history", []))
        history.append([x.model_dump() for x in post_audit])
        return {
            "script": repaired,
            "final_claims": claims,
            "final_verifications": verifications,
            "final_requirement_audit": post_audit,
            "final_requirement_audit_history": history,
            "repair_performed": repaired != state.get("script", ""),
            "pre_repair_script": state.get("script", "") if repaired != state.get("script", "") else "",
        }

    def invoke(self) -> EvidenceAwareState:
        return self.app.invoke({"prompt": self.prompt})
