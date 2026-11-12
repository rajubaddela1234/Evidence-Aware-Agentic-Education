from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import ClaimSet, Evidence, RequirementAssessment
from evidence_aware_edu.utils import clip


class ClaimExtractorAgent(AgentBase):
    SYSTEM_FROM_EVIDENCE = """You extract atomic factual claims for an evidence-grounded educational script.
Every claim must be independently checkable and directly supported by the supplied evidence.
Do not invent bridging facts. Aim for 15-30 useful claims rather than exhaustive trivia, with at
least two or three claims for every requirement the evidence covers, including definitions and
foundations, mechanism steps and any concrete examples or applications found in the evidence.
Set importance=core for claims essential to answering the user.

When ANSWER REQUIREMENTS are supplied, make claim extraction requirement-aware:
- for each requirement whose evidence status is covered, extract enough directly supported core
  claims to explain that requirement when the evidence permits;
- for partial/missing/conflicting requirements, do not invent a claim merely to fill the gap;
- do not select only the easiest supported facts while ignoring a covered requirement.
Return structured output only."""

    SYSTEM_FROM_SCRIPT = """You decompose an educational script into atomic factual claims for evaluation.
Ignore purely rhetorical, introductory, transition, and opinion sentences. Each factual proposition should be independently checkable.
If the script places evidence markers such as [E-abc] after a sentence, copy those markers into citation_ids for that claim.
Do not judge the claims. Return structured output only."""

    @staticmethod
    def _evidence_text(evidence: list[Evidence], max_chars: int = 26000) -> str:
        parts: list[str] = []
        used = 0
        for e in evidence:
            item = f"[{e.evidence_id}] {e.title}\n{clip(e.passage, 1800)}\n"
            if used + len(item) > max_chars:
                break
            parts.append(item)
            used += len(item)
        return "\n".join(parts)

    def from_evidence(
        self,
        query: str,
        evidence: list[Evidence],
        answer_requirements: list[str] | None = None,
        requirement_assessments: list[RequirementAssessment] | None = None,
    ) -> list:
        req_text = "\n".join(
            f"R{i}: {r}" for i, r in enumerate(answer_requirements or [], start=1)
        ) or "NONE"
        assessment_text = "\n".join(
            f"{a.requirement_id}: {a.status} ({a.confidence:.2f})"
            for a in (requirement_assessments or [])
        ) or "NONE"
        result, log = self.llm.generate_structured(
            task="candidate_claims",
            system_instruction=self.SYSTEM_FROM_EVIDENCE,
            prompt=(
                f"Question:\n{query}\n\n"
                f"ANSWER REQUIREMENTS:\n{req_text}\n\n"
                f"REQUIREMENT EVIDENCE STATUS:\n{assessment_text}\n\n"
                f"Evidence:\n{self._evidence_text(evidence)}"
            ),
            schema=ClaimSet,
        )
        self._record(log)
        return result.claims

    def from_script(self, script: str) -> list:
        result, log = self.llm.generate_structured(
            task="script_claims",
            system_instruction=self.SYSTEM_FROM_SCRIPT,
            prompt=script,
            schema=ClaimSet,
        )
        self._record(log)
        return result.claims
