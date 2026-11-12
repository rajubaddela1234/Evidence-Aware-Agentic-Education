from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.prompts import GROUNDED_EDUCATIONAL_SYSTEM
from evidence_aware_edu.schemas import Claim, Contradiction, Evidence, PedagogyPlan, RequirementAssessment


class SynthesizerAgent(AgentBase):
    SYSTEM = GROUNDED_EDUCATIONAL_SYSTEM

    def from_context(self, query: str, context: str, task: str = "synthesize") -> str:
        text, log = self.llm.generate_text(
            task=task,
            system_instruction=self.SYSTEM,
            prompt=(
                f"USER QUESTION:\n{query}\n\n"
                f"GROUNDING CONTEXT:\n{context}\n\n"
                "Write one complete spoken educational narration. The learner should hear a "
                "natural welcome, a concise introduction and roadmap, definitions/foundations, "
                "a detailed explanatory middle, at least one useful example/application when "
                "the context supports it, relevant nuance or limitations, a synthesized "
                "conclusion, and exactly one or two reflective questions at the end. Do not "
                "print section headings; make the structure clear through natural transitions."
            ),
            max_output_tokens=4500,
        )
        self._record(log)
        return text

    def from_verified_claims(
        self,
        query: str,
        claims: list[Claim],
        claim_to_evidence: dict[str, list[Evidence]],
        pedagogy: PedagogyPlan,
        answer_requirements: list[str] | None = None,
        requirement_assessments: list[RequirementAssessment] | None = None,
        contradictions: list[Contradiction] | None = None,
    ) -> str:
        blocks: list[str] = []
        for claim in claims:
            evs = claim_to_evidence.get(claim.claim_id, [])
            markers = ", ".join(f"[{e.evidence_id}]" for e in evs) or "NONE"
            blocks.append(
                f"CLAIM {claim.claim_id}: {claim.text}\n"
                f"ALLOWED EVIDENCE MARKERS: {markers}"
            )

        requirements = "\n".join(
            f"- {r}" for r in (answer_requirements or [])
        ) or "- NONE PROVIDED"
        assessments = "\n".join(
            f"- {a.requirement_id}: {a.requirement} => {a.status} ({a.confidence:.2f})"
            for a in (requirement_assessments or [])
        ) or "- NONE"
        contradiction_text = "\n".join(
            f"- {c.severity.upper()}: {c.description} | evidence={','.join(c.evidence_ids)}"
            for c in (contradictions or [])
        ) or "- NONE DETECTED"

        prompt = (
            f"USER QUESTION:\n{query}\n\n"
            f"ANSWER REQUIREMENTS:\n{requirements}\n\n"
            f"REQUIREMENT EVIDENCE STATUS:\n{assessments}\n\n"
            f"EVIDENCE DISAGREEMENTS:\n{contradiction_text}\n\n"
            f"PEDAGOGY / NARRATION PLAN:\n{pedagogy.model_dump_json(indent=2)}\n\n"
            "VERIFIED FACTUAL CLAIMS:\n"
            "Use only the factual claims below for facts. Do not invent additional facts. You may "
            "add a simple everyday analogy that introduces no new factual claim (see EXAMPLES AND "
            "ANALOGIES). Connect the claims into explanation: say how one leads to the next.\n\n"
            + "\n\n".join(blocks)
            + "\n\nWrite one complete continuous spoken narration, not an outline. Follow the pedagogy "
            "plan closely: welcome and orient the learner; introduce the topic and roadmap; "
            "define the central concepts; develop the detailed middle step by step; use a "
            "supported example/application when available; explain relevant limitations or "
            "misconceptions; synthesize the main takeaway in the conclusion; and end with "
            "exactly one or two reflective questions. Do not print section headings. Explicitly "
            "cover every answer requirement that has adequate evidence. If a requirement "
            "remains unresolved, mention it in one short natural sentence to the learner, without "
            "referring to 'the evidence' or 'the sources', rather than silently omitting it or "
            "inventing an answer.\n\n"
            "CONTRADICTION RULE: if an evidence disagreement is listed above, it MUST influence "
            "the narration. Do not present the disputed proposition as settled. Either qualify "
            "the statement, explain the disagreement using only the supplied verified material, "
            "or omit the disputed detail when it is not essential. Never invent a resolution."
        )
        text, log = self.llm.generate_text(
            task="p1_synthesis",
            system_instruction=self.SYSTEM,
            prompt=prompt,
            max_output_tokens=4500,
        )
        self._record(log)
        return text
