from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import Claim, Contradiction, PedagogyPlan, RequirementAssessment


class PedagogyAgent(AgentBase):
    SYSTEM = """You are a university-level educational narration planner.
Your job is to organize already verified factual claims into a teaching sequence before the final
script is written. You must not introduce new factual claims.

Use the USER QUESTION and ANSWER REQUIREMENTS as the completeness target. Use only the VERIFIED
FACTS for factual content. If a required point remains unresolved because evidence is missing,
plan a brief limitation statement instead of inventing information.

EVIDENCE DISAGREEMENTS are operational constraints, not decorative diagnostics. When meaningful
contradictions are supplied:
- do not plan the disputed proposition as an uncontested fact;
- qualify the explanation or explicitly acknowledge the disagreement when it is relevant;
- avoid choosing a winner unless the supplied verified facts themselves resolve the disagreement;
- include important disagreement in nuance_or_limitations or misconceptions_to_avoid.

Build a PedagogyPlan with these functions:
- learning_objectives: what the learner should understand by the end.
- opening_orientation: a concise way to welcome the learner and frame the topic.
- introduction_points: 2-4 ideas that establish importance and preview the explanation.
- key_definitions: central terms or concepts that must be defined before deeper reasoning.
- outline: the detailed explanatory middle in logical teaching order, foundations first.
- example_opportunities: concrete examples/applications/analogies supported by the verified facts;
  leave empty when the facts do not support a useful example.
- nuance_or_limitations: uncertainty, limitations, failure modes, or evidence disagreements worth noting.
- transition_plan: short descriptions of how major sections should connect.
- conclusion_takeaways: the 3-5 ideas that should be synthesized at the end without adding facts.
- reflective_questions: exactly one or two reasoning/application questions grounded in the topic.
- misconceptions_to_avoid: oversimplifications the narration should avoid.

The explanatory middle should receive the greatest attention. Do not merely reorder claims; plan
how the learner moves from definition to mechanism, example/application, nuance, and conclusion.
Return structured output only.
"""

    def run(
        self,
        query: str,
        verified_claims: list[Claim],
        answer_requirements: list[str] | None = None,
        requirement_assessments: list[RequirementAssessment] | None = None,
        contradictions: list[Contradiction] | None = None,
    ) -> PedagogyPlan:
        claims = "\n".join(f"- {c.text}" for c in verified_claims) or "- NONE"
        requirements = "\n".join(
            f"- {r}" for r in (answer_requirements or [])
        ) or "- NONE PROVIDED"
        status_text = "\n".join(
            f"- {a.requirement_id}: {a.requirement} => {a.status} ({a.confidence:.2f})"
            for a in (requirement_assessments or [])
        ) or "- NONE"
        contradiction_text = "\n".join(
            f"- {c.severity.upper()}: {c.description} | evidence={','.join(c.evidence_ids)}"
            for c in (contradictions or [])
        ) or "- NONE DETECTED"
        result, log = self.llm.generate_structured(
            task="pedagogy_plan",
            system_instruction=self.SYSTEM,
            prompt=(
                f"USER QUESTION:\n{query}\n\n"
                f"ANSWER REQUIREMENTS:\n{requirements}\n\n"
                f"REQUIREMENT EVIDENCE STATUS:\n{status_text}\n\n"
                f"EVIDENCE DISAGREEMENTS:\n{contradiction_text}\n\n"
                f"VERIFIED FACTS:\n{claims}"
            ),
            schema=PedagogyPlan,
            max_output_tokens=2400,
        )
        self._record(log)
        return result
