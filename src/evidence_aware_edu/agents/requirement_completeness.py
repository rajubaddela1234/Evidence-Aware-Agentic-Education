from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import RequirementCompletenessResult
from evidence_aware_edu.utils import dedupe_by


class RequirementCompletenessAgent(AgentBase):
    """Check whether the planner missed an important part of the question."""

    SYSTEM = """You are a requirement-completeness auditor for an educational question-answering system.
Do NOT answer the question and do NOT use retrieved evidence. Inspect only the USER QUESTION and
PROPOSED ANSWER REQUIREMENTS.

Your task is to decide whether the proposed requirements fully represent what a good answer must
explain. Preserve requirements that are useful, merge only obvious duplicates, and add missing
obligations that are clearly implied by the wording of the question.

Important rules:
- Never use or invent a benchmark/evaluation checklist.
- Derive requirements only from the user's question.
- Do not add optional trivia merely to make the list longer.
- For mechanism questions, check components, direction/sequence, transfers/flows, and the requested
  outcome.
- For comparison questions, check both compared items and every criterion explicitly requested.
- For causal questions, check the causal chain and any requested consequences/conditions.
- For questions containing "and", ensure both sides are represented.
- Return 3-10 concise, auditable requirements.
- added_requirements must contain only requirements that were absent from the proposed list.
Return structured output only."""

    def run(self, question: str, proposed_requirements: list[str]) -> RequirementCompletenessResult:
        proposed = "\n".join(f"- {x}" for x in proposed_requirements)
        result, log = self.llm.generate_structured(
            task="requirement_completeness_audit",
            system_instruction=self.SYSTEM,
            prompt=(
                f"USER QUESTION:\n{question}\n\n"
                f"PROPOSED ANSWER REQUIREMENTS:\n{proposed}\n\n"
                "Return the corrected complete requirement list and identify what you added."
            ),
            schema=RequirementCompletenessResult,
            max_output_tokens=1800,
        )
        self._record(log)

        # Clean up duplicates without adding new requirements in code.
        final = dedupe_by(result.final_requirements, lambda x: x)
        if len(final) < 3:
            final = dedupe_by(proposed_requirements + final, lambda x: x)
        result.final_requirements = final[:10]
        final_norm = {x.strip().lower() for x in result.final_requirements}
        proposed_norm = {x.strip().lower() for x in proposed_requirements}
        result.added_requirements = [
            x for x in result.final_requirements if x.strip().lower() not in proposed_norm
        ]
        # Fall back to the planner list if the checker returns nothing useful.
        assert final_norm
        return result
