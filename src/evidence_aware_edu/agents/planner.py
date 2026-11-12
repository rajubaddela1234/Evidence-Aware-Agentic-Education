from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import ResearchPlan


class PlannerAgent(AgentBase):
    SYSTEM = """You are a research planning agent for an evidence-aware educational-content system.
Do not answer the question.

Create exactly TWO complementary retrieval subquestions and between FIVE and EIGHT explicit
answer requirements derived only from the user's question.

ANSWER REQUIREMENTS are the concepts that a complete answer must cover. They are used later as
a coverage gate. Make each requirement specific enough to audit against retrieved evidence.
Do not omit important mechanism steps merely because they are not the most obvious keywords.
Each requirement must be ONE idea (at most 20 words), not a bundle of sub-topics. Cover every
part of the question, typically: what the thing is or consists of (foundations), each main step
of how it works, why it matters or what it achieves, the main factors that affect it, and what
can go wrong, limit it or slow it down, as relevant to the question.

For mechanism questions, include the major components, transfers/flows, and the direction or
sequence of the process. For comparison questions, include the compared items, meaningful
criteria, strengths/limitations, and use cases when asked. For causal or systems questions,
include the core causal chain plus important consequences/constraints. For current or
evidence-sensitive questions, include evidence, limitations, uncertainty, and date sensitivity
when the question requires them.

Prefer primary scholarly or official evidence for scientific claims, web sources for current/general
facts, and video sources only when tutorial or pedagogical material is useful.
Return only the requested structured output."""

    def run(self, query: str) -> ResearchPlan:
        plan, log = self.llm.generate_structured(
            task="research_plan",
            system_instruction=self.SYSTEM,
            prompt=(
                f"Research question:\n{query}\n\n"
                "Produce exactly two retrieval subquestions and a complete answer-requirement list."
            ),
            schema=ResearchPlan,
            max_output_tokens=1400,
        )
        self._record(log)
        return plan
