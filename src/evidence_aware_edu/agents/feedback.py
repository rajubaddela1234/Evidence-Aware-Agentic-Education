from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import FeedbackResult


class FeedbackAgent(AgentBase):
    SYSTEM = """You are the feedback agent in an educational multi-agent pipeline.
Evaluate the supplied script only against the supplied research context and the user question.
Do not use external knowledge.

Consider:
- factual grounding in the supplied context,
- whether the question is answered completely,
- clear introduction and topic framing,
- definitions/foundations before advanced explanation,
- depth and logical progression in the explanatory middle,
- useful supported examples/applications when appropriate,
- smooth spoken transitions,
- a concise conclusion/takeaway,
- one or two relevant reflective questions at the end.

The overall score is 0-10. Mark revision_needed when score is below the stated threshold.
Return structured output only.
"""

    def run(self, query: str, script: str, context: str, threshold: float) -> FeedbackResult:
        result, log = self.llm.generate_structured(
            task="original_feedback",
            system_instruction=self.SYSTEM,
            prompt=(
                f"Question: {query}\nThreshold: {threshold}\n\n"
                f"Research context:\n{context}\n\nScript:\n{script}"
            ),
            schema=FeedbackResult,
        )
        result.revision_needed = result.score < threshold
        self._record(log)
        return result
