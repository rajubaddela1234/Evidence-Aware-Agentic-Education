from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import SubQuestionSet


class DecomposerAgent(AgentBase):
    SYSTEM = """You are the question decomposer from a multi-agent research system.
Split a query into 1-3 semantically distinct sub-questions. Keep closely related concepts together.
Do not answer. Return only structured output."""

    def run(self, query: str) -> list[str]:
        result, log = self.llm.generate_structured(
            task="decompose",
            system_instruction=self.SYSTEM,
            prompt=query,
            schema=SubQuestionSet,
        )
        self._record(log)
        return result.subquestions
