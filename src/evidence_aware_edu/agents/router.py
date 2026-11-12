from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import RouteDecision

MIN_SOURCE_TYPES = 2
SOURCE_PREFERENCE = ("web", "academic", "video")


class RouterAgent(AgentBase):
    SYSTEM = """You are an adaptive evidence router. Select the smallest useful subset of source types.
Allowed source types: web, academic, video. Academic is preferred for scientific/technical evidence.
Web is preferred for current facts and official/general sources. Video is optional for tutorial or spoken-explanation context.
Avoid selecting all three unless the question genuinely benefits from source triangulation.
Return structured output only."""

    def run(self, query: str) -> RouteDecision:
        result, log = self.llm.generate_structured(
            task="adaptive_route",
            system_instruction=self.SYSTEM,
            prompt=f"Question/sub-question: {query}\nChoose source types and retrieval budget.",
            schema=RouteDecision,
        )
        self._record(log)
        # Remove duplicate source types from the model output.
        result.sources = list(dict.fromkeys(result.sources))
        # A single source type starved P1 of evidence compared with the fixed multi-source
        # baselines, so always triangulate across at least MIN_SOURCE_TYPES types.
        for source_type in SOURCE_PREFERENCE:
            if len(result.sources) >= MIN_SOURCE_TYPES:
                break
            if source_type not in result.sources:
                result.sources.append(source_type)
        return result
