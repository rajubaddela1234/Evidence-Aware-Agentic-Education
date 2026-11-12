from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import Contradiction, ContradictionBundle, Evidence
from evidence_aware_edu.utils import clip


class ContradictionAgent(AgentBase):
    SYSTEM = """You detect meaningful contradictions between evidence passages relevant to the research question.
Report only substantive factual disagreements, not differences in wording, scope, date, or emphasis.
When evidence reflects different time periods, identify that rather than calling it a contradiction.
Every evidence ID you return must be copied exactly from the supplied evidence. A contradiction must
reference at least two supplied evidence passages. Return structured output only."""

    def run(self, query: str, evidence: list[Evidence]) -> list[Contradiction]:
        if len(evidence) < 2:
            return []
        evidence_text = "\n\n".join(
            f"[{e.evidence_id}] {e.title}: {clip(e.passage, 1200)}" for e in evidence[:20]
        )
        result, log = self.llm.generate_structured(
            task="contradiction_detection",
            system_instruction=self.SYSTEM,
            prompt=f"Question: {query}\n\nEvidence:\n{evidence_text}",
            schema=ContradictionBundle,
        )
        self._record(log)

        valid = {e.evidence_id for e in evidence}
        normalized: list[Contradiction] = []
        for item in result.contradictions:
            ids = []
            for eid in item.evidence_ids:
                if eid in valid and eid not in ids:
                    ids.append(eid)
            if len(ids) < 2:
                continue
            item.evidence_ids = ids
            normalized.append(item)
        return normalized
