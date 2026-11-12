from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import (
    Claim,
    ClaimVerification,
    Contradiction,
    Evidence,
    NarrationRequirementAssessment,
)
from evidence_aware_edu.utils import clip


class RepairAgent(AgentBase):
    SYSTEM = """You repair an evidence-grounded educational narration using only supplied evidence.

FACTUAL RULES
- Remove, weaken, or rewrite claims identified as partial, unsupported, contradicted, or
  supported with verification confidence below the configured threshold.
- Preserve supported factual material and valid evidence markers.
- Do not introduce new factual claims.
- Never invent evidence IDs.
- If evidence is insufficient, remove the unsupported factual assertion or state the limitation
  cautiously.

REQUIREMENT-COVERAGE RULES
- When FINAL REQUIREMENT AUDIT marks a requirement partial or omitted and the supplied evidence is
  sufficient, add or strengthen the missing explanation using only supported material already in
  the narration/evidence.
- When evidence is genuinely unresolved, preserve an honest limitation statement instead of
  fabricating completion.
- Do not silently remove a requirement while repairing another sentence.

CONTRADICTION RULES
- EVIDENCE DISAGREEMENTS must affect the repaired narration.
- Do not present a disputed proposition as settled when supplied evidence conflicts.
- Qualify, acknowledge, or omit disputed detail using only the supplied information.
- Never invent which source is correct.

NARRATION-PRESERVATION RULES
The repaired result must remain a complete spoken educational narration, not a set of corrected
sentences. Keep it sounding natural when read aloud: write evidence markers back to back with no
commas between them (for example [E-1][E-2]), do not leave note-like fragments, and do not talk
about "the evidence" or "the sources" to the learner. Preserve or restore this flow:
1. natural welcome and topic orientation;
2. concise introduction and roadmap;
3. definitions and foundations before advanced explanation;
4. detailed explanatory middle with logical progression;
5. a useful evidence-supported example/application when available;
6. relevant nuance, limitation, or misconception where appropriate;
7. synthesized conclusion and takeaway;
8. exactly one or two reflective questions at the end.

Do not print section headings. Use natural transitions. Repairing one claim must not destroy the
coherence of neighbouring sentences or remove important supported explanations.

STYLE
- Sound like a knowledgeable instructor speaking to a learner.
- Use clear university-level English.
- Avoid unnecessary repetition, filler, and unrelated material.
- Do not mention verification, repair, prompts, or these instructions.
"""

    def whole_script_repair(
        self,
        query: str,
        script: str,
        claims: list[Claim],
        verifications: list[ClaimVerification],
        evidence: list[Evidence],
        requirement_audit: list[NarrationRequirementAssessment] | None = None,
        contradictions: list[Contradiction] | None = None,
    ) -> str:
        bad = {
            v.claim_id: v
            for v in verifications
            if v.status in {"unsupported", "contradicted", "partial"}
            or v.confidence < self.llm.settings.claim_support_threshold
        }
        coverage_issues = [
            a for a in (requirement_audit or []) if a.status in {"partial", "omitted"}
        ]
        contradiction_items = list(contradictions or [])
        if not bad and not coverage_issues:
            return script

        claim_map = {c.claim_id: c for c in claims}
        issue_text = "\n".join(
            f"- {cid}: {claim_map[cid].text} => {v.status}: {v.rationale}"
            for cid, v in bad.items()
            if cid in claim_map
        ) or "- NONE"
        coverage_text = "\n".join(
            f"- {a.requirement_id}: {a.requirement} => {a.status}. "
            f"Repair guidance: {a.suggested_repair or a.rationale}"
            for a in coverage_issues
        ) or "- NONE"
        contradiction_text = "\n".join(
            f"- {c.severity.upper()}: {c.description} | evidence={','.join(c.evidence_ids)}"
            for c in contradiction_items
        ) or "- NONE"
        ev_text = "\n".join(
            f"[{e.evidence_id}] {e.title}: {clip(e.passage, 1100)}" for e in evidence[:24]
        )
        text, log = self.llm.generate_text(
            task="script_repair",
            system_instruction=self.SYSTEM,
            prompt=(
                f"USER QUESTION:\n{query}\n\n"
                f"ORIGINAL NARRATION:\n{script}\n\n"
                f"CLAIM VERIFICATION ISSUES:\n{issue_text}\n\n"
                f"FINAL REQUIREMENT AUDIT ISSUES:\n{coverage_text}\n\n"
                f"EVIDENCE DISAGREEMENTS:\n{contradiction_text}\n\n"
                f"ALLOWED EVIDENCE:\n{ev_text}\n\n"
                "Return the repaired complete spoken narration. Preserve the educational flow, "
                "repair answerable omitted requirements, qualify genuine evidence disagreements, "
                "and end with one or two reflective questions."
            ),
            max_output_tokens=4500,
        )
        self._record(log)
        return text
