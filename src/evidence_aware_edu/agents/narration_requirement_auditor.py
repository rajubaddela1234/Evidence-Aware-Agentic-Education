from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import (
    NarrationRequirementAssessment,
    NarrationRequirementAuditBundle,
    RequirementAssessment,
)
from evidence_aware_edu.utils import clip


class NarrationRequirementAuditorAgent(AgentBase):
    """Audit whether the final narration actually uses the available requirement evidence."""

    SYSTEM = """You audit a finished educational narration against explicit answer requirements.
Do not use outside knowledge. The requirement-evidence status supplied to you describes whether the
system had enough retrieved evidence for each requirement. Your job is to determine what the final
narration did with those requirements.

Return exactly one assessment for every requirement using these statuses:
- answered_with_evidence: the narration substantively explains the requirement AND the supplied
  requirement-evidence status is covered with adequate confidence.
- partial: the narration mentions/explains only part of the requirement, is too vague, or presents a
  substantive answer when the supplied requirement evidence was only partial/conflicting.
- unresolved_acknowledged: the narration explicitly and appropriately says that the available
  evidence is insufficient/uncertain for that requirement rather than inventing an answer.
- omitted: the narration silently fails to address the requirement.

A keyword mention is not a substantive answer. Provide a short exact or near-exact script excerpt
when possible. suggested_repair should state how to repair partial/omitted items using only already
available supported material; leave it empty when the evidence itself is unresolved and the script
already acknowledges that limitation. Confidence is confidence in this audit judgement.
Return structured output only."""

    def run(
        self,
        question: str,
        requirements: list[str],
        evidence_assessments: list[RequirementAssessment],
        script: str,
        support_threshold: float,
    ) -> list[NarrationRequirementAssessment]:
        if not requirements:
            return []
        evidence_by_id = {a.requirement_id.upper(): a for a in evidence_assessments}
        status_lines = []
        for i, requirement in enumerate(requirements, start=1):
            rid = f"R{i}"
            a = evidence_by_id.get(rid)
            if a is None:
                status_lines.append(f"{rid}: {requirement} => missing (confidence 0.00)")
            else:
                status_lines.append(
                    f"{rid}: {requirement} => {a.status} (confidence {a.confidence:.2f})"
                )
        prompt = (
            f"USER QUESTION:\n{question}\n\n"
            "REQUIREMENTS AND RETRIEVED-EVIDENCE SUFFICIENCY:\n"
            + "\n".join(status_lines)
            + f"\n\nADEQUATE-EVIDENCE CONFIDENCE THRESHOLD: {support_threshold:.2f}\n\n"
            f"FINAL NARRATION:\n{clip(script, 30000)}"
        )
        result, log = self.llm.generate_structured(
            task="final_narration_requirement_audit",
            system_instruction=self.SYSTEM,
            prompt=prompt,
            schema=NarrationRequirementAuditBundle,
            max_output_tokens=2600,
        )
        self._record(log)

        # Keep one result per requirement and treat missing results conservatively.
        grouped: dict[str, list[NarrationRequirementAssessment]] = {}
        for item in result.assessments:
            grouped.setdefault(item.requirement_id.upper(), []).append(item)

        normalized: list[NarrationRequirementAssessment] = []
        severity = {
            "omitted": 3,
            "partial": 2,
            "unresolved_acknowledged": 1,
            "answered_with_evidence": 0,
        }
        for i, requirement in enumerate(requirements, start=1):
            rid = f"R{i}"
            options = grouped.get(rid, [])
            if not options:
                item = NarrationRequirementAssessment(
                    requirement_id=rid,
                    requirement=requirement,
                    status="omitted",
                    confidence=0.0,
                    rationale="No narration audit was returned; treated conservatively as omitted.",
                    suggested_repair=f"Explain this requirement explicitly: {requirement}",
                )
            else:
                item = max(options, key=lambda x: severity[x.status])
                item.requirement_id = rid
                item.requirement = requirement
            normalized.append(item)
        return normalized
