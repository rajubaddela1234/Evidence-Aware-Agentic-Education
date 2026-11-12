from __future__ import annotations

from evidence_aware_edu.agents.common import AgentBase
from evidence_aware_edu.schemas import Evidence, RequirementAssessment, RequirementSufficiencyBundle
from evidence_aware_edu.utils import clip


class RequirementAuditorAgent(AgentBase):
    SYSTEM = """You audit whether retrieved evidence is sufficient to answer explicit requirements.
You are NOT judging a generated script and you are NOT allowed to use outside knowledge.
Judge only whether the supplied evidence contains enough information to answer each requirement.

For every requirement return exactly one assessment:
- covered: evidence directly and adequately addresses the requirement.
- partial: evidence addresses some but not all important content.
- missing: useful evidence for the requirement is absent or too weak.
- conflicting: supplied evidence materially disagrees on the requirement.

Confidence is confidence in the sufficiency judgement, not confidence that a proposition is true.
For partial, missing, conflicting, or low-confidence cases, provide a short targeted retrieval query
that would likely obtain the missing/clarifying evidence. For covered items, suggested_query may be empty.
Return only exact evidence IDs supplied in the prompt; never invent an ID. Return exactly one
assessment for every R-number. Structured output only."""

    @staticmethod
    def _prompt(question: str, requirements: list[str], evidence: list[Evidence], max_chars: int) -> str:
        reqs = "\n".join(f"R{i}: {req}" for i, req in enumerate(requirements, start=1))
        parts: list[str] = []
        used = 0
        for e in evidence:
            item = f"[{e.evidence_id}] {e.title}\n{clip(e.passage, 1800)}\n"
            if used + len(item) > max_chars:
                break
            parts.append(item)
            used += len(item)
        return (
            f"USER QUESTION:\n{question}\n\nANSWER REQUIREMENTS:\n{reqs}\n\n"
            f"RETRIEVED EVIDENCE:\n{''.join(parts)}"
        )

    def run(self, question: str, requirements: list[str], evidence: list[Evidence]):
        if not requirements:
            return []
        result, log = self.llm.generate_structured(
            task="requirement_sufficiency_audit",
            system_instruction=self.SYSTEM,
            prompt=self._prompt(
                question,
                requirements,
                evidence,
                max_chars=self.llm.settings.max_context_chars,
            ),
            schema=RequirementSufficiencyBundle,
            max_output_tokens=2200,
        )
        self._record(log)

        valid_evidence_ids = {e.evidence_id for e in evidence}
        grouped: dict[str, list[RequirementAssessment]] = {}
        for item in result.assessments:
            grouped.setdefault(item.requirement_id.upper(), []).append(item)
        severity = {"covered": 0, "partial": 1, "conflicting": 2, "missing": 3}

        normalized = []
        for i, requirement in enumerate(requirements, start=1):
            rid = f"R{i}"
            options = grouped.get(rid, [])
            if not options:
                assessment = RequirementAssessment(
                    requirement_id=rid,
                    requirement=requirement,
                    status="missing",
                    confidence=0.0,
                    suggested_query=requirement,
                    rationale="No assessment was returned; treated conservatively as missing.",
                )
            else:
                assessment = max(options, key=lambda x: (severity[x.status], -x.confidence))
                assessment.requirement_id = rid
                assessment.requirement = requirement
                assessment.evidence_ids = list(
                    dict.fromkeys(
                        eid for eid in assessment.evidence_ids if eid in valid_evidence_ids
                    )
                )
                if assessment.status in {"covered", "partial", "conflicting"} and not assessment.evidence_ids:
                    assessment = RequirementAssessment(
                        requirement_id=rid,
                        requirement=requirement,
                        status="missing",
                        confidence=0.0,
                        evidence_ids=[],
                        suggested_query=assessment.suggested_query or requirement,
                        rationale=(
                            "The auditor returned no valid supplied evidence IDs for a non-missing "
                            "decision; treated conservatively as missing."
                        ),
                    )
            normalized.append(assessment)
        return normalized
