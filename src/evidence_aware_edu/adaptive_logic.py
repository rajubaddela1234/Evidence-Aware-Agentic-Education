from __future__ import annotations

from evidence_aware_edu.schemas import ClaimVerification, RequirementAssessment


def requirement_needs_more_evidence(
    assessment: RequirementAssessment,
    threshold: float,
) -> bool:
    return assessment.status != "covered" or assessment.confidence < threshold


def claim_needs_more_evidence(
    verification: ClaimVerification,
    threshold: float,
) -> bool:
    return verification.status != "supported" or verification.confidence < threshold
