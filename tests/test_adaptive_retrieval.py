from evidence_aware_edu.adaptive_logic import (
    claim_needs_more_evidence,
    requirement_needs_more_evidence,
)
from evidence_aware_edu.schemas import ClaimVerification, RequirementAssessment


def test_missing_requirement_triggers_more_evidence():
    assessment = RequirementAssessment(
        requirement_id="R4",
        requirement="external electron flow",
        status="missing",
        confidence=0.95,
    )
    assert requirement_needs_more_evidence(assessment, 0.70) is True


def test_low_confidence_covered_requirement_triggers_more_evidence():
    assessment = RequirementAssessment(
        requirement_id="R4",
        requirement="external electron flow",
        status="covered",
        confidence=0.52,
    )
    assert requirement_needs_more_evidence(assessment, 0.70) is True


def test_low_confidence_supported_claim_triggers_more_evidence():
    verification = ClaimVerification(
        claim_id="C1",
        status="supported",
        confidence=0.52,
    )
    assert claim_needs_more_evidence(verification, 0.70) is True


def test_high_confidence_supported_claim_does_not_trigger():
    verification = ClaimVerification(
        claim_id="C1",
        status="supported",
        confidence=0.90,
    )
    assert claim_needs_more_evidence(verification, 0.70) is False
