from __future__ import annotations

from evidence_aware_edu.schemas import Claim, ClaimVerification, QualityEvaluation


def support_counts(verifications: list[ClaimVerification]) -> dict[str, int | float]:
    counts = {"supported": 0, "partial": 0, "unsupported": 0, "contradicted": 0}
    for verification in verifications:
        counts[verification.status] += 1
    total = sum(counts.values())
    return {
        **counts,
        "total": total,
        "supported_claim_rate": counts["supported"] / total if total else 0.0,
    }


def reference_evidence_support_percent(verifications: list[ClaimVerification]) -> float:
    """Score how much of the extracted content is supported by reference evidence."""
    if not verifications:
        return 0.0
    weights = {
        "supported": 1.0,
        "partial": 0.5,
        "unsupported": 0.0,
        "contradicted": 0.0,
    }
    earned = sum(weights[v.status] for v in verifications)
    return 100.0 * earned / len(verifications)


def query_coverage_percent(evaluation: QualityEvaluation) -> float:
    if evaluation.aspect_scores:
        mean_score = sum(x.score for x in evaluation.aspect_scores) / len(evaluation.aspect_scores)
        return 100.0 * mean_score
    return 100.0 * evaluation.coverage_score


def narration_quality_percent(evaluation: QualityEvaluation) -> float:
    """Return the 10-part narration score as a percentage."""
    total = (
        evaluation.opening_orientation_score
        + evaluation.introduction_roadmap_score
        + evaluation.definitions_foundations_score
        + evaluation.explanatory_depth_score
        + evaluation.examples_applications_score
        + evaluation.organization_transitions_score
        + evaluation.clarity_accessibility_score
        + evaluation.spoken_naturalness_score
        + evaluation.conclusion_takeaway_score
        + evaluation.reflective_ending_score
    )
    return 100.0 * total / 40.0


def citation_metrics(
    claims: list[Claim],
    verifications: list[ClaimVerification],
) -> tuple[float, float, float]:
    """Calculate citation precision, recall and F1 using verified cited passages."""
    vmap = {v.claim_id: v for v in verifications}
    total_citations = 0
    supported_citations = 0
    recall_den = 0
    recall_num = 0

    for claim in claims:
        verification = vmap.get(claim.claim_id)
        cited_ids = set(claim.citation_ids)
        total_citations += len(cited_ids)

        supporting_ids: set[str] = set()
        if verification and verification.status == "supported":
            supporting_ids = set(verification.evidence_ids)
            supported_citations += len(cited_ids & supporting_ids)

        if claim.needs_citation:
            recall_den += 1
            if verification and verification.status == "supported" and cited_ids & supporting_ids:
                recall_num += 1

    precision = supported_citations / total_citations if total_citations else 0.0
    recall = recall_num / recall_den if recall_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

