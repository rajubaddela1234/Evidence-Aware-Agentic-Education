"""Five extra automated research measurements.

Retrieval metrics use the evidence actually supplied to the generator. Grounded
coverage uses a separate frozen-reference excerpt set for a fair comparison.
All labels are Claude judgments; none is an external ground-truth annotation.
"""
from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

from evidence_aware_edu.evaluation.rubric import generate_with_rubric, invalid_aspect_scores
from evidence_aware_edu.schemas import AspectCoverage, Evidence, QualityEvaluation


class AspectEvidenceRating(BaseModel):
    aspect: str
    score: float = Field(ge=0, le=1)
    rationale: str = ""


class PassageRelevanceRating(BaseModel):
    evidence_id: str
    relevant: bool


class EvidenceAssessment(BaseModel):
    # Evidence support for each checklist aspect *as explained in the script*.
    grounded_aspects: list[AspectEvidenceRating] = Field(default_factory=list)
    # Ranked passages are checked in their original order, before truncation to K.
    passage_relevance: list[PassageRelevanceRating] = Field(default_factory=list)
    # Can the retrieved context itself answer each checklist aspect?
    retrieved_aspects: list[AspectEvidenceRating] = Field(default_factory=list)


# Prompt budget (characters) before evidence is split into batches; the script is never cut.
DEFAULT_JUDGE_PROMPT_CHARS = 60000
MIN_EVIDENCE_BATCH_CHARS = 2000

SYSTEM_PROMPT = """You are an independent evaluator of educational-script evidence.
Use only the provided evidence extracts. Do not rely on model memory or external facts.
For every EXPECTED ASPECT return:
1. grounded_aspects: How strongly the REFERENCE evidence directly supports the script's
   explanation of this aspect? 0=not supported or contradicted, 0.5=partly supported,
   1=directly and clearly supported. A missing explanation gets 0.
2. retrieved_aspects: How completely the GENERATOR'S RETRIEVED PASSAGES alone provide the
   facts necessary to explain this aspect? 0=not covered, 0.5=partial, 1=complete.
For every ranked generator passage return a passage_relevance judgment. Relevant means
useful for answering the original QUESTION, not merely sharing a keyword.
Aspect scores must be exactly 0, 0.5 or 1; no other value is accepted.
The SCRIPT is always complete. Evidence extracts may arrive in several batches: judge only the
extracts supplied in this request, and score an aspect 0 when this batch has nothing for it.
Return every aspect/passage exactly once, preserving its provided identifier and wording.
If no retrieved passages exist, return no passage_relevance rows and score retrieved_aspects 0.
Never invent a citation or give credit for content absent from supplied excerpts.
"""


def _format_passages(items: list[Evidence]) -> str:
    """Show every passage in full.

    Passages are the same ~1,200-character extracts the generator received. The old
    450-character cut hid explanations that appear later in a passage, so grounded coverage and
    context recall were judged on less text than the generator saw. Length is handled by
    batching, never by truncation.
    """
    if not items:
        return "(none)"
    return "\n\n".join(f"[{item.evidence_id}] {item.passage}" for item in items)


def _chunk_passages(items: list[Evidence], budget: int) -> list[list[Evidence]]:
    """Split ranked passages into order-preserving batches of at most ``budget`` characters."""
    chunks: list[list[Evidence]] = []
    current: list[Evidence] = []
    used = 0
    for item in items:
        size = len(_format_passages([item])) + 2
        if current and used + size > budget:
            chunks.append(current)
            current, used = [], 0
        current.append(item)
        used += size
    if current:
        chunks.append(current)
    return chunks


def _scores_by_aspect(items: list[AspectEvidenceRating], expected: list[str]) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for item in items:
        grouped.setdefault(item.aspect.strip().casefold(), []).append(float(item.score))
    return [min(grouped.get(aspect.strip().casefold(), [0.0])) for aspect in expected]


def calculate_evidence_metrics(
    evaluation: EvidenceAssessment,
    expected: list[str],
    script_coverage: list[AspectCoverage],
    passages: list[Evidence],
    *,
    k: int = 5,
    have_reference: bool = True,
) -> dict[str, Any]:
    """Conservative normalization: missing, duplicate and unfamiliar IDs earn no credit."""
    coverage_map = {x.aspect.strip().casefold(): float(x.score) for x in script_coverage}
    grounded = _scores_by_aspect(evaluation.grounded_aspects, expected)
    retrieved = _scores_by_aspect(evaluation.retrieved_aspects, expected)
    grounded_coverage = [coverage_map.get(a.strip().casefold(), 0.0) * g
                         for a, g in zip(expected, grounded)]

    top = passages[:k]
    relevance: dict[str, list[bool]] = {}
    for item in evaluation.passage_relevance:
        relevance.setdefault(item.evidence_id, []).append(bool(item.relevant))
    precise = [all(relevance.get(x.evidence_id, [False])) for x in top]
    return {
        "grounded_requirement_coverage_percent": (
            round(100 * sum(grounded_coverage) / len(expected), 5)
            if expected and have_reference else None
        ),
        "context_precision_at_k": (sum(precise) / len(top) if top else None),
        "context_recall_percent": (
            100 * sum(retrieved) / len(expected) if expected and passages else None
        ),
        "retrieval_k": k if passages else None,
        "grounded_aspect_scores": grounded,
        "retrieved_aspect_scores": retrieved,
        "retrieved_passages_scored": len(top),
    }


def compare_revisions(before: QualityEvaluation, after: QualityEvaluation) -> dict[str, Any]:
    """Score revision effects on educational coverage and narration (not factual truth)."""
    before_cov = 100 * before.coverage_score
    after_cov = 100 * after.coverage_score
    narration_fields = [
        "opening_orientation_score", "introduction_roadmap_score",
        "definitions_foundations_score", "explanatory_depth_score",
        "examples_applications_score", "organization_transitions_score",
        "clarity_accessibility_score", "spoken_naturalness_score",
        "conclusion_takeaway_score", "reflective_ending_score",
    ]
    def narration(q: QualityEvaluation) -> float:
        return 100 * sum(getattr(q, name) for name in narration_fields) / 40

    prev = {x.aspect.strip().casefold(): x.score for x in before.aspect_scores}
    now = {x.aspect.strip().casefold(): x.score for x in after.aspect_scores}
    addressed = [key for key, score in prev.items() if score > 0]
    maintained = sum(now.get(key, 0) >= prev[key] for key in addressed)
    return {
        "repair_effectiveness_points": round(
            ((after_cov + narration(after)) - (before_cov + narration(before))) / 2, 5
        ),
        "requirement_preservation_rate": (
            maintained / len(addressed) if addressed else None
        ),
        "requirements_previously_addressed": len(addressed),
        "requirements_preserved_without_regression": maintained,
        "repair_coverage_delta_points": after_cov - before_cov,
        "repair_narration_delta_points": narration(after) - narration(before),
    }


def _merge_aspect_ratings(parts: list[list[AspectEvidenceRating]]) -> list[AspectEvidenceRating]:
    """Across evidence batches an aspect keeps its best score: support in any batch counts."""
    best: dict[str, AspectEvidenceRating] = {}
    for ratings in parts:
        for item in ratings:
            key = item.aspect.strip().casefold()
            if key not in best or item.score > best[key].score:
                best[key] = item
    return list(best.values())


def _merge_assessments(parts: list[EvidenceAssessment]) -> EvidenceAssessment:
    if len(parts) == 1:
        return parts[0]
    return EvidenceAssessment(
        grounded_aspects=_merge_aspect_ratings([p.grounded_aspects for p in parts]),
        retrieved_aspects=_merge_aspect_ratings([p.retrieved_aspects for p in parts]),
        passage_relevance=[row for p in parts for row in p.passage_relevance],
    )


def _validate_assessment(result: EvidenceAssessment) -> list[str]:
    return (
        invalid_aspect_scores(result.grounded_aspects, "grounded")
        + invalid_aspect_scores(result.retrieved_aspects, "retrieved")
    )


def _missing_judgements(
    result: EvidenceAssessment,
    expected: list[str],
    retrieved: list[Evidence],
) -> list[str]:
    """List aspects/passages the judge was asked about but did not return.

    An omitted item would otherwise be normalised to 0 / not relevant, turning a judge failure
    into a low system score. Omissions are re-asked instead and, if they persist, the case fails
    as an evaluation error.
    """
    problems: list[str] = []
    groups = [("grounded_aspects", result.grounded_aspects)]
    if retrieved:
        groups.append(("retrieved_aspects", result.retrieved_aspects))
    for label, rows in groups:
        returned = {row.aspect.strip().casefold() for row in rows}
        missing = [a for a in expected if a.strip().casefold() not in returned]
        if missing:
            problems.append(f"{label} omitted: " + "; ".join(missing))
    returned_ids = {row.evidence_id for row in result.passage_relevance}
    missing_ids = [p.evidence_id for p in retrieved if p.evidence_id not in returned_ids]
    if missing_ids:
        problems.append("passage_relevance omitted: " + ", ".join(missing_ids))
    return problems


def judge_evidence(
    llm: Any, question: str, expected: list[str], script: str,
    reference: list[Evidence], retrieved: list[Evidence],
    *, max_prompt_chars: int = DEFAULT_JUDGE_PROMPT_CHARS,
    record: Callable[[Any], None] | None = None,
) -> tuple[EvidenceAssessment, Any]:
    """Judge the complete script against all evidence, batching evidence when needed.

    The script is never truncated. When the prompt would exceed ``max_prompt_chars`` the
    evidence (not the script) is split into ordered batches, one call per batch, and the
    results are merged. ``record`` receives telemetry for every call except the last,
    which is returned.
    """
    checklist = "\n".join(f"- {x}" for x in expected)
    head = (
        f"QUESTION: {question}\nEXPECTED ASPECTS:\n{checklist}\n\n"
        f"SCRIPT:\n{script}\n\n"
    )
    evidence_budget = max(max_prompt_chars - len(head), MIN_EVIDENCE_BATCH_CHARS)
    shared = evidence_budget if not (reference and retrieved) else evidence_budget // 2
    ref_batches = _chunk_passages(reference, shared)
    ret_batches = _chunk_passages(retrieved, shared)
    n_batches = max(len(ref_batches), len(ret_batches), 1)

    parts: list[EvidenceAssessment] = []
    log: Any = None
    for index in range(n_batches):
        if index > 0 and record is not None:
            record(log)
        ref_chunk = ref_batches[index] if index < len(ref_batches) else []
        ret_chunk = ret_batches[index] if index < len(ret_batches) else []
        batch_note = f"EVIDENCE BATCH {index + 1} OF {n_batches}\n\n" if n_batches > 1 else ""
        prompt = (
            head
            + batch_note
            + f"REFERENCE CORPUS EXTRACTS:\n{_format_passages(ref_chunk)}\n\n"
            + f"GENERATOR RETRIEVED PASSAGES (RANKED):\n{_format_passages(ret_chunk)}"
        )
        result, log = generate_with_rubric(
            llm,
            prompt=prompt,
            validate=lambda r, chunk=ret_chunk: (
                _validate_assessment(r) + _missing_judgements(r, expected, chunk)
            ),
            record=record,
            rule=(
                "aspect scores must be exactly 0, 0.5 or 1, and every listed aspect and every "
                "ranked generator passage must be returned exactly once"
            ),
            task="grounded_requirements_and_retrieval_quality",
            system_instruction=SYSTEM_PROMPT,
            schema=EvidenceAssessment,
            max_output_tokens=3600,
        )
        parts.append(result)
    return _merge_assessments(parts), log
