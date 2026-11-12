from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, Field


SourceType = Literal["web", "academic", "video"]
VerificationStatus = Literal["supported", "partial", "unsupported", "contradicted"]
RequirementStatus = Literal["covered", "partial", "missing", "conflicting"]
NarrationRequirementStatus = Literal[
    "answered_with_evidence",
    "partial",
    "unresolved_acknowledged",
    "omitted",
]


class LLMCallLog(BaseModel):
    task: str
    model: str
    provider: str = ""
    latency_seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    attempts: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceDocument(BaseModel):
    source_id: str = Field(default_factory=lambda: f"S-{uuid4().hex[:10]}")
    source_type: SourceType
    title: str
    url: str
    text: str
    retrieval_query: str
    rank: int = 0
    author: str | None = None
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"E-{uuid4().hex[:10]}")
    source_id: str
    source_type: SourceType
    title: str
    url: str
    passage: str
    retrieval_query: str
    relevance_score: float = 0.0
    published_at: str | None = None


class ResearchPlan(BaseModel):
    # Some model responses use a different field name even when a schema is
    # supplied.  These validation aliases accept harmless naming differences while
    # keeping the stored/output field names stable for the experiment.
    subquestions: list[str] = Field(
        min_length=2,
        max_length=2,
        validation_alias=AliasChoices(
            "subquestions",
            "sub_questions",
            "retrieval_subquestions",
        ),
    )
    answer_requirements: list[str] = Field(
        min_length=3,
        max_length=10,
        validation_alias=AliasChoices(
            "answer_requirements",
            "requirements",
            "coverage_requirements",
        ),
    )
    freshness_required: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "freshness_required",
            "needs_freshness",
        ),
    )
    preferred_sources: list[SourceType] = Field(
        default_factory=lambda: ["web", "academic"],
        validation_alias=AliasChoices(
            "preferred_sources",
            "preferred_source_types",
            "source_types",
        ),
    )
    rationale: str = ""




class RequirementCompletenessResult(BaseModel):
    final_requirements: list[str] = Field(min_length=3, max_length=10)
    added_requirements: list[str] = Field(default_factory=list, max_length=7)
    rationale: str = ""


class SubQuestionSet(BaseModel):
    subquestions: list[str] = Field(
        min_length=1,
        max_length=3,
        validation_alias=AliasChoices(
            "subquestions",
            "sub_questions",
            "questions",
        ),
    )


class RouteDecision(BaseModel):
    sources: list[SourceType] = Field(min_length=1, max_length=3)
    max_per_source: int = Field(default=4, ge=1, le=8)
    rationale: str = ""


class RequirementAssessment(BaseModel):
    requirement_id: str
    requirement: str
    status: RequirementStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    suggested_query: str = ""
    rationale: str = ""


class RequirementSufficiencyBundle(BaseModel):
    assessments: list[RequirementAssessment] = Field(default_factory=list)




class NarrationRequirementAssessment(BaseModel):
    requirement_id: str
    requirement: str
    status: NarrationRequirementStatus
    confidence: float = Field(ge=0.0, le=1.0)
    script_excerpt: str = ""
    rationale: str = ""
    suggested_repair: str = ""


class NarrationRequirementAuditBundle(BaseModel):
    assessments: list[NarrationRequirementAssessment] = Field(default_factory=list)


class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: f"C-{uuid4().hex[:10]}")
    text: str
    needs_citation: bool = True
    importance: Literal["core", "supporting"] = "supporting"
    citation_ids: list[str] = Field(default_factory=list)


class ClaimSet(BaseModel):
    claims: list[Claim] = Field(default_factory=list, max_length=30)


class ClaimVerification(BaseModel):
    claim_id: str
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class VerificationBundle(BaseModel):
    verifications: list[ClaimVerification] = Field(default_factory=list)


class Contradiction(BaseModel):
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"


class ContradictionBundle(BaseModel):
    contradictions: list[Contradiction] = Field(default_factory=list)


class FeedbackResult(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    clarity: float = Field(ge=0.0, le=10.0)
    factual_grounding: float = Field(ge=0.0, le=10.0)
    completeness: float = Field(ge=0.0, le=10.0)
    revision_needed: bool = False
    feedback: str = ""


class PedagogyPlan(BaseModel):
    learning_objectives: list[str] = Field(default_factory=list, max_length=5)
    opening_orientation: str = ""
    introduction_points: list[str] = Field(default_factory=list, max_length=4)
    key_definitions: list[str] = Field(default_factory=list, max_length=6)
    outline: list[str] = Field(default_factory=list, max_length=8)
    example_opportunities: list[str] = Field(default_factory=list, max_length=4)
    nuance_or_limitations: list[str] = Field(default_factory=list, max_length=4)
    transition_plan: list[str] = Field(default_factory=list, max_length=6)
    conclusion_takeaways: list[str] = Field(default_factory=list, max_length=5)
    reflective_questions: list[str] = Field(default_factory=list, max_length=2)
    misconceptions_to_avoid: list[str] = Field(default_factory=list, max_length=5)
    audience_assumption: str = "General university-level learner"


class AspectCoverage(BaseModel):
    aspect: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class QualityEvaluation(BaseModel):
    aspect_scores: list[AspectCoverage] = Field(default_factory=list)
    covered_aspects: list[str] = Field(default_factory=list)
    partially_covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    coverage_score: float = Field(ge=0.0, le=1.0)
    coverage_rationale: str = ""
    relevance_score: float = Field(ge=0.0, le=1.0)
    relevance_rationale: str = ""

    # Narration quality dimensions: 0 absent/very poor, 1 weak, 2 adequate,
    # 3 strong, 4 excellent. These scores deliberately exclude factual truth.
    opening_orientation_score: int = Field(ge=0, le=4)
    introduction_roadmap_score: int = Field(ge=0, le=4)
    definitions_foundations_score: int = Field(ge=0, le=4)
    explanatory_depth_score: int = Field(ge=0, le=4)
    examples_applications_score: int = Field(ge=0, le=4)
    organization_transitions_score: int = Field(ge=0, le=4)
    clarity_accessibility_score: int = Field(ge=0, le=4)
    spoken_naturalness_score: int = Field(ge=0, le=4)
    conclusion_takeaway_score: int = Field(ge=0, le=4)
    reflective_ending_score: int = Field(ge=0, le=4)
    narration_rationale: str = ""


class RunTelemetry(BaseModel):
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: str | None = None
    wall_seconds: float = 0.0
    retrieval_calls: int = 0
    retrieved_documents: int = 0
    llm_calls: list[LLMCallLog] = Field(default_factory=list)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(x.prompt_tokens for x in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(x.output_tokens for x in self.llm_calls)

    @property
    def total_thinking_tokens(self) -> int:
        return sum(x.thinking_tokens for x in self.llm_calls)

    @property
    def total_billable_output_tokens(self) -> int:
        return self.total_output_tokens + self.total_thinking_tokens

    @property
    def total_llm_calls(self) -> int:
        return len(self.llm_calls)

    @property
    def estimated_llm_cost_usd(self) -> float:
        return float(sum(x.total_cost_usd for x in self.llm_calls))

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        xs = sorted(float(x) for x in values)
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        frac = pos - lo
        return xs[lo] * (1.0 - frac) + xs[hi] * frac

    @property
    def llm_call_latency_p50_seconds(self) -> float:
        return self._percentile([x.latency_seconds for x in self.llm_calls], 0.50)

    @property
    def llm_call_latency_p90_seconds(self) -> float:
        return self._percentile([x.latency_seconds for x in self.llm_calls], 0.90)


class PipelineResult(BaseModel):
    system_id: str
    system_name: str
    prompt_id: str
    query: str
    domain: str = ""
    reasoning_type: str = ""
    script: str
    sources: list[SourceDocument] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    verifications: list[ClaimVerification] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    telemetry: RunTelemetry = Field(default_factory=RunTelemetry)


class BenchmarkPrompt(BaseModel):
    prompt_id: str
    domain: str
    reasoning_type: str
    question: str


class EvaluationRecord(BaseModel):
    experiment_id: str = ""
    evaluated_script_sha256: str = ""
    reference_pool_sha256: str = ""
    generation_config_sha256: str = ""
    prompt_version: str = ""
    system_id: str
    prompt_id: str
    domain: str
    reasoning_type: str
    generation_provider: str = ""
    generation_access: str = ""
    generation_model: str = ""
    evaluator_provider: str = "anthropic"
    evaluator_model: str
    claim_count: int
    supported_claims: int
    partial_claims: int
    unsupported_claims: int
    contradicted_claims: int
    supported_claim_rate: float
    reference_evidence_support_percent: float = 0.0
    query_coverage_percent: float = 0.0
    query_relevance_percent: float = 0.0
    narration_quality_percent: float = 0.0
    citation_precision: float | None = None
    citation_recall: float | None = None
    citation_f1: float | None = None
    grounded_requirement_coverage_percent: float | None = None
    context_precision_at_k: float | None = None
    context_recall_percent: float | None = None
    repair_effectiveness_points: float | None = None
    requirement_preservation_rate: float | None = None
    source_count: int = 0
    source_type_count: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    billable_output_tokens: int = 0
    retrieval_calls: int = 0
    latency_seconds: float = 0.0
    word_count: int = 0
    estimated_llm_cost_usd: float = 0.0
    llm_call_latency_p50_seconds: float = 0.0
    llm_call_latency_p90_seconds: float = 0.0
