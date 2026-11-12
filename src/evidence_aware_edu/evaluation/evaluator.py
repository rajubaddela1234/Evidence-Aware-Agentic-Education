from __future__ import annotations

from collections import OrderedDict

from evidence_aware_edu.agents.claim_extractor import ClaimExtractorAgent
from evidence_aware_edu.agents.verifier import VerifierAgent
from evidence_aware_edu.config import Settings
from evidence_aware_edu.evaluation.checklist import (
    ChecklistIntegrityError,
    ChecklistMissing,
    load_or_derive_checklist,
)
from evidence_aware_edu.evaluation.judge import QualityJudge
from evidence_aware_edu.evaluation.additional_metrics import (
    calculate_evidence_metrics, compare_revisions, judge_evidence,
)
from evidence_aware_edu.evaluation.metrics import (
    citation_metrics,
    narration_quality_percent,
    query_coverage_percent,
    reference_evidence_support_percent,
    support_counts,
)
from evidence_aware_edu.evidence.extractor import extract_ranked_evidence
from evidence_aware_edu.experiments.identity import PROMPT_VERSION, sha256_text
from evidence_aware_edu.llm.claude import ClaudeEvaluatorClient
from evidence_aware_edu.observability import configure_langsmith, trace_span
from evidence_aware_edu.retrieval.frozen import FrozenCorpusHub
from evidence_aware_edu.schemas import (
    AspectCoverage,
    BenchmarkPrompt,
    Claim,
    ClaimVerification,
    EvaluationRecord,
    Evidence,
    PipelineResult,
    QualityEvaluation,
)
from evidence_aware_edu.telemetry import TelemetryCollector
from evidence_aware_edu.utils import strip_evidence_markers

# Reference-support verification: per claim, the top passages ranked against that claim, plus a
# few question-level passages for shared context. Five claims x five passages plus three general
# passages is roughly 34k characters, which fits CLAUDE_EVALUATOR_MAX_CONTEXT_CHARS=40000.
CLAIM_REFERENCE_PASSAGES = 5
GENERAL_REFERENCE_PASSAGES = 3


class ResearchEvaluator:
    """Evaluate finished scripts with the separate Claude evaluator."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.validate_independent_evaluator()
        self.frozen = FrozenCorpusHub(
            settings.reference_pool_dir,
            require_video_transcript=settings.youtube_require_transcript,
        )
        configure_langsmith(settings)
        # prompt_id -> aspects_sha256 of the checklist frozen for this experiment (see
        # freeze_checklists). Empty for ad-hoc evaluation, where checklists may be derived.
        self.frozen_checklists: dict[str, str] = {}

    def _checklist_dir(self, experiment_id: str):
        return self.settings.output_dir / "experiments" / (experiment_id or "adhoc") / "checklists"

    def freeze_checklists(
        self, prompts: list[BenchmarkPrompt], experiment_id: str, recorded: dict[str, str]
    ) -> dict[str, str]:
        """Derive every question's checklist before any case is evaluated and pin its hash.

        ``recorded`` holds hashes already stored in the experiment manifest. A recorded
        checklist must still exist unchanged; otherwise resuming would judge later systems
        against different obligations than earlier ones, so this raises instead. Questions
        without a recorded hash are derived (or loaded) now. Returns the full mapping.
        """
        eval_llm = None
        frozen: dict[str, str] = {}
        directory = self._checklist_dir(experiment_id)
        for prompt in prompts:
            expected = recorded.get(prompt.prompt_id)
            try:
                _, info = load_or_derive_checklist(
                    None, question=prompt.question, prompt_id=prompt.prompt_id,
                    directory=directory, allow_derive=False,
                )
            except ChecklistMissing:
                if expected is not None:
                    raise  # it was frozen earlier and has since disappeared
                if eval_llm is None:  # start Claude only when something must be derived
                    eval_llm = ClaudeEvaluatorClient(
                        self.settings, model=self.settings.evaluator_model
                    )
                _, info = load_or_derive_checklist(
                    eval_llm, question=prompt.question, prompt_id=prompt.prompt_id,
                    directory=directory,
                )
            if expected is not None and info.get("aspects_sha256") != expected:
                raise ChecklistIntegrityError(
                    f"Checklist for {prompt.prompt_id} no longer matches the experiment manifest."
                )
            frozen[prompt.prompt_id] = info["aspects_sha256"]
        self.frozen_checklists = dict(frozen)
        return frozen

    @staticmethod
    def _dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
        seen: OrderedDict[str, Evidence] = OrderedDict()
        for item in items:
            seen.setdefault(item.evidence_id, item)
        return list(seen.values())

    @staticmethod
    def _normalize_quality_aspects(
        expected_aspects: list[str], quality: QualityEvaluation
    ) -> QualityEvaluation:
        """Match coverage results back to the benchmark aspect list."""
        if not expected_aspects:
            return quality

        grouped: dict[str, list[AspectCoverage]] = {}
        for item in quality.aspect_scores:
            grouped.setdefault(item.aspect.strip().casefold(), []).append(item)

        normalized: list[AspectCoverage] = []
        for aspect in expected_aspects:
            options = grouped.get(aspect.strip().casefold(), [])
            if not options:
                normalized.append(
                    AspectCoverage(
                        aspect=aspect,
                        score=0.0,
                        rationale="Evaluator omitted this predeclared aspect; scored conservatively as missing.",
                    )
                )
            else:
                chosen = min(options, key=lambda x: x.score)
                normalized.append(
                    AspectCoverage(aspect=aspect, score=chosen.score, rationale=chosen.rationale)
                )

        quality.aspect_scores = normalized
        quality.coverage_score = sum(x.score for x in normalized) / len(normalized)
        quality.covered_aspects = [x.aspect for x in normalized if x.score >= 0.8]
        quality.partially_covered_aspects = [
            x.aspect for x in normalized if 0.0 < x.score < 0.8
        ]
        quality.missing_aspects = [x.aspect for x in normalized if x.score <= 0.0]
        return quality

    def _claim_specific_reference_support(
        self,
        question: str,
        claims: list[Claim],
        reference_docs,
        verifier: VerifierAgent,
    ) -> tuple[list, dict[str, list[str]]]:
        """Verify each claim against evidence selected for that claim."""
        all_verifications = []
        evidence_map: dict[str, list[str]] = {}
        batch_size = 5

        general = extract_ranked_evidence(
            question,
            reference_docs,
            max_total=GENERAL_REFERENCE_PASSAGES,
            per_document=1,
        )

        for start in range(0, len(claims), batch_size):
            batch = claims[start : start + batch_size]
            per_claim: list[list] = []
            for claim in batch:
                claim_evidence = extract_ranked_evidence(
                    claim.text,
                    reference_docs,
                    max_total=CLAIM_REFERENCE_PASSAGES,
                    per_document=1,
                )
                evidence_map[claim.claim_id] = [e.evidence_id for e in claim_evidence]
                per_claim.append(claim_evidence)
            # Claim-specific passages go first and are interleaved by rank (every claim's best
            # passage, then every claim's second, ...). If the context budget truncates the
            # list, each claim keeps its strongest passages instead of losing all of them to
            # the general question-level passages.
            batch_evidence = [
                ranked[rank]
                for rank in range(CLAIM_REFERENCE_PASSAGES)
                for ranked in per_claim
                if rank < len(ranked)
            ] + list(general)
            batch_evidence = self._dedupe_evidence(batch_evidence)
            all_verifications.extend(
                verifier.run(
                    batch,
                    batch_evidence,
                    task="reference_evidence_support",
                    max_chars=self.settings.claude_evaluator_max_context_chars,
                    require_complete=True,
                )
            )

        return all_verifications, evidence_map

    def _citation_entailment(
        self,
        claims: list[Claim],
        generation_evidence: list[Evidence],
        verifier: VerifierAgent,
    ) -> list:
        """Check each claim against the passages it actually cites, and only those.

        Claims are batched for cost, so one prompt holds several claims' cited passages. Each
        claim is therefore given an explicit ALLOWED EVIDENCE list (its own resolvable citation
        IDs), and the verifier's answer is filtered to that list, so a claim can never be
        credited with a passage that a different claim cited. Claims with no resolvable
        citation are unsupported by definition and are not sent to the model.
        """
        evidence_by_id = {e.evidence_id: e for e in generation_evidence}
        allowed_by_claim = {
            c.claim_id: {eid for eid in c.citation_ids if eid in evidence_by_id} for c in claims
        }
        uncited = [c for c in claims if not allowed_by_claim[c.claim_id]]
        cited = [c for c in claims if allowed_by_claim[c.claim_id]]
        decisions = {
            c.claim_id: ClaimVerification(
                claim_id=c.claim_id,
                status="unsupported",
                confidence=0.0,
                evidence_ids=[],
                rationale="Claim cites no passage the generator retrieved; unsupported by its citations.",
            )
            for c in uncited
        }

        batch_size = 5
        for start in range(0, len(cited), batch_size):
            batch = cited[start : start + batch_size]
            cited_union: list[Evidence] = []
            seen: set[str] = set()
            for claim in batch:
                for eid in sorted(allowed_by_claim[claim.claim_id]):
                    if eid not in seen:
                        cited_union.append(evidence_by_id[eid])
                        seen.add(eid)
            for verification in verifier.run(
                batch,
                cited_union,
                task="citation_entailment",
                max_chars=self.settings.claude_evaluator_max_context_chars,
                allowed_by_claim={c.claim_id: allowed_by_claim[c.claim_id] for c in batch},
                require_complete=True,
            ):
                decisions[verification.claim_id] = verification
        out = [decisions[c.claim_id] for c in claims]
        return out

    def evaluate(
        self,
        prompt: BenchmarkPrompt,
        result: PipelineResult,
        *,
        experiment_id: str = "",
        script_sha256: str = "",
        reference_pool_sha256: str = "",
        generation_config_sha256: str = "",
        prompt_version: str = PROMPT_VERSION,
    ) -> tuple[EvaluationRecord, dict]:
        evaluator_model = self.settings.evaluator_model
        script_sha256 = script_sha256 or sha256_text(result.script)
        with trace_span(
            self.settings,
            f"evaluation:{result.system_id}:{prompt.prompt_id}",
            run_type="chain",
            inputs={
                "system_id": result.system_id,
                "prompt_id": prompt.prompt_id,
                "question": prompt.question if self.settings.langsmith_trace_content else "[redacted]",
                "script": result.script if self.settings.langsmith_trace_content else "[redacted]",
            },
            metadata={
                "evaluation_role": "independent_provider_model",
                "evaluator_provider": self.settings.evaluator_provider,
                "evaluator_access": self.settings.evaluator_access,
                "evaluator_model": evaluator_model,
                "generation_provider": self.settings.generation_provider,
                "generation_access": self.settings.generation_access,
                "generation_model": self.settings.generation_model,
                "experiment_id": experiment_id,
                "script_sha256": script_sha256,
            },
            tags=["evidence-aware-education", "evaluation", result.system_id, prompt.prompt_id],
        ) as span:
            eval_telemetry = TelemetryCollector()
            eval_llm = ClaudeEvaluatorClient(self.settings, model=evaluator_model)
            extractor = ClaimExtractorAgent(eval_llm, eval_telemetry)
            verifier = VerifierAgent(eval_llm, eval_telemetry)
            quality_judge = QualityJudge(eval_llm)

            # One checklist per question, derived from the question text and shared by every
            # system, so coverage uses the same fixed denominator for all of them.
            frozen_hash = getattr(self, "frozen_checklists", {}).get(prompt.prompt_id)
            expected_aspects, checklist_info = load_or_derive_checklist(
                eval_llm,
                question=prompt.question,
                prompt_id=prompt.prompt_id,
                directory=self._checklist_dir(experiment_id),
                allow_derive=frozen_hash is None,
            )
            if frozen_hash is not None and checklist_info.get("aspects_sha256") != frozen_hash:
                raise ChecklistIntegrityError(
                    f"Checklist for {prompt.prompt_id} differs from the one frozen in the "
                    "experiment manifest; systems would be judged against different obligations."
                )
            reference_docs = self.frozen.load(prompt.prompt_id)
            # Evidence markers stay in the script for claim/citation verification only.
            # Coverage, relevance and narration are judged on the clean spoken text, so
            # systems that emit markers are not penalised (or rewarded) for them.
            narration_script = strip_evidence_markers(result.script)
            claims = extractor.from_script(result.script)
            ref_verifications, evidence_by_claim = self._claim_specific_reference_support(
                prompt.question,
                claims,
                reference_docs,
                verifier,
            )
            support = support_counts(ref_verifications)
            support_percent = reference_evidence_support_percent(ref_verifications)

            if result.evidence:
                citation_verifications = self._citation_entailment(
                    claims,
                    result.evidence,
                    verifier,
                )
                c_precision, c_recall, c_f1 = citation_metrics(claims, citation_verifications)
            else:
                citation_verifications = []
                c_precision = c_recall = c_f1 = None

            quality, quality_log = quality_judge.run(
                prompt.question,
                expected_aspects,
                narration_script,
                record=eval_telemetry.add_llm,
            )
            eval_telemetry.add_llm(quality_log)
            quality = self._normalize_quality_aspects(expected_aspects, quality)

            # These five measurements reuse the same evaluator and frozen corpus.
            # They do not change the generation side or any of the existing rubrics.
            reference_passages = extract_ranked_evidence(
                prompt.question, reference_docs, max_total=10, per_document=1
            )
            if reference_passages or result.evidence:
                evidence_judgments, evidence_log = judge_evidence(
                    eval_llm, prompt.question, expected_aspects,
                    narration_script, reference_passages, result.evidence,
                    max_prompt_chars=self.settings.claude_evaluator_max_judge_prompt_chars,
                    record=eval_telemetry.add_llm,
                )
                eval_telemetry.add_llm(evidence_log)
                extra_metrics = calculate_evidence_metrics(
                    evidence_judgments, expected_aspects,
                    quality.aspect_scores, result.evidence,
                    have_reference=bool(reference_passages),
                )
            else:
                evidence_judgments = None
                extra_metrics = {
                    "grounded_requirement_coverage_percent": None,
                    "context_precision_at_k": None,
                    "context_recall_percent": None,
                }

            # A before/after comparison is meaningful only when the saved draft
            # differs from the final script. Single-pass systems report N/A.
            pre_repair = result.metadata.get("pre_repair_script")
            revision = None
            if isinstance(pre_repair, str) and pre_repair.strip() and pre_repair != result.script:
                before, revision_log = quality_judge.run(
                    prompt.question, expected_aspects,
                    strip_evidence_markers(pre_repair),
                    record=eval_telemetry.add_llm,
                )
                eval_telemetry.add_llm(revision_log)
                before = self._normalize_quality_aspects(expected_aspects, before)
                revision = compare_revisions(before, quality)

            evaluation_telemetry = eval_telemetry.finish()
            resolved_evaluator_model = eval_llm.resolved_model or evaluator_model

            source_types = {s.source_type for s in result.sources}
            record = EvaluationRecord(
                experiment_id=experiment_id,
                evaluated_script_sha256=script_sha256,
                reference_pool_sha256=reference_pool_sha256,
                generation_config_sha256=generation_config_sha256,
                prompt_version=prompt_version,
                system_id=result.system_id,
                prompt_id=prompt.prompt_id,
                domain=prompt.domain,
                reasoning_type=prompt.reasoning_type,
                generation_provider=self.settings.generation_provider,
                generation_access=self.settings.generation_access,
                generation_model=self.settings.generation_model,
                evaluator_provider=self.settings.evaluator_provider,
                evaluator_model=resolved_evaluator_model,
                claim_count=int(support["total"]),
                supported_claims=int(support["supported"]),
                partial_claims=int(support["partial"]),
                unsupported_claims=int(support["unsupported"]),
                contradicted_claims=int(support["contradicted"]),
                supported_claim_rate=float(support["supported_claim_rate"]),
                reference_evidence_support_percent=support_percent,
                query_coverage_percent=query_coverage_percent(quality),
                query_relevance_percent=quality.relevance_score * 100.0,
                narration_quality_percent=narration_quality_percent(quality),
                citation_precision=c_precision,
                citation_recall=c_recall,
                citation_f1=c_f1,
                grounded_requirement_coverage_percent=extra_metrics.get(
                    "grounded_requirement_coverage_percent"
                ),
                context_precision_at_k=extra_metrics.get("context_precision_at_k"),
                context_recall_percent=extra_metrics.get("context_recall_percent"),
                repair_effectiveness_points=(
                    revision["repair_effectiveness_points"] if revision else None
                ),
                requirement_preservation_rate=(
                    revision["requirement_preservation_rate"] if revision else None
                ),
                source_count=len(result.sources),
                source_type_count=len(source_types),
                llm_calls=result.telemetry.total_llm_calls,
                prompt_tokens=result.telemetry.total_prompt_tokens,
                output_tokens=result.telemetry.total_output_tokens,
                thinking_tokens=result.telemetry.total_thinking_tokens,
                billable_output_tokens=result.telemetry.total_billable_output_tokens,
                retrieval_calls=result.telemetry.retrieval_calls,
                latency_seconds=result.telemetry.wall_seconds,
                word_count=len(result.script.split()),
                estimated_llm_cost_usd=result.telemetry.estimated_llm_cost_usd,
                llm_call_latency_p50_seconds=result.telemetry.llm_call_latency_p50_seconds,
                llm_call_latency_p90_seconds=result.telemetry.llm_call_latency_p90_seconds,
            )
            details = {
                "artifact_identity": {
                    "experiment_id": experiment_id,
                    "evaluated_script_sha256": script_sha256,
                    "reference_pool_sha256": reference_pool_sha256,
                    "generation_config_sha256": generation_config_sha256,
                    "prompt_version": prompt_version,
                },
                "narration_judged_script_sha256": sha256_text(narration_script),
                "question_checklist": {"aspects": expected_aspects, **checklist_info},
                "evaluation_claims": [x.model_dump() for x in claims],
                "reference_verifications": [x.model_dump() for x in ref_verifications],
                "reference_evidence_ids_by_claim": evidence_by_claim,
                "citation_entailment_verifications": [
                    x.model_dump() for x in citation_verifications
                ],
                "quality": quality.model_dump(),
                "additional_metric_details": {
                    **extra_metrics,
                    "retrieval_assessment": (
                        evidence_judgments.model_dump() if evidence_judgments else None
                    ),
                    "revision_comparison": revision,
                    "notes": "All scores are Claude judgments, not verified ground truth. "
                             "Repair effectiveness compares coverage/narration, not factual truth. "
                             "No revision or no retrieval is N/A rather than zero.",
                },
                "metric_semantics": {
                    "reference_evidence_support_percent": (
                        "Evidence entailment within the frozen reference corpus; not factual accuracy."
                    ),
                    "query_coverage_percent": (
                        "Coverage of a fixed six-aspect checklist derived once per question from the "
                        "question text and shared by all systems; missing/duplicate judge outputs "
                        "are normalized against that fixed denominator."
                    ),
                    "query_relevance_percent": "Independent-model relevance judgement.",
                    "narration_input": (
                        "Coverage, relevance, narration and grounded-requirement judgements use the "
                        "complete script with [E-...] evidence markers removed; claim extraction and "
                        "citation metrics use the marked script. Aspect scores are continuous in [0, 1]."
                    ),
                    "narration_quality_percent": (
                        "Ten-dimension spoken educational narration score; excludes factual truth."
                    ),
                    "citation_precision": (
                        "Share of emitted citation IDs that are identified as supporting the claim "
                        "when verification is restricted to cited passages."
                    ),
                    "citation_recall": (
                        "Share of all citation-required extracted factual claims that have at least "
                        "one cited passage judged to support the claim."
                    ),
                    "evaluation_estimated_llm_cost_usd": (
                        "Evaluator telemetry. CLI mode uses Claude Code's reported list-price "
                        "equivalent; API mode uses the configured Anthropic per-token rates."
                    ),
                },
                "evaluation_telemetry": evaluation_telemetry.model_dump(),
                "evaluation_provider": self.settings.evaluator_provider,
                "evaluation_access": self.settings.evaluator_access,
                "claude_access_method": self.settings.claude_access_method,
                "claude_cli_version": eval_llm.cli_version,
                "evaluation_model_requested": evaluator_model,
                "evaluation_model": resolved_evaluator_model,
                "evaluation_cli_reported_cost_usd": (
                    evaluation_telemetry.estimated_llm_cost_usd
                    if self.settings.claude_access_method == "cli"
                    else None
                ),
                "evaluation_estimated_llm_cost_usd": evaluation_telemetry.estimated_llm_cost_usd,
                "generation_provider": self.settings.generation_provider,
                "generation_access": self.settings.generation_access,
                "generation_model_requested": self.settings.generation_model,
                "generation_estimated_llm_cost_usd": result.telemetry.estimated_llm_cost_usd,
                "combined_generation_plus_evaluation_estimated_llm_cost_usd": (
                    result.telemetry.estimated_llm_cost_usd
                    + evaluation_telemetry.estimated_llm_cost_usd
                ),
            }
            span.set_outputs(
                {
                    "query_coverage_percent": record.query_coverage_percent,
                    "reference_evidence_support_percent": record.reference_evidence_support_percent,
                    "query_relevance_percent": record.query_relevance_percent,
                    "narration_quality_percent": record.narration_quality_percent,
                    "generation_estimated_llm_cost_usd": record.estimated_llm_cost_usd,
                    "evaluation_provider": self.settings.evaluator_provider,
                    "evaluation_access": self.settings.evaluator_access,
                    "evaluation_model_requested": evaluator_model,
                    "evaluation_model": resolved_evaluator_model,
                    "evaluation_cli_reported_cost_usd": (
                        evaluation_telemetry.estimated_llm_cost_usd
                    ),
                }
            )
            return record, details
