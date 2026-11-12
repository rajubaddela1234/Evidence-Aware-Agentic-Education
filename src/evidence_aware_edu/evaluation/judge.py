from __future__ import annotations

from typing import Any, Callable

from evidence_aware_edu.evaluation.rubric import generate_with_rubric
from evidence_aware_edu.schemas import QualityEvaluation

# Coverage scale: 0.0 means the aspect is absent; any aspect that appears scores at least 0.7.
MIN_PRESENT_SCORE = 0.7


class QualityJudge:
    """Score answer coverage, relevance and narration quality."""

    SYSTEM = """You are an independent evaluator of educational video narration.

Judge the script against the USER QUESTION. Use the same standard for every system. Do not
reward a script for being longer, for following a particular template, or for sounding more
formal. Do not use outside knowledge to decide whether factual claims are true; factual evidence
support and citation entailment are scored separately.

EXPECTED ASPECTS are a fixed benchmark checklist used only for coverage scoring. They help make
coverage reproducible, but they do not replace the meaning of the user question.

A. QUERY COVERAGE
For every supplied expected aspect, return exactly one AspectCoverage item using the same aspect
wording. Judge whether the script explains that aspect well enough to help the learner. Use
exactly one of these four levels; any value inside a band is allowed:
- 0.9-1.0 = the main idea of the aspect is explained clearly enough that a learner would
  understand it, even if briefly. This is the expected score for any aspect the script
  genuinely explains. Full credit does not require every listed example or sub-point, extra
  depth, or exhaustive detail; the checklist wording is illustrative, not an exhaustive list.
- 0.8-0.9 = explained, but one part of the idea is thin or only implicit.
- 0.7-0.8 = the aspect is present but only mentioned, partly explained, or vague.
- 0.0 = the aspect is absent, or materially mishandled within the script's own explanation.
Never use a value between 0.0 and 0.7: if the aspect appears in the script at all, score it at
least 0.7; if it does not appear, score 0.0.
Credit an aspect for explanation anywhere in the script, in any wording or order. Equivalent
ideas, synonyms and an explanation spread across several sentences all count, and the script
does not need to use the checklist's terms.

When no expected aspects are supplied, infer 3-6 concrete answer obligations from the user
question and score them in the same way. Do not give full credit for a bare keyword mention.
coverage_score must equal the arithmetic mean of the returned aspect scores.

B. QUERY RELEVANCE
Set relevance_score from 0.0 to 1.0. Definitions, mechanisms, examples, caveats and short
transitions are relevant when they help answer the question. Penalize real tangents, repetition
and generic filler, but do not penalize brief orientation or teaching context.

C. NARRATION QUALITY
Score each dimension independently from 0 to 4:
0 = absent or seriously poor
1 = weak
2 = acceptable but noticeably limited
3 = strong and suitable for an educational video
4 = excellent

Be generous and judge each dimension on whether it does its job for a learner, not against an
ideal script. If a dimension is handled well, give 4. If it is present and works but is plain
or brief, give 3. Reserve 2 or below for dimensions that are clearly missing or hard to follow.
Do not lower a score because the script is short or reads slightly formal, and do not lower it
for a single minor issue such as one awkward sentence or one long passage. When you are unsure
between two scores, choose the higher one.

The rubric is intentionally flexible. A good script does not need to use the same presentation
pattern every time. An explicit roadmap, formal welcome, analogy, or reflective question is not
mandatory when the narration is already clear and natural. Absence of an optional stylistic
feature should not by itself force a low score.

1. opening_orientation_score
   Does the opening quickly orient the learner to the topic? A formal greeting is optional.
2. introduction_roadmap_score
   Does the beginning make the purpose and direction clear? An explicit 'we will cover...' list
   is not required.
3. definitions_foundations_score
   Are central terms or foundations explained when the learner needs them?
4. explanatory_depth_score
   Does the script explain causes, mechanisms or reasoning at a level a learner can follow,
   rather than only listing facts? Moderate depth is enough.
5. examples_applications_score
   Is there at least one example, application, analogy or concrete illustration, or is the
   topic clear enough without one? Give 3 or 4 when an example or analogy is present, and do
   not penalize an omitted one if the topic is already concrete.
6. organization_transitions_score
   Are ideas ordered logically, with enough connection between sections to follow the explanation?
7. clarity_accessibility_score
   Is the explanation understandable to a general university-level learner without unnecessary
   jargon?
8. spoken_naturalness_score
   Could a presenter reasonably say it aloud? A somewhat formal or lecture-like tone is fine.
   Only score low for note fragments, bullet-like lists or text that is hard to read aloud.
9. conclusion_takeaway_score
   Does the ending synthesize the main answer and leave a clear takeaway without adding unrelated
   new material?
10. reflective_ending_score
   Does the ending encourage reflection, application or curiosity? This can be a question, a short
   implication, or a forward-looking thought. Do not require a literal question.

Judge narration independently of factual truth. A fluent unsupported statement can still score
well for narration and poorly for evidence support. Likewise, a well-supported script can score
poorly if it is difficult to teach from. Return structured output only.
"""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    @staticmethod
    def problems(quality: QualityEvaluation, expected_aspects: list[str]) -> list[str]:
        """Out-of-range scores and checklist aspects the judge did not score.

        An omitted aspect would otherwise be filled in as 0 and read as "the script did not
        cover this". It is re-asked instead; a persistent omission fails the case as an
        evaluation error rather than lowering the system's coverage.
        """
        out = [
            f"coverage: '{x.aspect}' scored {x.score!r}, which is not 0.0 or between "
            f"{MIN_PRESENT_SCORE} and 1.0"
            for x in quality.aspect_scores
            if not (float(x.score) == 0.0 or MIN_PRESENT_SCORE <= float(x.score) <= 1.0)
        ]
        returned = {x.aspect.strip().casefold() for x in quality.aspect_scores}
        out.extend(
            f"coverage: aspect '{a}' was not scored"
            for a in expected_aspects
            if a.strip().casefold() not in returned
        )
        return out

    def run(
        self,
        question: str,
        expected_aspects: list[str],
        script: str,
        *,
        record: Callable[[Any], None] | None = None,
    ):
        """Score one script. ``script`` is judged in full and should be marker-free narration.

        Aspect scores are 0.0 or within [0.7, 1.0] and are re-requested otherwise; ``record``
        receives the telemetry of any rejected attempt.
        """
        if expected_aspects:
            aspects = "\n".join(f"- {x}" for x in expected_aspects)
            checklist_note = (
                "Score every listed aspect exactly once and preserve its wording in aspect_scores."
            )
        else:
            aspects = "NONE PROVIDED"
            checklist_note = (
                "Infer 3-6 concrete answer obligations directly from the USER QUESTION, then score them."
            )

        prompt = (
            f"USER QUESTION:\n{question}\n\n"
            f"EXPECTED ASPECTS / BENCHMARK CHECKLIST:\n{aspects}\n\n"
            f"SCORING NOTE:\n{checklist_note}\n\n"
            f"GENERATED SCRIPT:\n{script}"
        )
        return generate_with_rubric(
            self.llm,
            prompt=prompt,
            validate=lambda q: self.problems(q, expected_aspects),
            record=record,
            rule=(
                "every listed aspect must be scored exactly once, using its exact wording, with "
                f"0.0 (absent) or a number from {MIN_PRESENT_SCORE} to 1.0 (present)"
            ),
            task="answer_fulfilment_and_narration_evaluation",
            system_instruction=self.SYSTEM,
            schema=QualityEvaluation,
            max_output_tokens=3000,
        )
