"""Question-derived coverage checklist, created once per question and shared by all systems.

Benchmark questions no longer carry hand-written ``expected_aspects``. Coverage, grounded
requirement coverage and context recall still need a fixed denominator, otherwise each script
would be judged against its own differently-sized, self-chosen list. The evaluator therefore asks
Claude once per question for a short checklist derived from the question text alone, stores it in
the experiment folder, and re-uses that same list for every system and every re-evaluation.

The checklist is evaluation-side only: it is never shown to a generator or written into a
reference pool.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evidence_aware_edu.experiments.identity import sha256_text, utc_now
from evidence_aware_edu.utils import json_dump

CHECKLIST_SIZE = 6
DERIVATION_RETRIES = 2

SYSTEM = f"""You prepare an evaluation checklist for educational-video scripts that answer one USER QUESTION.

Write exactly {CHECKLIST_SIZE} distinct answer obligations: the separate things a complete, accurate
explanation of this question must cover. Rules:
- Derive them from the question alone. Cover every part of it (for example both "how does X work"
  and "what can go wrong / what are the limits"), but stay at the level a clear introductory
  explanation would reach. Do not add adjacent topics the question does not ask for (for example
  cost, latency, security or evaluation methods) unless the question names them.
- Each item must be ONE central idea, not a bundle of sub-topics or a long list of examples.
- Order them from foundations to limitations or trade-offs.
- Each item is one short, concrete phrase (at most 15 words) naming something the script must explain.
  Do not write the answer itself, cite sources, or mention any system, script or scoring.
- Items must not overlap and must not be vague (no "general background" or "conclusion").
Return structured output only."""


class QuestionChecklist(BaseModel):
    aspects: list[str] = Field(default_factory=list)


def checklist_problems(aspects: list[str]) -> list[str]:
    problems: list[str] = []
    cleaned = [a.strip() for a in aspects]
    if len(cleaned) != CHECKLIST_SIZE:
        problems.append(f"expected exactly {CHECKLIST_SIZE} aspects, got {len(cleaned)}")
    if any(not a for a in cleaned):
        problems.append("contains an empty aspect")
    if len({a.casefold() for a in cleaned}) != len(cleaned):
        problems.append("contains duplicate aspects")
    return problems


def _read_cached(path: Path, question_sha256: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("question_sha256") != question_sha256:
        return None  # the question changed since the checklist was written
    if checklist_problems(data.get("aspects") or []):
        return None
    stored = data.get("aspects_sha256")
    if stored and stored != aspects_sha256(data["aspects"]):
        raise ChecklistIntegrityError(
            f"{path} was edited after it was derived (aspects_sha256 no longer matches). "
            "Restore the original checklist or start a new experiment."
        )
    return data


class ChecklistIntegrityError(RuntimeError):
    """A frozen checklist changed or disappeared; systems must not be judged against new ones."""


class ChecklistMissing(ChecklistIntegrityError):
    """No valid saved checklist exists and deriving a new one was not allowed."""


def aspects_sha256(aspects: list[str]) -> str:
    return sha256_text(json.dumps(aspects, ensure_ascii=False))


def load_or_derive_checklist(
    llm: Any,
    *,
    question: str,
    prompt_id: str,
    directory: Path,
    allow_derive: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Return ``(aspects, metadata)``; derive and save the checklist the first time only.

    The derivation call is not added to any system's evaluation telemetry (it would unfairly
    charge whichever system happened to be evaluated first); its cost is stored in the file.
    With ``allow_derive=False`` (an experiment whose checklists are already frozen) a missing
    or unreadable checklist raises instead of being silently re-derived.
    """
    question_sha256 = sha256_text(question)
    path = directory / f"{prompt_id}.json"
    cached = _read_cached(path, question_sha256)
    if cached is not None:
        return list(cached["aspects"]), _metadata(cached, path, derived_now=False)
    if not allow_derive:
        raise ChecklistMissing(
            f"Frozen checklist {path} is missing, unreadable or no longer matches its question. "
            "Restore it or start a new experiment."
        )

    prompt = f"USER QUESTION:\n{question}"
    problems: list[str] = []
    for attempt in range(DERIVATION_RETRIES + 1):
        note = (
            ""
            if not problems
            else "\n\nYour previous checklist was rejected: " + "; ".join(problems)
            + f". Return exactly {CHECKLIST_SIZE} distinct, non-empty aspects."
        )
        result, log = llm.generate_structured(
            task="question_coverage_checklist",
            system_instruction=SYSTEM,
            prompt=prompt + note,
            schema=QuestionChecklist,
            max_output_tokens=800,
        )
        aspects = [a.strip() for a in result.aspects]
        problems = checklist_problems(aspects)
        if not problems:
            payload = {
                "prompt_id": prompt_id,
                "question_sha256": question_sha256,
                "aspects": aspects,
                "aspects_sha256": aspects_sha256(aspects),
                "derived_at": utc_now(),
                "derived_by_model": getattr(llm, "resolved_model", None) or getattr(llm, "model", ""),
                "derivation_attempts": attempt + 1,
                "derivation_cost_usd": getattr(log, "total_cost_usd", None),
                "note": (
                    "Derived from the question text alone by the evaluator; shared by every "
                    "system in this experiment. Never shown to generators."
                ),
            }
            json_dump(path, payload)
            return aspects, _metadata(payload, path, derived_now=True)
    raise ValueError(
        f"Could not derive a valid {CHECKLIST_SIZE}-aspect checklist for {prompt_id}: "
        + "; ".join(problems)
    )


def _metadata(payload: dict[str, Any], path: Path, *, derived_now: bool) -> dict[str, Any]:
    return {
        "source": "derived_once_per_question_by_evaluator",
        "aspects_sha256": payload.get("aspects_sha256"),
        "derived_at": payload.get("derived_at"),
        "derived_by_model": payload.get("derived_by_model"),
        "derivation_cost_usd": payload.get("derivation_cost_usd"),
        "derived_in_this_call": derived_now,
        "file": str(path),
    }
