"""Enforce the 0 / 0.5 / 1 aspect-scoring rubric on structured judge output."""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable

ALLOWED_ASPECT_SCORES = (0.0, 0.5, 1.0)
DEFAULT_RUBRIC_RETRIES = 2


class RubricViolation(ValueError):
    """Raised when a judge keeps returning scores outside the allowed rubric values."""


def is_rubric_score(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return any(math.isclose(number, allowed, abs_tol=1e-9) for allowed in ALLOWED_ASPECT_SCORES)


def invalid_aspect_scores(items: Iterable[Any], label: str) -> list[str]:
    """Describe every aspect judgment whose score is not exactly 0, 0.5 or 1."""
    return [
        f"{label}: '{item.aspect}' scored {item.score!r}"
        for item in items
        if not is_rubric_score(item.score)
    ]


def generate_with_rubric(
    llm: Any,
    *,
    prompt: str,
    validate: Callable[[Any], list[str]],
    retries: int = DEFAULT_RUBRIC_RETRIES,
    record: Callable[[Any], None] | None = None,
    rule: str = "aspect scores must be exactly 0, 0.5 or 1",
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Call ``llm.generate_structured`` and re-ask when ``validate`` reports problems.

    ``validate`` returns a list of human-readable problems (empty means valid). Logs from
    rejected attempts are passed to ``record`` so their cost still appears in telemetry;
    the log of the accepted attempt is returned to the caller as usual.
    """
    current_prompt = prompt
    problems: list[str] = []
    for attempt in range(retries + 1):
        result, log = llm.generate_structured(prompt=current_prompt, **kwargs)
        problems = validate(result)
        if not problems:
            return result, log
        if record is not None:
            record(log)
        current_prompt = (
            f"{prompt}\n\nRUBRIC CORRECTION (attempt {attempt + 2}): your previous answer was "
            f"rejected because {rule}. Problems:\n- "
            + "\n- ".join(problems[:12])
            + f"\nReturn the complete answer again; {rule}."
        )
    raise RubricViolation(
        f"Judge returned off-rubric or incomplete judgements after {retries + 1} attempts: "
        + "; ".join(problems[:6])
    )
