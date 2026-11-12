from __future__ import annotations

import time
from typing import Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from evidence_aware_edu.config import Settings
from evidence_aware_edu.costs import estimate_gemini_cost
from evidence_aware_edu.observability import trace_span
from evidence_aware_edu.schemas import LLMCallLog

T = TypeVar("T", bound=BaseModel)


class EmptyGeneration(RuntimeError):
    """Gemini returned no text. Retried like a transient failure."""


class TruncatedGeneration(RuntimeError):
    """Gemini stopped at the output-token limit. Not retried: the same limit would cut it again."""


def _finish_reason(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    return str(getattr(candidates[0], "finish_reason", "") or "")


class GeminiClient:
    """Small Gemini wrapper with telemetry and optional LangSmith tracing."""

    def __init__(self, settings: Settings, model: str | None = None) -> None:
        self.settings = settings
        self.model = model or settings.gemini_model
        if settings.generation_provider == "gemini":
            settings.validate_generation()
        else:
            # Reference-pool preparation intentionally uses Gemini for both study arms.
            settings.validate_reference_pool_builder()
        self.client = genai.Client(api_key=settings.gemini_api_key)

    @staticmethod
    def _usage(response) -> tuple[int, int, int, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return 0, 0, 0, 0
        prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
        output = int(
            getattr(usage, "candidates_token_count", 0)
            or getattr(usage, "response_token_count", 0)
            or 0
        )
        thinking = int(getattr(usage, "thoughts_token_count", 0) or 0)
        total = int(
            getattr(usage, "total_token_count", 0)
            or (prompt + output + thinking)
        )
        return prompt, output, thinking, total

    def _config(
        self,
        system_instruction: str,
        max_output_tokens: int | None = None,
        response_schema: Type[BaseModel] | None = None,
    ) -> types.GenerateContentConfig:
        kwargs = {
            "system_instruction": system_instruction,
            "max_output_tokens": max_output_tokens or self.settings.max_output_tokens,
            "thinking_config": types.ThinkingConfig(
                thinking_level=self.settings.thinking_level
            ),
        }
        if response_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema
        return types.GenerateContentConfig(**kwargs)

    def _trace_inputs(self, system_instruction: str, prompt: str) -> dict:
        if self.settings.langsmith_trace_content:
            return {
                "system_instruction": system_instruction,
                "prompt": prompt,
                "model": self.model,
            }
        return {
            "model": self.model,
            "system_instruction_chars": len(system_instruction),
            "prompt_chars": len(prompt),
        }

    def _log(
        self,
        *,
        task: str,
        latency: float,
        prompt_tokens: int,
        output_tokens: int,
        thinking_tokens: int,
        total_tokens: int,
        attempts: int = 1,
    ) -> LLMCallLog:
        input_rate, output_rate = self.settings.gemini_pricing()
        billable_output_tokens = output_tokens + thinking_tokens
        cost = estimate_gemini_cost(
            prompt_tokens,
            billable_output_tokens,
            input_rate,
            output_rate,
        )
        return LLMCallLog(
            task=task,
            model=self.model,
            provider="google",
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            total_tokens=total_tokens,
            input_cost_usd=cost.input_cost_usd,
            output_cost_usd=cost.output_cost_usd,
            total_cost_usd=cost.total_cost_usd,
            attempts=attempts,
        )

    def _usage_for_langsmith(self, log: LLMCallLog) -> dict:
        return {
            "input_tokens": log.prompt_tokens,
            "output_tokens": log.output_tokens + log.thinking_tokens,
            "total_tokens": log.total_tokens,
            "output_token_details": {"reasoning": log.thinking_tokens},
            "input_cost": log.input_cost_usd,
            "output_cost": log.output_cost_usd,
            "total_cost": log.total_cost_usd,
        }

    def generate_text(
        self,
        *,
        task: str,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int | None = None,
    ) -> tuple[str, LLMCallLog]:
        with trace_span(
            self.settings,
            f"gemini:{task}",
            run_type="llm",
            inputs=self._trace_inputs(system_instruction, prompt),
            metadata={
                "ls_provider": "google",
                "ls_model_name": self.model,
                "task": task,
                "thinking_level": self.settings.thinking_level,
            },
            tags=["evidence-aware-education", "gemini", task],
        ) as span:
            last_error: Exception | None = None
            for attempt in range(self.settings.api_retry_attempts):
                started = time.perf_counter()
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=self._config(system_instruction, max_output_tokens),
                    )
                    latency = time.perf_counter() - started
                    text = (response.text or "").strip()
                    reason = _finish_reason(response)
                    if "MAX_TOKENS" in reason.upper():
                        raise TruncatedGeneration(
                            f"Gemini output for task={task} hit the output-token limit "
                            f"({max_output_tokens}); the text would be an incomplete script."
                        )
                    if not text:
                        raise EmptyGeneration(
                            f"Gemini returned no text for task={task} (finish_reason={reason or 'n/a'})."
                        )
                    p, o, th, t = self._usage(response)
                    log = self._log(
                        task=task,
                        latency=latency,
                        prompt_tokens=p,
                        output_tokens=o,
                        thinking_tokens=th,
                        total_tokens=t,
                        attempts=attempt + 1,
                    )
                    span.set_usage(self._usage_for_langsmith(log))
                    span.set_outputs(
                        {"text": text} if self.settings.langsmith_trace_content else {
                            "output_chars": len(text)
                        }
                    )
                    span.add_metadata({"attempts": attempt + 1})
                    return text, log
                except TruncatedGeneration:
                    raise
                except Exception as exc:  # SDK error classes change across releases.
                    last_error = exc
                    if attempt + 1 >= self.settings.api_retry_attempts:
                        break
                    time.sleep(min(2 ** attempt, 30))
            raise RuntimeError(f"Gemini call failed for task={task}: {last_error}") from last_error

    def generate_structured(
        self,
        *,
        task: str,
        system_instruction: str,
        prompt: str,
        schema: Type[T],
        max_output_tokens: int | None = None,
    ) -> tuple[T, LLMCallLog]:
        with trace_span(
            self.settings,
            f"gemini:{task}",
            run_type="llm",
            inputs=self._trace_inputs(system_instruction, prompt),
            metadata={
                "ls_provider": "google",
                "ls_model_name": self.model,
                "task": task,
                "response_schema": schema.__name__,
                "thinking_level": self.settings.thinking_level,
            },
            tags=["evidence-aware-education", "gemini", "structured", task],
        ) as span:
            last_error: Exception | None = None
            for attempt in range(self.settings.api_retry_attempts):
                started = time.perf_counter()
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=self._config(
                            system_instruction,
                            max_output_tokens,
                            response_schema=schema,
                        ),
                    )
                    latency = time.perf_counter() - started
                    parsed = getattr(response, "parsed", None)
                    if isinstance(parsed, schema):
                        obj = parsed
                    else:
                        obj = schema.model_validate_json(response.text)
                    p, o, th, t = self._usage(response)
                    log = self._log(
                        task=task,
                        latency=latency,
                        prompt_tokens=p,
                        output_tokens=o,
                        thinking_tokens=th,
                        total_tokens=t,
                        attempts=attempt + 1,
                    )
                    span.set_usage(self._usage_for_langsmith(log))
                    span.set_outputs(
                        obj.model_dump() if self.settings.langsmith_trace_content else {
                            "schema": schema.__name__
                        }
                    )
                    span.add_metadata({"attempts": attempt + 1})
                    return obj, log
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 >= self.settings.api_retry_attempts:
                        break
                    time.sleep(min(2 ** attempt, 30))
            raise RuntimeError(
                f"Structured Gemini call failed for task={task}: {last_error}"
            ) from last_error
