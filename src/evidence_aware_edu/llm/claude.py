from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from evidence_aware_edu.config import Settings
from evidence_aware_edu.costs import estimate_token_cost
from evidence_aware_edu.observability import trace_span
from evidence_aware_edu.schemas import LLMCallLog

T = TypeVar("T", bound=BaseModel)


class ClaudeEvaluatorClient:
    """Independent Claude evaluator with CLI and API access paths."""

    def __init__(self, settings: Settings, model: str | None = None) -> None:
        self.settings = settings
        self.settings.validate_independent_evaluator()
        self.model = model or settings.evaluator_model
        self.access_method = settings.claude_access_method
        self.resolved_model = ""
        self.executable: str | None = None
        self.cli_version = ""
        self._help = ""
        self._api_client = None

        if self.access_method == "cli":
            self.executable = shutil.which("claude") or shutil.which("claude.cmd")
            if not self.executable:
                raise RuntimeError(
                    "Claude Code was not found on PATH. Install it with "
                    "`npm install -g @anthropic-ai/claude-code`, run `claude`, and sign in."
                )
            self._help = self._read_help()
            self.cli_version = self._read_version()
        else:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "The Anthropic Python package is required when CLAUDE_ACCESS_METHOD=api."
                ) from exc
            self._api_client = Anthropic(api_key=settings.anthropic_api_key)

    @classmethod
    def cli_is_available(cls) -> bool:
        return bool(shutil.which("claude") or shutil.which("claude.cmd"))

    def _launch_command(self, args: list[str]) -> list[str]:
        executable = str(self.executable or "claude")
        if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
            command_line = subprocess.list2cmdline([executable, *args])
            return ["cmd.exe", "/d", "/s", "/c", command_line]
        return [executable, *args]

    def _read_help(self) -> str:
        try:
            result = subprocess.run(
                self._launch_command(["--help"]),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            return f"{result.stdout}\n{result.stderr}"
        except Exception:
            return ""

    def _read_version(self) -> str:
        try:
            result = subprocess.run(
                self._launch_command(["--version"]),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            return (result.stdout or result.stderr or "").strip()
        except Exception:
            return ""

    def _supports(self, flag: str) -> bool:
        return flag in self._help

    def _trace_inputs(self, system_instruction: str, prompt: str) -> dict[str, Any]:
        if self.settings.langsmith_trace_content:
            return {
                "system_instruction": system_instruction,
                "prompt": prompt,
                "requested_model": self.model,
                "access_method": self.access_method,
            }
        return {
            "requested_model": self.model,
            "access_method": self.access_method,
            "system_instruction_chars": len(system_instruction),
            "prompt_chars": len(prompt),
        }

    @staticmethod
    def _json_from_text(text: str):
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            starts = [x for x in (text.find("{"), text.find("[")) if x >= 0]
            if not starts:
                raise
            start = min(starts)
            for end in range(len(text), start, -1):
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    continue
            raise

    @staticmethod
    def _non_retryable(message: str) -> bool:
        lowered = message.lower()
        markers = (
            "failed to authenticate",
            "oauth session expired",
            "authentication",
            "invalid x-api-key",
            "invalid api key",
            "not logged in",
            "login required",
            "billing",
            "usage limit",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _first_model_info(payload: dict) -> tuple[str, dict]:
        usage = payload.get("modelUsage") or {}
        if not isinstance(usage, dict) or not usage:
            return "", {}
        name, info = next(iter(usage.items()))
        if not isinstance(info, dict):
            info = {}
        return str(info.get("canonicalModel") or name or ""), info

    def _cli_log(
        self,
        *,
        task: str,
        payload: dict,
        latency: float,
        attempts: int,
    ) -> LLMCallLog:
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}

        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        raw_output = int(usage.get("output_tokens", 0) or 0)
        details = usage.get("output_tokens_details") or {}
        thinking = int(details.get("thinking_tokens", 0) or 0) if isinstance(details, dict) else 0
        visible_output = max(raw_output - thinking, 0) if thinking else raw_output
        prompt_tokens = input_tokens + cache_creation + cache_read

        resolved, model_info = self._first_model_info(payload)
        if resolved:
            self.resolved_model = resolved

        reported_cost = float(payload.get("total_cost_usd", 0.0) or 0.0)
        if reported_cost == 0.0 and isinstance(model_info, dict):
            reported_cost = float(model_info.get("costUSD", 0.0) or 0.0)

        return LLMCallLog(
            task=task,
            model=resolved or self.model,
            provider="anthropic",
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            output_tokens=visible_output,
            thinking_tokens=thinking,
            total_tokens=prompt_tokens + raw_output,
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_cost_usd=reported_cost,
            attempts=attempts,
            metadata={
                "access_method": "cli",
                "access_mode": "claude_code_cli",
                "requested_model": self.model,
                "resolved_model": resolved or self.model,
                "cli_reported_cost_usd": reported_cost,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "session_id": payload.get("session_id", ""),
                "claude_cli_version": self.cli_version,
                "cost_semantics": "Claude Code reported list-price equivalent; subscription billing may differ",
            },
        )

    def _api_log(
        self,
        *,
        task: str,
        response: Any,
        latency: float,
        attempts: int,
    ) -> LLMCallLog:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
        details = getattr(usage, "output_tokens_details", None) if usage else None
        thinking = int(getattr(details, "thinking_tokens", 0) or 0) if details else 0
        visible_output = max(output_tokens - thinking, 0) if thinking else output_tokens
        resolved = str(getattr(response, "model", "") or self.model)
        self.resolved_model = resolved

        input_rate, output_rate = self.settings.anthropic_pricing()
        cost = estimate_token_cost(
            prompt_tokens + cache_creation + cache_read,
            output_tokens,
            input_rate,
            output_rate,
        )

        return LLMCallLog(
            task=task,
            model=resolved,
            provider="anthropic",
            latency_seconds=latency,
            prompt_tokens=prompt_tokens + cache_creation + cache_read,
            output_tokens=visible_output,
            thinking_tokens=thinking,
            total_tokens=prompt_tokens + cache_creation + cache_read + output_tokens,
            input_cost_usd=cost.input_cost_usd,
            output_cost_usd=cost.output_cost_usd,
            total_cost_usd=cost.total_cost_usd,
            attempts=attempts,
            metadata={
                "access_method": "api",
                "access_mode": "anthropic_api",
                "requested_model": self.model,
                "resolved_model": resolved,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "message_id": str(getattr(response, "id", "") or ""),
                "cost_semantics": "configured API list-price estimate",
            },
        )

    @staticmethod
    def _usage_for_langsmith(log: LLMCallLog) -> dict:
        return {
            "input_tokens": log.prompt_tokens,
            "output_tokens": log.output_tokens + log.thinking_tokens,
            "total_tokens": log.total_tokens,
            "output_token_details": {"reasoning": log.thinking_tokens},
            "total_cost": log.total_cost_usd,
        }

    def _invoke_cli(
        self,
        *,
        task: str,
        system_instruction: str,
        prompt: str,
        schema_json: str | None,
    ) -> tuple[dict, float]:
        with tempfile.TemporaryDirectory(prefix="eaa_claude_eval_") as tmp:
            tmp_path = Path(tmp)
            system_path = tmp_path / "system_prompt.txt"
            system_path.write_text(system_instruction, encoding="utf-8")

            args = ["-p", "--model", self.model, "--output-format", "json"]
            if self._supports("--system-prompt-file"):
                args.extend(["--system-prompt-file", str(system_path)])
            elif self._supports("--system-prompt"):
                args.extend(["--system-prompt", system_instruction])
            else:
                prompt = f"SYSTEM INSTRUCTIONS:\n{system_instruction}\n\nUSER INPUT:\n{prompt}"

            # The evaluator receives all evidence in the prompt, so tools are disabled.
            if self._supports("--tools"):
                args.extend(["--tools", ""])
            if self._supports("--disallowedTools"):
                args.extend(["--disallowedTools", "mcp__*"])
            if self._supports("--no-chrome"):
                args.append("--no-chrome")
            if self._supports("--no-session-persistence"):
                args.append("--no-session-persistence")
            if self._supports("--restricted"):
                args.append("--restricted")
            if schema_json is not None and self._supports("--json-schema"):
                args.extend(["--json-schema", schema_json])

            env = os.environ.copy()
            # CLI mode should use the logged-in Claude account, not a stray API key.
            env.pop("ANTHROPIC_API_KEY", None)

            if self.settings.claude_evaluator_show_progress:
                print(f"    Claude evaluator: {task}...", flush=True)

            started = time.perf_counter()
            try:
                process = subprocess.run(
                    self._launch_command(args),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.settings.claude_evaluator_timeout_seconds,
                    cwd=tmp,
                    env=env,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Claude Code timed out for task={task} after "
                    f"{self.settings.claude_evaluator_timeout_seconds}s."
                ) from exc

            latency = time.perf_counter() - started
            stdout = (process.stdout or "").strip()
            stderr = (process.stderr or "").strip()
            if not stdout:
                raise RuntimeError(
                    f"Claude Code returned no JSON for task={task}. "
                    f"Exit={process.returncode}. stderr={stderr[:1200]}"
                )
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Claude Code returned invalid wrapper JSON for task={task}: {stdout[:1200]}"
                ) from exc
            if process.returncode != 0 or payload.get("is_error"):
                message = str(payload.get("result") or stderr or payload)
                raise RuntimeError(f"Claude Code failed for task={task}: {message}")

            if self.settings.claude_evaluator_show_progress:
                resolved, _ = self._first_model_info(payload)
                suffix = f" · {resolved}" if resolved else ""
                print(f"    Claude evaluator: {task} complete in {latency:.1f}s{suffix}", flush=True)

            return payload, latency

    @staticmethod
    def _message_text(response: Any) -> str:
        chunks: list[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", "") == "text":
                chunks.append(str(getattr(block, "text", "") or ""))
        return "\n".join(x for x in chunks if x).strip()

    def _run_api_text(
        self,
        *,
        task: str,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int | None,
    ) -> tuple[str, LLMCallLog]:
        assert self._api_client is not None
        last_error: Exception | None = None
        for attempt in range(self.settings.claude_evaluator_retry_attempts):
            started = time.perf_counter()
            try:
                response = self._api_client.messages.create(
                    model=self.model,
                    max_tokens=max_output_tokens or self.settings.claude_evaluator_max_output_tokens,
                    system=system_instruction,
                    messages=[{"role": "user", "content": prompt}],
                )
                latency = time.perf_counter() - started
                text = self._message_text(response)
                if not text:
                    raise RuntimeError("Anthropic Messages API returned no text output.")
                log = self._api_log(
                    task=task,
                    response=response,
                    latency=latency,
                    attempts=attempt + 1,
                )
                return text, log
            except Exception as exc:
                last_error = exc
                if self._non_retryable(str(exc)) or attempt + 1 >= self.settings.claude_evaluator_retry_attempts:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Anthropic API call failed for task={task}: {last_error}") from last_error

    def _run_api_structured(
        self,
        *,
        task: str,
        system_instruction: str,
        prompt: str,
        schema: Type[T],
        max_output_tokens: int | None,
    ) -> tuple[T, LLMCallLog]:
        assert self._api_client is not None
        last_error: Exception | None = None
        schema_dict = schema.model_json_schema()
        for attempt in range(self.settings.claude_evaluator_retry_attempts):
            started = time.perf_counter()
            try:
                response = self._api_client.messages.create(
                    model=self.model,
                    max_tokens=max_output_tokens or self.settings.claude_evaluator_max_output_tokens,
                    system=system_instruction,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={
                        "format": {
                            "type": "json_schema",
                            "schema": schema_dict,
                        }
                    },
                )
                latency = time.perf_counter() - started
                text = self._message_text(response)
                if not text:
                    raise RuntimeError("Anthropic Messages API returned no structured output.")
                obj = schema.model_validate(self._json_from_text(text))
                log = self._api_log(
                    task=task,
                    response=response,
                    latency=latency,
                    attempts=attempt + 1,
                )
                return obj, log
            except Exception as exc:
                last_error = exc
                if self._non_retryable(str(exc)) or attempt + 1 >= self.settings.claude_evaluator_retry_attempts:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"Structured Anthropic API call failed for task={task}: {last_error}"
        ) from last_error

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
            f"claude:{task}",
            run_type="llm",
            inputs=self._trace_inputs(system_instruction, prompt),
            metadata={
                "ls_provider": "anthropic",
                "ls_model_name": self.model,
                "task": task,
                "evaluation_role": "independent_provider",
                "access_method": self.access_method,
            },
            tags=["evidence-aware-education", "anthropic", "claude", "evaluation", task],
        ) as span:
            if self.access_method == "api":
                text, log = self._run_api_text(
                    task=task,
                    system_instruction=system_instruction,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                )
            else:
                last_error: Exception | None = None
                for attempt in range(self.settings.claude_evaluator_retry_attempts):
                    try:
                        payload, latency = self._invoke_cli(
                            task=task,
                            system_instruction=system_instruction,
                            prompt=prompt,
                            schema_json=None,
                        )
                        text = str(payload.get("result", "")).strip()
                        if not text:
                            raise RuntimeError(f"Claude Code returned an empty result for task={task}.")
                        log = self._cli_log(
                            task=task,
                            payload=payload,
                            latency=latency,
                            attempts=attempt + 1,
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        if self._non_retryable(str(exc)) or attempt + 1 >= self.settings.claude_evaluator_retry_attempts:
                            raise RuntimeError(
                                f"Claude Code call failed for task={task}: {last_error}"
                            ) from last_error
                        time.sleep(min(2 ** attempt, 8))

            span.set_usage(self._usage_for_langsmith(log))
            span.set_outputs(
                {"text": text}
                if self.settings.langsmith_trace_content
                else {"output_chars": len(text)}
            )
            span.add_metadata({"resolved_model": log.model, "access_method": self.access_method})
            return text, log

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
            f"claude:{task}",
            run_type="llm",
            inputs=self._trace_inputs(system_instruction, prompt),
            metadata={
                "ls_provider": "anthropic",
                "ls_model_name": self.model,
                "task": task,
                "response_schema": schema.__name__,
                "evaluation_role": "independent_provider",
                "access_method": self.access_method,
            },
            tags=["evidence-aware-education", "anthropic", "claude", "evaluation", "structured", task],
        ) as span:
            if self.access_method == "api":
                obj, log = self._run_api_structured(
                    task=task,
                    system_instruction=system_instruction,
                    prompt=prompt,
                    schema=schema,
                    max_output_tokens=max_output_tokens,
                )
            else:
                schema_json = json.dumps(
                    schema.model_json_schema(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                fallback_prompt = (
                    f"{prompt}\n\nReturn only valid JSON matching this JSON Schema. "
                    "Do not use Markdown fences or add prose outside the JSON.\n"
                    f"SCHEMA:\n{schema_json}"
                )
                last_error: Exception | None = None
                for attempt in range(self.settings.claude_evaluator_retry_attempts):
                    try:
                        supports_schema = self._supports("--json-schema")
                        payload, latency = self._invoke_cli(
                            task=task,
                            system_instruction=system_instruction,
                            prompt=prompt if supports_schema else fallback_prompt,
                            schema_json=schema_json if supports_schema else None,
                        )
                        data = payload.get("structured_output") if supports_schema else None
                        if data is None:
                            data = self._json_from_text(str(payload.get("result", "")))
                        elif isinstance(data, str):
                            data = self._json_from_text(data)
                        obj = schema.model_validate(data)
                        log = self._cli_log(
                            task=task,
                            payload=payload,
                            latency=latency,
                            attempts=attempt + 1,
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        if self._non_retryable(str(exc)) or attempt + 1 >= self.settings.claude_evaluator_retry_attempts:
                            raise RuntimeError(
                                f"Structured Claude Code call failed for task={task}: {last_error}"
                            ) from last_error
                        time.sleep(min(2 ** attempt, 8))

            span.set_usage(self._usage_for_langsmith(log))
            span.set_outputs(
                obj.model_dump()
                if self.settings.langsmith_trace_content
                else {"schema": schema.__name__}
            )
            span.add_metadata({"resolved_model": log.model, "access_method": self.access_method})
            return obj, log
