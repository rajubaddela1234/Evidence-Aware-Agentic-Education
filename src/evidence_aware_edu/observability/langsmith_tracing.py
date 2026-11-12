from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from evidence_aware_edu.config import Settings

try:  # Keep tests usable when LangSmith is not installed.
    from langsmith import Client
    from langsmith.run_helpers import trace
except Exception:  # pragma: no cover
    Client = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]


class _NullSpan:
    def set_outputs(self, outputs: dict[str, Any]) -> None:
        return None

    def set_usage(self, usage: dict[str, Any]) -> None:
        return None

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        return None


class _LangSmithSpan:
    def __init__(self, run_tree: Any) -> None:
        self.run_tree = run_tree

    def set_outputs(self, outputs: dict[str, Any]) -> None:
        if self.run_tree is not None:
            self.run_tree.set(outputs=outputs)

    def set_usage(self, usage: dict[str, Any]) -> None:
        if self.run_tree is not None:
            self.run_tree.set(usage_metadata=usage)

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        if self.run_tree is not None:
            self.run_tree.add_metadata(metadata)


@lru_cache(maxsize=8)
def _client(api_key: str, endpoint: str):
    if Client is None:
        return None
    return Client(api_key=api_key, api_url=endpoint)


def configure_langsmith(settings: Settings) -> None:
    """Apply the project's tracing choice to the current Python process."""
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        # LangSmith can also be enabled by shell-level variables left over from
        # another project.  Turn those off explicitly so a disabled local setting
        # really means no background uploads.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGSMITH_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


@contextmanager
def trace_span(
    settings: Settings,
    name: str,
    *,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[_NullSpan | _LangSmithSpan]:
    """Create a LangSmith span, or a no-op span when tracing is disabled."""
    if (
        not settings.langsmith_tracing
        or not settings.langsmith_api_key
        or trace is None
        or Client is None
    ):
        yield _NullSpan()
        return

    client = _client(settings.langsmith_api_key, settings.langsmith_endpoint)
    with trace(
        name=name,
        run_type=run_type,
        inputs=inputs or {},
        metadata=metadata or {},
        tags=tags or [],
        project_name=settings.langsmith_project,
        client=client,
    ) as run_tree:
        yield _LangSmithSpan(run_tree)
