from __future__ import annotations

from time import perf_counter

from .schemas import LLMCallLog, RunTelemetry


class TelemetryCollector:
    def __init__(self) -> None:
        self.data = RunTelemetry()
        self._start = perf_counter()

    def add_llm(self, log: LLMCallLog) -> None:
        self.data.llm_calls.append(log)

    def add_retrieval(self, documents: int) -> None:
        self.data.retrieval_calls += 1
        self.data.retrieved_documents += documents

    def finish(self) -> RunTelemetry:
        self.data.wall_seconds = perf_counter() - self._start
        from datetime import datetime, timezone
        self.data.ended_at = datetime.now(timezone.utc).isoformat()
        return self.data
