from __future__ import annotations

from typing import Any
from evidence_aware_edu.telemetry import TelemetryCollector


class AgentBase:
    def __init__(self, llm: Any, telemetry: TelemetryCollector) -> None:
        self.llm = llm
        self.telemetry = telemetry

    def _record(self, log) -> None:
        self.telemetry.add_llm(log)
