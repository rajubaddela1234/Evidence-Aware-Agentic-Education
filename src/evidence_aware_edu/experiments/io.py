from __future__ import annotations

import json
from pathlib import Path

from evidence_aware_edu.schemas import BenchmarkPrompt


def load_benchmark(path: Path) -> list[BenchmarkPrompt]:
    rows: list[BenchmarkPrompt] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(BenchmarkPrompt.model_validate(json.loads(line)))
    return rows
