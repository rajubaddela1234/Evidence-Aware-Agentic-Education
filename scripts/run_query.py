from __future__ import annotations

import argparse

from evidence_aware_edu.config import get_settings
from evidence_aware_edu.pipelines.factory import make_pipeline
from evidence_aware_edu.schemas import BenchmarkPrompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="P1")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    settings = get_settings().model_copy(update={"use_frozen_reference_pool": False})
    prompt = BenchmarkPrompt(
        prompt_id="DEMO",
        domain="Demo",
        reasoning_type="demo",
        question=args.query,
    )
    result = make_pipeline(args.system, settings).run(prompt)
    print(result.script)
    print(
        f"\nSources: {len(result.sources)} | LLM calls: {result.telemetry.total_llm_calls} "
        f"| Cost: ${result.telemetry.estimated_llm_cost_usd:.5f} "
        f"| Latency: {result.telemetry.wall_seconds:.2f}s "
        f"| LLM P50/P90: {result.telemetry.llm_call_latency_p50_seconds:.2f}s/"
        f"{result.telemetry.llm_call_latency_p90_seconds:.2f}s"
    )


if __name__ == "__main__":
    main()
