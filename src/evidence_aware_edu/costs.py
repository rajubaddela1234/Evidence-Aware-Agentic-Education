from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def estimate_token_cost(
    prompt_tokens: int,
    output_tokens: int,
    input_usd_per_1m: float,
    output_usd_per_1m: float,
) -> CostBreakdown:
    """Estimate token cost from the configured per-million-token rates."""
    input_cost = max(prompt_tokens, 0) / 1_000_000 * input_usd_per_1m
    output_cost = max(output_tokens, 0) / 1_000_000 * output_usd_per_1m
    return CostBreakdown(
        input_cost_usd=float(input_cost),
        output_cost_usd=float(output_cost),
        total_cost_usd=float(input_cost + output_cost),
    )


def estimate_gemini_cost(
    prompt_tokens: int,
    output_tokens: int,
    input_usd_per_1m: float,
    output_usd_per_1m: float,
) -> CostBreakdown:
    """Gemini cost helper kept for the existing generation client."""
    return estimate_token_cost(
        prompt_tokens,
        output_tokens,
        input_usd_per_1m,
        output_usd_per_1m,
    )
