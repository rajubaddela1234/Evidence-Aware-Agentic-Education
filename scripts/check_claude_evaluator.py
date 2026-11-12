from __future__ import annotations

import argparse

from evidence_aware_edu.config import get_settings
from evidence_aware_edu.llm.claude import ClaudeEvaluatorClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Claude evaluator access path.")
    parser.add_argument(
        "--access",
        choices=["cli", "api"],
        default="",
        help="Use Claude Code login (cli) or ANTHROPIC_API_KEY (api).",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.access:
        settings = settings.model_copy(update={"claude_access_method": args.access})

    settings.validate_independent_evaluator()
    client = ClaudeEvaluatorClient(settings)
    text, log = client.generate_text(
        task="claude_connection_check",
        system_instruction="You are a connection test. Follow the user exactly.",
        prompt="Reply with exactly CLAUDE_WORKING",
        max_output_tokens=100,
    )

    print(f"Access: {settings.claude_access_method}")
    print(f"Requested model: {settings.claude_evaluator_model}")
    print(f"Resolved model: {log.model}")
    if client.cli_version:
        print(f"Claude CLI: {client.cli_version}")
    print(f"Result: {text}")
    print(f"Latency: {log.latency_seconds:.2f}s")
    if settings.claude_access_method == "cli":
        print(f"CLI reported cost equivalent: ${log.total_cost_usd:.6f}")
    elif log.total_cost_usd:
        print(f"Configured API cost estimate: ${log.total_cost_usd:.6f}")


if __name__ == "__main__":
    main()
