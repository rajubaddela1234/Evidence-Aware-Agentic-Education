import json
import sys
from types import ModuleType, SimpleNamespace

from evidence_aware_edu.config import Settings
from evidence_aware_edu.llm.claude import ClaudeEvaluatorClient
from evidence_aware_edu.schemas import ClaimSet


def _settings(**updates):
    values = {
        "GEMINI_API_KEY": "dummy-google",
        "CLAUDE_ACCESS_METHOD": "cli",
        "CLAUDE_EVALUATOR_MODEL": "claude-sonnet-5-5",
        "CLAUDE_EVALUATOR_SHOW_PROGRESS": False,
    }
    values.update(updates)
    return Settings(**values)


def _claim_payload():
    return {
        "claims": [
            {
                "claim_id": "c1",
                "text": "RAG retrieves external information.",
                "needs_citation": True,
                "importance": "core",
                "citation_ids": [],
            }
        ]
    }


def test_structured_claude_cli_output_is_validated(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude")
    help_text = "--output-format --system-prompt-file --tools --json-schema --no-session-persistence"

    def fake_run(command, **kwargs):
        if "--help" in command:
            return SimpleNamespace(returncode=0, stdout=help_text, stderr="")
        if "--version" in command:
            return SimpleNamespace(returncode=0, stdout="2.1.test", stderr="")
        payload = {
            "is_error": False,
            "result": "ok",
            "structured_output": _claim_payload(),
            "total_cost_usd": 0.01,
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 2,
                "output_tokens": 7,
                "output_tokens_details": {"thinking_tokens": 0},
            },
            "modelUsage": {
                "claude-sonnet-test": {
                    "canonicalModel": "claude-sonnet-test",
                    "costUSD": 0.01,
                }
            },
        }
        env = kwargs.get("env", {})
        assert "ANTHROPIC_API_KEY" not in env
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-reach-cli")

    client = ClaudeEvaluatorClient(_settings())
    result, log = client.generate_structured(
        task="script_claims",
        system_instruction="Extract claims.",
        prompt="Example script.",
        schema=ClaimSet,
    )

    assert result.claims[0].claim_id == "c1"
    assert log.provider == "anthropic"
    assert log.model == "claude-sonnet-test"
    assert log.prompt_tokens == 15
    assert log.total_cost_usd == 0.01
    assert log.metadata["access_method"] == "cli"
    assert client.resolved_model == "claude-sonnet-test"
    assert client.cli_version == "2.1.test"


def test_structured_claude_api_output_uses_anthropic_key(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                id="msg_test",
                model="claude-sonnet-5-5",
                content=[SimpleNamespace(type="text", text=json.dumps(_claim_payload()))],
                usage=SimpleNamespace(
                    input_tokens=40,
                    output_tokens=20,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    output_tokens_details=SimpleNamespace(thinking_tokens=5),
                ),
            )

    class FakeAnthropic:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    fake_module = ModuleType("anthropic")
    fake_module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    client = ClaudeEvaluatorClient(
        _settings(
            CLAUDE_ACCESS_METHOD="api",
            ANTHROPIC_API_KEY="test-anthropic-key",
        )
    )
    result, log = client.generate_structured(
        task="script_claims",
        system_instruction="Extract claims.",
        prompt="Example script.",
        schema=ClaimSet,
    )

    assert captured["api_key"] == "test-anthropic-key"
    assert captured["kwargs"]["model"] == "claude-sonnet-5-5"
    assert captured["kwargs"]["output_config"]["format"]["type"] == "json_schema"
    assert result.claims[0].claim_id == "c1"
    assert log.provider == "anthropic"
    assert log.metadata["access_method"] == "api"
    assert log.prompt_tokens == 40
    assert log.output_tokens == 15
    assert log.thinking_tokens == 5
    assert log.input_cost_usd == 40 * 2.0 / 1_000_000
    assert log.output_cost_usd == 20 * 10.0 / 1_000_000
    assert log.total_cost_usd > 0


def test_claude_sonnet_standard_rate_defaults():
    settings = _settings()
    assert settings.anthropic_pricing() == (2.0, 10.0)
