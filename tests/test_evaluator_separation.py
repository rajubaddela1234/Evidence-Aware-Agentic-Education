import pytest

from evidence_aware_edu.config import Settings


def _settings(**updates):
    base = dict(
        GEMINI_API_KEY="dummy-google",
        GEMINI_MODEL="generation-model",
        CLAUDE_ACCESS_METHOD="cli",
        CLAUDE_EVALUATOR_MODEL="claude-sonnet-5-5",
    )
    base.update(updates)
    return Settings(**base)


def test_claude_cli_evaluator_is_accepted_without_api_key():
    settings = _settings(ANTHROPIC_API_KEY="")
    settings.validate_independent_evaluator()
    assert settings.evaluator_provider == "anthropic"
    assert settings.evaluator_access == "claude_code_cli"
    assert settings.evaluator_model == "claude-sonnet-5-5"


def test_claude_api_requires_an_api_key():
    with pytest.raises(ValueError):
        _settings(CLAUDE_ACCESS_METHOD="api", ANTHROPIC_API_KEY="").validate_independent_evaluator()


def test_claude_api_evaluator_is_accepted_with_key():
    settings = _settings(CLAUDE_ACCESS_METHOD="api", ANTHROPIC_API_KEY="test-key")
    settings.validate_independent_evaluator()
    assert settings.evaluator_access == "anthropic_api"


def test_missing_claude_evaluator_model_is_rejected():
    with pytest.raises(ValueError):
        _settings(CLAUDE_EVALUATOR_MODEL="").validate_independent_evaluator()


def test_too_short_claude_timeout_is_rejected():
    with pytest.raises(ValueError):
        _settings(CLAUDE_EVALUATOR_TIMEOUT_SECONDS=10).validate_independent_evaluator()
