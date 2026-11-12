import pytest
from pydantic import ValidationError

from evidence_aware_edu.config import Settings
from evidence_aware_edu.llm.factory import make_generation_client


def test_gemini_needs_an_api_key():
    settings = Settings(GEMINI_API_KEY="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        settings.validate_generation()


def test_removed_generation_provider_is_rejected():
    with pytest.raises(ValidationError):
        Settings(GENERATION_PROVIDER="codex")


def test_factory_rejects_unknown_provider():
    # model_copy skips pydantic validation. The client factory still checks it.
    settings = Settings(GEMINI_API_KEY="test").model_copy(
        update={"generation_provider": "unused_provider"}
    )
    with pytest.raises(ValueError, match="Only Gemini"):
        make_generation_client(settings)
